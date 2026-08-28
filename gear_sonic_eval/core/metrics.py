"""Per-episode recording and metric computation.

Backend-agnostic on purpose: a backend only has to fill in a :class:`StepSample`
per control step, and every metric below is computed by the same code for MuJoCo
and IsaacLab.  Fields a backend cannot provide are left ``None`` and the
dependent metrics come out as ``nan`` -- nothing is estimated from data the
simulator did not give us.

Metric definitions follow ``IsaacLab/scripts/myscripts/eval.md`` (the existing G1
flat-walking benchmark) so numbers are comparable with the RL baseline:
steady-state error / RMSE after a settling window, rise & settling time and
overshoot for the startup transient, foot slip as contact-time foot speed, and
cost of transport from positive mechanical power.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Sequence

import numpy as np


@dataclass
class StepSample:
    """One control-step observation. ``None`` = not available from this backend."""

    t: float
    # command actually in force at this step (body frame)
    cmd_vx: float
    cmd_vy: float
    cmd_yaw_rate: float
    # base state
    base_pos: Sequence[float]  # world xyz
    base_quat: Sequence[float]  # world, (w, x, y, z)
    lin_vel_body: Sequence[float]  # body frame xyz
    ang_vel_body: Sequence[float]  # body frame xyz
    # optional
    joint_pos: Optional[Sequence[float]] = None
    joint_vel: Optional[Sequence[float]] = None
    joint_torque: Optional[Sequence[float]] = None
    foot_contact: Optional[Sequence[float]] = None  # per foot, 0/1
    foot_pos: Optional[Sequence[Sequence[float]]] = None  # per foot, world xyz
    pushed: bool = False
    # planner-side introspection (MuJoCo/deploy only)
    locomotion_mode: Optional[int] = None
    planner_speed: Optional[float] = None


@dataclass
class EpisodeRecord:
    """Accumulated samples plus episode-level bookkeeping."""

    condition: str
    group: str
    backend: str
    seed: int
    episode_index: int
    cmd: tuple[float, float, float]
    disturbance: Optional[dict] = None
    samples: list[StepSample] = field(default_factory=list)
    fell: bool = False
    fall_time: float = float("nan")
    #: "none" | "settle" (before the command was applied -- the episode never
    #: tested anything and is invalid) | "command".
    fall_phase: str = "none"
    #: Number of reset retries that preceded this (valid) episode.
    reset_retries: int = 0
    duration: float = 0.0
    notes: str = ""
    #: "const" or "sine"; sine episodes additionally yield a tracking lag.
    waveform: str = "const"
    #: Primary command axis ("vx" / "vy" / "yaw_rate"), set by the runner.
    axis: str = "vx"
    #: Sine frequency [Hz] (0 for constant commands).
    frequency: float = 0.0

    def add(self, sample: StepSample) -> None:
        self.samples.append(sample)

    # ------------------------------------------------------------------ arrays
    def array(self, attr: str) -> np.ndarray:
        vals = [getattr(s, attr) for s in self.samples]
        if not vals or any(v is None for v in vals):
            return np.zeros((0,))
        return np.asarray(vals, dtype=float)

    def timeseries(self) -> dict[str, np.ndarray]:
        """Flat per-step arrays, used for CSV dumps and time-series plots."""
        t = self.array("t")
        quat = self.array("base_quat")
        rpy = quat_to_rpy(quat) if quat.size else np.zeros((0, 3))
        pos = self.array("base_pos")
        lin = self.array("lin_vel_body")
        ang = self.array("ang_vel_body")
        out = {
            "t": t,
            "cmd_vx": self.array("cmd_vx"),
            "cmd_vy": self.array("cmd_vy"),
            "cmd_yaw_rate": self.array("cmd_yaw_rate"),
            "vx": lin[:, 0] if lin.size else np.zeros((0,)),
            "vy": lin[:, 1] if lin.size else np.zeros((0,)),
            "vz": lin[:, 2] if lin.size else np.zeros((0,)),
            "yaw_rate": ang[:, 2] if ang.size else np.zeros((0,)),
            "roll_rate": ang[:, 0] if ang.size else np.zeros((0,)),
            "pitch_rate": ang[:, 1] if ang.size else np.zeros((0,)),
            "base_height": pos[:, 2] if pos.size else np.zeros((0,)),
            "roll": rpy[:, 0] if rpy.size else np.zeros((0,)),
            "pitch": rpy[:, 1] if rpy.size else np.zeros((0,)),
            "yaw": rpy[:, 2] if rpy.size else np.zeros((0,)),
            "pushed": np.asarray([float(s.pushed) for s in self.samples]),
        }
        return out


# --------------------------------------------------------------------- helpers
def quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    """(w, x, y, z) -> (roll, pitch, yaw), ZYX convention."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1)


def tilt_angle(quat: np.ndarray) -> np.ndarray:
    """Angle [rad] between the body z axis and world z."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    body_z_world_z = 1.0 - 2.0 * (x * x + y * y)
    return np.arccos(np.clip(body_z_world_z, -1.0, 1.0))


def _nan() -> float:
    return float("nan")


def _rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2))) if err.size else _nan()


def rise_settling_overshoot(
    t: np.ndarray, v: np.ndarray, target: float, t0: float, band: float = 0.05
) -> dict[str, float]:
    """Step-response metrics of ``v`` against a step to ``target`` at ``t0``.

    ``eval.md`` 1-3 / 1-5: rise time is 10%->90% of the target, settling time the
    first instant after which |v - target| stays inside ``band * |target|``, and
    overshoot the largest excursion beyond the target, in percent.
    """
    out = {"rise_time": _nan(), "settling_time": _nan(), "overshoot_pct": _nan()}
    if v.size == 0 or abs(target) < 1e-6:
        return out
    mask = t >= t0
    tt, vv = t[mask], v[mask]
    if tt.size == 0:
        return out
    sgn = math.copysign(1.0, target)
    prog = vv * sgn
    goal = abs(target)

    def _first(th):
        idx = np.nonzero(prog >= th)[0]
        return tt[idx[0]] if idx.size else None

    t10, t90 = _first(0.1 * goal), _first(0.9 * goal)
    if t10 is not None and t90 is not None:
        out["rise_time"] = float(t90 - t10)

    inside = np.abs(vv - target) <= band * goal
    # last index where the signal leaves the band; settling starts after it
    outside = np.nonzero(~inside)[0]
    if inside.any():
        start = 0 if outside.size == 0 else outside[-1] + 1
        if start < tt.size:
            out["settling_time"] = float(tt[start] - t0)
    peak = float(np.max(prog)) if prog.size else 0.0
    out["overshoot_pct"] = float(max(0.0, (peak - goal) / goal) * 100.0)
    return out


def gait_metrics(rec: EpisodeRecord, dt: float, mask: np.ndarray) -> dict[str, float]:
    """Contact-derived gait quality. All-nan when the backend gives no contacts."""
    out = {
        "contact_fraction": _nan(),
        "duty_factor": _nan(),
        "step_frequency": _nan(),
        "step_length": _nan(),
        "foot_slip": _nan(),
        "double_support_fraction": _nan(),
        "flight_fraction": _nan(),
    }
    contact = rec.array("foot_contact")
    if contact.size == 0 or not mask.any():
        return out
    c = contact[mask]  # [T, F]
    out["contact_fraction"] = float(c.mean())
    out["duty_factor"] = float(c.mean())  # per-foot stance fraction of the cycle
    n_contact = c.sum(axis=1)
    out["double_support_fraction"] = float(np.mean(n_contact >= 2))
    out["flight_fraction"] = float(np.mean(n_contact == 0))

    # step frequency: rising contact edges per foot, averaged
    rises = np.sum((c[1:] > 0.5) & (c[:-1] <= 0.5), axis=0)
    span = max((c.shape[0] - 1) * dt, 1e-6)
    if c.shape[1] > 0:
        out["step_frequency"] = float(rises.mean() / span)

    lin = rec.array("lin_vel_body")[mask]
    speed = float(np.mean(np.linalg.norm(lin[:, :2], axis=1))) if lin.size else _nan()
    if out["step_frequency"] and out["step_frequency"] > 1e-6 and not math.isnan(speed):
        out["step_length"] = speed / out["step_frequency"]

    foot_pos = rec.array("foot_pos")  # [T, F, 3]
    if foot_pos.size:
        fp = foot_pos[mask]
        disp = np.linalg.norm(np.diff(fp[:, :, :2], axis=0), axis=-1)  # [T-1, F]
        stance = c[1:] > 0.5
        if stance.any():
            out["foot_slip"] = float(disp[stance].mean() / dt)
    return out


def base_jerk(pos: np.ndarray, dt: float, mask: np.ndarray) -> float:
    """Mean |3rd position difference| of the base [m/s^3].

    Same estimator as ``run_sweep`` in IsaacLab's ``eval_locomotion.py``:
    ``(p_t - 3 p_{t-1} + 3 p_{t-2} - p_{t-3}) / dt^3``.
    """
    if pos.shape[0] < 4:
        return _nan()
    j = (pos[3:] - 3 * pos[2:-1] + 3 * pos[1:-2] - pos[:-3]) / dt**3
    m = mask[3:] if mask.shape[0] == pos.shape[0] else np.ones(j.shape[0], dtype=bool)
    if not m.any():
        return _nan()
    return float(np.linalg.norm(j[m], axis=-1).mean())


def tracking_lag(cmd: np.ndarray, actual: np.ndarray, dt: float, max_lag: float = 0.6) -> float:
    """Cross-correlation lag [s] between command and response (eval.md 1-4).

    Identical search to ``run_sine``: both signals are mean-centred and the lag
    maximising their dot product over ``0..max_lag`` is returned.
    """
    n = min(cmd.size, actual.size)
    if n < 10:
        return _nan()
    c = cmd[:n] - cmd[:n].mean()
    if np.allclose(c, 0.0):  # constant command: lag is undefined
        return _nan()
    best, best_val = 0, -np.inf
    for lag in range(0, int(round(max_lag / dt)) + 1):
        if n - lag < 10:
            break
        a = actual[lag:n]
        corr = float(np.dot(c[: n - lag], a - a.mean()))
        if corr > best_val:
            best_val, best = corr, lag
    return float(best * dt)


def compute_metrics(rec: EpisodeRecord, *, transient: float, dt: float, mass: float | None = None) -> dict:
    """Reduce one episode to the flat scalar row written to the result CSVs."""
    ts = rec.timeseries()
    t = ts["t"]
    row: dict[str, float | str | bool] = {
        "backend": rec.backend,
        "condition": rec.condition,
        "group": rec.group,
        "seed": rec.seed,
        "episode_index": rec.episode_index,
        "cmd_vx": rec.cmd[0],
        "cmd_vy": rec.cmd[1],
        "cmd_yaw_rate": rec.cmd[2],
        "push_force": rec.disturbance["force"] if rec.disturbance else 0.0,
        "push_direction": rec.disturbance["direction"] if rec.disturbance else "none",
        "push_impulse": rec.disturbance["impulse"] if rec.disturbance else 0.0,
        "duration": rec.duration,
        "fell": bool(rec.fell and rec.fall_phase == "command"),
        "success": bool(not rec.fell),
        "invalid": bool(rec.fall_phase == "settle"),
        "fall_phase": rec.fall_phase,
        "reset_retries": rec.reset_retries,
        "time_to_fall": rec.fall_time,
        "num_samples": len(rec.samples),
    }
    if t.size == 0:
        return row

    mask = t >= (t[0] + transient)
    if not mask.any():  # episode terminated inside the transient window
        mask = np.ones_like(t, dtype=bool)

    # ------------------------------------------------------------- tracking
    for axis, key in (("vx", "vx"), ("vy", "vy"), ("yaw_rate", "yaw_rate")):
        actual = ts[key][mask]
        cmd = ts[f"cmd_{key}"][mask]
        err = cmd - actual
        row[f"{axis}_cmd"] = float(np.mean(cmd)) if cmd.size else _nan()
        row[f"{axis}_actual_mean"] = float(np.mean(actual)) if actual.size else _nan()
        row[f"{axis}_steady_err"] = float(np.mean(err)) if err.size else _nan()
        row[f"{axis}_mae"] = float(np.mean(np.abs(err))) if err.size else _nan()
        row[f"{axis}_rmse"] = _rmse(err)
        row[f"{axis}_max_abs_err"] = float(np.max(np.abs(err))) if err.size else _nan()
        row[f"{axis}_std"] = float(np.std(actual)) if actual.size else _nan()

    # ------------------------------------------------------------ stability
    quat = rec.array("base_quat")
    tilt = tilt_angle(quat) if quat.size else np.zeros((0,))
    row["base_height_mean"] = float(np.mean(ts["base_height"][mask]))
    row["base_height_std"] = float(np.std(ts["base_height"][mask]))
    row["roll_mean"] = float(np.mean(ts["roll"][mask]))
    row["pitch_mean"] = float(np.mean(ts["pitch"][mask]))
    row["roll_std"] = float(np.std(ts["roll"][mask]))
    row["pitch_std"] = float(np.std(ts["pitch"][mask]))
    row["yaw_std"] = float(np.std(np.unwrap(ts["yaw"][mask])))
    row["tilt_rms_deg"] = float(np.degrees(np.sqrt(np.mean(tilt[mask] ** 2)))) if tilt.size else _nan()
    ang = rec.array("ang_vel_body")
    row["angvel_xy_rms"] = (
        float(np.sqrt(np.mean(ang[mask][:, 0] ** 2 + ang[mask][:, 1] ** 2))) if ang.size else _nan()
    )
    row["vz_std"] = float(np.std(ts["vz"][mask]))

    # --------------------------------------------------------- startup / step
    t0 = float(t[0])
    for axis, key, target in (
        ("vx", "vx", rec.cmd[0]),
        ("vy", "vy", rec.cmd[1]),
        ("yaw_rate", "yaw_rate", rec.cmd[2]),
    ):
        step = rise_settling_overshoot(t, ts[key], target, t0)
        for k, v in step.items():
            row[f"{axis}_{k}"] = v

    pos = rec.array("base_pos")
    row["jerk_base"] = base_jerk(pos, dt, mask) if pos.size else _nan()

    row["waveform"] = rec.waveform
    row["axis"] = rec.axis
    row["frequency"] = rec.frequency
    if rec.waveform == "sine":
        key = rec.axis
        row["tracking_lag"] = tracking_lag(ts[f"cmd_{key}"], ts[key], dt)
    else:
        row["tracking_lag"] = _nan()

    # ------------------------------------------------------------------ gait
    row.update(gait_metrics(rec, dt, mask))

    # --------------------------------------------------- joints / energetics
    jv = rec.array("joint_vel")
    if jv.size:
        row["joint_vel_rms"] = float(np.sqrt(np.mean(jv[mask] ** 2)))
        row["joint_vel_max"] = float(np.max(np.abs(jv[mask])))
        acc = np.diff(jv, axis=0) / dt
        amask = mask[1:] if mask.size == jv.shape[0] else np.ones(acc.shape[0], dtype=bool)
        row["joint_acc_rms"] = float(np.sqrt(np.mean(acc[amask] ** 2))) if acc.size else _nan()
    else:
        row["joint_vel_rms"] = row["joint_vel_max"] = row["joint_acc_rms"] = _nan()

    tau = rec.array("joint_torque")
    if tau.size:
        row["joint_torque_rms"] = float(np.sqrt(np.mean(tau[mask] ** 2)))
        row["joint_torque_max"] = float(np.max(np.abs(tau[mask])))
    else:
        row["joint_torque_rms"] = row["joint_torque_max"] = _nan()

    if tau.size and jv.size and tau.shape == jv.shape:
        power = np.sum(np.abs(tau * jv), axis=1)  # |tau . dq|, absolute convention
        row["mech_power_mean"] = float(np.mean(power[mask]))
        row["energy_total"] = float(np.sum(power) * dt)
        lin = rec.array("lin_vel_body")
        dist = float(np.sum(np.linalg.norm(lin[mask][:, :2], axis=1)) * dt) if lin.size else 0.0
        if mass and dist > 1e-3:
            row["cost_of_transport"] = float(np.sum(power[mask]) * dt / (mass * 9.81 * dist))
        else:
            row["cost_of_transport"] = _nan()
    else:
        row["mech_power_mean"] = row["energy_total"] = row["cost_of_transport"] = _nan()

    # -------------------------------------------------------------- recovery
    row["recovery_time"] = _recovery_time(rec, ts)
    return row


def _recovery_time(rec: EpisodeRecord, ts: dict[str, np.ndarray], band: float = 0.15, hold: float = 0.5) -> float:
    """Time after the push until |v - v_cmd| stays inside ``band`` for ``hold`` s."""
    if not rec.disturbance:
        return _nan()
    pushed = ts["pushed"] > 0.5
    if not pushed.any():
        return _nan()
    t = ts["t"]
    t_push_end = float(t[np.nonzero(pushed)[0][-1]])
    mask = t > t_push_end
    if not mask.any() or rec.fell:
        return _nan()
    err = np.sqrt((ts["cmd_vx"][mask] - ts["vx"][mask]) ** 2 + (ts["cmd_vy"][mask] - ts["vy"][mask]) ** 2)
    tt = t[mask]
    dt = float(np.median(np.diff(tt))) if tt.size > 1 else 0.02
    need = max(int(round(hold / dt)), 1)
    inside = err <= band
    run = 0
    for i, ok in enumerate(inside):
        run = run + 1 if ok else 0
        if run >= need:
            return float(tt[i - need + 1] - t_push_end)
    return _nan()
