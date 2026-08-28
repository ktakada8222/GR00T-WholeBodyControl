"""Metric computation shared by the MuJoCo and IsaacLab evaluation runners.

All functions operate on plain numpy arrays so they have no dependency on
either simulator backend. An EpisodeBuffer accumulates per-step samples during
a rollout; EpisodeMetrics.from_buffer() reduces them into the scalar metrics
written to CSV.

Only metrics that are actually derivable from the available simulator state
are implemented (see the investigation notes in ../README-less docstring at
the top of evaluate_sonic_planner.py). In particular:
  - Mechanical power / energy IS computed, since joint torque and joint
    velocity are both available from MuJoCo (mj_data.actuator_force,
    mj_data.qvel) and from IsaacLab (Articulation.data.applied_torque,
    joint_vel).
  - Foot slip requires foot contact state + contact-point velocity; only
    implemented where the runner supplies contact_state and foot velocities.
  - Step length / step frequency are estimated from contact-state transitions
    (heel-strike timestamps) and are approximate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err)))) if len(err) else float("nan")


def mae(err: np.ndarray) -> float:
    return float(np.mean(np.abs(err))) if len(err) else float("nan")


def max_abs(err: np.ndarray) -> float:
    return float(np.max(np.abs(err))) if len(err) else float("nan")


@dataclass
class EpisodeBuffer:
    """Per-step samples collected during one episode rollout."""

    t: list = field(default_factory=list)
    cmd_vx: list = field(default_factory=list)
    cmd_vy: list = field(default_factory=list)
    cmd_yaw_rate: list = field(default_factory=list)
    act_vx: list = field(default_factory=list)
    act_vy: list = field(default_factory=list)
    act_yaw_rate: list = field(default_factory=list)
    base_height: list = field(default_factory=list)
    roll: list = field(default_factory=list)
    pitch: list = field(default_factory=list)
    yaw: list = field(default_factory=list)
    ang_vel_norm: list = field(default_factory=list)
    lin_vel_norm: list = field(default_factory=list)
    joint_vel_abs_sum: list = field(default_factory=list)
    joint_torque_abs_sum: list = field(default_factory=list)
    mech_power: list = field(default_factory=list)
    contact_state: list = field(default_factory=list)  # e.g. dict{"left":bool,"right":bool} per step
    fell: bool = False
    fall_time: float = float("nan")

    def append(self, **kwargs) -> None:
        for k, v in kwargs.items():
            getattr(self, k).append(v)


@dataclass
class EpisodeMetrics:
    condition: str
    seed: int
    success: bool
    fell: bool
    time_to_fall: float

    vx_rmse: float
    vx_mae: float
    vx_max_err: float
    vy_rmse: float
    vy_mae: float
    vy_max_err: float
    yaw_rate_rmse: float
    yaw_rate_mae: float
    yaw_rate_max_err: float

    base_height_mean: float
    base_height_std: float
    roll_std: float
    pitch_std: float
    settling_time: float  # time from episode start until tracking error enters +/-10% band and stays

    mech_energy_j: float  # integral of |power| dt, only if torque+vel available (else nan)
    mean_joint_torque_abs: float

    step_frequency_hz: float
    duty_factor: float

    @staticmethod
    def from_buffer(buf: EpisodeBuffer, condition: str, seed: int,
                     settle_band: float = 0.1, min_settle_time: float = 0.5) -> "EpisodeMetrics":
        t = np.asarray(buf.t)
        cmd_vx, act_vx = np.asarray(buf.cmd_vx), np.asarray(buf.act_vx)
        cmd_vy, act_vy = np.asarray(buf.cmd_vy), np.asarray(buf.act_vy)
        cmd_wz, act_wz = np.asarray(buf.cmd_yaw_rate), np.asarray(buf.act_yaw_rate)

        err_vx = act_vx - cmd_vx
        err_vy = act_vy - cmd_vy
        err_wz = act_wz - cmd_wz

        base_height = np.asarray(buf.base_height)
        roll = np.asarray(buf.roll)
        pitch = np.asarray(buf.pitch)

        cmd_speed = float(np.hypot(cmd_vx[-1], cmd_vy[-1])) if len(cmd_vx) else 0.0
        settling_time = _settling_time(t, err_vx, cmd_speed, settle_band, min_settle_time)

        mech_power = np.asarray(buf.mech_power) if buf.mech_power else np.array([])
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        mech_energy = float(_trapz(np.abs(mech_power), t)) if len(mech_power) == len(t) and len(t) > 1 else float("nan")
        mean_torque = float(np.mean(buf.joint_torque_abs_sum)) if buf.joint_torque_abs_sum else float("nan")

        step_freq, duty = _gait_metrics(t, buf.contact_state)

        return EpisodeMetrics(
            condition=condition,
            seed=seed,
            success=not buf.fell,
            fell=buf.fell,
            time_to_fall=buf.fall_time,
            vx_rmse=rmse(err_vx), vx_mae=mae(err_vx), vx_max_err=max_abs(err_vx),
            vy_rmse=rmse(err_vy), vy_mae=mae(err_vy), vy_max_err=max_abs(err_vy),
            yaw_rate_rmse=rmse(err_wz), yaw_rate_mae=mae(err_wz), yaw_rate_max_err=max_abs(err_wz),
            base_height_mean=float(np.mean(base_height)) if len(base_height) else float("nan"),
            base_height_std=float(np.std(base_height)) if len(base_height) else float("nan"),
            roll_std=float(np.std(roll)) if len(roll) else float("nan"),
            pitch_std=float(np.std(pitch)) if len(pitch) else float("nan"),
            settling_time=settling_time,
            mech_energy_j=mech_energy,
            mean_joint_torque_abs=mean_torque,
            step_frequency_hz=step_freq,
            duty_factor=duty,
        )


def _settling_time(t: np.ndarray, err: np.ndarray, cmd_speed: float,
                    band_frac: float, min_hold: float) -> float:
    if len(t) < 2 or cmd_speed <= 1e-6:
        return float("nan")
    band = max(band_frac * cmd_speed, 0.02)
    inside = np.abs(err) <= band
    dt = float(np.median(np.diff(t)))
    hold_steps = max(1, int(min_hold / dt))
    for i in range(len(inside)):
        if np.all(inside[i:i + hold_steps]) and i + hold_steps <= len(inside):
            return float(t[i])
    return float("nan")


def _gait_metrics(t: np.ndarray, contact_state: list):
    """Estimate step frequency [Hz] and duty factor from a list of per-step
    contact dicts {"left": bool, "right": bool}. Returns (nan, nan) if no
    contact data was recorded."""
    if not contact_state or len(contact_state) != len(t):
        return float("nan"), float("nan")
    left = np.array([c.get("left", False) for c in contact_state])
    right = np.array([c.get("right", False) for c in contact_state])
    duration = float(t[-1] - t[0]) if len(t) > 1 else float("nan")
    if not duration or duration <= 0:
        return float("nan"), float("nan")

    def count_strikes(contact: np.ndarray) -> int:
        rising = np.diff(contact.astype(int)) == 1
        return int(np.sum(rising))

    strikes = count_strikes(left) + count_strikes(right)
    step_freq = strikes / duration if duration > 0 else float("nan")
    duty = float(np.mean(left | right)) if len(left) else float("nan")
    return step_freq, duty


def quat_wxyz_to_euler(quat_wxyz: np.ndarray):
    """Roll/pitch/yaw (rad) from a [w,x,y,z] quaternion (MuJoCo / IsaacLab
    convention). Returns a tuple (roll, pitch, yaw)."""
    w, x, y, z = quat_wxyz
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return float(roll), float(pitch), float(yaw)
