"""Metric-layer tests on synthetic episodes with known answers."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic_eval.core.metrics import (  # noqa: E402
    EpisodeRecord,
    StepSample,
    compute_metrics,
    quat_to_rpy,
    rise_settling_overshoot,
)
from gear_sonic_eval.core.results import aggregate  # noqa: E402


def _episode(vx_actual, cmd_vx=1.0, n=500, dt=0.02, contacts=True):
    rec = EpisodeRecord(
        condition="c", group="forward", backend="test", seed=0, episode_index=0,
        cmd=(cmd_vx, 0.0, 0.0),
    )
    for i in range(n):
        t = i * dt
        rec.add(StepSample(
            t=t, cmd_vx=cmd_vx, cmd_vy=0.0, cmd_yaw_rate=0.0,
            base_pos=[vx_actual * t, 0.0, 0.8],
            base_quat=[1.0, 0.0, 0.0, 0.0],
            lin_vel_body=[vx_actual, 0.0, 0.0],
            ang_vel_body=[0.0, 0.0, 0.0],
            joint_vel=np.ones(29),
            joint_torque=np.full(29, 2.0),
            foot_contact=[float(math.sin(2 * math.pi * 1.5 * t) > 0),
                          float(math.sin(2 * math.pi * 1.5 * t) <= 0)] if contacts else None,
            foot_pos=[[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]] if contacts else None,
        ))
    rec.duration = n * dt
    return rec


def test_constant_tracking_error():
    row = compute_metrics(_episode(0.8, cmd_vx=1.0), transient=1.0, dt=0.02, mass=35.0)
    assert abs(row["vx_steady_err"] - 0.2) < 1e-9
    assert abs(row["vx_rmse"] - 0.2) < 1e-9
    assert abs(row["vx_mae"] - 0.2) < 1e-9
    assert row["success"] is True and row["fell"] is False


def test_energy_and_power():
    row = compute_metrics(_episode(1.0), transient=1.0, dt=0.02, mass=35.0)
    # 29 joints * |2 Nm * 1 rad/s| = 58 W
    assert abs(row["mech_power_mean"] - 58.0) < 1e-6
    assert abs(row["energy_total"] - 58.0 * 500 * 0.02) < 1e-6
    assert row["cost_of_transport"] > 0.0


def test_gait_metrics_from_contacts():
    row = compute_metrics(_episode(1.0), transient=1.0, dt=0.02, mass=35.0)
    assert abs(row["step_frequency"] - 1.5) < 0.1
    assert 0.4 < row["duty_factor"] < 0.6
    assert row["foot_slip"] == 0.0  # feet do not move in the synthetic episode


def test_missing_signals_stay_nan():
    row = compute_metrics(_episode(1.0, contacts=False), transient=1.0, dt=0.02, mass=None)
    assert math.isnan(row["step_frequency"])
    assert math.isnan(row["foot_slip"])
    assert math.isnan(row["cost_of_transport"])


def test_step_response():
    dt = 0.02
    t = np.arange(0, 5, dt)
    v = 1.0 * (1 - np.exp(-t / 0.3))  # first-order step, no overshoot
    out = rise_settling_overshoot(t, v, 1.0, 0.0)
    assert 0.5 < out["rise_time"] < 0.8  # 10->90% of a tau=0.3 s lag
    assert out["overshoot_pct"] < 1.0
    assert out["settling_time"] > 0.0


def test_quat_to_rpy_roundtrip():
    rpy = quat_to_rpy(np.array([[0.9238795, 0.0, 0.0, 0.3826834]]))[0]
    assert abs(rpy[2] - math.pi / 4) < 1e-6


def test_aggregate_success_rate():
    rows = [
        {"condition": "c", "cmd_vx": 1.0, "cmd_vy": 0.0, "cmd_yaw_rate": 0.0,
         "push_force": 0.0, "push_direction": "none", "seed": s, "episode_index": s,
         "fell": s == 0, "success": s != 0, "vx_rmse": 0.1 * (s + 1)}
        for s in range(4)
    ]
    agg = aggregate(rows)[0]
    assert agg["n"] == 4
    assert abs(agg["success_rate"] - 0.75) < 1e-9
    assert abs(agg["fall_rate"] - 0.25) < 1e-9
    assert abs(agg["vx_rmse"] - 0.25) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
