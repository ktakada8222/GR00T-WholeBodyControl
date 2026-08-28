"""The export must satisfy exactly the schema report_locomotion.py reads."""

import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic_eval.core.isaaclab_report import ci95, export  # noqa: E402
from gear_sonic_eval.core.results import write_csv  # noqa: E402


def _rows():
    rows = []
    for cmd in (0.4, 0.8):
        for seed in range(3):
            rows.append({
                "backend": "test", "condition": f"forward_vx+{cmd:.2f}", "group": "forward",
                "waveform": "const", "axis": "vx", "frequency": 0.0,
                "seed": seed, "episode_index": seed,
                "cmd_vx": cmd, "cmd_vy": 0.0, "cmd_yaw_rate": 0.0,
                "push_force": 0.0, "push_direction": "none", "push_impulse": 0.0,
                "fell": False, "success": True,
                "vx_actual_mean": cmd - 0.05 - 0.01 * seed, "vx_steady_err": 0.05,
                "vx_rmse": 0.06, "vx_rise_time": 0.5, "vx_settling_time": 1.2,
                "vx_overshoot_pct": 3.0, "tilt_rms_deg": 2.0, "foot_slip": 0.05,
                "cost_of_transport": 0.4, "jerk_base": 50.0, "base_height_std": 0.01,
                "angvel_xy_rms": 0.1, "tracking_lag": float("nan"),
            })
    for freq in (0.25, 0.5):
        for seed in range(3):
            rows.append({
                "backend": "test", "condition": f"sine_vx_f{freq:.2f}", "group": "sine_vx",
                "waveform": "sine", "axis": "vx", "frequency": freq,
                "seed": seed, "episode_index": seed,
                "cmd_vx": 0.6, "cmd_vy": 0.0, "cmd_yaw_rate": 0.0,
                "push_force": 0.0, "push_direction": "none", "push_impulse": 0.0,
                "fell": False, "success": True, "tracking_lag": 0.1 + freq,
            })
    for force, direction in ((100.0, "front"), (200.0, "left")):
        for seed in range(3):
            rows.append({
                "backend": "test", "condition": "forward_vx+0.80", "group": "forward",
                "waveform": "const", "axis": "vx", "frequency": 0.0,
                "seed": seed, "episode_index": seed,
                "cmd_vx": 0.8, "cmd_vy": 0.0, "cmd_yaw_rate": 0.0,
                "push_force": force, "push_direction": direction,
                "push_impulse": force * 0.1,
                "fell": force > 150.0, "success": force <= 150.0,
                "recovery_time": 0.4,
            })
    for vx in (0.4, 0.8):
        for wz in (-0.5, 0.5):
            for seed in range(3):
                rows.append({
                    "backend": "test", "condition": f"circle_{vx:+.2f}_{wz:+.2f}",
                    "group": "circle", "waveform": "const", "axis": "vx", "frequency": 0.0,
                    "seed": seed, "episode_index": seed,
                    "cmd_vx": vx, "cmd_vy": 0.0, "cmd_yaw_rate": wz,
                    "push_force": 0.0, "push_direction": "none", "push_impulse": 0.0,
                    "fell": False, "success": True,
                    "vx_rmse": 0.07, "yaw_rate_rmse": 0.03, "tilt_rms_deg": 2.5,
                    "foot_slip": 0.06, "cost_of_transport": 0.45,
                })
    return rows


def _export_tmp():
    tmp = Path(tempfile.mkdtemp())
    write_csv(tmp / "episodes.csv", _rows())
    export(tmp, run_info={"mass_kg": 35.0})
    return dict(np.load(tmp / "eval.npz", allow_pickle=True)), json.loads((tmp / "eval.json").read_text())


def test_ci95_matches_reference():
    """Same estimator as eval_locomotion.ci95 (t-quantile, ddof=1)."""
    mean, half = ci95(np.array([[1.0], [2.0], [3.0]]))
    assert abs(mean[0] - 2.0) < 1e-12
    # n = 3 -> t quantile is looked up at n-1 = 2 (4.303), std uses ddof=1 (=1.0)
    assert abs(half[0] - 4.303 * 1.0 / math.sqrt(3)) < 1e-9
    _, zero = ci95(np.array([[5.0]]))
    assert zero[0] == 0.0


def test_keys_report_locomotion_reads():
    npz, summary = _export_tmp()
    for key in ("sweep_vx_points", "sweep_vx_v_axis_actual_mean", "sweep_vx_v_axis_actual_ci",
                "sweep_vx_cot_mean", "sweep_vx_tilt_rms_deg_mean", "sweep_vx_foot_slip_mean",
                "sweep_vx_fall_rate_mean", "step_vx_targets", "step_vx_rise_time_mean",
                "step_vx_settle_time_mean", "step_vx_overshoot_mean",
                "sine_vx_freqs", "sine_vx_tracking_lag_mean",
                "push_speeds", "push_dirs", "push_mags", "push_fall_grid_mean",
                "circle_speeds", "circle_yaws", "circle_fall_grid_mean"):
        assert key in npz, key
    for key in ("task", "axes", "seeds", "mass_kg", "power_convention", "overall"):
        assert key in summary, key
    for key in ("sweep_vx_steady_err", "sweep_vx_track_rmse", "sweep_vx_v_max", "sweep_vx_v_min",
                "step_vx_rise_time", "sine_vx_tracking_lag", "push_fall", "push_recover",
                "circle_fall", "circle_rmse_vx"):
        assert key in summary["overall"], key
        assert set(summary["overall"][key]) == {"mean", "ci95"}


def test_grid_shapes_and_values():
    npz, summary = _export_tmp()
    # push grid is [speeds, dirs, impulses]
    assert npz["push_fall_grid_mean"].shape == (
        len(npz["push_speeds"]), len(npz["push_dirs"]), len(npz["push_mags"]))
    # 200 N x 0.1 s = 20 N*s cell fell, 10 N*s cell did not
    mags = list(npz["push_mags"])
    grid = npz["push_fall_grid_mean"]
    assert np.nanmax(grid[:, :, mags.index(20.0)]) == 1.0
    assert np.nanmax(grid[:, :, mags.index(10.0)]) == 0.0
    assert npz["circle_fall_grid_mean"].shape == (2, 2)
    # sweep points sorted, achieved velocity averaged over the three seeds
    assert list(npz["sweep_vx_points"]) == [0.4, 0.8]
    assert abs(npz["sweep_vx_v_axis_actual_mean"][0] - (0.4 - 0.05 - 0.01)) < 1e-9
    assert list(npz["sine_vx_freqs"]) == [0.25, 0.5]
    assert abs(npz["sine_vx_tracking_lag_mean"][1] - 0.6) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
