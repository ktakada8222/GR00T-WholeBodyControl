"""Export benchmark results in IsaacLab's ``eval.npz`` / ``eval.json`` schema.

The point is to reuse the existing report generator verbatim::

    python IsaacLab/scripts/reinforcement_learning/rsl_rl/report_locomotion.py \\
        --inputs sonic=results/mujoco rl=IsaacLab/results/2026-08-07_19-01-19_g1_flat_v1 \\
        --out results

so the Sonic planner is reported in exactly the same format, with the same
metric names, the same tables and the same plots as the G1 RL benchmark -- and
can be put side by side with it in one report.

Schema produced (mirrors ``eval_locomotion.py::main``):

* ``sweep_{ax}_points`` + ``sweep_{ax}_{metric}_{mean,ci}`` for
  ``v_axis_actual, steady_err, track_rmse, tilt_rms_deg, foot_slip, cot,
  jerk_base, fall_rate, height_std, angvel_rms``
* ``step_{ax}_targets`` + ``step_{ax}_{rise_time,settle_time,overshoot}_{mean,ci}``
* ``sine_{ax}_freqs`` + ``sine_{ax}_tracking_lag_{mean,ci}``
* ``push_speeds / push_dirs / push_mags`` + ``push_{fall,recover}_grid_{mean,ci}``
* ``circle_speeds / circle_yaws`` + ``circle_{rmse_vx,rmse_wz,tilt_deg,foot_slip,
  cot,fall}_grid_{mean,ci}``
* ``eval.json``: ``overall[key] = {mean, ci95}`` for every curve/grid/scalar.

Scenario mapping (differences from the RL benchmark are intentional and listed
in the README):

* A constant-command condition feeds **both** ``sweep_{ax}`` (steady-state
  window) and ``step_{ax}`` (the 0 -> target transient at the start of the
  episode), because every Sonic episode begins from a standstill.  The RL
  benchmark runs those as two separate scenarios.
* ``ci95`` is taken across the episodes of a condition (each has its own seed),
  which is the same "across seeds" statistic the RL benchmark reports.
* ``push_mags`` are impulses (force x duration), matching the RL benchmark's
  ``--push_impulses`` axis.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import warnings

import numpy as np

from gear_sonic_eval.core.results import read_csv

#: Student's t 0.975 quantile, copied from eval_locomotion.py so the confidence
#: intervals of the two benchmarks are computed identically.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228}

AX_ORDER = ["vx", "vy", "wz"]
#: our internal axis names -> the report's axis names
AXIS_OUT = {"vx": "vx", "vy": "vy", "yaw_rate": "wz"}
CMD_KEY = {"vx": "cmd_vx", "vy": "cmd_vy", "yaw_rate": "cmd_yaw_rate"}

#: sweep curves: report key -> our episodes.csv column (axis-formatted)
SWEEP_CURVES = {
    "v_axis_actual": "{ax}_actual_mean",
    "steady_err": "{ax}_steady_err",
    "track_rmse": "{ax}_rmse",
    "tilt_rms_deg": "tilt_rms_deg",
    "foot_slip": "foot_slip",
    "cot": "cost_of_transport",
    "jerk_base": "jerk_base",
    "height_std": "base_height_std",
    "angvel_rms": "angvel_xy_rms",
}
STEP_CURVES = {
    "rise_time": "{ax}_rise_time",
    "settle_time": "{ax}_settling_time",
    "overshoot": "{ax}_overshoot_pct",
}
CIRCLE_GRIDS = {
    "circle_rmse_vx_grid": "vx_rmse",
    "circle_rmse_wz_grid": "yaw_rate_rmse",
    "circle_tilt_deg_grid": "tilt_rms_deg",
    "circle_foot_slip_grid": "foot_slip",
    "circle_cot_grid": "cost_of_transport",
}
PUSH_DIR_ORDER = ["front", "back", "left", "right"]


def ci95(values: np.ndarray, axis: int = 0):
    """mean and 95% CI half-width across ``axis`` -- same as eval_locomotion.py."""
    n = values.shape[axis]
    # all-nan slices are expected (a metric a backend cannot provide); they must
    # propagate as nan, not as a warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(values, axis=axis)
        if n < 2:
            return mean, np.zeros_like(mean)
        std = np.nanstd(values, axis=axis, ddof=1)
    tcrit = _T975.get(n - 1, 1.96)
    return mean, tcrit * std / math.sqrt(n)


def _val(row: dict, key: str) -> float:
    v = row.get(key, float("nan"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _cells(rows: list[dict], key):
    """Group episode rows by ``key(row)``, preserving first-seen order."""
    out: dict = {}
    for r in rows:
        out.setdefault(key(r), []).append(r)
    return out


def _stack(cells: dict, order: list, column: str) -> np.ndarray:
    """[n_episodes, n_points] array of one column, padded with nan."""
    depth = max(len(cells[p]) for p in order)
    arr = np.full((depth, len(order)), np.nan)
    for j, point in enumerate(order):
        for i, row in enumerate(cells[point]):
            arr[i, j] = _val(row, column)
    return arr


def export(
    result_root: str | Path,
    *,
    label: str = "sonic_planner",
    task: str = "gear_sonic_planner",
    run_info: dict | None = None,
) -> tuple[Path, Path]:
    """Write ``eval.npz`` and ``eval.json`` next to the result CSVs."""
    root = Path(result_root)
    episodes = read_csv(root / "episodes.csv")
    if not episodes:
        raise ValueError(f"no episodes.csv in {root}")

    info = run_info or {}
    npz: dict = {}
    overall: dict = {}

    unpushed = [r for r in episodes if str(r.get("push_direction", "none")) == "none"]
    pushed = [r for r in episodes if str(r.get("push_direction", "none")) not in ("none", "")]

    _export_sweep_and_step(unpushed, npz, overall)
    _export_sine(unpushed, npz, overall)
    _export_push(pushed, npz, overall)
    _export_circle(unpushed, npz, overall)

    seeds = sorted({int(_val(r, "seed")) for r in episodes})
    npz["scenarios"] = np.array(sorted({k.split("_")[0] for k in npz if k.endswith("_points")
                                        or k.endswith("_targets") or k.endswith("_freqs")} |
                                       ({"push"} if pushed else set()) |
                                       ({"circle"} if "circle_fall_grid_mean" in npz else set())))
    npz["axes"] = np.array(AX_ORDER)
    npz["seeds"] = np.array(seeds)
    npz["mass"] = float(info.get("mass_kg", float("nan")))
    npz["power_convention"] = "abs"

    summary = {
        "task": task,
        "checkpoint": info.get("planner_model", info.get("requires", label)),
        "axes": AX_ORDER,
        "seeds": seeds,
        "mass_kg": float(info.get("mass_kg", float("nan"))),
        # our mechanical power is sum|tau * dq|; the RL benchmark defaults to
        # "positive" (clamped), so CoT is only comparable when both use "abs".
        "power_convention": "abs",
        "run_dir": str(root),
        "overall": overall,
    }

    npz_path, json_path = root / "eval.npz", root / "eval.json"
    np.savez(npz_path, **npz)
    json_path.write_text(json.dumps(summary, indent=2))
    return npz_path, json_path


# ----------------------------------------------------------------- scenarios
def _record_curve(npz, overall, name, points, stack, exclude_zero=False):
    mean, half = ci95(stack)
    npz[f"{name}_mean"] = mean
    npz[f"{name}_ci"] = half
    cols = stack[:, np.abs(points) > 1e-6] if exclude_zero else stack
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_seed = np.nanmean(cols, axis=1) if cols.size else np.array([np.nan])
    m, h = ci95(per_seed[:, None])
    overall[name] = {"mean": float(m[0]), "ci95": float(h[0])}


def _export_sweep_and_step(rows: list[dict], npz: dict, overall: dict) -> None:
    """Constant-command conditions -> sweep (steady) + step (transient) curves."""
    const = [r for r in rows
             if str(r.get("waveform", "const")) == "const"
             and str(r.get("group", "")) not in ("circle", "compound", "custom")]
    for axis_in, ax in AXIS_OUT.items():
        sub = [r for r in const if str(r.get("axis", "vx")) == axis_in]
        if not sub:
            continue
        cells = _cells(sub, lambda r: round(_val(r, CMD_KEY[axis_in]), 6))
        points = sorted(cells)
        pts = np.array(points, dtype=float)
        npz[f"sweep_{ax}_points"] = pts
        npz[f"step_{ax}_targets"] = pts

        for out_key, col in SWEEP_CURVES.items():
            stack = _stack(cells, points, col.format(ax=axis_in))
            _record_curve(npz, overall, f"sweep_{ax}_{out_key}", pts, stack,
                          exclude_zero=out_key in ("cot", "steady_err"))
        fall = _stack(cells, points, "fell")
        _record_curve(npz, overall, f"sweep_{ax}_fall_rate", pts, np.nan_to_num(fall, nan=0.0))

        # achieved-velocity extremes, as in run_sweep's scalars
        achieved = npz[f"sweep_{ax}_v_axis_actual_mean"]
        if achieved.size:
            for key, value in (("v_max", np.nanmax(achieved)), ("v_min", np.nanmin(achieved))):
                overall[f"sweep_{ax}_{key}"] = {"mean": float(value), "ci95": 0.0}

        for out_key, col in STEP_CURVES.items():
            stack = _stack(cells, points, col.format(ax=axis_in))
            _record_curve(npz, overall, f"step_{ax}_{out_key}", pts, stack)


def _export_sine(rows: list[dict], npz: dict, overall: dict) -> None:
    sine = [r for r in rows if str(r.get("waveform", "const")) == "sine"]
    for axis_in, ax in AXIS_OUT.items():
        sub = [r for r in sine if str(r.get("axis", "vx")) == axis_in]
        if not sub:
            continue
        cells = _cells(sub, lambda r: str(r["condition"]))
        order = sorted(cells, key=lambda c: _val(cells[c][0], "frequency"))
        freqs = np.array([_val(cells[c][0], "frequency") for c in order], dtype=float)
        npz[f"sine_{ax}_freqs"] = freqs
        _record_curve(npz, overall, f"sine_{ax}_tracking_lag", freqs,
                      _stack(cells, order, "tracking_lag"))
        _record_curve(npz, overall, f"sine_{ax}_fall_rate", freqs,
                      np.nan_to_num(_stack(cells, order, "fell"), nan=0.0))


def _export_push(rows: list[dict], npz: dict, overall: dict) -> None:
    """Push episodes -> [speed, direction, impulse] fall/recover grids."""
    if not rows:
        return
    speeds = sorted({round(_val(r, "cmd_vx"), 6) for r in rows})
    dirs = [d for d in PUSH_DIR_ORDER
            if d in {str(r["push_direction"]) for r in rows}]
    dirs += sorted({str(r["push_direction"]) for r in rows} - set(dirs))
    mags = sorted({round(_val(r, "push_impulse"), 6) for r in rows})
    shape = (len(speeds), len(dirs), len(mags))

    depth = 1
    index: dict = {}
    for r in rows:
        key = (round(_val(r, "cmd_vx"), 6), str(r["push_direction"]), round(_val(r, "push_impulse"), 6))
        index.setdefault(key, []).append(r)
        depth = max(depth, len(index[key]))

    fall = np.full((depth, *shape), np.nan)
    rec = np.full((depth, *shape), np.nan)
    for (spd, direction, mag), group in index.items():
        s, d, m = speeds.index(spd), dirs.index(direction), mags.index(mag)
        for i, row in enumerate(group):
            fall[i, s, d, m] = 1.0 if str(row.get("fell")).lower() in ("true", "1", "1.0") else 0.0
            rec[i, s, d, m] = _val(row, "recovery_time")

    npz["push_speeds"] = np.array(speeds, dtype=float)
    npz["push_dirs"] = np.array(dirs)
    npz["push_mags"] = np.array(mags, dtype=float)
    for name, stack in (("push_fall_grid", fall), ("push_recover_grid", rec)):
        mean, half = ci95(stack)
        npz[f"{name}_mean"] = mean
        npz[f"{name}_ci"] = half
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            overall[name[:-5]] = {"mean": float(np.nanmean(stack)), "ci95": 0.0}


def _export_circle(rows: list[dict], npz: dict, overall: dict) -> None:
    """Compound vx+wz grid -> the circle scenario's grids."""
    circle = [r for r in rows if str(r.get("group", "")) == "circle"]
    if not circle:
        return
    speeds = sorted({round(_val(r, "cmd_vx"), 6) for r in circle})
    yaws = sorted({round(_val(r, "cmd_yaw_rate"), 6) for r in circle})
    index: dict = {}
    depth = 1
    for r in circle:
        key = (round(_val(r, "cmd_vx"), 6), round(_val(r, "cmd_yaw_rate"), 6))
        index.setdefault(key, []).append(r)
        depth = max(depth, len(index[key]))

    npz["circle_speeds"] = np.array(speeds, dtype=float)
    npz["circle_yaws"] = np.array(yaws, dtype=float)
    grids = dict(CIRCLE_GRIDS)
    grids["circle_fall_grid"] = "fell"
    for name, column in grids.items():
        stack = np.full((depth, len(speeds), len(yaws)), np.nan)
        for (vx, wz), group in index.items():
            i, j = speeds.index(vx), yaws.index(wz)
            for k, row in enumerate(group):
                if column == "fell":
                    stack[k, i, j] = 1.0 if str(row.get("fell")).lower() in ("true", "1", "1.0") else 0.0
                else:
                    stack[k, i, j] = _val(row, column)
        mean, half = ci95(stack)
        npz[f"{name}_mean"] = mean
        npz[f"{name}_ci"] = half
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            overall[name[:-5]] = {"mean": float(np.nanmean(stack)), "ci95": 0.0}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("result_dir", help="results/<backend> directory")
    ap.add_argument("--task", default="gear_sonic_planner")
    ap.add_argument("--label", default="sonic_planner")
    args = ap.parse_args()
    info_path = Path(args.result_dir) / "run_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    for path in export(args.result_dir, label=args.label, task=args.task, run_info=info):
        print(f"wrote {path}")
