"""Result layout, CSV/JSON writers and per-condition aggregation.

Layout (identical for every backend, so MuJoCo and IsaacLab runs can be diffed
file-by-file)::

    results/<backend>/
        config.json          # the exact EvalConfig used (reproducibility)
        run_info.json        # backend, timestamps, code versions, notes
        episodes.csv         # one row per episode, all metrics
        summary.csv          # one row per (condition, push) cell, aggregated
        velocity_tracking.csv
        stability.csv
        timeseries/<episode_id>.csv
        plots/*.png

Only the stdlib ``csv`` module and numpy are used, so the writers work in a bare
IsaacSim python as well as in the gear_sonic conda env.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import datetime as _dt
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

TRACKING_COLUMNS = [
    "condition", "group", "cmd_vx", "cmd_vy", "cmd_yaw_rate", "push_force", "push_direction",
    "n", "success_rate",
    "vx_actual_mean", "vx_steady_err", "vx_mae", "vx_rmse", "vx_max_abs_err",
    "vy_actual_mean", "vy_steady_err", "vy_mae", "vy_rmse", "vy_max_abs_err",
    "yaw_rate_actual_mean", "yaw_rate_steady_err", "yaw_rate_mae", "yaw_rate_rmse", "yaw_rate_max_abs_err",
    "vx_rise_time", "vx_settling_time", "vx_overshoot_pct",
]

STABILITY_COLUMNS = [
    "condition", "group", "cmd_vx", "cmd_vy", "cmd_yaw_rate", "push_force", "push_direction", "n",
    "success_rate", "base_height_mean", "base_height_std", "roll_mean", "pitch_mean",
    "roll_std", "pitch_std", "yaw_std", "tilt_rms_deg", "angvel_xy_rms", "vz_std",
    "time_to_fall", "recovery_time",
]


def episode_id(row: dict) -> str:
    push = "" if row.get("push_direction", "none") == "none" else "_push{:g}{}".format(
        row.get("push_force", 0.0), row.get("push_direction")
    )
    return f"{row['condition']}{push}_ep{row['episode_index']}_seed{row['seed']}"


class ResultWriter:
    """Creates the directory tree and writes every artefact of one run."""

    def __init__(self, output_dir: str | Path, backend: str, config=None, run_info: dict | None = None):
        self.root = Path(output_dir) / backend
        self.timeseries_dir = self.root / "timeseries"
        self.plots_dir = self.root / "plots"
        for d in (self.root, self.timeseries_dir, self.plots_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.rows: list[dict] = []
        if config is not None:
            (self.root / "config.json").write_text(
                json.dumps(config.to_dict() if hasattr(config, "to_dict") else asdict(config), indent=2)
            )
        info = {"backend": backend, "started": _dt.datetime.now().isoformat(timespec="seconds")}
        info.update(run_info or {})
        (self.root / "run_info.json").write_text(json.dumps(info, indent=2))

    # ------------------------------------------------------------- recording
    def add_episode(self, row: dict, timeseries: dict[str, np.ndarray] | None = None) -> None:
        self.rows.append(row)
        if timeseries:
            self.write_timeseries(episode_id(row), timeseries)

    def write_timeseries(self, name: str, series: dict[str, np.ndarray]) -> Path:
        path = self.timeseries_dir / f"{name}.csv"
        keys = [k for k, v in series.items() if getattr(v, "size", 0) > 0]
        if not keys:
            return path
        length = len(series[keys[0]])
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for i in range(length):
                w.writerow([_fmt(series[k][i]) for k in keys])
        return path

    # -------------------------------------------------------------- finalize
    def finalize(self) -> dict[str, Path]:
        paths = {}
        paths["episodes"] = write_csv(self.root / "episodes.csv", self.rows)
        summary = aggregate(self.rows)
        paths["summary"] = write_csv(self.root / "summary.csv", summary)
        paths["velocity_tracking"] = write_csv(
            self.root / "velocity_tracking.csv", [_project(r, TRACKING_COLUMNS) for r in summary], TRACKING_COLUMNS
        )
        paths["stability"] = write_csv(
            self.root / "stability.csv", [_project(r, STABILITY_COLUMNS) for r in summary], STABILITY_COLUMNS
        )
        return paths


def _project(row: dict, cols: Sequence[str]) -> dict:
    return {c: row.get(c, "") for c in cols}


def _fmt(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_csv(path: Path, rows: Iterable[dict], columns: Sequence[str] | None = None) -> Path:
    rows = list(rows)
    if not rows:
        path.write_text("")
        return path
    if columns is None:
        columns = list(rows[0].keys())
        for r in rows:  # union, keeps first-seen order
            for k in r:
                if k not in columns:
                    columns.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in columns})
    return path


def aggregate(rows: list[dict]) -> list[dict]:
    """Mean over episodes within each (condition, push force, push direction) cell.

    Fall statistics are aggregated as rates; every other numeric column is a
    nan-aware mean over the episodes of the cell.
    """
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["condition"], r.get("push_force", 0.0), r.get("push_direction", "none"))
        cells.setdefault(key, []).append(r)

    out = []
    for (condition, force, direction), group in cells.items():
        agg = {
            "condition": condition,
            "group": group[0].get("group", ""),
            "backend": group[0].get("backend", ""),
            "cmd_vx": group[0]["cmd_vx"],
            "cmd_vy": group[0]["cmd_vy"],
            "cmd_yaw_rate": group[0]["cmd_yaw_rate"],
            "push_force": force,
            "push_direction": direction,
            "push_impulse": group[0].get("push_impulse", 0.0),
            "n": len(group),
            "success_rate": float(np.mean([1.0 if g["success"] else 0.0 for g in group])),
            "fall_rate": float(np.mean([1.0 if g["fell"] else 0.0 for g in group])),
            # episodes that collapsed before the command was applied: they
            # measure the reset, not the planner, and must not be read as falls
            "invalid_rate": float(np.mean([1.0 if g.get("invalid") else 0.0 for g in group])),
        }
        skip = set(agg) | {"seed", "episode_index", "fell", "success", "invalid"}
        for key in group[0]:
            if key in skip:
                continue
            values = [g.get(key) for g in group]
            numeric = [float(v) for v in values if isinstance(v, (int, float, np.floating)) and not _isnan(v)]
            if numeric:
                agg[key] = float(np.mean(numeric))
            elif isinstance(values[0], str):
                agg[key] = values[0]
            else:
                agg[key] = float("nan")
        out.append(agg)
    return out


def _isnan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def read_csv(path: str | Path) -> list[dict]:
    """Read a result CSV back into dicts, converting numeric fields to float."""
    path = Path(path)
    if not path.exists() or not path.read_text().strip():
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if v == "":
                r[k] = float("nan")
                continue
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows
