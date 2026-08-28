#!/usr/bin/env python3
"""Compare two backend result directories (MuJoCo vs IsaacLab) side by side.

    python compare_sims.py --results results --backends mujoco isaaclab

Writes ``<results>/comparison/`` with the sim-to-sim plots and a per-condition
delta table (``comparison.csv``) so a planner change can be judged for both
engines at once.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gear_sonic_eval.core.plots import plot_comparison  # noqa: E402
from gear_sonic_eval.core.results import read_csv, write_csv  # noqa: E402

COMPARE_KEYS = [
    "vx_actual_mean", "vx_rmse", "vx_mae", "vy_rmse", "yaw_rate_rmse",
    "base_height_mean", "base_height_std", "tilt_rms_deg", "success_rate", "fall_rate",
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results", help="Root results directory.")
    p.add_argument("--backends", nargs=2, default=["mujoco", "isaaclab"])
    p.add_argument("--out", default=None, help="Output dir (default <results>/comparison).")
    args = p.parse_args(argv)

    root = Path(args.results)
    a, b = args.backends
    out = Path(args.out) if args.out else root / "comparison"
    out.mkdir(parents=True, exist_ok=True)

    rows_a = {r["condition"]: r for r in read_csv(root / a / "summary.csv")}
    rows_b = {r["condition"]: r for r in read_csv(root / b / "summary.csv")}
    shared = [c for c in rows_a if c in rows_b]
    if not shared:
        print(f"no shared conditions between {a} and {b}")
        return 1

    table = []
    for cond in shared:
        ra, rb = rows_a[cond], rows_b[cond]
        row = {"condition": cond, "cmd_vx": ra.get("cmd_vx"), "cmd_vy": ra.get("cmd_vy"),
               "cmd_yaw_rate": ra.get("cmd_yaw_rate")}
        for key in COMPARE_KEYS:
            va, vb = ra.get(key), rb.get(key)
            row[f"{a}_{key}"] = va
            row[f"{b}_{key}"] = vb
            try:
                row[f"delta_{key}"] = float(vb) - float(va)
            except (TypeError, ValueError):
                row[f"delta_{key}"] = float("nan")
        table.append(row)

    path = write_csv(out / "comparison.csv", table)
    print(f"wrote {path} ({len(table)} conditions)")
    for plot in plot_comparison({a: root / a, b: root / b}, out):
        print(f"  plot {plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
