#!/usr/bin/env python3
"""CLI entrypoint for evaluating the Gear Sonic locomotion planner's walking
performance, in MuJoCo and/or IsaacLab, under a shared set of velocity and
disturbance conditions defined in a YAML config.

Examples
--------
    python evaluate_sonic_planner.py --sim mujoco --config configs/walking_eval.yaml
    python evaluate_sonic_planner.py --sim isaaclab --config configs/walking_eval.yaml
    python evaluate_sonic_planner.py --sim mujoco --config configs/walking_eval.yaml --disturbance
    python evaluate_sonic_planner.py --sim both --config configs/walking_eval.yaml \\
        --output-dir results --seed 0 1 2 --headless

Results are written to <output-dir>/<sim>/{summary,velocity_tracking,stability,episodes}.csv
and <output-dir>/<sim>/plots/*.png. When --sim both is used, an additional
<output-dir>/comparison/plots/mujoco_vs_isaaclab_*.png is generated.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.common.config import EvalConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", choices=["mujoco", "isaaclab", "both"], required=True)
    p.add_argument("--config", required=True, help="path to a walking_eval.yaml-style config")
    p.add_argument("--disturbance", action="store_true", help="also run the disturbance conditions")
    p.add_argument("--num-episodes", type=int, default=None,
                   help="override: cap the number of velocity conditions evaluated (for quick smoke tests)")
    p.add_argument("--seed", type=int, nargs="+", default=None, help="override the config's seed list")
    p.add_argument("--output-dir", default=None, help="override the config's output_dir")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--visualize", dest="headless", action="store_false",
                   help="open a viewer (MuJoCo passive viewer / IsaacLab GUI)")
    p.add_argument("--gear-sonic-root", default=None,
                   help="path to the gear_sonic package root (defaults to the repo containing this script)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = EvalConfig.load(args.config)

    if args.seed:
        config.seeds = args.seed
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_episodes:
        config.velocity_conditions = config.velocity_conditions[: args.num_episodes]

    os.makedirs(config.output_dir, exist_ok=True)
    config.save(os.path.join(config.output_dir, "resolved_config.yaml"))

    gear_sonic_root = args.gear_sonic_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.sim in ("mujoco", "both"):
        from evaluation.runners.mujoco_runner import run_mujoco_evaluation
        print(f"[evaluate_sonic_planner] running MuJoCo evaluation "
              f"({len(config.velocity_conditions)} velocity conditions, "
              f"{len(config.seeds)} seeds, disturbance={args.disturbance})")
        run_mujoco_evaluation(config, gear_sonic_root, with_disturbance=args.disturbance, headless=args.headless)

    if args.sim in ("isaaclab", "both"):
        from evaluation.runners.isaaclab_runner import run_isaaclab_evaluation
        print(f"[evaluate_sonic_planner] running IsaacLab evaluation "
              f"({len(config.velocity_conditions)} velocity conditions, "
              f"{len(config.seeds)} seeds, disturbance={args.disturbance})")
        run_isaaclab_evaluation(config, with_disturbance=args.disturbance, headless=args.headless)

    if args.sim == "both":
        import pandas as pd
        from evaluation.common.plotting import plot_mujoco_vs_isaaclab
        m = pd.read_csv(os.path.join(config.output_dir, "mujoco", "summary.csv"))
        i = pd.read_csv(os.path.join(config.output_dir, "isaaclab", "summary.csv"))
        cmp_dir = os.path.join(config.output_dir, "comparison", "plots")
        os.makedirs(cmp_dir, exist_ok=True)
        plot_mujoco_vs_isaaclab(m, i, cmp_dir)
        print(f"[evaluate_sonic_planner] sim-to-sim comparison plots written to {cmp_dir}")

    print(f"[evaluate_sonic_planner] done. results in {os.path.abspath(config.output_dir)}")


if __name__ == "__main__":
    main()
