#!/usr/bin/env python3
"""Sonic planner walking benchmark -- entry point.

Examples
--------
    # MuJoCo (the deploy binary must already be running, see backends/mujoco_backend.py)
    python evaluate_sonic_planner.py --sim mujoco --config configs/walking_eval.yaml

    # add the push grid
    python evaluate_sonic_planner.py --sim mujoco --config configs/walking_eval.yaml --disturbance

    # IsaacLab (sim-to-sim), replaying planner trajectories
    python evaluate_sonic_planner.py --sim isaaclab --config configs/walking_eval.yaml --headless

    # plumbing smoke test, no simulator required
    python evaluate_sonic_planner.py --sim mock --config configs/smoke_eval.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running the file directly from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gear_sonic_eval.core.config import EvalConfig  # noqa: E402
from gear_sonic_eval.core.runner import run_evaluation  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", choices=["mujoco", "isaaclab", "mock"], required=True,
                   help="Which backend to evaluate.")
    p.add_argument("--config", required=True, help="YAML/JSON benchmark configuration.")
    p.add_argument("--output-dir", default="results", help="Root results directory.")
    p.add_argument("--num-episodes", type=int, default=None, help="Override repeats per condition.")
    p.add_argument("--seed", type=int, default=None, help="Override the base random seed.")
    p.add_argument("--episode-duration", type=float, default=None, help="Override episode length [s].")
    p.add_argument("--conditions", nargs="*", default=None,
                   help="Only run these condition names (or groups).")
    p.add_argument("--disturbance", action="store_true", help="Enable the push grid.")
    p.add_argument("--no-disturbance", action="store_true", help="Force the push grid off.")
    p.add_argument("--headless", action="store_true", help="No GUI (IsaacLab / MuJoCo viewer).")
    p.add_argument("--visualize", action="store_true", help="Open the simulator viewer.")
    p.add_argument("--real-time", action="store_true", help="Throttle MuJoCo to wall-clock time.")
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    p.add_argument("--backend-opt", action="append", default=[], metavar="KEY=VALUE",
                   help="Override one backend setting for this run, e.g. "
                        "--backend-opt physics_parity=default --backend-opt static_friction=1.0. "
                        "Repeatable. Applies to the selected --sim backend's config block.")
    p.add_argument("--check-order", action="store_true",
                   help="IsaacLab dds mode: print the DDS-motor -> Isaac-joint mapping at startup.")
    p.add_argument("--no-isaaclab-report", action="store_true",
                   help="Skip the eval.npz/eval.json export used by report_locomotion.py.")
    p.add_argument("--no-timeseries", action="store_true", help="Do not dump per-step CSVs.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the episode manifest and exit (no simulator needed).")
    return p


def apply_overrides(config: EvalConfig, args) -> EvalConfig:
    if args.num_episodes is not None:
        config.num_episodes = args.num_episodes
    if args.seed is not None:
        config.seed = args.seed
    if args.episode_duration is not None:
        config.sim.episode_duration = args.episode_duration
    if args.disturbance:
        config.disturbance.enabled = True
    if args.no_disturbance:
        config.disturbance.enabled = False
    if args.real_time:
        config.sim.real_time = True
    if args.backend_opt:
        block = config.backend.setdefault(args.sim, {})
        for item in args.backend_opt:
            if "=" not in item:
                raise SystemExit(f"--backend-opt expects KEY=VALUE, got {item!r}")
            key, _, raw = item.partition("=")
            block[key.strip()] = _coerce(raw.strip())
            print(f"  backend override: {key.strip()} = {block[key.strip()]!r}")
    if args.conditions:
        wanted = set(args.conditions)
        config.conditions = [c for c in config.conditions if c.name in wanted or c.group in wanted]
        if not config.conditions:
            raise SystemExit(f"no conditions match {sorted(wanted)}")
    return config


def _coerce(raw: str):
    """YAML-ish scalar parsing for --backend-opt values."""
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def make_backend(args, config: EvalConfig):
    if args.sim == "mock":
        from gear_sonic_eval.backends.mock import MockBackend

        return MockBackend(config)
    if args.sim == "mujoco":
        from gear_sonic_eval.backends.mujoco_backend import MujocoBackend

        return MujocoBackend(config, onscreen=args.visualize and not args.headless)
    mode = config.backend.get("isaaclab", {}).get("mode", "dds")
    headless = args.headless or not args.visualize
    if mode == "dds":
        # full Sonic stack (planner + WBC policy) driving the Isaac robot
        from gear_sonic_eval.backends.isaaclab_dds import IsaacLabDDSBackend

        if args.check_order:
            config.backend.setdefault("isaaclab", {})["check_order"] = True
        return IsaacLabDDSBackend(config, headless=headless)

    from gear_sonic_eval.backends.isaaclab_backend import IsaacLabBackend

    return IsaacLabBackend(config, headless=headless)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = apply_overrides(EvalConfig.from_file(args.config), args)

    episodes = config.episodes()
    print(f"benchmark '{config.name}': {len(config.conditions)} conditions, "
          f"{len(episodes)} episodes, backend={args.sim}")
    if args.dry_run:
        for ep in episodes:
            c = ep["condition"]
            push = ep["disturbance"]
            print(f"  {c.name:24s} vx={c.vx:+.2f} vy={c.vy:+.2f} wz={c.yaw_rate:+.2f} "
                  f"seed={ep['seed']} push={push and (push['force'], push['direction'])}")
        return 0

    backend = make_backend(args, config)
    try:
        writer = run_evaluation(
            backend, config, args.output_dir, save_timeseries=not args.no_timeseries
        )
    finally:
        backend.close()

    if not args.no_isaaclab_report:
        from gear_sonic_eval.core.isaaclab_report import export

        npz_path, json_path = export(
            writer.root, task=f"gear_sonic_planner_{args.sim}", run_info=backend.info
        )
        print(f"\nIsaacLab-format export: {npz_path.name}, {json_path.name}")
        print("  generate report.md (identical format to the G1 RL benchmark) with:")
        print(f"  python <IsaacLab>/scripts/reinforcement_learning/rsl_rl/report_locomotion.py \\")
        print(f"      --inputs sonic={writer.root} --out {args.output_dir}")

    if not args.no_plots:
        try:
            from gear_sonic_eval.core.plots import plot_all
        except ImportError as exc:
            print(f"\nskipping plots: {exc}")
            print("  the CSV results above are complete; install matplotlib and run")
            print(f"  python -m gear_sonic_eval.core.plots {writer.root}")
        else:
            print("\ngenerating plots:")
            plot_all(writer.root)
    return 0


if __name__ == "__main__":
    status = main()
    # The DDS subscriber threads and the MuJoCo passive viewer crash during
    # interpreter teardown (segfault after a completed run). Everything we own
    # is already closed and flushed at this point, so leave immediately instead.
    sys.stdout.flush()
    sys.stderr.flush()
    import os

    os._exit(status)
