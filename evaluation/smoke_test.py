#!/usr/bin/env python3
"""Framework smoke test: exercises config parsing, command conversion, the
MockPlannerClient, metric computation, CSV writing, and plot generation
end-to-end WITHOUT requiring mujoco or IsaacLab (neither is installed in this
environment). This does NOT validate the Sonic planner or either physics
backend -- see evaluate_sonic_planner.py for that, once mujoco/onnxruntime or
an IsaacLab install are available.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from evaluation.common.command import HeadingIntegrator, VelocityCommand, velocity_command_to_movement_state
from evaluation.common.config import EvalConfig
from evaluation.common.episode import ResultsWriter
from evaluation.common.metrics import EpisodeBuffer, EpisodeMetrics, quat_wxyz_to_euler
from evaluation.common.planner_client import MockPlannerClient
from evaluation.common.plotting import generate_all_summary_plots


def run_kinematic_episode(vel_cond, seed, dist_cond=None, duration_s=6.0, dt=0.02):
    np.random.seed(seed)
    planner = MockPlannerClient()
    planner.initialize(np.array([1.0, 0, 0, 0]), np.zeros(29))

    heading = HeadingIntegrator()
    cmd = VelocityCommand(vx=vel_cond["vx"], vy=vel_cond["vy"], yaw_rate=vel_cond["yaw_rate"])
    buf = EpisodeBuffer()

    pos = np.zeros(3)
    act_vel = np.zeros(2)
    n_steps = int(duration_s / dt)
    gen_frame = 0
    for step in range(n_steps):
        t = step * dt
        heading.step(cmd.yaw_rate, dt)
        mstate = velocity_command_to_movement_state(cmd, heading.heading_rad)
        chunk = planner.update(mstate, gen_frame)
        gen_frame += len(chunk["joint_pos"])

        target = np.array(mstate["movement_direction"][:2]) * mstate["movement_speed"]
        act_vel += (dt / 0.3) * (target - act_vel)
        pos[:2] += act_vel * dt

        if dist_cond and dist_cond["apply_at_s"] <= t < dist_cond["apply_at_s"] + dist_cond["duration_s"]:
            act_vel += 0.02 * dist_cond["force_n"] * np.array([0, 1.0])

        c, s = np.cos(-heading.heading_rad), np.sin(-heading.heading_rad)
        hvx = c * act_vel[0] - s * act_vel[1]
        hvy = s * act_vel[0] + c * act_vel[1]
        height = 0.79 - 0.02 * min(1.0, np.linalg.norm(act_vel) / 3.0) * (1 if not dist_cond else 1.5)

        fell = height < 0.4
        buf.append(
            t=t, cmd_vx=cmd.vx, cmd_vy=cmd.vy, cmd_yaw_rate=cmd.yaw_rate,
            act_vx=hvx, act_vy=hvy, act_yaw_rate=cmd.yaw_rate * 0.95,
            base_height=height, roll=0.01 * act_vel[1], pitch=0.01 * act_vel[0], yaw=heading.heading_rad,
            ang_vel_norm=abs(cmd.yaw_rate), lin_vel_norm=float(np.linalg.norm(act_vel)),
            joint_vel_abs_sum=0.0, joint_torque_abs_sum=0.0, mech_power=0.0,
            contact_state={"left": step % 25 < 12, "right": step % 25 >= 12},
        )
        if fell and not buf.fell:
            buf.fell, buf.fall_time = True, t
            break

    cond_name = vel_cond["name"] if dist_cond is None else f"{vel_cond['name']}__{dist_cond['name']}"
    return EpisodeMetrics.from_buffer(buf, condition=cond_name, seed=seed), buf


def main():
    config = EvalConfig.load(os.path.join(os.path.dirname(__file__), "configs", "walking_eval.yaml"))
    config.output_dir = os.path.join(os.path.dirname(__file__), "results_smoke")
    config.seeds = [0, 1]
    config.velocity_conditions = config.velocity_conditions[:8]
    config.disturbance_conditions = config.disturbance_conditions[:3]

    writer = ResultsWriter(config.output_dir, "mock")
    for vc in config.velocity_conditions:
        vel_cond = {"name": vc.name, "vx": vc.vx, "vy": vc.vy, "yaw_rate": vc.yaw_rate}
        for seed in config.seeds:
            metrics, buf = run_kinematic_episode(vel_cond, seed)
            writer.log_episode(metrics, buf, {"vx": vc.vx, "vy": vc.vy, "yaw_rate": vc.yaw_rate})
        for dc in config.disturbance_conditions:
            dist_cond = {"name": dc.name, "force_n": dc.force_n, "duration_s": dc.duration_s, "apply_at_s": dc.apply_at_s}
            for seed in config.seeds:
                metrics, buf = run_kinematic_episode(vel_cond, seed, dist_cond)
                writer.log_episode(metrics, buf, {"vx": vc.vx, "vy": vc.vy, "yaw_rate": vc.yaw_rate},
                                    {"force_n": dc.force_n})

    summary_df = writer.finalize()
    generate_all_summary_plots(summary_df, writer.plots_dir)
    print(f"OK: wrote {len(summary_df)} episode rows to {writer.root}")
    print(summary_df[["condition", "seed", "vx_rmse", "success"]].head(10).to_string())


if __name__ == "__main__":
    main()
