"""IsaacLab evaluation runner.

UNEXECUTED in this environment: IsaacLab requires Omniverse / Isaac Sim, which
is not installed here. This module was written against the APIs reported by
inspecting ~/tron/IsaacLab's G1 locomotion task
(source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/
flat_env_cfg.py, and the existing bespoke benchmark
scripts/reinforcement_learning/rsl_rl/eval_locomotion.py), but has not been
run. Treat the Articulation API calls (set_joint_position_target /
write_data_to_sim) as the documented IsaacLab pattern, not as verified against
this repo's exact IsaacLab version.

Design: the Sonic planner has no Python bindings (see planner_client.py), so
it cannot be wrapped as an RSL-RL policy the way eval_locomotion.py's target
policy is. Instead this runner:
  1. Builds the existing `Isaac-Velocity-Flat-G1-Play-v0` scene/env config
     directly (isaaclab_tasks...config.g1.flat_env_cfg.G1FlatEnvCfg_PLAY),
     reusing its G1 asset, initial pose, ground plane, and push_robot event
     term -- so the *scene* is identical to the MuJoCo condition as far as
     IsaacLab and MuJoCo's respective physics engines allow.
  2. Does NOT use the manager-based env's action pipeline (that expects an
     RL policy's joint-position-delta actions). Instead it steps the
     underlying Articulation directly with PlannerClient.update()'s 50Hz
     joint-position chunks, mirroring the MuJoCo runner's PD-target tracking
     loop as closely as IsaacLab's API allows.
  3. Reuses the disturbance mechanism already implemented for G1 training,
     `isaaclab.envs.mdp.events.push_by_setting_velocity` (adds a sampled
     world-frame velocity to the root), configured here as a one-shot "mode"
     call at `apply_at_s` rather than the periodic training-time interval,
     to match the MuJoCo runner's push timing. NOTE: this is a *velocity*
     disturbance (kg-independent) rather than the MuJoCo runner's *force*
     disturbance (xfrc_applied over duration_s) -- IsaacLab's G1 event
     library does not currently expose a timed external-force push (only
     `apply_external_force_torque`, which sets a constant force/torque, and
     `push_by_setting_velocity`, an instantaneous velocity kick). This is a
     known, documented sim-to-sim asymmetry: converting an N*duration_s
     impulse into an equivalent velocity kick (dv = F*dt/mass) is possible
     but was not done here since G1's mass properties were not confirmed in
     this session -- flagged for follow-up rather than guessed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..common.command import HeadingIntegrator, VelocityCommand, velocity_command_to_movement_state
from ..common.config import DisturbanceCondition, EvalConfig, VelocityCondition
from ..common.episode import ResultsWriter
from ..common.metrics import EpisodeBuffer, EpisodeMetrics, quat_wxyz_to_euler
from ..common.planner_client import PlannerClient, make_planner_client


class IsaacLabEvalRunner:
    def __init__(self, config: EvalConfig, headless: bool = True):
        try:
            import gymnasium as gym
            import isaaclab_tasks  # noqa: F401  (registers gym ids)
            from isaaclab.envs import ManagerBasedRLEnv
        except ImportError as e:
            raise RuntimeError(
                "IsaacLabEvalRunner requires an IsaacLab / Isaac Sim installation "
                "(omni, isaaclab, isaaclab_tasks), not available in this environment."
            ) from e
        self._gym = gym
        self.config = config

        env_cfg = gym.spec("Isaac-Velocity-Flat-G1-Play-v0").kwargs["cfg"]
        env_cfg.scene.num_envs = 1
        env_cfg.sim.dt = config.sim_params.sim_timestep
        env_cfg.episode_length_s = config.sim_params.episode_duration_s
        # We drive joints directly; disable the training reward/action wrapper's
        # influence by leaving actions at the default posture each manager step.
        self.env: ManagerBasedRLEnv = gym.make("Isaac-Velocity-Flat-G1-Play-v0", cfg=env_cfg).unwrapped
        self.robot = self.env.scene["robot"]
        self.planner: Optional[PlannerClient] = None

    def _read_state(self):
        import torch
        root_state = self.robot.data.root_state_w[0]
        pos, quat = root_state[0:3], root_state[3:7]  # IsaacLab convention: (w,x,y,z) quat
        lin_vel_w, ang_vel_w = root_state[7:10], root_state[10:13]
        quat_np = quat.detach().cpu().numpy()
        roll, pitch, yaw = quat_wxyz_to_euler(quat_np)
        c, s = np.cos(-yaw), np.sin(-yaw)
        lv = lin_vel_w.detach().cpu().numpy()
        heading_vx = c * lv[0] - s * lv[1]
        heading_vy = s * lv[0] + c * lv[1]
        av = ang_vel_w.detach().cpu().numpy()
        joint_pos = self.robot.data.joint_pos[0].detach().cpu().numpy()
        joint_vel = self.robot.data.joint_vel[0].detach().cpu().numpy()
        joint_torque = self.robot.data.applied_torque[0].detach().cpu().numpy()
        return {
            "base_height": float(pos[2].item()), "roll": roll, "pitch": pitch, "yaw": yaw,
            "act_vx": float(heading_vx), "act_vy": float(heading_vy), "act_yaw_rate": float(av[2]),
            "ang_vel_norm": float(np.linalg.norm(av)), "lin_vel_norm": float(np.linalg.norm(lv)),
            "joint_pos": joint_pos, "joint_vel": joint_vel, "joint_torque": joint_torque,
        }

    def _apply_push(self, dist: DisturbanceCondition) -> None:
        from isaaclab.envs.mdp.events import push_by_setting_velocity
        import torch
        if dist.axis == "x":
            vr = {"x": (dist.force_n / 100.0, dist.force_n / 100.0)}
        elif dist.axis == "y":
            vr = {"y": (dist.force_n / 100.0, dist.force_n / 100.0)}
        else:
            vr = {"x": (-dist.force_n / 100.0, dist.force_n / 100.0),
                  "y": (-dist.force_n / 100.0, dist.force_n / 100.0)}
        push_by_setting_velocity(self.env, torch.tensor([0], device=self.env.device), velocity_range=vr)

    def run_episode(self, vel_cond: VelocityCondition, seed: int,
                     dist_cond: Optional[DisturbanceCondition] = None) -> tuple[EpisodeMetrics, EpisodeBuffer]:
        import torch
        self.env.seed(seed)
        obs, _ = self.env.reset()

        self.planner = make_planner_client(self.config.planner_backend, self.config.planner_model_path)
        state0 = self._read_state()
        quat0 = self.robot.data.root_state_w[0, 3:7].detach().cpu().numpy()
        self.planner.initialize(quat0, state0["joint_pos"])

        heading = HeadingIntegrator()
        cmd = VelocityCommand(vx=vel_cond.vx, vy=vel_cond.vy, yaw_rate=vel_cond.yaw_rate,
                               height=vel_cond.height, locomotion_mode=vel_cond.locomotion_mode)

        buf = EpisodeBuffer()
        sim_dt = self.env.sim.get_physics_dt()
        control_dt = 1.0 / self.config.sim_params.control_frequency
        duration = self.config.sim_params.episode_duration_s
        n_steps = int(duration / sim_dt)
        replan_every = max(1, int(control_dt / sim_dt))
        decimation = self.env.cfg.decimation

        joint_target = state0["joint_pos"].copy()
        gen_frame, pushed = 0, False
        default_action = torch.zeros(self.env.action_space.shape, device=self.env.device)

        for step in range(n_steps):
            t = step * sim_dt
            if step % replan_every == 0:
                heading.step(cmd.yaw_rate, control_dt)
                mstate = velocity_command_to_movement_state(cmd, heading.heading_rad)
                chunk = self.planner.update(mstate, gen_frame)
                gen_frame += len(chunk["joint_pos"])
                joint_target = chunk["joint_pos"][0]

            self.robot.set_joint_position_target(
                torch.as_tensor(joint_target, device=self.env.device).unsqueeze(0)
            )
            self.robot.write_data_to_sim()

            if dist_cond is not None and not pushed and t >= dist_cond.apply_at_s:
                self._apply_push(dist_cond)
                pushed = True

            if step % decimation == 0:
                self.env.step(default_action)
            else:
                self.env.sim.step(render=False)
                self.env.scene.update(dt=sim_dt)

            state = self._read_state()
            buf.append(
                t=t, cmd_vx=cmd.vx, cmd_vy=cmd.vy, cmd_yaw_rate=cmd.yaw_rate,
                act_vx=state["act_vx"], act_vy=state["act_vy"], act_yaw_rate=state["act_yaw_rate"],
                base_height=state["base_height"], roll=state["roll"], pitch=state["pitch"], yaw=state["yaw"],
                ang_vel_norm=state["ang_vel_norm"], lin_vel_norm=state["lin_vel_norm"],
                joint_vel_abs_sum=float(np.sum(np.abs(state["joint_vel"]))),
                joint_torque_abs_sum=float(np.sum(np.abs(state["joint_torque"]))),
                mech_power=float(np.sum(np.abs(state["joint_torque"] * state["joint_vel"]))),
                contact_state={},  # IsaacLab foot-contact wiring left to a ContactSensor lookup, not done here
            )
            if state["base_height"] < 0.2 and not buf.fell:
                buf.fell = True
                buf.fall_time = t
                break

        cond_name = vel_cond.name if dist_cond is None else f"{vel_cond.name}__{dist_cond.name}"
        metrics = EpisodeMetrics.from_buffer(buf, condition=cond_name, seed=seed)
        self.planner.close()
        return metrics, buf

    def close(self) -> None:
        self.env.close()


def run_isaaclab_evaluation(config: EvalConfig, with_disturbance: bool, headless: bool = True) -> None:
    runner = IsaacLabEvalRunner(config, headless=headless)
    writer = ResultsWriter(config.output_dir, "isaaclab")

    dist_conditions = config.disturbance_conditions if with_disturbance else [None]
    for vel_cond in config.velocity_conditions:
        for dist_cond in dist_conditions:
            for seed in config.seeds:
                metrics, buf = runner.run_episode(vel_cond, seed, dist_cond)
                cmd_fields = {"vx": vel_cond.vx, "vy": vel_cond.vy, "yaw_rate": vel_cond.yaw_rate}
                extra = {"force_n": dist_cond.force_n} if dist_cond else {}
                writer.log_episode(metrics, buf, cmd_fields, extra)

    summary_df = writer.finalize()
    from ..common.plotting import generate_all_summary_plots
    generate_all_summary_plots(summary_df, writer.plots_dir)
    runner.close()
