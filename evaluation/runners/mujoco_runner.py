"""MuJoCo evaluation runner.

Design decision (see investigation notes in evaluate_sonic_planner.py): the
production path is two OS processes (gear_sonic/scripts/run_sim_loop.py +
the compiled g1_deploy_onnx_ref binary) talking over Unitree DDS. That is the
right architecture for teleop/real-time use, but it is a poor fit for
scripted, many-episode automated evaluation (process lifecycle, DDS domain
setup, and non-determinism per episode). This runner therefore reuses the
same G1 MJCF asset and the same physics stepping pattern as
gear_sonic/utils/mujoco_sim/base_sim.py::DefaultEnv (mujoco.mj_step at
SIMULATE_DT, obs extraction identical to prepare_obs()), but drives it
in-process: PlannerClient.update() produces a 50Hz joint-position chunk,
which is tracked with a PD position controller each physics step, exactly
mirroring compute_body_torques() in base_sim.py. No DDS/ZMQ is used, so this
only exercises the ONNX or Mock planner backends -- the ZMQ backend is
intended for the two-process, DDS-based path and is not wired into this
in-process loop.

NOT EXECUTED in this environment: `mujoco` is not installed here. The MJCF
loading path and joint layout below follow
gear_sonic/utils/mujoco_sim/base_sim.py and
gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml exactly,
but have not been run against the real assets from this session.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ..common.command import HeadingIntegrator, VelocityCommand, velocity_command_to_movement_state
from ..common.config import DisturbanceCondition, EvalConfig, VelocityCondition
from ..common.episode import ResultsWriter
from ..common.metrics import EpisodeBuffer, EpisodeMetrics, quat_wxyz_to_euler
from ..common.planner_client import PlannerClient, make_planner_client

GEAR_SONIC_DEFAULT_XML = (
    "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
)


class MuJoCoEvalRunner:
    def __init__(self, config: EvalConfig, gear_sonic_root: str,
                 mjcf_relpath: str = GEAR_SONIC_DEFAULT_XML,
                 kp: float = 100.0, kd: float = 2.0, headless: bool = True):
        try:
            import mujoco
        except ImportError as e:
            raise RuntimeError(
                "MuJoCoEvalRunner requires the 'mujoco' package, not installed "
                "in this environment ('pip install mujoco')."
            ) from e
        self._mj = mujoco
        self.config = config
        self.kp, self.kd = kp, kd
        self.headless = headless

        xml_path = os.path.join(gear_sonic_root, mjcf_relpath)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = config.sim_params.sim_timestep
        self.data = mujoco.MjData(self.model)

        self.viewer = None
        if not headless:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        self.n_joints = self.model.nq - 7  # floating base = 7 qpos
        self.planner: Optional[PlannerClient] = None

    def _reset(self, seed: int) -> None:
        self._mj.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.79
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self._mj.mj_forward(self.model, self.data)

    def _read_state(self):
        base_quat = self.data.qpos[3:7].copy()
        roll, pitch, yaw = quat_wxyz_to_euler(base_quat)
        lin_vel = self.data.qvel[0:3].copy()
        ang_vel = self.data.qvel[3:6].copy()
        c, s = np.cos(-yaw), np.sin(-yaw)
        heading_vx = c * lin_vel[0] - s * lin_vel[1]
        heading_vy = s * lin_vel[0] + c * lin_vel[1]
        return {
            "base_height": float(self.data.qpos[2]),
            "roll": roll, "pitch": pitch, "yaw": yaw,
            "act_vx": float(heading_vx), "act_vy": float(heading_vy),
            "act_yaw_rate": float(ang_vel[2]),
            "ang_vel_norm": float(np.linalg.norm(ang_vel)),
            "lin_vel_norm": float(np.linalg.norm(lin_vel)),
            "joint_pos": self.data.qpos[7:7 + self.n_joints].copy(),
            "joint_vel": self.data.qvel[6:6 + self.n_joints].copy(),
            "joint_torque": self.data.actuator_force.copy() if self.model.nu else np.zeros(self.n_joints),
        }

    def _contact_state(self) -> dict:
        def touching(body_id):
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                b1 = self.model.geom_bodyid[c.geom1]
                b2 = self.model.geom_bodyid[c.geom2]
                if body_id in (b1, b2) and self.floor_geom_id in (c.geom1, c.geom2):
                    return True
            return False
        return {"left": touching(self.left_foot_id), "right": touching(self.right_foot_id)}

    def _apply_push(self, dist: DisturbanceCondition, t_in_episode: float) -> None:
        if dist is None or not (dist.apply_at_s <= t_in_episode < dist.apply_at_s + dist.duration_s):
            return
        if dist.axis == "x":
            force = np.array([dist.force_n, 0.0, 0.0])
        elif dist.axis == "y":
            force = np.array([0.0, dist.force_n, 0.0])
        else:  # random, fixed per-episode direction would be better; simple per-step random here
            theta = np.random.uniform(0, 2 * np.pi)
            force = dist.force_n * np.array([np.cos(theta), np.sin(theta), 0.0])
        self.data.xfrc_applied[self.torso_id, 0:3] = force

    def run_episode(self, vel_cond: VelocityCondition, seed: int,
                     dist_cond: Optional[DisturbanceCondition] = None) -> tuple[EpisodeMetrics, EpisodeBuffer]:
        np.random.seed(seed)
        self._reset(seed)
        self.planner = make_planner_client(self.config.planner_backend, self.config.planner_model_path)
        self.planner.initialize(self.data.qpos[3:7].copy(), self.data.qpos[7:7 + self.n_joints].copy())

        heading = HeadingIntegrator()
        cmd = VelocityCommand(vx=vel_cond.vx, vy=vel_cond.vy, yaw_rate=vel_cond.yaw_rate,
                               height=vel_cond.height, locomotion_mode=vel_cond.locomotion_mode)

        buf = EpisodeBuffer()
        dt = self.model.opt.timestep
        control_dt = 1.0 / self.config.sim_params.control_frequency
        duration = self.config.sim_params.episode_duration_s
        n_steps = int(duration / dt)
        replan_every = max(1, int(control_dt / dt))

        joint_target = self.data.qpos[7:7 + self.n_joints].copy()
        gen_frame = 0
        for step in range(n_steps):
            t = step * dt
            if step % replan_every == 0:
                heading.step(cmd.yaw_rate, control_dt)
                mstate = velocity_command_to_movement_state(cmd, heading.heading_rad)
                chunk = self.planner.update(mstate, gen_frame)
                gen_frame += len(chunk["joint_pos"])
                joint_target = chunk["joint_pos"][0]

            if self.model.nu:
                self.data.ctrl[:] = self.kp * (joint_target - self.data.qpos[7:7 + self.n_joints]) \
                    - self.kd * self.data.qvel[6:6 + self.n_joints]

            self._apply_push(dist_cond, t)
            self._mj.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()

            state = self._read_state()
            buf.append(
                t=t, cmd_vx=cmd.vx, cmd_vy=cmd.vy, cmd_yaw_rate=cmd.yaw_rate,
                act_vx=state["act_vx"], act_vy=state["act_vy"], act_yaw_rate=state["act_yaw_rate"],
                base_height=state["base_height"], roll=state["roll"], pitch=state["pitch"], yaw=state["yaw"],
                ang_vel_norm=state["ang_vel_norm"], lin_vel_norm=state["lin_vel_norm"],
                joint_vel_abs_sum=float(np.sum(np.abs(state["joint_vel"]))),
                joint_torque_abs_sum=float(np.sum(np.abs(state["joint_torque"]))),
                mech_power=float(np.sum(np.abs(state["joint_torque"] * state["joint_vel"]))),
                contact_state=self._contact_state(),
            )

            if state["base_height"] < 0.2 and not buf.fell:
                buf.fell = True
                buf.fall_time = t
                break

        cond_name = vel_cond.name if dist_cond is None else f"{vel_cond.name}__{dist_cond.name}"
        metrics = EpisodeMetrics.from_buffer(buf, condition=cond_name, seed=seed)
        self.planner.close()
        return metrics, buf


def run_mujoco_evaluation(config: EvalConfig, gear_sonic_root: str, with_disturbance: bool,
                           headless: bool = True) -> None:
    runner = MuJoCoEvalRunner(config, gear_sonic_root, headless=headless)
    writer = ResultsWriter(config.output_dir, "mujoco")

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
