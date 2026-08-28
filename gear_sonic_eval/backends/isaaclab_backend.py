"""IsaacLab backend for sim-to-sim comparison.

Two modes, because the Sonic stack cannot be dropped into IsaacLab unchanged
(the planner and the WBC policy are a C++/TensorRT process that speaks Unitree
DDS; there is no DDS<->IsaacLab bridge in this repo):

``trajectory`` (default, planner-faithful)
    Replays a **Sonic planner trajectory** in IsaacLab.  The trajectory is the
    50 Hz ``MotionSequence`` the deploy binary already dumps with
    ``--planner-motion-logfile`` (see ``g1_deploy_onnx_ref.cpp``: body position,
    body quaternion, then 29 joint angles in MuJoCo order, one line per frame,
    blocks separated by a blank line).  Joint positions are commanded as PD
    targets on the IsaacLab G1 articulation, so the *same planner output* is
    executed by a second physics engine and scored with the same metrics.
    This is the sim-to-sim comparison the benchmark is for.

``velocity_policy`` (reference baseline)
    Drives an ``Isaac-Velocity-*-G1-Play`` env with a trained RL locomotion
    policy under the identical command grid.  It does **not** run the Sonic
    planner; it is the baseline the planner is compared *against*, and it reuses
    the existing benchmark in ``IsaacLab/scripts/reinforcement_learning/rsl_rl/
    eval_locomotion.py`` conceptually (same metric definitions from eval.md).

Both modes need IsaacSim; the imports are deferred so importing this module on a
laptop is harmless.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gear_sonic_eval.backends.base import EvalBackend
from gear_sonic_eval.core.commands import MovementState
from gear_sonic_eval.core.metrics import StepSample


def load_planner_motion_csv(path: str | Path, block: int = -1) -> dict[str, np.ndarray]:
    """Parse a ``--planner-motion-logfile`` dump into arrays.

    Each blank-line-separated block is one snapshot of the whole planner motion
    buffer; ``block=-1`` (default) takes the last, i.e. the most complete one.
    Returns ``{"body_pos": [T,3], "body_quat": [T,4] (w,x,y,z),
    "joint_pos": [T,29] (MuJoCo order)}``.
    """
    blocks: list[list[list[float]]] = [[]]
    for line in Path(path).read_text().splitlines():
        line = line.strip().rstrip(",")
        if not line:
            if blocks[-1]:
                blocks.append([])
            continue
        blocks[-1].append([float(v) for v in line.split(",") if v != ""])
    blocks = [b for b in blocks if b]
    if not blocks:
        raise ValueError(f"no frames found in {path}")
    arr = np.asarray(blocks[block], dtype=float)
    if arr.shape[1] < 7 + 29:
        raise ValueError(f"expected >=36 columns per frame, got {arr.shape[1]}")
    return {"body_pos": arr[:, 0:3], "body_quat": arr[:, 3:7], "joint_pos": arr[:, 7:36]}


class IsaacLabBackend(EvalBackend):
    """Common IsaacLab plumbing; mode-specific behaviour in ``step``/``reset``."""

    name = "isaaclab"

    def __init__(self, config, *, headless: bool = True):
        super().__init__(config)
        cfg = dict(config.backend.get("isaaclab", {}))
        self.mode = cfg.get("mode", "trajectory")
        self.task = cfg.get("task", "Isaac-Velocity-Flat-G1-Play-v0")
        self.checkpoint = cfg.get("checkpoint")
        self.trajectory_dir = cfg.get("trajectory_dir")
        self.headless = headless
        self.env = None
        self.info = {"backend": "isaaclab", "mode": self.mode, "task": self.task}

        if self.mode not in ("trajectory", "velocity_policy"):
            raise ValueError(f"unknown isaaclab mode: {self.mode}")

        self._launch()

    # ----------------------------------------------------------------- setup
    def _launch(self) -> None:
        from isaaclab.app import AppLauncher  # noqa: PLC0415

        self._app = AppLauncher({"headless": self.headless}).app

        import gymnasium as gym  # noqa: PLC0415
        import isaaclab_tasks  # noqa: F401,PLC0415
        from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415

        env_cfg = parse_env_cfg(self.task, device="cuda:0", num_envs=1)
        env_cfg.sim.dt = self.config.sim.physics_dt
        env_cfg.decimation = max(int(round(self.config.sim.control_dt / self.config.sim.physics_dt)), 1)
        # Disable the env's own command resampling and pushes: the benchmark
        # runner owns both, so MuJoCo and IsaacLab see the same schedule.
        for attr in ("base_external_force_torque", "push_robot"):
            if hasattr(env_cfg, "events") and hasattr(env_cfg.events, attr):
                setattr(env_cfg.events, attr, None)
        if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None

        self.env = gym.make(self.task, cfg=env_cfg).unwrapped
        self.robot = self.env.scene["robot"]
        self.dt = self.config.sim.control_dt
        self.mass = float(self.robot.root_physx_view.get_masses()[0].sum().item())
        self.info["episode_dt"] = self.dt

        if self.mode == "velocity_policy":
            self._policy = _load_policy(self.checkpoint)
            self._cmd_term = self.env.command_manager.get_term("base_velocity")
        else:
            self._trajectory = None
            self._traj_index = 0
        self._push_force = np.zeros(3)
        self.t = 0.0

    # --------------------------------------------------------------- episode
    def reset(self, seed: int) -> None:
        import torch  # noqa: PLC0415

        self.env.seed(seed)
        self.env.reset()
        init = self.config.init
        root = self.robot.data.default_root_state.clone()
        root[:, 0:2] = torch.tensor(init.base_xy, device=root.device)
        root[:, 2] = init.base_height
        root[:, 3:7] = torch.tensor(_yaw_quat(init.base_yaw), device=root.device)
        root[:, 7:10] = torch.tensor(init.base_lin_vel, device=root.device)
        root[:, 10:13] = torch.tensor(init.base_ang_vel, device=root.device)
        self.robot.write_root_state_to_sim(root)
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos.clone(), self.robot.data.default_joint_vel.clone()
        )
        self.t = 0.0
        self._push_force = np.zeros(3)
        self._traj_index = 0

    def begin_episode(self, condition, disturbance=None) -> None:
        """In ``trajectory`` mode, load ``<trajectory_dir>/<condition>.csv``.

        Those files are produced by running the deploy binary with
        ``--planner-motion-logfile`` under the same benchmark conditions (see
        ``tools/export_planner_trajectories.md``), which is what makes the
        IsaacLab run a replay of the *same* planner output as the MuJoCo run.
        """
        if self.mode != "trajectory":
            return
        if not self.trajectory_dir:
            raise RuntimeError(
                "isaaclab trajectory mode requires backend.isaaclab.trajectory_dir"
            )
        path = Path(self.trajectory_dir) / f"{condition.name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"no planner trajectory for condition '{condition.name}': {path}"
            )
        self.set_trajectory(path)

    def send_movement_state(self, state: MovementState) -> None:
        """Consume the planner command.

        ``velocity_policy``: the MovementState is converted back to the (vx, vy,
        wz) twist the RL command term expects.
        ``trajectory``: the command selects which pre-generated planner
        trajectory is replayed (one per benchmark condition).
        """
        if self.mode == "velocity_policy":
            import torch  # noqa: PLC0415

            vx, vy, wz = _movement_state_to_twist(state, self._yaw(), self.config.sim.command_dt)
            self._cmd_term.vel_command_b[:, 0] = torch.as_tensor(vx, device=self.env.device)
            self._cmd_term.vel_command_b[:, 1] = torch.as_tensor(vy, device=self.env.device)
            self._cmd_term.vel_command_b[:, 2] = torch.as_tensor(wz, device=self.env.device)

    def set_trajectory(self, path: str | Path) -> None:
        """Load the planner trajectory replayed by ``trajectory`` mode."""
        self._trajectory = load_planner_motion_csv(path)
        self._traj_index = 0
        self.info["trajectory"] = str(path)

    def step(self) -> StepSample:
        import torch  # noqa: PLC0415

        if np.linalg.norm(self._push_force) > 0.0:
            forces = torch.zeros((1, 1, 3), device=self.env.device)
            forces[0, 0] = torch.tensor(self._push_force, device=self.env.device)
            self.robot.set_external_force_and_torque(
                forces, torch.zeros_like(forces), body_ids=[0]
            )

        if self.mode == "velocity_policy":
            with torch.inference_mode():
                obs = self.env.observation_manager.compute()["policy"]
                action = self._policy(obs)
                self.env.step(action)
        else:
            if self._trajectory is None:
                raise RuntimeError("trajectory mode requires set_trajectory() before stepping")
            joint_pos = self._trajectory["joint_pos"]
            idx = min(self._traj_index, joint_pos.shape[0] - 1)
            target = torch.tensor(joint_pos[idx], dtype=torch.float32, device=self.env.device)
            self.robot.set_joint_position_target(target.unsqueeze(0))
            self.robot.write_data_to_sim()
            self.env.sim.step()
            self.robot.update(self.dt)
            self._traj_index += 1

        self.t += self.dt
        return self._observe()

    def apply_push(self, force_world, duration: float) -> None:
        self._push_force = np.asarray(force_world, dtype=float)

    def base_yaw(self) -> float:
        return self._yaw()

    def _yaw(self) -> float:
        q = self.robot.data.root_quat_w[0].cpu().numpy()
        w, x, y, z = q
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def _observe(self) -> StepSample:
        d = self.robot.data
        contact_sensor = self.env.scene.sensors.get("contact_forces") if hasattr(self.env.scene, "sensors") else None
        contacts = None
        foot_pos = None
        if contact_sensor is not None:
            ids, _ = self.robot.find_bodies(".*ankle_roll.*")
            if ids:
                forces = contact_sensor.data.net_forces_w[0, ids, :].cpu().numpy()
                contacts = (np.linalg.norm(forces, axis=-1) > 1.0).astype(float).tolist()
                foot_pos = d.body_pos_w[0, ids, :].cpu().numpy()
        return StepSample(
            t=self.t,
            cmd_vx=0.0, cmd_vy=0.0, cmd_yaw_rate=0.0,
            base_pos=d.root_pos_w[0].cpu().numpy(),
            base_quat=d.root_quat_w[0].cpu().numpy(),
            lin_vel_body=d.root_lin_vel_b[0].cpu().numpy(),
            ang_vel_body=d.root_ang_vel_b[0].cpu().numpy(),
            joint_pos=d.joint_pos[0].cpu().numpy(),
            joint_vel=d.joint_vel[0].cpu().numpy(),
            joint_torque=d.applied_torque[0].cpu().numpy(),
            foot_contact=contacts,
            foot_pos=foot_pos,
        )

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
        if getattr(self, "_app", None) is not None:
            self._app.close()


def _load_policy(checkpoint):
    import torch

    if checkpoint is None:
        raise ValueError("isaaclab.velocity_policy mode requires backend.isaaclab.checkpoint")
    policy = torch.jit.load(checkpoint) if str(checkpoint).endswith(".pt") else torch.load(checkpoint)
    policy.eval()
    return policy


def _movement_state_to_twist(state: MovementState, current_yaw: float, dt: float):
    """MovementState (world-frame) -> body-frame (vx, vy, yaw_rate)."""
    speed = max(state.movement_speed, 0.0)
    fx, fy, _ = state.facing_direction
    facing_yaw = float(np.arctan2(fy, fx))
    mx, my, _ = state.movement_direction
    if speed > 1e-6 and abs(mx) + abs(my) > 1e-6:
        rel = float(np.arctan2(my, mx)) - facing_yaw
        vx, vy = speed * np.cos(rel), speed * np.sin(rel)
    else:
        vx = vy = 0.0
    yaw_err = (facing_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi
    return float(vx), float(vy), float(yaw_err / max(dt, 1e-6))


def _yaw_quat(yaw: float):
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
