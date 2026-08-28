"""IsaacLab backend that runs the **real Sonic stack** (planner + WBC policy).

This is the true sim-to-sim counterpart of the MuJoCo backend: the very same
``g1_deploy_onnx_ref`` process controls the robot, only the physics engine
changes.

    evaluate_sonic_planner.py --sim isaaclab           (backend.isaaclab.mode: dds)
        | ZMQ  command + planner topics
        v
    g1_deploy_onnx_ref  --input-type zmq_manager
        | DDS  rt/lowstate  <---- published by THIS backend from Isaac state
        | DDS  rt/lowcmd    ----> consumed by THIS backend, PD -> joint efforts
        v
    IsaacLab / PhysX G1 articulation

What makes it possible: ``UnitreeSdk2Bridge`` (gear_sonic/utils/mujoco_sim/
unitree_sdk2py_bridge.py) never touches MuJoCo -- it publishes from a plain obs
dict and exposes the received ``low_cmd``.  So the same bridge is reused here,
fed from ``robot.data`` instead of ``mj_data``, and the PD law from
``DefaultEnv.compute_body_torques`` is replicated verbatim:

    tau_i = tau_ff_i + kp_i * (q_des_i - q_i) + kd_i * (dq_des_i - dq_i)

The actuators must therefore be configured with zero stiffness/damping so PhysX
does not add a second PD loop on top (see ``_zero_actuator_gains``).

**Joint order.** DDS motor index ``i`` is defined by the MuJoCo model's joint
order (``configs/g1_motor_order.json``, regenerate with
``tools/dump_motor_order.py``).  The Isaac articulation orders joints
differently, so every exchange goes through a name-resolved permutation.  A
wrong permutation is the most likely cause of a robot that instantly collapses;
``--check-order`` prints the resolved mapping before the run.

Untested: this module has never been executed -- no IsaacSim in the development
environment.  Treat the first run as a bring-up, and use ``--check-order`` plus
a single low-speed condition before trusting any numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from gear_sonic_eval.backends.base import EvalBackend
from gear_sonic_eval.core.commands import MovementState
from gear_sonic_eval.core.metrics import StepSample

MOTOR_ORDER_FILE = Path(__file__).resolve().parent.parent / "configs" / "g1_motor_order.json"
FOOT_JOINT_PATTERN = ".*ankle_roll.*"
#: Net force [N] above which a foot counts as in contact. Matches the default
#: of IsaacLab's eval_locomotion.py (--contact_force 1.0).
CONTACT_FORCE_THRESHOLD = 1.0


def load_motor_order() -> dict:
    return json.loads(MOTOR_ORDER_FILE.read_text())


class IsaacLabDDSBackend(EvalBackend):
    """Sonic planner + WBC policy over DDS, with IsaacLab as the physics engine."""

    name = "isaaclab"

    def __init__(self, config, *, headless: bool = True):
        super().__init__(config)
        cfg = dict(config.backend.get("isaaclab", {}))
        self.headless = headless
        self.zmq_host = cfg.get("zmq_host", "*")
        self.zmq_port = int(cfg.get("zmq_port", 5556))
        self.startup_wait = float(cfg.get("startup_wait", 2.0))
        self.arm_wait = float(cfg.get("arm_wait", 8.0))
        self.settle_after_reset = float(cfg.get("settle_after_reset", 1.0))
        self.deploy_timeout = float(cfg.get("deploy_timeout", 120.0))
        self.pace_real_time = bool(cfg.get("real_time", True))
        # GUI mode renders at most render_fps, not once per physics step: at
        # 200 Hz physics a render per step cannot keep real time, and this
        # backend must stay in lockstep with the wall-clock deploy process.
        self.render_fps = float(cfg.get("render_fps", 30.0))
        self.robot_cfg_path = cfg.get(
            "robot_cfg", "gear_sonic.envs.manager_env.robots.g1:G1_CYLINDER_MODEL_12_DEX_CFG"
        )
        self.check_order = bool(cfg.get("check_order", False))
        # Teleporting the joints back to the articulation default fights the
        # controller, which is still holding its own target pose: the resulting
        # PD spike can collapse the robot during the settle phase. false keeps
        # the current joint pose and resets only the base.
        self.reset_joints = bool(cfg.get("reset_joints", True))
        # Which simulator this run is supposed to be comparable with. See
        # PHYSICS_PARITY in this module and configs/g1_physics_reference.json.
        self.physics_parity = cfg.get("physics_parity", "training")
        if self.physics_parity not in ("training", "mujoco", "default"):
            raise ValueError(f"unknown physics_parity: {self.physics_parity}")
        self.match_mass = bool(cfg.get("match_mujoco_mass",
                                       self.physics_parity == "mujoco"))
        # Per-knob overrides on top of the preset, so the three differences
        # (friction, foot geometry, mass) can be changed one at a time -- which
        # is the only way to attribute a behaviour change to one of them.
        self.parity_overrides = {
            k: cfg[k] for k in (
                "static_friction", "dynamic_friction", "friction_combine_mode",
                "restitution_combine_mode", "replace_cylinders_with_capsules",
            ) if k in cfg
        }
        config.sim.real_time = False  # paced here, per physics step

        _preflight()
        self._launch_app()
        self._build_scene()
        self._setup_bridge()
        self._armed = False
        self.t = 0.0
        self._push = np.zeros(3)
        self._next_step_wall = None
        self._step_count = 0
        self._render_every = max(
            int(round(1.0 / (self.render_fps * self.config.sim.physics_dt))), 1
        ) if self.render_fps > 0 else 0

    # ------------------------------------------------------------------ setup
    def _launch_app(self) -> None:
        from isaaclab.app import AppLauncher

        self._app_launcher = AppLauncher({"headless": self.headless})
        self._app = self._app_launcher.app

    def _build_scene(self) -> None:
        """Minimal scene: ground plane, light, one G1 -- no RL env, no commands."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab.sim import SimulationCfg, SimulationContext

        physics_dt = self.config.sim.physics_dt
        self.sim = SimulationContext(SimulationCfg(dt=physics_dt, device="cuda:0"))
        self.sim_dt = physics_dt
        self.substeps = max(int(round(self.config.sim.control_dt / physics_dt)), 1)

        parity = dict(PHYSICS_PARITY[self.physics_parity])
        parity.update(self.parity_overrides)
        print(f"[isaaclab-dds] physics: parity={self.physics_parity} "
              f"friction={parity['static_friction']}/{parity['friction_combine_mode']} "
              f"capsules={parity['replace_cylinders_with_capsules']} "
              f"match_mass={self.match_mass}")
        ground_cfg = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=parity["static_friction"],
                dynamic_friction=parity["dynamic_friction"],
                restitution=0.0,
                friction_combine_mode=parity["friction_combine_mode"],
                restitution_combine_mode=parity["restitution_combine_mode"],
            )
        )
        ground_cfg.func("/World/ground", ground_cfg)
        sim_utils.DomeLightCfg(intensity=2000.0).func(
            "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0)
        )

        module_name, _, attr = self.robot_cfg_path.partition(":")
        import importlib

        robot_cfg = getattr(importlib.import_module(module_name), attr).copy()
        robot_cfg.prim_path = "/World/Robot"
        if parity["replace_cylinders_with_capsules"] is not None and hasattr(
            robot_cfg.spawn, "replace_cylinders_with_capsules"
        ):
            robot_cfg.spawn.replace_cylinders_with_capsules = parity[
                "replace_cylinders_with_capsules"
            ]
        robot_cfg.init_state.pos = (
            self.config.init.base_xy[0],
            self.config.init.base_xy[1],
            self.config.init.base_height,
        )
        self.robot = Articulation(robot_cfg)
        self.sim.reset()

        self._resolve_joint_order()
        self._zero_actuator_gains()
        self.mass = float(self.robot.root_physx_view.get_masses()[0].sum().item())
        self.mass_scale = 1.0
        if self.match_mass:
            self.mass_scale = self._match_mujoco_mass()
            self.mass = float(self.robot.root_physx_view.get_masses()[0].sum().item())
        foot_ids, foot_names = self.robot.find_bodies(FOOT_JOINT_PATTERN)
        self.foot_ids, self.foot_names = foot_ids, foot_names
        self.contact_sensor = self._make_contact_sensor()

        self.info = {
            "backend": "isaaclab",
            "mode": "dds",
            "robot_cfg": self.robot_cfg_path,
            "physics_dt": physics_dt,
            "control_dt": self.config.sim.control_dt,
            "mass_kg": self.mass,
            "zmq_endpoint": f"tcp://{self.zmq_host}:{self.zmq_port}",
            "requires": "g1_deploy_onnx_ref --input-type zmq_manager on the same DDS domain",
            "physics_parity": self.physics_parity,
            "parity_overrides": self.parity_overrides,
            "static_friction": parity["static_friction"],
            "friction_combine_mode": parity["friction_combine_mode"],
            "replace_cylinders_with_capsules": parity["replace_cylinders_with_capsules"],
            "mass_scale": self.mass_scale,
            "contact_detection": "sensor" if self.contact_sensor is not None else "foot_height",
            "headless": self.headless,
            "render_fps": self.render_fps if not self.headless else 0.0,
        }

    def _resolve_joint_order(self) -> None:
        """Permutations between DDS motor index and Isaac joint index."""
        order = load_motor_order()
        names = order["body_joints"]
        ids, resolved = self.robot.find_joints(names, preserve_order=True)
        if len(ids) != len(names):
            missing = set(names) - set(resolved)
            raise RuntimeError(
                f"robot articulation is missing DDS motor joints: {sorted(missing)}. "
                f"Check backend.isaaclab.robot_cfg -- it must be the same 29-DoF G1 the "
                f"deploy binary was built for."
            )
        self.motor_to_isaac = np.asarray(ids, dtype=int)  # motor i -> isaac joint index
        self.motor_names = resolved
        if self.check_order:
            print("[isaaclab-dds] DDS motor -> Isaac joint mapping")
            for i, (name, idx) in enumerate(zip(resolved, ids)):
                print(f"  motor {i:2d} {name:32s} -> isaac joint {idx}")

    def _make_contact_sensor(self):
        """Real foot contact forces, so foot slip / duty factor mean the same
        thing here as in MuJoCo (which uses actual contacts).

        Falls back to a foot-height heuristic if the sensor cannot be created;
        ``run_info.json`` records which one was used.
        """
        try:
            from isaaclab.sensors import ContactSensor, ContactSensorCfg

            cfg = ContactSensorCfg(
                prim_path="/World/Robot/.*ankle_roll.*",
                update_period=0.0,
                history_length=0,
                track_air_time=False,
            )
            sensor = ContactSensor(cfg)
            self.sim.reset()
            print(f"[isaaclab-dds] contact sensor active on {sensor.num_instances} feet")
            return sensor
        except Exception as exc:  # noqa: BLE001
            print(f"[isaaclab-dds] contact sensor unavailable ({exc}); "
                  "falling back to a foot-height contact heuristic")
            return None

    def _match_mujoco_mass(self) -> float:
        """Scale every body mass so the robot's total matches the MuJoCo model.

        The two assets partition the robot differently (the URDF splits off
        head_link, logo_link, hand palms, ... while MuJoCo folds them into the
        parent), and the URDF totals 34.394 kg against MuJoCo's 36.165 kg.  A
        uniform scale fixes the total and leaves the URDF's distribution intact;
        per-link masses and inertia tensors still differ, so CoT and push
        results remain only approximately comparable.
        """
        import torch

        ref = _load_physics_reference()
        target = float(ref["mujoco"]["total_mass_kg"])
        view = self.robot.root_physx_view
        masses = view.get_masses()
        current = float(masses[0].sum().item())
        if current <= 0.0:
            return 1.0
        scale = target / current
        # set_masses/set_inertias need an explicit index tensor (indices=None is
        # not accepted by omni.physics.tensors).
        env_ids = torch.arange(masses.shape[0], dtype=torch.int32)
        view.set_masses(masses * scale, env_ids)
        # Inertia is stored separately: scaling mass alone would leave the
        # rotational dynamics inconsistent with the new mass.
        try:
            inertias = view.get_inertias()
            view.set_inertias(inertias * scale, env_ids)
            inertia_note = "mass+inertia"
        except Exception as exc:  # noqa: BLE001
            inertia_note = f"mass only (inertia unchanged: {exc})"
        print(f"[isaaclab-dds] {inertia_note} matched to MuJoCo: {current:.3f} -> "
              f"{current * scale:.3f} kg (x{scale:.4f})")
        return scale

    def _zero_actuator_gains(self) -> None:
        """PhysX must not run its own PD: the deploy binary's kp/kd are the only ones."""
        import torch

        zeros = torch.zeros_like(self.robot.data.joint_stiffness)
        self.robot.write_joint_stiffness_to_sim(zeros)
        self.robot.write_joint_damping_to_sim(zeros)

    def _setup_bridge(self) -> None:
        import zmq
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        from gear_sonic.utils.mujoco_sim.configs import BaseConfig
        from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge

        cfg = dict(self.config.backend.get("isaaclab", {}))
        sim_cfg = BaseConfig(
            wbc_version=cfg.get("wbc_version", "sonic_model12"),
            interface=cfg.get("interface", "sim"),
            sim_frequency=int(round(1.0 / self.config.sim.physics_dt)),
            control_frequency=int(round(1.0 / self.config.sim.control_dt)),
        )
        self.wbc_config = sim_cfg.load_wbc_yaml()
        # ChannelFactoryInitialize directly rather than via
        # gear_sonic.utils.mujoco_sim.simulator_factory.init_channel: that module
        # imports base_sim, which imports mujoco -- an dependency this backend
        # has no reason to require in an IsaacSim environment.
        if self.wbc_config.get("INTERFACE", None):
            ChannelFactoryInitialize(self.wbc_config["DOMAIN_ID"], self.wbc_config["INTERFACE"])
        else:
            ChannelFactoryInitialize(self.wbc_config["DOMAIN_ID"])
        self.bridge = UnitreeSdk2Bridge(self.wbc_config)
        self.num_motors = self.bridge.num_body_motor
        self.torque_limit = np.asarray(self.wbc_config["motor_effort_limit_list"], dtype=float)

        ctx = zmq.Context.instance()
        self.socket = ctx.socket(zmq.PUB)
        self.socket.bind(f"tcp://{self.zmq_host}:{self.zmq_port}")
        time.sleep(0.5)

    # -------------------------------------------------------------------- ZMQ
    def _send_command(self, start: bool, stop: bool, planner: bool = True) -> None:
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message

        self.socket.send(build_command_message(start=start, stop=stop, planner=planner))

    def send_movement_state(self, state: MovementState) -> None:
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_planner_message

        self.socket.send(
            build_planner_message(
                mode=state.locomotion_mode,
                movement=state.movement_direction,
                facing=state.facing_direction,
                speed=state.movement_speed,
                height=state.height,
            )
        )

    # ---------------------------------------------------------------- physics
    def _observe_dict(self) -> dict:
        """The obs dict UnitreeSdk2Bridge.PublishLowState expects, in motor order."""
        d = self.robot.data
        q = d.joint_pos[0].cpu().numpy()[self.motor_to_isaac]
        dq = d.joint_vel[0].cpu().numpy()[self.motor_to_isaac]
        tau = d.applied_torque[0].cpu().numpy()[self.motor_to_isaac]
        acc = d.joint_acc[0].cpu().numpy()[self.motor_to_isaac] if hasattr(d, "joint_acc") \
            else np.zeros_like(q)
        pos = d.root_pos_w[0].cpu().numpy()
        quat = d.root_quat_w[0].cpu().numpy()  # (w, x, y, z), same convention as MuJoCo
        lin = d.root_lin_vel_w[0].cpu().numpy()
        ang = d.root_ang_vel_b[0].cpu().numpy()
        hands = np.zeros(self.bridge.num_hand_motor)
        return {
            "body_q": q, "body_dq": dq, "body_ddq": acc, "body_tau_est": tau,
            "floating_base_pose": np.concatenate([pos, quat]),
            "floating_base_vel": np.concatenate([lin, ang]),
            # PhysX exposes no base linear acceleration; the IMU accelerometer
            # channel is left at zero (the Sonic policy does not observe it).
            "floating_base_acc": np.zeros(6),
            "secondary_imu_quat": quat,
            "secondary_imu_vel": np.concatenate([lin, ang]),
            "left_hand_q": hands, "left_hand_dq": hands,
            "right_hand_q": hands, "right_hand_dq": hands,
            "time": self.t,
        }

    def _torques_from_lowcmd(self) -> np.ndarray:
        """Replica of DefaultEnv.compute_body_torques (same PD law, same order)."""
        d = self.robot.data
        q = d.joint_pos[0].cpu().numpy()[self.motor_to_isaac]
        dq = d.joint_vel[0].cpu().numpy()[self.motor_to_isaac]
        tau = np.zeros(self.num_motors)
        cmd = self.bridge.low_cmd
        if cmd is None:
            return tau
        with self.bridge.low_cmd_lock:
            for i in range(self.num_motors):
                m = cmd.motor_cmd[i]
                tau[i] = m.tau + m.kp * (m.q - q[i]) + m.kd * (m.dq - dq[i])
        return np.clip(tau, -self.torque_limit[: self.num_motors],
                       self.torque_limit[: self.num_motors])

    def _physics_step(self) -> None:
        import torch

        self.bridge.PublishLowState(self._observe_dict())
        tau = self._torques_from_lowcmd()
        efforts = torch.zeros((1, self.robot.num_joints), device=self.robot.device)
        efforts[0, self.motor_to_isaac] = torch.tensor(
            tau, dtype=efforts.dtype, device=efforts.device
        )
        self.robot.set_joint_effort_target(efforts)
        if np.linalg.norm(self._push) > 0.0:
            forces = torch.zeros((1, 1, 3), device=self.robot.device)
            forces[0, 0] = torch.tensor(self._push, dtype=forces.dtype, device=forces.device)
            self.robot.set_external_force_and_torque(forces, torch.zeros_like(forces), body_ids=[0])
        self.robot.write_data_to_sim()
        self._step_count += 1
        render = (
            not self.headless
            and self._render_every
            and self._step_count % self._render_every == 0
        )
        self.sim.step(render=bool(render))
        self.robot.update(self.sim_dt)
        if self.contact_sensor is not None:
            self.contact_sensor.update(self.sim_dt)

        if self.pace_real_time:
            now = time.monotonic()
            if self._next_step_wall is None:
                self._next_step_wall = now
            self._next_step_wall += self.sim_dt
            delay = self._next_step_wall - now
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:
                self._next_step_wall = now

    # --------------------------------------------------------------- protocol
    def prepare(self) -> None:
        if self._armed:
            return
        print("waiting for g1_deploy_onnx_ref (LowCmd on rt/lowcmd) ...")
        deadline = time.monotonic() + self.deploy_timeout
        seen = False
        while time.monotonic() < deadline:
            self._physics_step()
            if self.bridge.cmd_received():
                seen = True
                break
        if not seen:
            print("  WARNING: no LowCmd received; is the deploy binary running with "
                  "--input-type zmq_manager on the same DDS interface?")
        else:
            end = time.monotonic() + self.startup_wait
            while time.monotonic() < end:
                self._physics_step()
        print("arming the controller (command{start=true, planner=true}) ...")
        self._send_command(start=True, stop=False, planner=True)
        end = time.monotonic() + self.arm_wait
        while time.monotonic() < end:
            self._physics_step()
        self._armed = True

    def reset(self, seed: int) -> None:
        import torch

        rng = np.random.default_rng(seed)
        init = self.config.init
        root = self.robot.data.default_root_state.clone()
        root[:, 0:2] = torch.tensor(init.base_xy, device=root.device, dtype=root.dtype)
        root[:, 2] = init.base_height
        root[:, 3:7] = torch.tensor(_yaw_quat(init.base_yaw), device=root.device, dtype=root.dtype)
        root[:, 7:10] = torch.tensor(init.base_lin_vel, device=root.device, dtype=root.dtype)
        root[:, 10:13] = torch.tensor(init.base_ang_vel, device=root.device, dtype=root.dtype)
        self.robot.write_root_state_to_sim(root)

        if self.reset_joints:
            joint_pos = self.robot.data.default_joint_pos.clone()
            if init.joint_noise > 0.0:
                noise = rng.normal(0.0, init.joint_noise, joint_pos.shape[-1])
                joint_pos += torch.tensor(noise, device=joint_pos.device, dtype=joint_pos.dtype)
        else:
            joint_pos = self.robot.data.joint_pos.clone()
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))

        self._push = np.zeros(3)
        self.t = 0.0
        self._next_step_wall = None
        self.send_movement_state(_idle_state())
        end = time.monotonic() + self.settle_after_reset
        while time.monotonic() < end:
            self._physics_step()

    def step(self) -> StepSample:
        for _ in range(self.substeps):
            self._physics_step()
        self.t += self.config.sim.control_dt
        return self._sample()

    def apply_push(self, force_world, duration: float) -> None:
        self._push = np.asarray(force_world, dtype=float)

    def base_yaw(self) -> float:
        w, x, y, z = self.robot.data.root_quat_w[0].cpu().numpy()
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def _sample(self) -> StepSample:
        d = self.robot.data
        contacts, foot_pos = None, None
        if self.foot_ids:
            foot_pos = d.body_pos_w[0, self.foot_ids, :].cpu().numpy()
            if self.contact_sensor is not None:
                forces = self.contact_sensor.data.net_forces_w[0].cpu().numpy()
                contacts = (np.linalg.norm(forces, axis=-1) > CONTACT_FORCE_THRESHOLD
                            ).astype(float).tolist()
            else:
                contacts = (foot_pos[:, 2] < 0.06).astype(float).tolist()
        return StepSample(
            t=self.t,
            cmd_vx=0.0, cmd_vy=0.0, cmd_yaw_rate=0.0,
            base_pos=d.root_pos_w[0].cpu().numpy(),
            base_quat=d.root_quat_w[0].cpu().numpy(),
            lin_vel_body=d.root_lin_vel_b[0].cpu().numpy(),
            ang_vel_body=d.root_ang_vel_b[0].cpu().numpy(),
            joint_pos=d.joint_pos[0].cpu().numpy()[self.motor_to_isaac],
            joint_vel=d.joint_vel[0].cpu().numpy()[self.motor_to_isaac],
            joint_torque=d.applied_torque[0].cpu().numpy()[self.motor_to_isaac],
            foot_contact=contacts,
            foot_pos=foot_pos,
        )

    def close(self) -> None:
        socket = getattr(self, "socket", None)
        if socket is not None:
            try:
                self.send_movement_state(_idle_state())
            except Exception as exc:  # noqa: BLE001
                print(f"[isaaclab-dds] could not send final command: {exc}")
            socket.close(linger=200)
            self.socket = None
        if getattr(self, "_app", None) is not None:
            self._app.close()


def _load_physics_reference() -> dict:
    return json.loads(
        (Path(__file__).resolve().parent.parent / "configs" / "g1_physics_reference.json").read_text()
    )


#: What "comparable" means for this run.
#:
#: ``training`` reproduces the environment the WBC policy was trained in
#: (gear_sonic/envs/manager_env/modular_tracking_env_cfg.py: friction 1.0/1.0
#: combined by multiply, capsule feet).  Numbers are then interpretable as
#: "how the planner behaves in its own training physics".
#:
#: ``mujoco`` moves the Isaac scene toward the MuJoCo model instead: MuJoCo
#: gives every geom ``friction="1.0"`` and combines with max, so the effective
#: coefficient is 1.0 rather than the product with the robot material; the
#: cylinder->capsule replacement is switched off because MuJoCo's sole is a flat
#: box; and the total robot mass is scaled to the MuJoCo model's.  Use it when
#: the point of the run is a like-for-like MuJoCo/Isaac comparison.
#:
#: ``default`` is IsaacLab's own defaults (static/dynamic friction 0.5, average
#: combine) -- kept only to reproduce earlier runs.
PHYSICS_PARITY = {
    "training": {
        "static_friction": 1.0, "dynamic_friction": 1.0,
        "friction_combine_mode": "multiply", "restitution_combine_mode": "multiply",
        "replace_cylinders_with_capsules": True,
    },
    "mujoco": {
        "static_friction": 1.0, "dynamic_friction": 1.0,
        "friction_combine_mode": "max", "restitution_combine_mode": "max",
        "replace_cylinders_with_capsules": False,
    },
    "default": {
        "static_friction": 0.5, "dynamic_friction": 0.5,
        "friction_combine_mode": "average", "restitution_combine_mode": "average",
        "replace_cylinders_with_capsules": None,
    },
}


def _preflight() -> None:
    """Fail early, with a readable list, if the interpreter lacks a dependency.

    This backend needs the IsaacSim stack *and* the gear_sonic DDS stack in the
    same interpreter, which is the usual stumbling block: the IsaacLab
    environment normally has neither unitree_sdk2py nor gear_sonic on its path.
    """
    import importlib.util

    required = {
        "isaaclab": "IsaacLab (run with IsaacLab's python, e.g. ./isaaclab.sh -p)",
        "torch": "PyTorch (part of the IsaacSim environment)",
        "zmq": "pyzmq",
        "yaml": "PyYAML",
        "scipy": "scipy",
        "unitree_sdk2py": "unitree_sdk2py (pip install -e external_dependencies/unitree_sdk2_python)",
        "gear_sonic": "gear_sonic (pip install -e . in GR00T-WholeBodyControl, or set PYTHONPATH)",
    }
    missing = [f"  {mod:16s} {hint}" for mod, hint in required.items()
               if importlib.util.find_spec(mod) is None]
    if missing:
        raise SystemExit(
            "the IsaacLab dds backend needs IsaacSim and the gear_sonic DDS stack in the "
            "same interpreter; missing here:\n" + "\n".join(missing)
        )


def _idle_state() -> MovementState:
    return MovementState(0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.0, -1.0)


def _yaw_quat(yaw: float):
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
