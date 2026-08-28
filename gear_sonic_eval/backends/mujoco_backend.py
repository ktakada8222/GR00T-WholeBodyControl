"""MuJoCo backend: the real Sonic stack (planner -> WBC policy -> G1) in sim.

Data flow actually used by the benchmark::

    evaluate_sonic_planner.py
        | ZMQ PUB tcp://*:5556, topics "command" + "planner"   (wire format from
        |   gear_sonic/utils/teleop/zmq/zmq_planner_sender.py, decoded by
        |   ZMQManager -> PlannerMessage -> MovementState)
        v
    g1_deploy_onnx_ref  --input-type zmq_manager --planner-file <planner.onnx>
        | LocalMotionPlanner(TensorRT)  ->  30 Hz trajectory
        | ResampleGeneratedSequence50Hz ->  MotionSequence @50 Hz
        | control policy (TensorRT)     ->  MotorCommand
        | DDS  (unitree_sdk2 LowCmd)
        v
    this backend's MuJoCo DefaultEnv  (PD torques, physics, LowState publish)

So the deploy binary must already be running against the same DDS domain/interface
(``INTERFACE: lo`` in the wbc yaml) *before* the evaluation starts.  The backend
never fabricates planner behaviour: it only sends the planner's own input format
and measures the resulting robot state.

The **elastic band is disabled** (``env.elastic_band = None``).  In the
interactive workflow it is on by default and is released by pressing ``9`` in
the MuJoCo window; a benchmark must not depend on a key press, and
``DefaultEnv.sim_step`` zeroes ``xfrc_applied`` on the band body every step,
which would silently cancel the push forces.

Push disturbances use ``mj_data.xfrc_applied[pelvis]`` (a real MuJoCo API, held
for ``duration`` seconds so the impulse is force*duration).  ``apply_perturbation``
in ``base_sim.py`` was *not* reused because it adds an instantaneous velocity
step, which cannot be expressed in newtons.
"""

from __future__ import annotations

import time

import numpy as np

from gear_sonic_eval.backends.base import EvalBackend
from gear_sonic_eval.core.commands import MovementState
from gear_sonic_eval.core.metrics import StepSample

FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]


class _EvalEnv:
    """Thin wrapper adding eval hooks to gear_sonic's ``DefaultEnv``.

    The only behavioural change is that a fall no longer auto-resets the sim:
    the runner owns episode boundaries.
    """

    def __init__(self, wbc_config, onscreen: bool):
        from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv

        self.env = DefaultEnv(config=wbc_config, env_name="default", onscreen=onscreen)
        self.env.check_fall = self._check_fall  # type: ignore[method-assign]
        self.fall = False

    def _check_fall(self):
        self.fall = bool(self.env.mj_data.qpos[2] < 0.2)


class MujocoBackend(EvalBackend):
    name = "mujoco"

    def __init__(self, config, *, onscreen: bool = False):
        super().__init__(config)
        import mujoco  # noqa: F401  (import here so core stays dependency-free)
        import zmq

        from gear_sonic.utils.mujoco_sim.configs import BaseConfig
        from gear_sonic.utils.mujoco_sim.simulator_factory import init_channel

        self.mujoco = mujoco
        backend_cfg = dict(config.backend.get("mujoco", {}))
        self.zmq_port = int(backend_cfg.get("zmq_port", 5556))
        self.zmq_host = backend_cfg.get("zmq_host", "*")
        self.rearm_wait = float(backend_cfg.get("rearm_wait", 3.0))
        self.startup_wait = float(backend_cfg.get("startup_wait", 2.0))
        self.deploy_timeout = float(backend_cfg.get("deploy_timeout", 120.0))
        self.use_elastic_band = bool(backend_cfg.get("elastic_band", False))

        sim_cfg = BaseConfig(
            wbc_version=backend_cfg.get("wbc_version", "sonic_model12"),
            interface=backend_cfg.get("interface", "sim"),
            sim_frequency=int(round(1.0 / config.sim.physics_dt)),
            control_frequency=int(round(1.0 / config.sim.control_dt)),
            enable_onscreen=onscreen,
        )
        self.wbc_config = sim_cfg.load_wbc_yaml()
        self.wbc_config["ENV_NAME"] = "default"
        init_channel(config=self.wbc_config)

        self.wrapper = _EvalEnv(self.wbc_config, onscreen=onscreen)
        self.env = self.wrapper.env
        self.model, self.data = self.env.mj_model, self.env.mj_data
        self.pelvis_id = self.model.body("pelvis").id
        self.substeps = max(int(round(config.sim.control_dt / config.sim.physics_dt)), 1)
        self.mass = float(self.model.body_subtreemass[self.pelvis_id])
        self._foot_geoms = _foot_geom_ids(self.model, mujoco)
        self._floor_geoms = _floor_geom_ids(self.model, mujoco)
        self._qpos0 = self.data.qpos.copy()

        if not self.use_elastic_band:
            # See the module docstring: the band both suspends the robot and
            # clears xfrc_applied on the pelvis every physics step.
            self.env.elastic_band = None

        ctx = zmq.Context.instance()
        self.socket = ctx.socket(zmq.PUB)
        self.socket.bind(f"tcp://{self.zmq_host}:{self.zmq_port}")
        time.sleep(0.5)  # let the deploy subscriber connect before the first send

        self.info = {
            "backend": "mujoco",
            "robot_scene": self.wbc_config["ROBOT_SCENE"],
            "zmq_endpoint": f"tcp://{self.zmq_host}:{self.zmq_port}",
            "physics_dt": self.model.opt.timestep,
            "control_dt": config.sim.control_dt,
            "mass_kg": self.mass,
            "requires": "g1_deploy_onnx_ref --input-type zmq_manager --planner-file <planner> running on the same DDS domain",
        }
        self.t = 0.0
        self._push = np.zeros(3)
        self._armed = False

    # ------------------------------------------------------------------ ZMQ
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

    # --------------------------------------------------------------- episode
    def wait_for_deploy(self, verbose: bool = True) -> bool:
        """Step the sim until the deploy binary starts publishing LowCmd.

        The deploy process needs LowState from this simulator to run its INIT
        ramp, so the sim must already be stepping while we wait.  Detection uses
        ``UnitreeSdk2Bridge.cmd_received()``.
        """
        if verbose:
            print("waiting for g1_deploy_onnx_ref (LowCmd on rt/lowcmd) ...")
        deadline = time.monotonic() + self.deploy_timeout
        bridge = self.env.unitree_bridge
        while time.monotonic() < deadline:
            self._physics_step()
            if bridge.cmd_received():
                if verbose:
                    print(f"  deploy detected; waiting {self.startup_wait:.1f}s for its INIT ramp")
                end = time.monotonic() + self.startup_wait
                while time.monotonic() < end:
                    self._physics_step()
                return True
        print("  WARNING: no LowCmd received; is the deploy binary running with "
              "--input-type zmq_manager on the same DDS interface?")
        return False

    def reset(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        init = self.config.init

        if not self._armed:
            self.wait_for_deploy()
            self._armed = True

        # Stop the controller, restore the initial state, re-arm in planner mode.
        self._send_command(start=False, stop=True)
        time.sleep(0.2)

        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._qpos0
        self.data.qpos[0:2] = init.base_xy
        self.data.qpos[2] = init.base_height
        self.data.qpos[3:7] = _yaw_quat(init.base_yaw)
        if init.joint_noise > 0.0:
            n = self.data.qpos.shape[0] - 7
            self.data.qpos[7:] += rng.normal(0.0, init.joint_noise, n)
        self.data.qvel[:] = 0.0
        self.data.qvel[0:3] = init.base_lin_vel
        self.data.qvel[3:6] = init.base_ang_vel
        self.data.xfrc_applied[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        self.wrapper.fall = False
        self._push = np.zeros(3)
        self.t = 0.0

        self._send_command(start=True, stop=False, planner=True)
        # let the deploy binary ramp to the default pose and initialise the planner
        deadline = time.monotonic() + self.rearm_wait
        while time.monotonic() < deadline:
            self._physics_step()

    def _physics_step(self) -> None:
        self.data.xfrc_applied[self.pelvis_id, :3] = self._push
        self.env.sim_step()

    def step(self) -> StepSample:
        for _ in range(self.substeps):
            self._physics_step()
        self.t += self.config.sim.control_dt
        if self.env.viewer is not None:
            self.env.update_viewer()
        return self._observe()

    def apply_push(self, force_world, duration: float) -> None:
        self._push = np.asarray(force_world, dtype=float)

    def base_yaw(self) -> float:
        w, x, y, z = self.data.qpos[3:7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def is_fallen(self) -> bool:
        return bool(self.wrapper.fall)

    # ------------------------------------------------------------ observation
    def _observe(self) -> StepSample:
        d = self.data
        quat = np.asarray(d.qpos[3:7], dtype=float)  # (w, x, y, z)
        lin_world = np.asarray(d.qvel[0:3], dtype=float)
        lin_body = _rotate_inv(quat, lin_world)
        ang_body = np.asarray(d.qvel[3:6], dtype=float)  # free joint: body frame

        idx = self.env.body_joint_index
        joint_pos = np.asarray(d.qpos[idx + self.env.qpos_offset - 1], dtype=float)
        joint_vel = np.asarray(d.qvel[idx + self.env.qvel_offset - 1], dtype=float)
        torque = np.asarray(d.actuator_force[idx - 1], dtype=float)

        contacts = _foot_contacts(d, self._foot_geoms, self._floor_geoms)
        foot_pos = [d.body(name).xpos.copy() for name in FOOT_BODIES]

        return StepSample(
            t=self.t,
            cmd_vx=0.0, cmd_vy=0.0, cmd_yaw_rate=0.0,  # filled in by the runner
            base_pos=np.asarray(d.qpos[0:3], dtype=float),
            base_quat=quat,
            lin_vel_body=lin_body,
            ang_vel_body=ang_body,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            joint_torque=torque,
            foot_contact=contacts,
            foot_pos=foot_pos,
        )

    def close(self) -> None:
        try:
            self._send_command(start=False, stop=True)
            self.socket.close(linger=200)
        finally:
            if getattr(self.env, "viewer", None) is not None:
                self.env.viewer.close()


# ------------------------------------------------------------------- helpers
def _foot_geom_ids(model, mujoco):
    out = []
    for body in FOOT_BODIES:
        bid = model.body(body).id
        start = model.body_geomadr[bid]
        out.append(set(range(start, start + model.body_geomnum[bid])))
    return out


def _floor_geom_ids(model, mujoco):
    ids = set()
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name in ("floor", "ground") or model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            ids.add(i)
    return ids


def _foot_contacts(data, foot_geoms, floor_geoms) -> list[float]:
    hit = [0.0] * len(foot_geoms)
    for i in range(data.ncon):
        c = data.contact[i]
        for f, geoms in enumerate(foot_geoms):
            if (c.geom1 in geoms and c.geom2 in floor_geoms) or (
                c.geom2 in geoms and c.geom1 in floor_geoms
            ):
                hit[f] = 1.0
    return hit


def _rotate_inv(quat, vec):
    w, x, y, z = quat
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return r.T @ vec


def _yaw_quat(yaw: float):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
