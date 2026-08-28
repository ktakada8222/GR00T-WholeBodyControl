"""Planner client backends.

The Gear Sonic planner (LocalMotionPlannerBase / LocalMotionPlannerTensorRT,
gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner*.hpp)
has no Python bindings. Three ways to drive it were identified during the
investigation of this repo (2026-08-28); all three are implemented behind the
same `PlannerClient` interface so runners never need to know which backend is
active:

1. OnnxPlannerClient (preferred once a model file is available): loads the
   planner's exported .onnx directly with onnxruntime and reproduces
   UpdatePlanning / ResampleGeneratedSequence50Hz from localmotion_kplanner.hpp
   in Python. Single process, no DDS/ZMQ needed. This mirrors the *deprecated*
   ONNX C++ backend (localmotion_kplanner_onnx.hpp), which is why it's marked
   version 0/1 below (see that header for version 2's 27-mode tensor layout,
   not reproduced here since no v2 model was available to confirm shapes).
   NOT YET VERIFIED END TO END: no .onnx file for the Sonic planner exists in
   this repository snapshot (only unrelated WBC balance/walk policies do), and
   onnxruntime is not installed in this environment. The tensor names/shapes
   below come directly from the C++ header and should be checked against the
   actual exported model before trusting numerical output.

2. ZmqPlannerClient: drives the real compiled `g1_deploy_onnx_ref` binary
   (run as a subprocess with `--input-type zmq_manager`) over the ZMQ wire
   protocol demonstrated in
   gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/tests/test_zmq_manager.py.
   This is the only backend that runs the *exact* deployed TensorRT model,
   but requires: the binary to be built, a DDS domain shared with the MuJoCo
   sim (see runners/mujoco_runner.py), and pyzmq installed. It is implemented
   as a thin subprocess+socket wrapper; NOT exercised in this environment
   (no compiled binary / DDS stack available here).

3. MockPlannerClient: a constant-acceleration kinematic integrator that
   accepts the same MovementState dict and produces a plausible 50 Hz
   root-pose trajectory (no joint motion). Used to validate the evaluation
   framework itself (config parsing, metrics, plotting, CLI) without any of
   the above dependencies. Never represents actual Sonic planner behavior.
"""
from __future__ import annotations

import subprocess
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class PlannerClient(ABC):
    """Common interface: feed a MovementState dict, get back a 50Hz motion
    chunk (root position/quaternion + joint positions/velocities)."""

    @abstractmethod
    def initialize(self, base_quat_wxyz: np.ndarray, joint_positions: np.ndarray) -> None:
        ...

    @abstractmethod
    def update(self, movement_state: dict, gen_frame: int) -> dict:
        """Returns a dict with keys:
            body_pos: (N,3), body_quat_wxyz: (N,4),
            joint_pos: (N, num_joints), joint_vel: (N, num_joints)
        sampled at 50 Hz, N up to 64 frames (per ResampleGeneratedSequence50Hz's
        cap in the C++ implementation)."""
        ...

    def close(self) -> None:
        pass


class OnnxPlannerClient(PlannerClient):
    NUM_JOINTS = 29
    CONTEXT_FRAMES = 4

    def __init__(self, model_path: str):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "OnnxPlannerClient requires onnxruntime ('pip install onnxruntime' or "
                "'onnxruntime-gpu'). Not installed in this environment."
            ) from e
        self._ort = ort
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self.session.get_inputs()}
        self._context = None  # (4, 36) mujoco_qpos context buffer

    def initialize(self, base_quat_wxyz: np.ndarray, joint_positions: np.ndarray) -> None:
        qpos0 = np.concatenate([[0, 0, 0.79], base_quat_wxyz, joint_positions]).astype(np.float32)
        self._context = np.tile(qpos0, (self.CONTEXT_FRAMES, 1))

    def update(self, movement_state: dict, gen_frame: int) -> dict:
        if self._context is None:
            raise RuntimeError("call initialize() first")

        feed = {
            "context_mujoco_qpos": self._context.astype(np.float32),
            "target_vel": np.array([movement_state["movement_speed"]], dtype=np.float32),
            "mode": np.array([movement_state["locomotion_mode"]], dtype=np.int64),
            "movement_direction": np.array(movement_state["movement_direction"], dtype=np.float32),
            "facing_direction": np.array(movement_state["facing_direction"], dtype=np.float32),
            "random_seed": np.array([-1], dtype=np.int64),
        }
        if "height" in self._input_names:
            feed["height"] = np.array([movement_state.get("height", -1.0)], dtype=np.float32)
        if "has_specific_target" in self._input_names:
            feed["has_specific_target"] = np.array([0], dtype=np.int64)
            feed["specific_target_positions"] = np.zeros(12, dtype=np.float32)
            feed["specific_target_headings"] = np.zeros(4, dtype=np.float32)
            feed["allowed_pred_num_tokens"] = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int64)
        feed = {k: v for k, v in feed.items() if k in self._input_names}

        outputs = self.session.run(None, feed)
        qpos_30hz, num_frames = outputs[0], int(outputs[1])
        qpos_30hz = qpos_30hz[:num_frames]

        motion_50hz = _resample_30hz_to_50hz(qpos_30hz)
        self._context = _update_context_from_motion(motion_50hz, gen_frame, self.CONTEXT_FRAMES)

        body_pos = motion_50hz[:, 0:3]
        body_quat = motion_50hz[:, 3:7]
        joint_pos = motion_50hz[:, 7:7 + self.NUM_JOINTS]
        joint_vel = np.gradient(joint_pos, 1.0 / 50.0, axis=0) if len(joint_pos) > 1 else np.zeros_like(joint_pos)
        return {"body_pos": body_pos, "body_quat_wxyz": body_quat, "joint_pos": joint_pos, "joint_vel": joint_vel}


def _slerp(q0: np.ndarray, q1: np.ndarray, w: float) -> np.ndarray:
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        return (q0 + w * (q1 - q0)) / np.linalg.norm(q0 + w * (q1 - q0))
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta0 * w
    q2 = q1 - q0 * dot
    q2 /= np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def _resample_30hz_to_50hz(qpos_30hz: np.ndarray) -> np.ndarray:
    """Direct port of LocalMotionPlannerBase::ResampleGeneratedSequence50Hz
    (localmotion_kplanner.hpp:395-519)."""
    timesteps_30hz = len(qpos_30hz)
    if timesteps_30hz < 2:
        return qpos_30hz.copy()
    motion_seconds = timesteps_30hz / 30.0
    n50 = int(np.floor(motion_seconds * 50))
    out = np.zeros((n50, qpos_30hz.shape[1]), dtype=np.float32)
    for f in range(n50):
        t = f / 50.0
        f30 = t * 30.0
        f0 = min(int(np.floor(f30)), timesteps_30hz - 1)
        f1 = min(f0 + 1, timesteps_30hz - 1)
        w1 = f30 - f0
        w0 = 1.0 - w1
        out[f, 0:3] = w0 * qpos_30hz[f0, 0:3] + w1 * qpos_30hz[f1, 0:3]
        out[f, 7:] = w0 * qpos_30hz[f0, 7:] + w1 * qpos_30hz[f1, 7:]
        out[f, 3:7] = _slerp(qpos_30hz[f0, 3:7], qpos_30hz[f1, 3:7], w1)
    return out


def _update_context_from_motion(motion_50hz: np.ndarray, gen_frame: int, n_context: int) -> np.ndarray:
    """Approximates UpdateContextFromMotion (localmotion_kplanner.hpp:628-678):
    resample n_context frames spaced at 30Hz intervals starting at gen_time."""
    if len(motion_50hz) == 0:
        raise RuntimeError("planner produced an empty motion chunk")
    gen_time = gen_frame / 50.0
    idxs = np.clip(
        np.round((gen_time + np.arange(n_context) / 30.0) * 50.0).astype(int),
        0, len(motion_50hz) - 1,
    )
    return motion_50hz[idxs]


class ZmqPlannerClient(PlannerClient):
    """Drives the real compiled g1_deploy_onnx_ref binary via ZMQ, matching
    tests/test_zmq_manager.py. Requires the binary to already be built and a
    Unitree DDS domain shared with whatever consumes rt/lowcmd (e.g. the
    MuJoCo runner's UnitreeSdk2Bridge). Not exercised in this environment."""

    def __init__(self, binary_path: str, planner_model_path: str,
                 zmq_host: str = "localhost", zmq_port: int = 5556,
                 extra_args: Optional[list] = None):
        try:
            import zmq  # noqa: F401
        except ImportError as e:
            raise RuntimeError("ZmqPlannerClient requires pyzmq ('pip install pyzmq').") from e
        self._zmq = __import__("zmq")
        self.binary_path = binary_path
        self.planner_model_path = planner_model_path
        self.zmq_host, self.zmq_port = zmq_host, zmq_port
        self.extra_args = extra_args or []
        self._proc: Optional[subprocess.Popen] = None
        self._socket = None

    def initialize(self, base_quat_wxyz: np.ndarray, joint_positions: np.ndarray) -> None:
        args = [
            self.binary_path,
            "--input-type", "zmq_manager",
            "--planner-file", self.planner_model_path,
            "--zmq-host", self.zmq_host,
            "--zmq-port", str(self.zmq_port),
            "--disable-crc-check",
            *self.extra_args,
        ]
        self._proc = subprocess.Popen(args)
        time.sleep(2.0)  # let the binary bind its sockets / initialize the planner
        ctx = self._zmq.Context()
        self._socket = ctx.socket(self._zmq.PUB)
        self._socket.connect(f"tcp://{self.zmq_host}:{self.zmq_port}")
        self._send_command(start=True, planner=True)

    def _send_command(self, **kwargs) -> None:
        # Wire format matches CommandMessage / PlannerMessage in
        # input_interface/input_command.hpp; left as a documented stub since
        # the exact struct packing could not be executed/verified here.
        raise NotImplementedError(
            "ZmqPlannerClient wire protocol must be packed to match "
            "input_interface/input_command.hpp's CommandMessage/PlannerMessage "
            "struct layout exactly (struct packing, byte order). Port "
            "test_zmq_manager.py's ZMQPublisher.send_command/send_planner here "
            "once the binary is available to validate against."
        )

    def update(self, movement_state: dict, gen_frame: int) -> dict:
        raise NotImplementedError(
            "ZmqPlannerClient reads the resulting trajectory back from the "
            "MuJoCo-side DDS rt/lowstate feed (there is no synchronous "
            "request/response channel back from the planner over ZMQ); a "
            "runner using this backend should read robot state from the "
            "MuJoCo sim directly rather than from this method's return value."
        )

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=5)


class MockPlannerClient(PlannerClient):
    """Kinematic stand-in used only to exercise the evaluation framework
    itself when no real planner backend is available. Integrates the
    commanded movement_direction * movement_speed and facing_direction (yaw)
    with a first-order lag to emulate acceleration limits, and leaves all
    joints at a fixed standing pose (no gait). Metrics computed against this
    backend describe the harness, NOT the Sonic planner."""

    NUM_JOINTS = 29
    TIME_CONSTANT_S = 0.3

    def __init__(self, standing_joint_pos: Optional[np.ndarray] = None):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.standing_joint_pos = (
            standing_joint_pos if standing_joint_pos is not None else np.zeros(self.NUM_JOINTS)
        )

    def initialize(self, base_quat_wxyz: np.ndarray, joint_positions: np.ndarray) -> None:
        self.quat = base_quat_wxyz.copy()
        self.standing_joint_pos = joint_positions.copy()
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)

    def update(self, movement_state: dict, gen_frame: int) -> dict:
        n = 32  # ~0.64s chunk at 50Hz, similar order to the real planner's cap of 64
        dt = 1.0 / 50.0
        speed = movement_state["movement_speed"]
        mdir = np.array(movement_state["movement_direction"])
        target_vel = mdir * speed
        fdir = movement_state["facing_direction"]
        yaw = float(np.arctan2(fdir[1], fdir[0]))

        body_pos = np.zeros((n, 3))
        body_quat = np.zeros((n, 4))
        alpha = dt / self.TIME_CONSTANT_S
        for i in range(n):
            self.vel += alpha * (target_vel - self.vel)
            self.pos += self.vel * dt
            body_pos[i] = self.pos
            body_quat[i] = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        self.quat = body_quat[-1]

        joint_pos = np.tile(self.standing_joint_pos, (n, 1))
        joint_vel = np.zeros_like(joint_pos)
        return {"body_pos": body_pos, "body_quat_wxyz": body_quat, "joint_pos": joint_pos, "joint_vel": joint_vel}


def make_planner_client(backend: str, model_path: Optional[str] = None, **kwargs) -> PlannerClient:
    if backend == "onnx":
        if not model_path:
            raise ValueError("planner_backend='onnx' requires planner_model_path in the eval config")
        return OnnxPlannerClient(model_path)
    if backend == "zmq":
        return ZmqPlannerClient(binary_path=kwargs["binary_path"], planner_model_path=model_path, **kwargs)
    if backend == "mock":
        return MockPlannerClient()
    raise ValueError(f"unknown planner backend: {backend}")
