"""Analytic stand-in backend -- no simulator, no planner, no GPU.

It exists so the benchmark plumbing (config -> schedule -> metrics -> CSV ->
plots) can be exercised and unit-tested anywhere, including on machines without
MuJoCo/IsaacSim.  The dynamics are a deliberately crude first-order velocity
lag with a speed-dependent tracking droop, a bobbing base and a fixed-frequency
contact pattern; **its numbers say nothing about the real planner.**
"""

from __future__ import annotations

import math

import numpy as np

from gear_sonic_eval.backends.base import EvalBackend
from gear_sonic_eval.core.commands import MovementState
from gear_sonic_eval.core.metrics import StepSample


class MockBackend(EvalBackend):
    name = "mock"
    mass = 35.0

    def __init__(self, config):
        super().__init__(config)
        self.dt = config.sim.control_dt
        self.info = {"backend": "mock", "warning": "analytic placeholder, not a simulator"}
        self.reset(config.seed)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        init = self.config.init
        self.t = 0.0
        self.pos = np.array([init.base_xy[0], init.base_xy[1], init.base_height], dtype=float)
        self.yaw = float(init.base_yaw)
        self.vel = np.zeros(3)
        self.wz = 0.0
        self.cmd = np.zeros(3)
        self.push = np.zeros(3)
        self.fallen = False

    def send_movement_state(self, state: MovementState) -> None:
        # Undo the world-frame planner encoding to recover the body-frame twist
        # the mock dynamics track (a real backend just forwards the message).
        speed = max(state.movement_speed, 0.0)
        mx, my, _ = state.movement_direction
        fx, fy, _ = state.facing_direction
        facing_yaw = math.atan2(fy, fx)
        if speed > 1e-6 and (abs(mx) + abs(my)) > 1e-6:
            move_yaw = math.atan2(my, mx)
            rel = move_yaw - facing_yaw
            self.cmd[0], self.cmd[1] = speed * math.cos(rel), speed * math.sin(rel)
        else:
            self.cmd[0] = self.cmd[1] = 0.0
        self.cmd[2] = _wrap(facing_yaw - self.yaw) / max(self.config.sim.command_dt, 1e-6)

    def apply_push(self, force_world, duration: float) -> None:
        self.push = np.asarray(force_world, dtype=float)

    def base_yaw(self) -> float:
        return self.yaw

    def is_fallen(self) -> bool:
        return self.fallen

    def step(self) -> StepSample:
        dt, rng = self.dt, self.rng
        # tracking droop: the faster the command, the larger the shortfall
        target = self.cmd[:2] * (1.0 - 0.06 * abs(self.cmd[0]))
        self.vel[:2] += (target - self.vel[:2]) * min(dt / 0.35, 1.0)
        self.vel[:2] += self.push[:2] / self.mass * dt
        self.vel[:2] += rng.normal(0.0, 0.01, 2)
        self.wz += (self.cmd[2] - self.wz) * min(dt / 0.3, 1.0)

        self.yaw = _wrap(self.yaw + self.wz * dt)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.pos[0] += (c * self.vel[0] - s * self.vel[1]) * dt
        self.pos[1] += (s * self.vel[0] + c * self.vel[1]) * dt
        self.t += dt

        push_mag = float(np.linalg.norm(self.push))
        tip = 0.0006 * push_mag
        # a strong push while walking fast tips the mock robot over
        if push_mag > 180.0 and abs(self.vel[0]) > 0.8:
            self.fallen = True
        self.pos[2] = self.config.init.base_height + 0.01 * math.sin(2 * math.pi * 1.6 * self.t)
        if self.fallen:
            self.pos[2] = 0.25

        roll = tip * math.cos(self.t) + 0.01 * rng.normal()
        pitch = -0.02 * self.vel[0] + tip + 0.01 * rng.normal()
        quat = _rpy_to_quat(roll, pitch, self.yaw)

        phase = 2 * math.pi * 1.6 * self.t
        moving = np.linalg.norm(self.cmd[:2]) > 1e-3
        contact = [1.0, 1.0] if not moving else [
            float(math.sin(phase) > -0.3), float(math.sin(phase + math.pi) > -0.3)
        ]
        foot_pos = [
            [self.pos[0] + 0.1 * math.cos(phase), self.pos[1] + 0.1, 0.03],
            [self.pos[0] + 0.1 * math.cos(phase + math.pi), self.pos[1] - 0.1, 0.03],
        ]
        joint_vel = 0.5 * np.sin(phase + np.arange(29) * 0.2) * (1.0 + abs(self.cmd[0]))
        joint_torque = 20.0 * np.cos(phase + np.arange(29) * 0.2)

        return StepSample(
            t=self.t,
            cmd_vx=self.cmd[0], cmd_vy=self.cmd[1], cmd_yaw_rate=self.cmd[2],
            base_pos=self.pos.copy(),
            base_quat=quat,
            lin_vel_body=[self.vel[0], self.vel[1], 0.0],
            ang_vel_body=[0.0, 0.0, self.wz],
            joint_pos=np.zeros(29),
            joint_vel=joint_vel,
            joint_torque=joint_torque,
            foot_contact=contact,
            foot_pos=foot_pos,
        )


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _rpy_to_quat(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]
