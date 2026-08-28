"""Conversion between (vx, vy, yaw_rate) velocity commands and the Gear Sonic
planner's native MovementState representation.

The Sonic planner (gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner.hpp)
does NOT take (vx, vy, yaw_rate). Its input struct is:

    struct MovementState {
        int locomotion_mode;
        std::array<double,3> movement_direction;   // world-frame unit vector
        std::array<double,3> facing_direction;      // world-frame unit vector (desired heading)
        double movement_speed;                      // scalar speed, m/s (-1 = mode default)
        double height;                               // -1 = mode default
    };

This is exactly what the keyboard input handler
(gear_sonic_deploy/.../input_interface/keyboard_handler.hpp) builds: forward/strafe
keys are combined into a world-frame movement_direction, while a separately
tracked heading angle (incremented by turn keys) becomes facing_direction. A
turn-rate command has no direct field in MovementState -- the real deployment
integrates yaw_rate into a heading angle every control tick and re-sends it as
facing_direction. We reproduce that integration here so evaluation scripts can
issue a standard (vx, vy, yaw_rate) command like any other legged-robot
benchmark and have it converted into the planner's native input on each tick.

ASSUMPTION (not directly verified against source, since the keyboard handler
only demonstrates pure-forward and pure-turn presses, not simultaneous
strafe+turn): vx, vy are expressed in the robot's current heading frame
(forward / lateral), matching the convention used by IsaacLab's
track_lin_vel_xy_yaw_frame_exp for G1. movement_direction is therefore the
world-frame rotation of [vx, vy] by the current heading, and movement_speed is
its magnitude. This should be revisited once the planner's actual
strafe+turn behavior can be checked against ground truth (e.g. by comparing
MuJoCo rollouts against the real robot).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VelocityCommand:
    """A standard legged-robot velocity command."""

    vx: float = 0.0  # forward velocity in robot heading frame [m/s]
    vy: float = 0.0  # lateral velocity in robot heading frame [m/s], +y = left
    yaw_rate: float = 0.0  # [rad/s]
    height: float = -1.0  # desired body height, -1 = planner default
    locomotion_mode: int = 1  # default: WALK (see LocomotionMode enum)


class HeadingIntegrator:
    """Integrates yaw_rate into a heading angle, mirroring
    SimpleKeyboard::handle_input's `planner_facing_angle` accumulator."""

    def __init__(self, initial_heading_rad: float = 0.0):
        self.heading_rad = initial_heading_rad

    def reset(self, heading_rad: float = 0.0) -> None:
        self.heading_rad = heading_rad

    def step(self, yaw_rate: float, dt: float) -> float:
        self.heading_rad = _wrap_to_pi(self.heading_rad + yaw_rate * dt)
        return self.heading_rad


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def velocity_command_to_movement_state(
    cmd: VelocityCommand,
    heading_rad: float,
) -> dict:
    """Convert a VelocityCommand + current integrated heading into the
    planner's native MovementState fields.

    Returns a plain dict with keys matching the C++ struct so it can be
    passed directly to any PlannerClient backend:
        locomotion_mode, movement_direction, facing_direction,
        movement_speed, height
    """
    speed = math.hypot(cmd.vx, cmd.vy)
    if speed < 1e-6:
        movement_direction = (0.0, 0.0, 0.0)
    else:
        # rotate the heading-frame (vx, vy) into world frame by `heading_rad`
        c, s = math.cos(heading_rad), math.sin(heading_rad)
        wx = c * cmd.vx - s * cmd.vy
        wy = s * cmd.vx + c * cmd.vy
        movement_direction = (wx / speed, wy / speed, 0.0)

    facing_direction = (math.cos(heading_rad), math.sin(heading_rad), 0.0)

    return {
        "locomotion_mode": int(cmd.locomotion_mode) if speed > 1e-6 else 0,  # IDLE if stationary
        "movement_direction": movement_direction,
        "facing_direction": facing_direction,
        "movement_speed": speed if speed > 1e-6 else 0.0,
        "height": cmd.height,
    }
