"""Conversion between benchmark velocity commands and Sonic planner MovementState.

The Sonic planner is *not* a velocity controller.  Its per-tick command
(``MovementState`` in ``gear_sonic_deploy/.../localmotion_kplanner.hpp``) is

    locomotion_mode     : LocomotionMode enum (IDLE / SLOW_WALK / WALK / RUN / ...)
    movement_direction  : world-frame unit vector of desired travel
    facing_direction    : world-frame unit vector the pelvis should face
    movement_speed      : scalar speed [m/s]  (-1 = mode default, 0 = stationary)
    height              : body height [m]     (-1 = mode default)

A benchmark condition is expressed the usual way, as a *body-frame* twist
``(vx, vy, yaw_rate)``.  This module performs the mapping the keyboard handler
performs interactively (see ``keyboard_handler.hpp::handle_input``):

  * the commanded heading is an integrated angle
    ``yaw_cmd(t+dt) = yaw_cmd(t) + yaw_rate * dt``  (the keyboard nudges the same
    ``planner_facing_angle`` by +/- 0.1 rad or +/- pi/6 per key press),
  * ``facing_direction = [cos(yaw_cmd), sin(yaw_cmd), 0]``,
  * ``movement_speed   = hypot(vx, vy)``,
  * ``movement_direction = Rz(yaw_cmd) @ [vx, vy, 0] / speed``  (so vx<0 gives the
    ``-facing`` vector used by the backward key, and vy>0 gives the left-strafe
    vector ``[-sin, cos, 0]``),
  * the locomotion mode is picked from the speed using the ranges documented on
    the enum (SLOW_WALK 0.1-0.8, WALK 0.8-2.5, RUN 2.5-7.5 m/s).

Pure turning (``speed == 0``, ``yaw_rate != 0``) is emitted as IDLE with a
rotating ``facing_direction``.  The planner replans on any facing change
(``facing_direction_changed`` in ``g1_deploy_onnx_ref.cpp::Planner``), so this is
the turn-in-place command the planner actually understands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


class LocomotionMode:
    """Mirror of ``LocomotionMode`` (localmotion_kplanner.hpp). Only the subset
    used by walking evaluation is listed; values must match the C++ enum."""

    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3


#: Speed ranges of the walking modes, from the enum comments in the C++ header.
MODE_SPEED_RANGES = {
    LocomotionMode.SLOW_WALK: (0.1, 0.8),
    LocomotionMode.WALK: (0.8, 2.5),
    LocomotionMode.RUN: (2.5, 7.5),
}


@dataclass(frozen=True)
class VelocityCommand:
    """Body-frame twist used as the benchmark-level command."""

    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.vx, self.vy, self.yaw_rate)


@dataclass
class MovementState:
    """Python mirror of the C++ ``MovementState`` struct."""

    locomotion_mode: int = LocomotionMode.IDLE
    movement_direction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    facing_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    movement_speed: float = -1.0
    height: float = -1.0

    def as_dict(self) -> dict:
        return {
            "locomotion_mode": self.locomotion_mode,
            "movement_direction": list(self.movement_direction),
            "facing_direction": list(self.facing_direction),
            "movement_speed": self.movement_speed,
            "height": self.height,
        }


def select_mode(speed: float, *, min_walk_speed: float = 1e-3) -> int:
    """Pick the locomotion mode whose documented range contains ``speed``."""
    if speed < min_walk_speed:
        return LocomotionMode.IDLE
    if speed <= MODE_SPEED_RANGES[LocomotionMode.SLOW_WALK][1]:
        return LocomotionMode.SLOW_WALK
    if speed <= MODE_SPEED_RANGES[LocomotionMode.WALK][1]:
        return LocomotionMode.WALK
    return LocomotionMode.RUN


@dataclass
class CommandConverter:
    """Stateful ``(vx, vy, yaw_rate)`` -> ``MovementState`` converter.

    The heading is integrated, so the converter must be stepped at a fixed rate
    and reset between episodes (``reset()``) for runs to be reproducible.
    """

    initial_yaw: float = 0.0
    height: float = -1.0
    #: If the mode's speed range does not contain the requested speed we still
    #: send the requested speed (the planner clamps internally); set this to
    #: True to clamp on our side instead, which makes the command we log equal
    #: to the command the planner receives.
    clamp_speed_to_mode: bool = False
    #: Pin ``locomotion_mode`` instead of deriving it from the speed.  Needed
    #: whenever the commanded speed varies within an episode (the sine
    #: scenario): without it, crossing a mode boundary -- or passing through
    #: zero -- switches the planner's mode mid-episode, which triggers a replan
    #: (``movement_mode_changed`` in g1_deploy_onnx_ref.cpp) and measures the
    #: mode transition instead of the velocity-tracking response.
    forced_mode: int | None = None
    yaw_cmd: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.yaw_cmd = self.initial_yaw

    def reset(self, initial_yaw: float | None = None) -> None:
        self.yaw_cmd = self.initial_yaw if initial_yaw is None else initial_yaw

    def step(self, cmd: VelocityCommand, dt: float) -> MovementState:
        """Advance the integrated heading by ``dt`` and build the MovementState."""
        self.yaw_cmd = _wrap_pi(self.yaw_cmd + cmd.yaw_rate * dt)
        return self.to_movement_state(cmd)

    def to_movement_state(self, cmd: VelocityCommand) -> MovementState:
        """Build the MovementState for the current heading without advancing it."""
        c, s = math.cos(self.yaw_cmd), math.sin(self.yaw_cmd)
        facing = (c, s, 0.0)

        speed = cmd.speed
        mode = self.forced_mode if self.forced_mode is not None else select_mode(speed)
        if mode == LocomotionMode.IDLE or (self.forced_mode is None and speed < 1e-3):
            # Stationary (possibly turning in place): zero movement vector, and
            # speed 0 rather than -1 so the planner does not fall back to the
            # mode default speed.
            return MovementState(mode, (0.0, 0.0, 0.0), facing, 0.0, self.height)

        if self.clamp_speed_to_mode and mode in MODE_SPEED_RANGES:
            lo, hi = MODE_SPEED_RANGES[mode]
            speed = min(max(speed, lo), hi)

        if speed < 1e-9:  # pinned mode, momentary zero crossing of a sine
            return MovementState(mode, (0.0, 0.0, 0.0), facing, 0.0, self.height)

        # Rotate the body-frame direction into the world frame.
        ux, uy = cmd.vx / cmd.speed, cmd.vy / cmd.speed
        movement = (c * ux - s * uy, s * ux + c * uy, 0.0)
        return MovementState(mode, movement, facing, speed, self.height)


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
