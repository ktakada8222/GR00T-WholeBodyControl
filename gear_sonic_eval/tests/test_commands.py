"""Checks that the (vx, vy, yaw_rate) -> MovementState mapping matches the
semantics of keyboard_handler.hpp (the reference producer of MovementState)."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic_eval.core.commands import (  # noqa: E402
    CommandConverter,
    LocomotionMode,
    VelocityCommand,
    select_mode,
)


def approx(a, b, tol=1e-6):
    assert abs(a - b) < tol, f"{a} != {b}"


def test_mode_selection():
    assert select_mode(0.0) == LocomotionMode.IDLE
    assert select_mode(0.5) == LocomotionMode.SLOW_WALK
    assert select_mode(1.5) == LocomotionMode.WALK
    assert select_mode(3.0) == LocomotionMode.RUN


def test_forward_is_facing_direction():
    """Forward key: movement_direction == facing_direction (keyboard_handler)."""
    c = CommandConverter()
    s = c.to_movement_state(VelocityCommand(vx=0.6))
    approx(s.movement_speed, 0.6)
    assert s.locomotion_mode == LocomotionMode.SLOW_WALK
    for m, f in zip(s.movement_direction, s.facing_direction):
        approx(m, f)


def test_backward_is_negated_facing():
    c = CommandConverter()
    s = c.to_movement_state(VelocityCommand(vx=-0.4))
    approx(s.movement_speed, 0.4)
    for m, f in zip(s.movement_direction, s.facing_direction):
        approx(m, -f)


def test_left_strafe_matches_keyboard():
    """Left key: movement = [-sin(angle), cos(angle), 0]."""
    c = CommandConverter(initial_yaw=0.3)
    s = c.to_movement_state(VelocityCommand(vy=0.5))
    approx(s.movement_direction[0], -math.sin(0.3))
    approx(s.movement_direction[1], math.cos(0.3))
    approx(s.movement_speed, 0.5)


def test_yaw_rate_integrates_heading():
    c = CommandConverter()
    dt = 0.1
    for _ in range(10):
        s = c.step(VelocityCommand(vx=0.0, yaw_rate=0.5), dt)
    approx(c.yaw_cmd, 0.5)  # 10 * 0.1 * 0.5
    approx(math.atan2(s.facing_direction[1], s.facing_direction[0]), 0.5)
    # pure turning is IDLE with a rotating facing vector
    assert s.locomotion_mode == LocomotionMode.IDLE
    approx(s.movement_speed, 0.0)


def test_pinned_mode_survives_sine_sweep():
    """A sine that crosses a mode boundary / zero must not switch the mode."""
    from gear_sonic_eval.core.commands import CommandConverter as CC

    free, pinned = CC(), CC(forced_mode=LocomotionMode.SLOW_WALK)
    # 0.2 .. 1.0 m/s crosses the SLOW_WALK / WALK boundary at 0.8
    modes_free = {free.to_movement_state(VelocityCommand(vx=v)).locomotion_mode
                  for v in (0.2, 0.6, 0.9, 1.0)}
    modes_pinned = {pinned.to_movement_state(VelocityCommand(vx=v)).locomotion_mode
                    for v in (0.2, 0.6, 0.9, 1.0)}
    assert len(modes_free) > 1  # this is the problem the pin solves
    assert modes_pinned == {LocomotionMode.SLOW_WALK}
    # zero crossing of a lateral sine keeps the mode too, with speed 0
    zero = pinned.to_movement_state(VelocityCommand(vy=0.0))
    assert zero.locomotion_mode == LocomotionMode.SLOW_WALK
    approx(zero.movement_speed, 0.0)


def test_compound_command_direction():
    c = CommandConverter()
    cmd = VelocityCommand(vx=0.5, vy=0.3)
    s = c.to_movement_state(cmd)
    approx(s.movement_speed, math.hypot(0.5, 0.3))
    approx(math.atan2(s.movement_direction[1], s.movement_direction[0]), math.atan2(0.3, 0.5))


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
