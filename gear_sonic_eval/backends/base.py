"""Backend interface shared by the MuJoCo and IsaacLab evaluators.

A backend owns the simulator and the robot; it does *not* own the benchmark
protocol.  The runner (``core/runner.py``) decides when episodes start, which
command is active, when a push is applied and when an episode is a failure, so
MuJoCo and IsaacLab execute exactly the same schedule.
"""

from __future__ import annotations

import abc
from typing import Optional

from gear_sonic_eval.core.commands import MovementState
from gear_sonic_eval.core.config import EvalConfig
from gear_sonic_eval.core.metrics import StepSample


class EvalBackend(abc.ABC):
    """One simulator + one Sonic-planner command channel."""

    #: Human-readable name, used as the results sub-directory.
    name: str = "base"
    #: Robot mass [kg] for cost-of-transport; None disables that metric.
    mass: Optional[float] = None
    #: Free-form dict written into run_info.json (versions, model paths, ...).
    info: dict

    def __init__(self, config: EvalConfig):
        self.config = config
        self.info = {}

    @abc.abstractmethod
    def reset(self, seed: int) -> None:
        """Put the robot back into the configured initial state.

        Must be fully determined by ``seed`` and ``config.init`` so that a rerun
        with the same config reproduces the same episode.
        """

    @abc.abstractmethod
    def send_movement_state(self, state: MovementState) -> None:
        """Push one planner command (the planner's real input format)."""

    @abc.abstractmethod
    def step(self) -> StepSample:
        """Advance one control step (``config.sim.control_dt``) and observe."""

    @abc.abstractmethod
    def apply_push(self, force_world: tuple[float, float, float], duration: float) -> None:
        """Apply an external force on the pelvis for ``duration`` seconds."""

    @abc.abstractmethod
    def base_yaw(self) -> float:
        """Current world yaw of the pelvis [rad] (used to orient pushes)."""

    def prepare(self) -> None:
        """One-off setup before the first episode (handshakes, warm-up).

        The MuJoCo backend waits here for the deploy binary to come up, so the
        wait is not printed in the middle of the first episode's log line.
        """

    def retry_hint(self, attempt: int) -> None:
        """Called before re-running an episode that collapsed during settle.

        Backends should perturb something so the retry is not a bit-identical
        repeat of a deterministic failure.
        """

    def begin_episode(self, condition, disturbance=None) -> None:
        """Optional hook called after ``reset`` with the episode descriptor.

        Backends that need to know *which* condition is about to run (the
        IsaacLab trajectory-replay mode selects the planner trajectory here)
        override this; command-driven backends can ignore it.
        """

    def is_fallen(self) -> bool:
        """Optional backend-native fall signal; the runner also applies the
        geometric criteria in ``config.fall``."""
        return False

    def close(self) -> None:
        pass
