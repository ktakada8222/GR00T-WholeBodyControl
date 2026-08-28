"""Declarative, reproducible configuration for the Sonic planner walking benchmark.

Everything that affects a run -- seeds, timestep, episode length, the command
grid, the initial pose, the disturbance grid -- lives in a YAML/JSON file so the
same benchmark can be replayed before and after a planner change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import itertools
import json
import math
from pathlib import Path
from typing import Any

from gear_sonic_eval.core.commands import VelocityCommand


@dataclass
class SimConfig:
    """Simulation / control timing. Shared by every backend so MuJoCo and
    IsaacLab run the same schedule."""

    control_dt: float = 0.02  # 50 Hz control loop (deploy Control thread)
    physics_dt: float = 0.005  # 200 Hz MuJoCo physics (SIMULATE_DT)
    command_dt: float = 0.1  # 10 Hz planner command rate (deploy planner_dt_)
    episode_duration: float = 10.0  # [s] recorded per episode
    settle_duration: float = 2.0  # [s] standing before the command is applied
    transient_duration: float = 1.5  # [s] dropped from steady-state statistics
    real_time: bool = False  # throttle MuJoCo to wall-clock


@dataclass
class RobotInitConfig:
    """Deterministic initial state, identical across backends where possible."""

    base_height: float = 0.793
    base_xy: tuple[float, float] = (0.0, 0.0)
    base_yaw: float = 0.0
    base_lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Joint-position noise [rad] applied with the episode seed (0 = deterministic).
    joint_noise: float = 0.0


@dataclass
class FallConfig:
    """Fall / termination criteria (shared definition across backends)."""

    min_base_height: float = 0.4  # [m] pelvis height below this = fallen
    max_tilt_deg: float = 50.0  # [deg] angle between body z and world z
    #: Consecutive control steps a criterion must hold before declaring a fall.
    hold_steps: int = 5


@dataclass
class DisturbanceConfig:
    """External push applied to the pelvis.

    In MuJoCo the push is a constant force written to ``mj_data.xfrc_applied``
    on the pelvis body for ``duration`` seconds (impulse = force * duration).
    In IsaacLab the equivalent is a root-velocity kick (``push_by_setting_velocity``)
    or an external wrench; the runner reports both force and impulse so the two
    can be compared.
    """

    enabled: bool = False
    forces: list[float] = field(default_factory=lambda: [50.0, 100.0, 150.0, 200.0, 250.0])
    directions: list[str] = field(default_factory=lambda: ["front", "back", "left", "right"])
    duration: float = 0.1  # [s]
    time: float = 3.0  # [s] after the command starts
    #: Command conditions (by name) the push grid is crossed with. Empty = all.
    conditions: list[str] = field(default_factory=list)
    #: Also run one un-pushed episode set per condition, so the tracking and
    #: stability baselines are available in the same result directory.
    include_baseline: bool = True
    #: "random" is also accepted in ``directions``; it draws a uniform heading
    #: from the episode RNG, which keeps it reproducible for a given seed.


@dataclass
class ConditionConfig:
    """One benchmark condition = one constant velocity command."""

    name: str
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    group: str = "custom"  # forward / backward / lateral / turn / compound

    def command(self) -> VelocityCommand:
        return VelocityCommand(self.vx, self.vy, self.yaw_rate)


@dataclass
class EvalConfig:
    """Top-level benchmark description."""

    name: str = "sonic_walking_eval"
    seed: int = 0
    num_episodes: int = 1  # repeats per condition (each gets seed + index)
    sim: SimConfig = field(default_factory=SimConfig)
    init: RobotInitConfig = field(default_factory=RobotInitConfig)
    fall: FallConfig = field(default_factory=FallConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    conditions: list[ConditionConfig] = field(default_factory=list)
    #: Free-form per-backend settings (zmq port, task id, ...); see backends/.
    backend: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_file(cls, path: str | Path) -> "EvalConfig":
        path = Path(path)
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            import yaml  # local import: JSON configs need no PyYAML

            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
        return cls.from_dict(raw or {})

    @classmethod
    def from_dict(cls, raw: dict) -> "EvalConfig":
        raw = dict(raw)
        sim = SimConfig(**raw.pop("sim", {}) or {})
        init = RobotInitConfig(**raw.pop("init", {}) or {})
        fall = FallConfig(**raw.pop("fall", {}) or {})
        dist = DisturbanceConfig(**raw.pop("disturbance", {}) or {})
        conditions = _expand_conditions(raw.pop("conditions", []) or [])
        return cls(sim=sim, init=init, fall=fall, disturbance=dist, conditions=conditions, **raw)

    def to_dict(self) -> dict:
        return asdict(self)

    # ---------------------------------------------------------------- helpers
    def disturbance_specs(self, condition: ConditionConfig) -> list[dict | None]:
        """Push variants evaluated for one condition (``[None]`` when disabled)."""
        d = self.disturbance
        if not d.enabled:
            return [None]
        if d.conditions and condition.name not in d.conditions:
            return [None]
        specs: list[dict | None] = [None] if d.include_baseline else []
        for force, direction in itertools.product(d.forces, d.directions):
            specs.append(
                {
                    "force": float(force),
                    "direction": direction,
                    "duration": d.duration,
                    "time": d.time,
                    "impulse": float(force) * d.duration,
                }
            )
        return specs

    def episodes(self) -> list[dict]:
        """Full, ordered list of episodes -- the reproducible run manifest."""
        out = []
        for cond in self.conditions:
            for spec in self.disturbance_specs(cond):
                for ep in range(self.num_episodes):
                    out.append(
                        {
                            "condition": cond,
                            "disturbance": spec,
                            "episode_index": ep,
                            "seed": self.seed + ep,
                        }
                    )
        return out


def _expand_conditions(raw: list) -> list[ConditionConfig]:
    """Accept both explicit conditions and compact sweep entries.

    Compact form::

        - group: forward
          sweep: vx
          values: [0.2, 0.4, 0.6]
    """
    conditions: list[ConditionConfig] = []
    for entry in raw:
        if "sweep" in entry:
            axis = entry["sweep"]
            group = entry.get("group", axis)
            base = {k: entry.get(k, 0.0) for k in ("vx", "vy", "yaw_rate")}
            for value in entry["values"]:
                kwargs = dict(base)
                kwargs[axis] = float(value)
                name = entry.get("name_format", "{group}_{axis}{value:+.2f}").format(
                    group=group, axis=axis, value=float(value)
                )
                conditions.append(ConditionConfig(name=name, group=group, **kwargs))
        else:
            entry = dict(entry)
            entry.setdefault(
                "name",
                "cmd_vx{vx:+.2f}_vy{vy:+.2f}_wz{yaw_rate:+.2f}".format(
                    vx=entry.get("vx", 0.0),
                    vy=entry.get("vy", 0.0),
                    yaw_rate=entry.get("yaw_rate", 0.0),
                ),
            )
            conditions.append(ConditionConfig(**entry))
    return conditions


def direction_to_world_force(direction: str, force: float, yaw: float, rng) -> tuple[float, float, float]:
    """Map a named push direction + robot heading to a world-frame force vector."""
    if direction == "random":
        angle = rng.uniform(-math.pi, math.pi)
    else:
        angle = yaw + {
            "front": 0.0,
            "back": math.pi,
            "left": math.pi / 2,
            "right": -math.pi / 2,
        }[direction]
    return (force * math.cos(angle), force * math.sin(angle), 0.0)
