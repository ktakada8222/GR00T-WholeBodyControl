"""YAML-driven evaluation configuration, shared by both the MuJoCo and
IsaacLab runners so a single config file defines identical test conditions
for a sim-to-sim comparison."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class VelocityCondition:
    name: str
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    height: float = -1.0
    locomotion_mode: int = 1


@dataclass
class DisturbanceCondition:
    name: str
    axis: str = "y"  # "x", "y", or "random"
    force_n: float = 100.0
    duration_s: float = 0.2
    apply_at_s: float = 3.0  # time into the episode to apply the push, after settling


@dataclass
class SimParams:
    sim_timestep: float = 0.005      # physics dt, matches gear_sonic SIMULATE_DT (200 Hz)
    control_frequency: float = 50.0  # planner trajectory / control loop rate
    episode_duration_s: float = 10.0
    settle_time_s: float = 1.0       # warm-up excluded from steady-state metrics
    initial_pose: str = "default_standing"


@dataclass
class EvalConfig:
    seeds: list = field(default_factory=lambda: [0])
    sim_params: SimParams = field(default_factory=SimParams)
    velocity_conditions: list = field(default_factory=list)
    disturbance_conditions: list = field(default_factory=list)
    output_dir: str = "results"
    planner_backend: str = "mock"  # "onnx" | "zmq" | "mock"
    planner_model_path: Optional[str] = None

    @staticmethod
    def load(path: str) -> "EvalConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        sim_params = SimParams(**raw.get("sim_params", {}))
        vconds = [VelocityCondition(**c) for c in raw.get("velocity_conditions", [])]
        dconds = [DisturbanceCondition(**c) for c in raw.get("disturbance_conditions", [])]
        return EvalConfig(
            seeds=raw.get("seeds", [0]),
            sim_params=sim_params,
            velocity_conditions=vconds,
            disturbance_conditions=dconds,
            output_dir=raw.get("output_dir", "results"),
            planner_backend=raw.get("planner_backend", "mock"),
            planner_model_path=raw.get("planner_model_path"),
        )

    def save(self, path: str) -> None:
        def as_dict(x):
            return dataclasses.asdict(x) if dataclasses.is_dataclass(x) else x
        raw = {
            "seeds": self.seeds,
            "sim_params": as_dict(self.sim_params),
            "velocity_conditions": [as_dict(c) for c in self.velocity_conditions],
            "disturbance_conditions": [as_dict(c) for c in self.disturbance_conditions],
            "output_dir": self.output_dir,
            "planner_backend": self.planner_backend,
            "planner_model_path": self.planner_model_path,
        }
        with open(path, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False)
