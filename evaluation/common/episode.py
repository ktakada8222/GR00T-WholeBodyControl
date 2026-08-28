"""CSV logging shared by both runners so `results/mujoco/` and
`results/isaaclab/` come out in an identical format for direct comparison."""
from __future__ import annotations

import dataclasses
import os

import pandas as pd

from .metrics import EpisodeBuffer, EpisodeMetrics


def episode_buffer_to_df(buf: EpisodeBuffer) -> pd.DataFrame:
    return pd.DataFrame({
        "t": buf.t,
        "cmd_vx": buf.cmd_vx, "cmd_vy": buf.cmd_vy, "cmd_yaw_rate": buf.cmd_yaw_rate,
        "act_vx": buf.act_vx, "act_vy": buf.act_vy, "act_yaw_rate": buf.act_yaw_rate,
        "base_height": buf.base_height, "roll": buf.roll, "pitch": buf.pitch, "yaw": buf.yaw,
        "ang_vel_norm": buf.ang_vel_norm, "lin_vel_norm": buf.lin_vel_norm,
    })


class ResultsWriter:
    """Accumulates EpisodeMetrics rows and per-episode timeseries, writing the
    results/<sim>/{summary,velocity_tracking,stability,episodes}.csv layout
    plus plots/ requested in the spec."""

    def __init__(self, output_dir: str, sim_name: str):
        self.sim_name = sim_name
        self.root = os.path.join(output_dir, sim_name)
        self.plots_dir = os.path.join(self.root, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        self._summary_rows: list[dict] = []

    def log_episode(self, metrics: EpisodeMetrics, buf: EpisodeBuffer, cmd: dict,
                     extra_condition_fields: dict | None = None) -> None:
        row = {**dataclasses.asdict(metrics), **cmd, **(extra_condition_fields or {})}
        self._summary_rows.append(row)

        ts_dir = os.path.join(self.root, "episodes")
        os.makedirs(ts_dir, exist_ok=True)
        episode_buffer_to_df(buf).to_csv(
            os.path.join(ts_dir, f"{metrics.condition}_seed{metrics.seed}.csv"), index=False
        )

    def finalize(self) -> pd.DataFrame:
        df = pd.DataFrame(self._summary_rows)
        df.to_csv(os.path.join(self.root, "summary.csv"), index=False)

        vt_cols = ["condition", "seed", "vx", "vy", "yaw_rate",
                   "vx_rmse", "vx_mae", "vx_max_err", "vy_rmse", "vy_mae", "vy_max_err",
                   "yaw_rate_rmse", "yaw_rate_mae", "yaw_rate_max_err", "settling_time"]
        df[[c for c in vt_cols if c in df.columns]].to_csv(
            os.path.join(self.root, "velocity_tracking.csv"), index=False)

        st_cols = ["condition", "seed", "base_height_mean", "base_height_std",
                   "roll_std", "pitch_std", "success", "fell", "time_to_fall",
                   "mech_energy_j", "mean_joint_torque_abs", "step_frequency_hz", "duty_factor"]
        df[[c for c in st_cols if c in df.columns]].to_csv(
            os.path.join(self.root, "stability.csv"), index=False)

        ep_cols = ["condition", "seed", "success", "fell", "time_to_fall"]
        df[[c for c in ep_cols if c in df.columns]].to_csv(
            os.path.join(self.root, "episodes.csv"), index=False)

        return df
