"""Generates the required comparison plots from summary/episode CSVs.
Shared by both sim backends and by the mujoco-vs-isaaclab comparison report.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_tracking_timeseries(episode_df: pd.DataFrame, out_dir: str, condition: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for ax, cmd_c, act_c, label in [
        (axes[0], "cmd_vx", "act_vx", "vx [m/s]"),
        (axes[1], "cmd_vy", "act_vy", "vy [m/s]"),
        (axes[2], "cmd_yaw_rate", "act_yaw_rate", "yaw rate [rad/s]"),
    ]:
        ax.plot(episode_df["t"], episode_df[cmd_c], "--", label="commanded")
        ax.plot(episode_df["t"], episode_df[act_c], "-", label="actual")
        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"Velocity tracking - {condition}")
    _save(fig, out_dir, f"velocity_tracking_{condition}.png")


def plot_tracking_error(episode_df: pd.DataFrame, out_dir: str, condition: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episode_df["t"], episode_df["act_vx"] - episode_df["cmd_vx"], label="vx error")
    ax.plot(episode_df["t"], episode_df["act_vy"] - episode_df["cmd_vy"], label="vy error")
    ax.plot(episode_df["t"], episode_df["act_yaw_rate"] - episode_df["cmd_yaw_rate"], label="yaw rate error")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error")
    ax.legend(fontsize=8)
    ax.set_title(f"Tracking error - {condition}")
    _save(fig, out_dir, f"tracking_error_{condition}.png")


def plot_base_orientation_height(episode_df: pd.DataFrame, out_dir: str, condition: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(episode_df["t"], episode_df["base_height"])
    axes[0].set_ylabel("base height [m]")
    axes[1].plot(episode_df["t"], np.degrees(episode_df["roll"]), label="roll")
    axes[1].plot(episode_df["t"], np.degrees(episode_df["pitch"]), label="pitch")
    axes[1].plot(episode_df["t"], np.degrees(episode_df["yaw"]), label="yaw")
    axes[1].set_ylabel("deg")
    axes[1].set_xlabel("time [s]")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Base height / orientation - {condition}")
    _save(fig, out_dir, f"base_height_orientation_{condition}.png")


def plot_rmse_vs_speed(summary_df: pd.DataFrame, out_dir: str) -> None:
    df = summary_df.copy()
    df["cmd_speed"] = np.hypot(df["vx"], df["vy"])
    g = df.groupby("cmd_speed").agg(vx_rmse=("vx_rmse", "mean"), vy_rmse=("vy_rmse", "mean"),
                                     yaw_rate_rmse=("yaw_rate_rmse", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g["cmd_speed"], g["vx_rmse"], "o-", label="vx RMSE")
    ax.plot(g["cmd_speed"], g["vy_rmse"], "o-", label="vy RMSE")
    ax.plot(g["cmd_speed"], g["yaw_rate_rmse"], "o-", label="yaw rate RMSE")
    ax.set_xlabel("commanded speed [m/s]")
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=8)
    ax.set_title("Tracking RMSE vs commanded speed")
    _save(fig, out_dir, "rmse_vs_speed.png")


def plot_stability_vs_speed(summary_df: pd.DataFrame, out_dir: str) -> None:
    df = summary_df.copy()
    df["cmd_speed"] = np.hypot(df["vx"], df["vy"])
    g = df.groupby("cmd_speed").agg(base_height_std=("base_height_std", "mean"),
                                     roll_std=("roll_std", "mean"),
                                     pitch_std=("pitch_std", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g["cmd_speed"], g["base_height_std"], "o-", label="base height std")
    ax.plot(g["cmd_speed"], g["roll_std"], "o-", label="roll std")
    ax.plot(g["cmd_speed"], g["pitch_std"], "o-", label="pitch std")
    ax.set_xlabel("commanded speed [m/s]")
    ax.set_ylabel("std")
    ax.legend(fontsize=8)
    ax.set_title("Stability vs commanded speed")
    _save(fig, out_dir, "stability_vs_speed.png")


def plot_success_rate_vs_speed(summary_df: pd.DataFrame, out_dir: str) -> None:
    df = summary_df.copy()
    df["cmd_speed"] = np.hypot(df["vx"], df["vy"])
    g = df.groupby("cmd_speed").agg(success_rate=("success", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g["cmd_speed"], g["success_rate"], "o-")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("commanded speed [m/s]")
    ax.set_ylabel("success rate")
    ax.set_title("Success rate vs commanded speed")
    _save(fig, out_dir, "success_rate_vs_speed.png")


def plot_disturbance_vs_success(summary_df: pd.DataFrame, out_dir: str) -> None:
    if "force_n" not in summary_df.columns:
        return
    g = summary_df.groupby("force_n").agg(success_rate=("success", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g["force_n"], g["success_rate"], "o-")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("push force [N]")
    ax.set_ylabel("success rate")
    ax.set_title("Disturbance strength vs success rate")
    _save(fig, out_dir, "disturbance_vs_success_rate.png")


def plot_heatmap_speed_yaw_success(summary_df: pd.DataFrame, out_dir: str) -> None:
    df = summary_df.copy()
    if "yaw_rate" not in df.columns:
        return
    df["cmd_speed"] = np.hypot(df["vx"], df["vy"])
    pivot = df.pivot_table(index="yaw_rate", columns="cmd_speed", values="success", aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])
    ax.set_xlabel("speed [m/s]")
    ax.set_ylabel("yaw rate [rad/s]")
    fig.colorbar(im, ax=ax, label="success rate")
    ax.set_title("Success rate: speed x yaw rate")
    _save(fig, out_dir, "heatmap_speed_yaw_success.png")


def plot_heatmap_speed_disturbance_fall(summary_df: pd.DataFrame, out_dir: str) -> None:
    if "force_n" not in summary_df.columns:
        return
    df = summary_df.copy()
    df["cmd_speed"] = np.hypot(df["vx"], df["vy"])
    df["fell"] = ~df["success"]
    pivot = df.pivot_table(index="force_n", columns="cmd_speed", values="fell", aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.0f}" for r in pivot.index])
    ax.set_xlabel("speed [m/s]")
    ax.set_ylabel("push force [N]")
    fig.colorbar(im, ax=ax, label="fall rate")
    ax.set_title("Fall rate: speed x disturbance strength")
    _save(fig, out_dir, "heatmap_speed_disturbance_fall.png")


def plot_mujoco_vs_isaaclab(mujoco_summary: pd.DataFrame, isaaclab_summary: pd.DataFrame, out_dir: str) -> None:
    m, i = mujoco_summary.copy(), isaaclab_summary.copy()
    m["cmd_speed"], i["cmd_speed"] = np.hypot(m["vx"], m["vy"]), np.hypot(i["vx"], i["vy"])
    mg = m.groupby("cmd_speed")["vx_rmse"].mean()
    ig = i.groupby("cmd_speed")["vx_rmse"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mg.index, mg.values, "o-", label="MuJoCo")
    ax.plot(ig.index, ig.values, "s-", label="IsaacLab")
    ax.set_xlabel("commanded speed [m/s]")
    ax.set_ylabel("vx RMSE")
    ax.legend()
    ax.set_title("MuJoCo vs IsaacLab: vx tracking RMSE")
    _save(fig, out_dir, "mujoco_vs_isaaclab_vx_rmse.png")


def generate_all_summary_plots(summary_df: pd.DataFrame, out_dir: str) -> None:
    plot_rmse_vs_speed(summary_df, out_dir)
    plot_stability_vs_speed(summary_df, out_dir)
    plot_success_rate_vs_speed(summary_df, out_dir)
    plot_disturbance_vs_success(summary_df, out_dir)
    plot_heatmap_speed_yaw_success(summary_df, out_dir)
    plot_heatmap_speed_disturbance_fall(summary_df, out_dir)
