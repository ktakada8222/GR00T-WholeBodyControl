"""Plot generation from the result CSVs (matplotlib only, headless-safe)."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic_eval.core.results import read_csv  # noqa: E402

AXES = [("vx", "v_x [m/s]"), ("vy", "v_y [m/s]"), ("yaw_rate", "yaw rate [rad/s]")]


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _no_push(rows):
    return [r for r in rows if str(r.get("push_direction", "none")) == "none"]


def _num(row, key):
    try:
        return float(row.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def plot_all(result_root: str | Path, verbose: bool = True) -> list[Path]:
    """Generate every benchmark plot for one backend's result directory."""
    root = Path(result_root)
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    summary = read_csv(root / "summary.csv")
    episodes = read_csv(root / "episodes.csv")
    made: list[Path] = []
    if not summary:
        return made

    made += _tracking_curves(summary, plots)
    made += _error_curves(summary, plots)
    made += _stability_vs_speed(summary, plots)
    made += _success_rate(summary, plots)
    made += _timeseries(root, plots)
    made += _disturbance(summary, plots)
    made += _heatmaps(summary, plots)
    if verbose:
        for p in made:
            print(f"  plot {p}")
    return made


def _by_group(rows, groups):
    return [r for r in rows if str(r.get("group", "")) in groups]


def _tracking_curves(summary, plots) -> list[Path]:
    """(1) commanded vs actual velocity, per axis."""
    out = []
    for axis, label in AXES:
        cmd_key = {"vx": "cmd_vx", "vy": "cmd_vy", "yaw_rate": "cmd_yaw_rate"}[axis]
        rows = [r for r in _no_push(summary) if abs(_num(r, cmd_key)) > 1e-9]
        if not rows:
            continue
        rows.sort(key=lambda r: _num(r, cmd_key))
        cmd = [_num(r, cmd_key) for r in rows]
        act = [_num(r, f"{axis}_actual_mean") for r in rows]
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        lim = [min(cmd + act), max(cmd + act)]
        ax.plot(lim, lim, "k--", lw=1, label="ideal")
        ax.plot(cmd, act, "o-", label="achieved")
        ax.set_xlabel(f"commanded {label}")
        ax.set_ylabel(f"actual {label}")
        ax.set_title(f"Command tracking ({axis})")
        ax.grid(alpha=0.3)
        ax.legend()
        out.append(_save(fig, plots / f"01_tracking_{axis}.png"))
    return out


def _error_curves(summary, plots) -> list[Path]:
    """(2)+(3) tracking error and RMSE against commanded speed."""
    out = []
    for axis, label in AXES:
        cmd_key = {"vx": "cmd_vx", "vy": "cmd_vy", "yaw_rate": "cmd_yaw_rate"}[axis]
        rows = [r for r in _no_push(summary) if abs(_num(r, cmd_key)) > 1e-9]
        if not rows:
            continue
        rows.sort(key=lambda r: _num(r, cmd_key))
        cmd = [_num(r, cmd_key) for r in rows]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(cmd, [_num(r, f"{axis}_rmse") for r in rows], "o-", label="RMSE")
        ax.plot(cmd, [_num(r, f"{axis}_mae") for r in rows], "s-", label="MAE")
        ax.plot(cmd, [abs(_num(r, f"{axis}_steady_err")) for r in rows], "^-", label="|steady err|")
        ax.set_xlabel(f"commanded {label}")
        ax.set_ylabel("error")
        ax.set_title(f"Tracking error vs command ({axis})")
        ax.grid(alpha=0.3)
        ax.legend()
        out.append(_save(fig, plots / f"03_tracking_error_{axis}.png"))
    return out


def _stability_vs_speed(summary, plots) -> list[Path]:
    """(6) stability metrics against commanded forward speed."""
    rows = [r for r in _no_push(summary) if str(r.get("group")) in ("forward", "backward")]
    if not rows:
        rows = _no_push(summary)
    rows.sort(key=lambda r: _num(r, "cmd_vx"))
    if not rows:
        return []
    vx = [_num(r, "cmd_vx") for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, key, title in (
        (axes[0][0], "base_height_std", "base height std [m]"),
        (axes[0][1], "tilt_rms_deg", "tilt RMS [deg]"),
        (axes[1][0], "roll_std", "roll std [rad]"),
        (axes[1][1], "pitch_std", "pitch std [rad]"),
    ):
        ax.plot(vx, [_num(r, key) for r in rows], "o-")
        ax.set_xlabel("commanded v_x [m/s]")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle("Stability vs commanded speed")
    return [_save(fig, plots / "06_stability_vs_speed.png")]


def _success_rate(summary, plots) -> list[Path]:
    """(7) success rate against commanded speed, per command group."""
    rows = _no_push(summary)
    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = sorted({str(r.get("group", "")) for r in rows})
    for g in groups:
        sub = [r for r in rows if str(r.get("group", "")) == g]
        key = "cmd_yaw_rate" if g == "turn" else ("cmd_vy" if g == "lateral" else "cmd_vx")
        sub.sort(key=lambda r: _num(r, key))
        ax.plot([_num(r, key) for r in sub], [_num(r, "success_rate") for r in sub], "o-", label=g)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("command value")
    ax.set_ylabel("success rate")
    ax.set_title("Success rate vs command")
    ax.grid(alpha=0.3)
    ax.legend()
    return [_save(fig, plots / "07_success_rate.png")]


def _timeseries(root: Path, plots: Path) -> list[Path]:
    """(4)+(5) base height and roll/pitch over time, for every episode."""
    out = []
    ts_dir = root / "timeseries"
    files = sorted(ts_dir.glob("*.csv"))[:12]  # keep the plot directory manageable
    if not files:
        return out
    fig_h, ax_h = plt.subplots(figsize=(7, 4))
    fig_rp, ax_rp = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    fig_v, ax_v = plt.subplots(figsize=(7, 4))
    for path in files:
        rows = read_csv(path)
        if not rows:
            continue
        t = np.array([_num(r, "t") for r in rows])
        label = path.stem
        ax_h.plot(t, [_num(r, "base_height") for r in rows], lw=1, label=label)
        ax_rp[0].plot(t, [_num(r, "roll") for r in rows], lw=1)
        ax_rp[1].plot(t, [_num(r, "pitch") for r in rows], lw=1, label=label)
        ax_v.plot(t, [_num(r, "vx") for r in rows], lw=1, label=f"{label} actual")
        ax_v.plot(t, [_num(r, "cmd_vx") for r in rows], lw=0.8, ls="--", alpha=0.6)
    ax_h.set_xlabel("t [s]"); ax_h.set_ylabel("base height [m]"); ax_h.grid(alpha=0.3)
    ax_h.set_title("Base height over time")
    ax_h.legend(fontsize=6, ncol=2)
    out.append(_save(fig_h, plots / "04_base_height_time.png"))
    ax_rp[0].set_ylabel("roll [rad]"); ax_rp[1].set_ylabel("pitch [rad]")
    ax_rp[1].set_xlabel("t [s]")
    for a in ax_rp:
        a.grid(alpha=0.3)
    ax_rp[0].set_title("Roll / pitch over time")
    out.append(_save(fig_rp, plots / "05_roll_pitch_time.png"))
    ax_v.set_xlabel("t [s]"); ax_v.set_ylabel("v_x [m/s]"); ax_v.grid(alpha=0.3)
    ax_v.set_title("v_x: command (dashed) vs actual")
    ax_v.legend(fontsize=6, ncol=2)
    out.append(_save(fig_v, plots / "02_vx_time.png"))
    return out


def _disturbance(summary, plots) -> list[Path]:
    """(8) success rate against push strength."""
    rows = [r for r in summary if str(r.get("push_direction", "none")) != "none"]
    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(6, 4))
    for direction in sorted({str(r["push_direction"]) for r in rows}):
        sub = sorted((r for r in rows if str(r["push_direction"]) == direction),
                     key=lambda r: _num(r, "push_force"))
        ax.plot([_num(r, "push_force") for r in sub], [_num(r, "success_rate") for r in sub],
                "o-", label=direction)
    ax.set_xlabel("push force [N]")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Disturbance rejection")
    ax.grid(alpha=0.3)
    ax.legend()
    return [_save(fig, plots / "08_push_success_rate.png")]


def _heatmap(fig_path: Path, xs, ys, values, xlabel, ylabel, title, cmap="viridis"):
    grid = np.full((len(ys), len(xs)), np.nan)
    for (xi, yi), v in values.items():
        grid[ys.index(yi), xs.index(xi)] = v
    fig, ax = plt.subplots(figsize=(1.2 * len(xs) + 3, 0.9 * len(ys) + 2.5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
    ax.set_yticks(range(len(ys)), [f"{y}" for y in ys])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    for i in range(len(ys)):
        for j in range(len(xs)):
            if not math.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w")
    fig.colorbar(im, ax=ax)
    return _save(fig, fig_path)


def _heatmaps(summary, plots) -> list[Path]:
    """Heat maps: speed x yaw-rate -> success, speed x push force -> fall rate."""
    out = []
    rows = _no_push(summary)
    xs = sorted({_num(r, "cmd_vx") for r in rows})
    ys = sorted({_num(r, "cmd_yaw_rate") for r in rows})
    if len(xs) > 1 and len(ys) > 1:
        vals = {(_num(r, "cmd_vx"), _num(r, "cmd_yaw_rate")): _num(r, "success_rate") for r in rows}
        out.append(_heatmap(plots / "10_heatmap_speed_yaw_success.png", xs, ys, vals,
                            "commanded v_x [m/s]", "commanded yaw rate [rad/s]",
                            "Success rate: speed x yaw rate"))
    pushed = [r for r in summary if str(r.get("push_direction", "none")) != "none"]
    xs = sorted({_num(r, "cmd_vx") for r in pushed})
    ys = sorted({_num(r, "push_force") for r in pushed})
    if len(xs) >= 1 and len(ys) > 1:
        vals = {(_num(r, "cmd_vx"), _num(r, "push_force")): _num(r, "fall_rate") for r in pushed}
        out.append(_heatmap(plots / "11_heatmap_speed_push_fall.png", xs, ys, vals,
                            "commanded v_x [m/s]", "push force [N]",
                            "Fall rate: speed x push force", cmap="magma"))
    return out


def plot_comparison(roots: dict[str, str | Path], out_dir: str | Path) -> list[Path]:
    """(9) MuJoCo vs IsaacLab: tracking, error and success rate side by side."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {name: _no_push(read_csv(Path(root) / "summary.csv")) for name, root in roots.items()}
    data = {k: v for k, v in data.items() if v}
    if len(data) < 2:
        return []
    made = []
    for axis, label in AXES:
        cmd_key = {"vx": "cmd_vx", "vy": "cmd_vy", "yaw_rate": "cmd_yaw_rate"}[axis]
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
        any_data = False
        for name, rows in data.items():
            sub = sorted((r for r in rows if abs(_num(r, cmd_key)) > 1e-9), key=lambda r: _num(r, cmd_key))
            if not sub:
                continue
            any_data = True
            cmd = [_num(r, cmd_key) for r in sub]
            axs[0].plot(cmd, [_num(r, f"{axis}_actual_mean") for r in sub], "o-", label=name)
            axs[1].plot(cmd, [_num(r, f"{axis}_rmse") for r in sub], "o-", label=name)
            axs[2].plot(cmd, [_num(r, "success_rate") for r in sub], "o-", label=name)
        if not any_data:
            plt.close(fig)
            continue
        lim = axs[0].get_xlim()
        axs[0].plot(lim, lim, "k--", lw=1)
        for ax, title in zip(axs, ("achieved", "RMSE", "success rate")):
            ax.set_xlabel(f"commanded {label}")
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle(f"Sim-to-sim comparison ({axis})")
        made.append(_save(fig, out_dir / f"09_sim2sim_{axis}.png"))
    return made
