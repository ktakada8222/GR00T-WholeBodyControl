"""Backend-agnostic episode driver: the benchmark protocol itself.

One episode is::

    reset(seed)
    |<-- settle -->|<---------------- recorded episode ---------------->|
      IDLE command    constant (vx, vy, yaw_rate), push at t = push.time

Commands are re-sent at ``sim.command_dt`` (the deploy planner runs at 10 Hz and
treats planner messages older than ~1 s as stale), while state is sampled every
``sim.control_dt``.  Falls are detected with the shared geometric criteria in
``config.fall`` so MuJoCo and IsaacLab agree on what "fell" means.
"""

from __future__ import annotations

import math
import time

import numpy as np

from gear_sonic_eval.backends.base import EvalBackend
from gear_sonic_eval.core.commands import CommandConverter, VelocityCommand
from gear_sonic_eval.core.config import ConditionConfig, EvalConfig, direction_to_world_force
from gear_sonic_eval.core.metrics import EpisodeRecord, compute_metrics, quat_to_rpy
from gear_sonic_eval.core.results import ResultWriter


def run_episode(
    backend: EvalBackend,
    config: EvalConfig,
    condition: ConditionConfig,
    *,
    seed: int,
    episode_index: int,
    disturbance: dict | None = None,
    verbose: bool = True,
) -> EpisodeRecord:
    sim = config.sim
    rng = np.random.default_rng(seed)
    backend.reset(seed)
    backend.begin_episode(condition, disturbance)

    converter = CommandConverter(
        initial_yaw=config.init.base_yaw, height=-1.0, forced_mode=condition.locomotion_mode
    )
    converter.reset(config.init.base_yaw)

    record = EpisodeRecord(
        condition=condition.name,
        group=condition.group,
        backend=backend.name,
        seed=seed,
        episode_index=episode_index,
        cmd=(condition.vx, condition.vy, condition.yaw_rate),
        disturbance=disturbance,
    )

    idle = VelocityCommand(0.0, 0.0, 0.0)
    record.waveform = condition.waveform
    record.axis = condition.axis
    record.frequency = condition.frequency

    n_settle = int(round(sim.settle_duration / sim.control_dt))
    n_episode = int(round(sim.episode_duration / sim.control_dt))
    command_every = max(int(round(sim.command_dt / sim.control_dt)), 1)

    push_start = push_end = None
    if disturbance:
        push_start = int(round(disturbance["time"] / sim.control_dt))
        push_end = push_start + max(int(round(disturbance["duration"] / sim.control_dt)), 1)

    fall_counter = 0
    t_wall = time.monotonic()

    # The command actually in force between two sends (the planner receives a
    # 10 Hz staircase, so that -- not the continuous ideal -- is what we log).
    current = idle
    for i in range(-n_settle, n_episode):
        active = i >= 0

        if (i + n_settle) % command_every == 0:
            current = condition.command_at(i * sim.control_dt) if active else idle
            state = converter.step(current, sim.command_dt if active else 0.0)
            backend.send_movement_state(state)

        pushing = bool(disturbance) and active and push_start <= i < push_end
        if pushing:
            force = direction_to_world_force(
                disturbance["direction"], disturbance["force"], backend.base_yaw(), rng
            )
            backend.apply_push(force, sim.control_dt)
        elif disturbance and active and i == push_end:
            backend.apply_push((0.0, 0.0, 0.0), sim.control_dt)

        sample = backend.step()
        sample.cmd_vx, sample.cmd_vy, sample.cmd_yaw_rate = current.as_tuple()
        sample.pushed = pushing

        if active:
            sample.t = i * sim.control_dt
            record.add(sample)
            record.duration = sample.t + sim.control_dt

        if _fallen(sample, config, backend):
            fall_counter += 1
            if fall_counter >= config.fall.hold_steps:
                record.fell = True
                record.fall_phase = "command" if active else "settle"
                record.fall_time = (i * sim.control_dt) if active else 0.0
                if verbose:
                    where = "while walking" if active else "during settle (before the command)"
                    print(f"    fall detected at t={record.fall_time:.2f}s {where}")
                break
        else:
            fall_counter = 0

        if sim.real_time:
            t_wall += sim.control_dt
            delay = t_wall - time.monotonic()
            if delay > 0:
                time.sleep(delay)

    return record


def _fallen(sample, config: EvalConfig, backend: EvalBackend) -> bool:
    if backend.is_fallen():
        return True
    height = float(sample.base_pos[2])
    if height < config.fall.min_base_height:
        return True
    quat = np.asarray([sample.base_quat], dtype=float)
    rpy = quat_to_rpy(quat)[0]
    tilt = max(abs(rpy[0]), abs(rpy[1]))
    return tilt > math.radians(config.fall.max_tilt_deg)


def run_evaluation(
    backend: EvalBackend,
    config: EvalConfig,
    output_dir: str,
    *,
    save_timeseries: bool = True,
    verbose: bool = True,
) -> ResultWriter:
    """Run every episode of the manifest and write the full result tree."""
    backend.prepare()
    writer = ResultWriter(output_dir, backend.name, config=config, run_info=backend.info)
    episodes = config.episodes()
    for n, ep in enumerate(episodes, start=1):
        cond: ConditionConfig = ep["condition"]
        if verbose:
            push = ep["disturbance"]
            tag = "" if not push else f" push={push['force']:g}N/{push['direction']}"
            print(f"[{n}/{len(episodes)}] {cond.name} (vx={cond.vx:+.2f} vy={cond.vy:+.2f} "
                  f"wz={cond.yaw_rate:+.2f}){tag} seed={ep['seed']}")
        # An episode that collapses during the settle phase never applied the
        # command, so it measures the reset, not the planner. Retry it instead
        # of letting it pollute the fall statistics.
        retries = 0
        while True:
            record = run_episode(
                backend,
                config,
                cond,
                seed=ep["seed"] + 1000 * retries,
                episode_index=ep["episode_index"],
                disturbance=ep["disturbance"],
                verbose=verbose,
            )
            record.reset_retries = retries
            if record.fall_phase != "settle" or retries >= config.max_reset_retries:
                break
            retries += 1
            if verbose:
                print(f"    settle-phase collapse -> retrying ({retries}/{config.max_reset_retries})")
        if record.fall_phase == "settle" and verbose:
            print("    WARNING: episode still collapsed during settle; recorded as invalid")
        row = compute_metrics(
            record,
            transient=config.sim.transient_duration,
            dt=config.sim.control_dt,
            mass=backend.mass,
        )
        writer.add_episode(row, record.timeseries() if save_timeseries else None)
        if verbose:
            print(
                "    vx {:+.3f}/{:+.3f} rmse {:.3f} | height {:.3f}+-{:.3f} | {}".format(
                    row.get("vx_cmd", float("nan")),
                    row.get("vx_actual_mean", float("nan")),
                    row.get("vx_rmse", float("nan")),
                    row.get("base_height_mean", float("nan")),
                    row.get("base_height_std", float("nan")),
                    "FELL" if row["fell"] else "ok",
                )
            )
    paths = writer.finalize()
    if verbose:
        print(f"\nresults written to {writer.root}")
        for key, path in paths.items():
            print(f"  {key:18s} {path}")
    return writer
