# Sonic planner walking benchmark

A reproducible evaluation framework for the **Gear Sonic locomotion planner** on
the Unitree G1, runnable in **MuJoCo** (the real deploy stack) and in
**IsaacLab** (sim-to-sim), with a shared configuration, shared metric code and a
shared result layout.

It is built to answer two questions repeatably, before and after a planner
change:

* how good is the Sonic planner's walking (tracking, stability, gait, energy)?
* at which speed / direction / disturbance does it degrade?

## What the planner actually consumes

The planner is **not** a velocity controller.  Its input is a `MovementState`
(`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner.hpp`):

| field | meaning |
|---|---|
| `locomotion_mode` | `IDLE=0, SLOW_WALK=1 (0.1–0.8 m/s), WALK=2 (0.8–2.5), RUN=3 (2.5–7.5)` |
| `movement_direction` | world-frame unit vector of travel |
| `facing_direction` | world-frame unit vector the pelvis faces |
| `movement_speed` | scalar [m/s] (`-1` = mode default, `0` = stationary) |
| `height` | body height [m] (`-1` = mode default) |

The benchmark speaks the usual `(vx, vy, yaw_rate)` body twist and converts it
the way `keyboard_handler.hpp` does (`core/commands.py`):

```
yaw_cmd  += yaw_rate * dt                     # integrated heading
facing    = [cos(yaw_cmd), sin(yaw_cmd), 0]
speed     = hypot(vx, vy)                      # mode picked from this
movement  = Rz(yaw_cmd) @ [vx, vy, 0] / speed  # == +facing forward, -facing backward
```

Pure turning (`speed == 0, yaw_rate != 0`) is sent as `IDLE` with a rotating
`facing_direction`; the deploy planner replans on any facing change
(`facing_direction_changed` in `g1_deploy_onnx_ref.cpp::Planner`), so this is a
command the planner genuinely understands.  Nothing is injected further down the
pipeline — the trajectory, the resampling to 50 Hz and the WBC policy are the
deploy binary's own.

## Layout

```
gear_sonic_eval/
  evaluate_sonic_planner.py   # CLI entry point
  compare_sims.py             # MuJoCo vs IsaacLab comparison
  configs/                    # benchmark definitions (YAML)
  core/commands.py            # (vx,vy,wz) <-> MovementState
  core/config.py              # EvalConfig: seeds, dt, episodes, command grid, pushes
  core/runner.py              # the benchmark protocol (backend agnostic)
  core/metrics.py             # StepSample -> all metrics
  core/results.py             # CSV/JSON writers + per-condition aggregation
  core/plots.py               # all plots + heat maps
  backends/mujoco_backend.py  # real Sonic stack over ZMQ + DDS
  backends/isaaclab_backend.py# trajectory replay / RL baseline
  backends/mock.py            # analytic stand-in for testing the plumbing
  tests/                      # unit tests (no simulator required)
  tools/                      # planner-trajectory export procedure
```

## Running

```bash
# 0) inspect the episode manifest (no simulator needed)
python evaluate_sonic_planner.py --sim mock --config configs/walking_eval.yaml --dry-run

# 1) MuJoCo — start the deploy binary FIRST, then:
python evaluate_sonic_planner.py --sim mujoco --config configs/walking_eval.yaml
python evaluate_sonic_planner.py --sim mujoco --config configs/walking_eval.yaml --disturbance

# 2) IsaacLab (sim-to-sim)
python evaluate_sonic_planner.py --sim isaaclab --config configs/walking_eval.yaml --headless

# 3) compare
python compare_sims.py --results results --backends mujoco isaaclab
```

Plots need matplotlib.  If it is missing the run still completes and writes every
CSV; regenerate the figures afterwards with

```bash
python -m gear_sonic_eval.core.plots results/mujoco
```

Useful flags: `--num-episodes`, `--seed`, `--episode-duration`, `--conditions`,
`--output-dir`, `--headless`, `--visualize`, `--real-time`, `--no-plots`,
`--no-timeseries`, `--dry-run`.

### MuJoCo: how the interactive workflow maps to the benchmark

The manual workflow is 2 terminals + key presses:

| manual step | benchmark equivalent |
|---|---|
| Terminal 1: `python gear_sonic/scripts/run_sim_loop.py` | **replaced** by `evaluate_sonic_planner.py --sim mujoco` (the benchmark *is* the simulator, so it can step deterministically, reset per episode and apply pushes) |
| Terminal 2: `bash deploy.sh --input-type keyboard sim` | `bash deploy.sh --input-type zmq_manager sim` |
| `]` (start policy) | `command{start=true, stop=false, planner=true}` on the ZMQ `command` topic |
| `Enter` (planner mode) | same message — `ZMQManager::handlePlannerInput` enables the planner and waits for its init. Sent **once per run**, not per episode |
| `O` / Ctrl-C (stop) | never sent between episodes: `stop` sets `operator_state.stop`, which `main()` loops on, so it kills the deploy process |
| `9` in MuJoCo (release elastic band) | the band is disabled in code (`backend.mujoco.elastic_band: false`) |
| W/A/S/D/Q/E (movement keys) | `planner` topic messages generated from the config's `(vx, vy, yaw_rate)` grid |

Run exactly two terminals:

```bash
# Terminal 1 (host) -- simulator + benchmark. Start this FIRST; it waits for the
# deploy binary and keeps stepping so the deploy INIT ramp can run.
cd ~/GR00T-WholeBodyControl && source .venv_sim/bin/activate
python gear_sonic_eval/evaluate_sonic_planner.py --sim mujoco \
    --config gear_sonic_eval/configs/walking_eval.yaml --visualize

# Terminal 2 (docker) -- the Sonic planner + WBC policy
cd ~/GR00T-WholeBodyControl/gear_sonic_deploy
export TensorRT_ROOT="$HOME/TensorRT"
./docker/run-ros2-dev-takada.sh
bash deploy.sh --input-type zmq_manager sim
```

Do **not** start `run_sim_loop.py` as well: two MuJoCo processes would both
publish `rt/lowstate` on the same DDS domain and the controller would see
interleaved states.  No key presses are needed in either terminal.

The container runs with `--network host`, so the deploy binary's ZMQ subscriber
(`--zmq-host localhost`, port 5556 by default) reaches the PUB socket the
benchmark binds on the host.

## report.md — identical format to the IsaacLab G1 benchmark

Every run also writes `results/<backend>/eval.npz` + `eval.json` in the schema of
`IsaacLab/scripts/reinforcement_learning/rsl_rl/eval_locomotion.py`, so the
report is generated by **that repo's own** `report_locomotion.py` — same tables,
same metric names, same plots, and the Sonic planner can be put next to an
existing RL run in one report:

```bash
python <IsaacLab>/scripts/reinforcement_learning/rsl_rl/report_locomotion.py \
    --inputs sonic=results/mujoco rl=<IsaacLab>/results/2026-08-07_19-01-19_g1_flat_v1 \
    --out results
# -> results/<timestamp>/report.md + report_*.png
```

Use `configs/eval_md_benchmark.yaml`: it lays the conditions out as the eval.md
scenarios (sweep / step / sine / push / circle) so every table in `report.md` is
filled. `python -m gear_sonic_eval.core.isaaclab_report results/mujoco`
regenerates the export from existing CSVs.

Scenario mapping, and how it differs from the RL benchmark:

| report.md item | how the Sonic benchmark produces it |
|---|---|
| `sweep_{ax}` | constant-command conditions of that axis (`group: forward/backward/lateral/turn`), steady-state window after `transient_duration` |
| `step_{ax}` | the 0→target transient at the start of **the same** episodes (every Sonic episode starts from a standstill), so sweep and step are not separate rollouts as they are in the RL benchmark |
| `sine_{ax}` | `waveform: sine` conditions; the command is a 10 Hz staircase because that is the planner's command rate, and the logged command is the value actually in force |
| `push` | `disturbance` grid; `push_mags` are impulses (force × duration) to match the RL benchmark's axis |
| `circle` | `group: circle` grid of vx × yaw_rate |
| `± 95% CI` | across the episodes of a condition (one seed each), the same estimator (`ci95`, t-quantile, ddof=1) |
| `power_convention` | `abs` (Σ&#124;τ·q̇&#124;). The RL benchmark defaults to `positive`, so **CoT is only comparable if that run also used `--power abs`** |

## Metrics

Per episode (`episodes.csv`), aggregated per condition (`summary.csv`,
`velocity_tracking.csv`, `stability.csv`) and dumped per step
(`timeseries/*.csv`).  Definitions follow `IsaacLab/scripts/myscripts/eval.md`,
so numbers line up with the existing G1 RL benchmark.

* **Tracking** (per axis, after `transient_duration`): commanded, achieved,
  steady-state error, MAE, RMSE, max abs error, std.
* **Transient**: rise time (10→90%), settling time (±5% band), overshoot.
* **Stability**: base height mean/std, roll/pitch/yaw mean & std, tilt RMS,
  angular-velocity RMS, vertical-velocity std.
* **Fall**: success/failure, fall rate, time to fall, recovery time after a push.
* **Gait**: contact fraction, duty factor, step frequency, step length, foot
  slip, double-support and flight fraction.
* **Joints / energy**: joint velocity & acceleration RMS, torque RMS/max,
  mechanical power, total energy, cost of transport.

A metric whose input the backend cannot provide stays `nan`; nothing is
estimated from missing data.

## Reproducibility

`seed`, `control_dt`, `physics_dt`, `command_dt`, `episode_duration`,
`settle_duration`, initial pose/velocity, the command grid and the push grid all
live in the config file, which is copied verbatim into
`results/<backend>/config.json` next to `run_info.json`.  Re-running the same
config reproduces the same episode manifest (`--dry-run` prints it).
