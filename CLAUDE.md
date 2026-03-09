# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build
cargo build --release

# Run
cargo run --release -- <config.yaml> <output.parquet>

# Test
cargo test

# Coverage
cargo llvm-cov --html

# Generate synthetic berth timing data (Python, requires uv)
uv run scripts/make_timing_parquet.py

# Generate a 100-train config
uv run scripts/run_100trains.py

# Plot results
uv run scripts/plot.py
```

## Architecture

Rusty-trains is a **Discrete Event Simulator (DES)** for train networks. Instead of a fixed time-step loop, time jumps event-to-event. Two train types coexist: physics-based (Davis equation) and timing-based (historical Parquet data).

### Key modules

- **`main.rs`** — CLI, config loading, event loop, Rayon-parallel train advancement, Parquet output with row-group flushing
- **`scheduler.rs`** — Min-heap event queue with lazy cancellation (O(1) cancel, token-based invalidation)
- **`physics.rs`** — Davis equation force model, terminal velocity via bisection, closed-form constant-acceleration kinematics
- **`timing.rs`** — Loads Parquet berth-timing traces; binary-search + linear interpolation for position at arbitrary time
- **`model.rs`** — Core structs: `TrainDescription`, `SimulatedState`, `DriverInput`, `TrainType` enum
- **`assets.rs`** — Empty stub for future signal/infrastructure logic

### Simulation loop

1. One self-scheduling `PhysicsTick` event drives all physics trains in parallel (Rayon `par_iter_mut`)
2. Random placeholder events (`Departure`, `Arrival`, `SignalChange`, `SpeedChange`) advance single trains — these are future expansion points
3. Output rows are buffered and flushed to Parquet every `flush_rows` rows (default 1M) to bound memory

### Physics model

```
F_net = F_traction − F_gravity − F_drag − F_rolling − F_braking
```
Terminal velocity is found via bisection (52 iterations). Integration uses closed-form constant-acceleration kinematics to avoid overshoot past equilibrium.

### Output schema (Parquet)

| Column | Type | Notes |
|--------|------|-------|
| `train_id` | String | |
| `event_kind` | String | `"physics_tick"` or event name |
| `time_s` | Float64 | Simulation seconds |
| `position_m` | Float64? | Null for timing trains outside data range |
| `speed_kmh` | Float64? | Null for timing trains |
| `acceleration_mss` | Float64? | Null for timing trains |

### Configuration

YAML config files live in `config/`. See `config/config_mixed.yaml` for a minimal mixed physics+timing example. Timing trains reference Parquet files with schema `(train_id, berth_id, timestamp_ms, position_m)`.
