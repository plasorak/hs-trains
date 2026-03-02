# rusty-trains

A train network simulator written in Rust. Supports two simulation modes:

- **Physics** — integrates Newton's equations of motion using a configurable train description, environment, and driver input.
- **Timing** — replays real berth timing data from a Parquet file, producing a position trace without kinematic assumptions.

## Prerequisites

| Tool | Purpose |
|------|---------|
| [Rust + Cargo](https://rustup.rs) | Build the simulator |
| [uv](https://docs.astral.sh/uv/) | Run the Python data-generation scripts |

## Build

```sh
cargo build --release
```

The binary is written to `target/release/rusty-trains`.

## Usage

```
rusty-trains <config.yaml> <output.parquet>
```

| Argument | Description |
|----------|-------------|
| `<config.yaml>` | Simulation configuration (see below) |
| `<output.parquet>` | Destination for the result table |

Run `rusty-trains --help` for a brief usage summary.

## Simulation modes

### Physics simulation

Integrate train motion using the Davis equation, aerodynamic drag, and gravity.

Config (`config_physics.yaml`):

```yaml
simulation:
  type: physics

  train:
    power: 2460000.0                        # W
    traction_force_at_standstill: 409000.0  # N
    max_speed: 120.0                        # km/h
    mass: 2000000.0                         # kg
    drag_coeff: 10.0                        # kg/m
    braking_force: 800000.0                 # N

  environment:
    gradient: 0.01      # rise/run (positive = uphill)
    wind_speed: 0.0     # m/s (head-wind positive)

  driver:
    power_ratio: 0.8    # 0–1 throttle
    break_ratio: 0.0    # 0–1 braking

  time_step_s: 0.1      # integration step
  duration_s: 2000.0    # total simulated time
```

Output columns: `time_s`, `position_m`, `speed_kmh`, `acceleration_mss`.

Run:

```sh
cargo run --release -- config_physics.yaml output_physics.parquet
```

### Timing-based replay

Load berth step events for a specific train from a Parquet file and
produce a position trace ordered by timestamp.

Config (`config_timing.yaml`):

```yaml
simulation:
  type: timing
  parquet_file: berth_timing.parquet
  train_id: "1A23"
```

Output columns: `timestamp_ms`, `position_m`.

Speed and acceleration are not available from berth timing data.

Run:

```sh
cargo run --release -- config_timing.yaml output_timing.parquet
```

## Generating sample timing data

`make_timing_parquet.py` generates a synthetic berth timing Parquet file
that matches the expected schema:

| Column | Type | Description |
|--------|------|-------------|
| `train_id` | UTF8 | Train identifier / headcode |
| `berth_id` | UTF8 | Berth name |
| `timestamp_ms` | INT64 | Unix timestamp, milliseconds |
| `position_m` | DOUBLE | Along-track distance from route origin, metres |

```sh
# Default: trains 1A23, 2B45, 3C67 × 12 berths each
uv run make_timing_parquet.py berth_timing.parquet

# Custom trains and berth count
uv run make_timing_parquet.py berth_timing.parquet --trains 1A23 2B45 --berths 20
```

## Running the tests

```sh
cargo test
```
