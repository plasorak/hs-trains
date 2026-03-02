# rusty-trains

A train network simulator written in Rust. Physics and timing trains can be freely mixed in a single simulation — time is always the outer loop, so every train advances together at each step.

## Prerequisites

| Tool | Purpose |
|------|---------|
| [Rust + Cargo](https://rustup.rs) | Build the simulator |
| [uv](https://docs.astral.sh/uv/) | Run the Python helper scripts |

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

## Configuration

All configs share the same top-level structure. `time_step_s` and `duration_s`
apply to all trains. Each entry in `trains` has a `kind` field that selects
the simulation mode for that train.

### Physics trains (`kind: physics`)

Integrate train motion using the Davis equation, aerodynamic drag, and gravity.

```yaml
simulation:
  time_step_s: 0.1       # integration step (s)
  duration_s: 2000.0     # total simulated time (s)

  trains:
    - id: "express_1"
      kind: physics
      train:
        power: 2460000.0                        # W  (max traction power)
        traction_force_at_standstill: 409000.0  # N
        max_speed: 120.0                        # km/h
        mass: 2000000.0                         # kg
        drag_coeff: 10.0                        # kg/m (aerodynamic)
        braking_force: 800000.0                 # N
      environment:
        gradient: 0.01    # rise/run (positive = uphill)
        wind_speed: 0.0   # m/s (head-wind positive)
      driver:
        power_ratio: 0.8  # 0–1 throttle
        break_ratio: 0.0  # 0–1 braking
```

### Timing trains (`kind: timing`)

Interpolate position from real berth timing data read from a Parquet file.
Timestamps are normalised to `t = 0` at the first record for that train.
`speed_kmh` and `acceleration_mss` are `null` in the output for these trains.

```yaml
simulation:
  time_step_s: 10.0
  duration_s: 3600.0

  trains:
    - id: "1A23"
      kind: timing
      parquet_file: berth_timing.parquet

    - id: "2B45"
      kind: timing
      parquet_file: berth_timing.parquet
```

### Mixed simulation

Physics and timing trains can coexist in the same config:

```yaml
simulation:
  time_step_s: 0.1
  duration_s: 3600.0

  trains:
    - id: "simulated_express"
      kind: physics
      train: { ... }
      environment: { ... }
      driver: { ... }

    - id: "1A23"
      kind: timing
      parquet_file: berth_timing.parquet
```

See `config/config_mixed.yaml` for a complete example.

## Output columns

| Column | Type | Physics | Timing |
|--------|------|---------|--------|
| `train_id` | String | ✓ | ✓ |
| `time_s` | Float64 | ✓ | ✓ |
| `position_m` | Float64 | ✓ | ✓ (interpolated; `null` outside data range) |
| `speed_kmh` | Float64 | ✓ | `null` |
| `acceleration_mss` | Float64 | ✓ | `null` |

## Example runs

```sh
# Physics only (2 trains)
cargo run --release -- config/config_physics.yaml output_physics.parquet

# Timing only (2 trains)
cargo run --release -- config/config_timing.yaml output_timing.parquet

# Mixed (1 physics + 2 timing trains)
cargo run --release -- config/config_mixed.yaml output_mixed.parquet
```

## Python helper scripts

Scripts live in `scripts/`.

### Generate sample timing data

`scripts/make_timing_parquet.py` generates a synthetic berth timing Parquet file
that matches the expected schema:

| Column | Type | Description |
|--------|------|-------------|
| `train_id` | UTF8 | Train identifier / headcode |
| `berth_id` | UTF8 | Berth name |
| `timestamp_ms` | INT64 | Unix timestamp, milliseconds |
| `position_m` | DOUBLE | Along-track distance from route origin, metres |

```sh
# Default: trains 1A23, 2B45, 3C67 × 12 berths each
uv run scripts/make_timing_parquet.py berth_timing.parquet

# Custom trains and berth count
uv run scripts/make_timing_parquet.py berth_timing.parquet --trains 1A23 2B45 --berths 20
```

### Generate a large-scale config

`scripts/run_100trains.py` generates a config for N physics trains with
randomised parameters and prints the command to run it. Defaults to 100 trains
for 1 hour at 1-second steps (360,000 output rows).

```sh
uv run scripts/run_100trains.py

# Custom options
uv run scripts/run_100trains.py --trains 50 --dt 0.5 --duration 2
uv run scripts/run_100trains.py --help
```

## Running the tests

```sh
cargo test
```
