# Rusty-Trains — Design Document

## Overview

Rusty-Trains is a train network simulator written in Rust. It models the motion of one or more trains using either realistic physics or real berth timing data, and supports mixed simulations where both kinds of train coexist. Simulation results are streamed to a Parquet file for downstream analysis.

---

## Repository Layout

```
src/
├── main.rs        CLI, config parsing, event-driven simulation loop, Parquet output
├── scheduler.rs   Discrete event queue (BinaryHeap + lazy cancellation)
├── physics.rs     Davis-equation force model, constant-acceleration kinematics
├── model.rs       Core data structures (trains, states, environment, infrastructure stubs)
├── timing.rs      Parquet loader and linear interpolator for timing trains
└── assets.rs      Placeholder for future infrastructure (currently empty)
config/
├── config_physics.yaml   Example: two physics trains
├── config_timing.yaml    Example: two timing trains
└── config_mixed.yaml     Example: one physics + two timing trains
```

---

## Architecture

### High-Level Data Flow

```
YAML config
    │
    ▼
Config / TrainConfig
    │
    ▼
SimState per train ──────────────────────────┐
    │                                        │
    ▼                                        │
EventQueue (seeded)                          │
    │                                        │
    ▼                                        │
Event loop ──► advance_train / position_at ──┘
    │
    ▼
Row buffer ──► flush to Parquet (row-groups)
```

### Modules

| Module | Responsibility |
|---|---|
| `main.rs` | CLI, config deserialisation, event loop, parallel train advancement, Parquet I/O |
| `scheduler.rs` | Priority-ordered event queue; lazy cancellation |
| `physics.rs` | Force calculation (Davis equation), terminal-velocity capping, kinematics |
| `model.rs` | Data structures for train parameters, kinematic states, environment, driver input |
| `timing.rs` | Load and linearly interpolate historical berth-timing data from Parquet |

---

## Discrete Event Scheduler

Rather than advancing every train at every fixed time step, the simulator uses a **Discrete Event Scheduler (DES)**. Time jumps from event to event; trains are only advanced when an event requires their state.

### `EventQueue` (`scheduler.rs`)

```
EventQueue {
    heap:      BinaryHeap<Event>   // min-heap by time
    cancelled: HashSet<EventId>    // lazy cancellation set
    next_id:   u64
}
```

- **`push(event) → EventId`** — inserts an event and returns its ID.
- **`pop() → Option<Event>`** — removes and returns the earliest non-cancelled event.
- **`cancel(id)`** — marks an event as cancelled in O(1); removed lazily on the next `pop`.

### Event Types

| Variant | Description |
|---|---|
| `PhysicsTick` | Periodic integration step; self-schedules the next tick after each execution |
| `Random(kind)` | Placeholder for future events: `Departure`, `Arrival`, `SignalChange`, `SpeedChange` |

### Event Loop (main.rs)

```
seed queue: one PhysicsTick at t = dt, scattered random events
─────────────────────────────────────────────────────────────
while event = queue.pop():
    if event.time > duration: break

    PhysicsTick:
        schedule next tick if time remains and trains are still moving
        advance all trains in parallel (Rayon) from last_times[i] → event.time
        record one row per train

    Random event:
        advance target train from last_times[i] → event.time
        record one row for that train

    if buffer ≥ flush_rows:
        write Parquet row-group, clear buffer

write final Parquet row-group
```

The `last_times` array tracks each train's most-recently computed time so that partial-step advancement is always correct when events arrive at irregular intervals.

---

## Physics Engine (`physics.rs`)

### Force Model (Davis Equation)

```
F_net = F_traction − F_gravity − F_drag − F_rolling − F_braking
```

| Force | Formula |
|---|---|
| Traction | `min(F_standstill × power_ratio, P × power_ratio / v)` |
| Gravity | `mass × g × gradient` |
| Aerodynamic drag | `drag_coeff × (v + wind_speed)²` |
| Rolling resistance | `0.002 × mass × g` (constant) |
| Braking | `braking_force × brake_ratio` |

### Terminal Velocity

The equilibrium speed where `F_net = 0` is found by **bisection** (`terminal_speed`), running 52 iterations for double-precision convergence. This is called once per `advance_train` invocation.

### Constant-Acceleration Kinematics (`advance_train`)

To avoid overshoot past the equilibrium speed, integration is split into at most two phases:

1. **Accelerating phase** — constant acceleration `a` until terminal velocity `v_eq` is reached (or `dt` expires).
2. **Cruising phase** — constant speed `v_eq` for the remaining time.

Closed-form kinematics are used throughout (`x = x₀ + v₀ dt + ½ a dt²`), so no iteration is needed. The function also accepts an `AdvanceTarget::Distance` target for future distance-based event scheduling.

### Parallelisation

Physics ticks advance all trains concurrently using **Rayon** (`par_iter_mut`). Each train's state is independent, so no synchronisation is required.

---

## Timing Trains (`timing.rs`)

Timing trains replay recorded berth-level data instead of computing forces.

### `TimingTrace`

```
TimingTrace {
    times_s:     Vec<f64>   // normalised to 0-based seconds
    positions_m: Vec<f64>   // along-track distance
}
```

Loaded from a Parquet file with schema `(train_id, berth_id, timestamp_ms, position_m)`. Timestamps are normalised so `t = 0` corresponds to the first observed berth event.

### `position_at(t) → Option<f64>`

Binary search (`partition_point`) locates the surrounding bracket; linear interpolation gives the position. Returns `None` outside the observed time range.

---

## Data Structures (`model.rs`)

### Train Parameters

```rust
TrainDescription {
    power:                        f64,  // W
    traction_force_at_standstill: f64,  // N
    max_speed:                    f64,  // km/h
    mass:                         f64,  // kg
    drag_coeff:                   f64,  // kg/m
    braking_force:                f64,  // N
}
```

### Kinematic States

```rust
SimulatedState { position: Position, speed: f64, acceleration: f64 }
ObservedState  { position: Position, timestamp_ms: i64 }
```

`SimState` (main.rs) wraps either a `SimulatedState` (physics train) or a `TimingTrace` (timing train).

### Environment & Driver

```rust
Environment  { wind_speed: f64, gradient: f64 }
DriverInput  { power_ratio: f64, brake_ratio: f64 }
```

### Infrastructure Stubs

`SignalDescription` and `BerthDescription` are defined in `model.rs` but not yet wired into the simulation. They are placeholders for future traffic-management logic (signal aspects, route conflicts, overlap release).

---

## Configuration

Top-level YAML structure:

```yaml
simulation:
  time_step_s: 1.0
  duration_s:  7200.0
  flush_rows:  1000000      # optional, default 1 000 000
  trains:
    - id: "train_1"
      kind: physics
      train:        { power: 2460000, traction_force_at_standstill: 409000,
                      max_speed: 120, mass: 2000000, drag_coeff: 10,
                      braking_force: 800000 }
      environment:  { gradient: 0.01, wind_speed: 0.0 }
      driver:       { power_ratio: 0.8, brake_ratio: 0.0 }

    - id: "1A23"
      kind: timing
      parquet_file: berth_timing.parquet
```

Physics and timing trains can be freely mixed in the same `trains` list.

---

## Output Format

Results are written as a Parquet file with one row per train per event:

| Column | Type | Notes |
|---|---|---|
| `train_id` | String | From config `id` |
| `event_kind` | String | `"physics_tick"` or random event name |
| `time_s` | Float64 | Simulation time in seconds |
| `position_m` | Float64 (nullable) | Along-track distance; null if outside timing data range |
| `speed_kmh` | Float64 (nullable) | Null for timing trains |
| `acceleration_mss` | Float64 (nullable) | Null for timing trains |

Rows are buffered in memory and flushed to Parquet row-groups every `flush_rows` rows to bound memory usage.

---

## Key Design Decisions

### Event-Driven vs. Fixed Time-Step
The DES allows large time steps between events while still recording high-fidelity snapshots at moments of interest (signal changes, departures, etc.). It also makes it straightforward to add non-periodic events in the future without changing the core loop.

### Constant-Acceleration Kinematics with Terminal-Velocity Capping
A single force evaluation per step is sufficient because the force model is approximated as constant over `dt`. Capping at terminal velocity prevents the integrator from violating energy conservation at coarse time steps, without requiring iteration.

### Lazy Event Cancellation
Cancelled events remain in the heap until popped. This trades a small amount of memory for O(1) cancel operations, which is beneficial when many future events may need to be revoked (e.g., when a signal clears early).

### Mixed Physics / Timing Simulation
Both train types share the same event loop and output schema, enabling direct comparison of simulated trajectories against observed data in the same Parquet file.

### Row-Based Flushing
Writing Parquet row-groups only when a row-count threshold is reached decouples memory usage from simulation duration and train count, making it practical to run multi-hour, 100-train simulations without large memory footprints.
