mod physics;
mod model;
mod timing;
mod scheduler;

use model::{SimulatedState, TrainDescription, Environment, DriverInput, Position};
use physics::{advance_train, AdvanceTarget};
use timing::TimingTrace;
use polars::prelude::*;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Train network simulator.
///
/// Runs a unified simulation where physics and timing trains coexist.
/// Time is always the outer loop: all trains advance together each step.
/// Results are flushed to Parquet in row-group batches to bound memory use.
#[derive(Parser)]
#[command(name = "rusty-trains", version)]
struct Cli {
    /// Path to the simulation config YAML file.
    config: std::path::PathBuf,

    /// Path to write the output Parquet file.
    output: std::path::PathBuf,
}

// ---------------------------------------------------------------------------
// Configuration structs (deserialized from YAML)
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct Config {
    simulation: SimulationConfig,
}

/// Per-train configuration, tagged by `kind: physics | timing`.
#[derive(serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum TrainConfig {
    Physics {
        id: String,
        train: TrainDescription,
        environment: Environment,
        driver: DriverInput,
    },
    Timing {
        id: String,
        parquet_file: String,
    },
}

impl TrainConfig {
    fn id(&self) -> &str {
        match self {
            TrainConfig::Physics { id, .. } | TrainConfig::Timing { id, .. } => id,
        }
    }
}

fn default_flush_rows() -> usize { 1_000_000 }

#[derive(serde::Deserialize)]
struct SimulationConfig {
    time_step_s: f64,
    duration_s: f64,
    trains: Vec<TrainConfig>,
    /// Maximum number of rows to buffer before flushing a Parquet row group.
    /// Row-based (not step-based) so the buffer size stays bounded regardless
    /// of the number of trains. Defaults to 1 000 000.
    #[serde(default = "default_flush_rows")]
    flush_rows: usize,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    let config_path = &cli.config;
    let output_path = &cli.output;

    let config_str = std::fs::read_to_string(config_path)
        .unwrap_or_else(|e| { eprintln!("Cannot read config '{}': {e}", config_path.display()); std::process::exit(1) });
    let config: Config = serde_yaml_ng::from_str(&config_str)
        .unwrap_or_else(|e| { eprintln!("Invalid config: {e}"); std::process::exit(1) });

    let sim = config.simulation;
    println!(
        "Running simulation: {} train(s), dt={}s, duration={}s, flush every {} rows",
        sim.trains.len(), sim.time_step_s, sim.duration_s, sim.flush_rows
    );

    run_simulation(&sim.trains, sim.time_step_s, sim.duration_s, output_path, sim.flush_rows);
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

/// Per-train runtime state, parallel to `TrainConfig`.
enum SimState {
    Physics(SimulatedState),
    Timing(TimingTrace),
}

/// Run the simulation, streaming results to `output` as Parquet row groups.
///
/// Time is the outer loop: every train is advanced for each time step before
/// moving to the next step. Physics and timing trains may be freely mixed.
///
/// Per-step physics is parallelised with Rayon across all trains.
///
/// In-memory buffers hold at most `flush_rows` rows at a time; they are
/// written as a row group and cleared on every flush boundary.
///
/// Output columns:
/// - `train_id`         — string identifier
/// - `time_s`           — elapsed simulation time in seconds
/// - `position_m`       — position along route in metres (null if timing train
///                        is outside its data range)
/// - `speed_kmh`        — speed in km/h (null for timing trains)
/// - `acceleration_mss` — acceleration in m/s² (null for timing trains)

fn build_batch<'a>(
    train_id_data:   &[&'a str],
    event_kind_data: &[&'a str],
    time_s_data:     &[f64],
    position_m_data: &[Option<f64>],
    speed_kmh_data:  &[Option<f64>],
    accel_mss_data:  &[Option<f64>],
) -> DataFrame {
    let n = time_s_data.len();
    DataFrame::new(
        n,
        vec![
            Series::new("train_id".into(),         train_id_data).into(),
            Series::new("event_kind".into(),        event_kind_data).into(),
            Series::new("time_s".into(),            time_s_data).into(),
            Series::new("position_m".into(),        position_m_data).into(),
            Series::new("speed_kmh".into(),         speed_kmh_data).into(),
            Series::new("acceleration_mss".into(),  accel_mss_data).into(),
        ],
    ).unwrap()
}

fn run_simulation(trains: &[TrainConfig], dt: f64, duration: f64, output: &std::path::Path, flush_rows: usize) {
    let steps = (duration / dt).round() as usize;
    let buf_cap = flush_rows.min(steps * trains.len());

    // Pre-cache IDs as &str slices — avoids per-step String allocations in the hot loop.
    let train_id_cache: Vec<&str> = trains.iter().map(|c| c.id()).collect();

    let mut train_id_data    = Vec::<&str>::with_capacity(buf_cap);
    let mut event_kind_data  = Vec::<&'static str>::with_capacity(buf_cap);
    let mut time_s_data      = Vec::<f64>::with_capacity(buf_cap);
    let mut position_m_data  = Vec::<Option<f64>>::with_capacity(buf_cap);
    let mut speed_kmh_data   = Vec::<Option<f64>>::with_capacity(buf_cap);
    let mut accel_mss_data   = Vec::<Option<f64>>::with_capacity(buf_cap);

    // Fixed output schema — must match the Series built at flush time.
    let schema = Schema::from_iter([
        Field::new("train_id".into(),         DataType::String),
        Field::new("event_kind".into(),       DataType::String),
        Field::new("time_s".into(),           DataType::Float64),
        Field::new("position_m".into(),       DataType::Float64),
        Field::new("speed_kmh".into(),        DataType::Float64),
        Field::new("acceleration_mss".into(), DataType::Float64),
    ]);

    let file = std::fs::File::create(output)
        .unwrap_or_else(|e| { eprintln!("Cannot create '{}': {e}", output.display()); std::process::exit(1) });
    let mut writer = ParquetWriter::new(file)
        .with_compression(ParquetCompression::Lz4Raw)
        .set_parallel(true)
        .batched(&schema)
        .unwrap_or_else(|e| { eprintln!("Cannot initialise Parquet writer: {e}"); std::process::exit(1) });

    // Initialise per-train state.
    let mut states: Vec<SimState> = trains.iter().map(|cfg| match cfg {
        TrainConfig::Physics { .. } => SimState::Physics(SimulatedState {
            position: Position { x: 0.0, y: 0.0, z: 0.0 },
            speed: 0.0,
            acceleration: 0.0,
        }),
        TrainConfig::Timing { id, parquet_file } => {
            let trace = TimingTrace::load(std::path::Path::new(parquet_file), id)
                .unwrap_or_else(|e| { eprintln!("Error loading timing data for '{id}': {e}"); std::process::exit(1) });
            SimState::Timing(trace)
        }
    }).collect();

    // -----------------------------------------------------------------------
    // Build the event queue
    // -----------------------------------------------------------------------

    let mut queue = scheduler::EventQueue::new();

    // Seed only the first physics tick — subsequent ticks are self-scheduled
    // from inside the event loop, keeping the queue small at all times.
    queue.push(dt, None, scheduler::EventKind::PhysicsTick);

    // Seed random placeholder events. Each carries an EntityRef so the
    // dispatch code can route it to the right object later.
    {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let n_trains = trains.len().max(1);
        let n_random = steps / 10;
        for _ in 0..n_random {
            let t = rng.gen_range(0.0f64..duration);
            let (target, kind) = match rng.gen_range(0u8..4) {
                0 => (
                    Some(scheduler::EntityRef::Train(rng.gen_range(0..n_trains))),
                    scheduler::EventKind::Random(scheduler::RandomEventKind::Departure),
                ),
                1 => (
                    Some(scheduler::EntityRef::Train(rng.gen_range(0..n_trains))),
                    scheduler::EventKind::Random(scheduler::RandomEventKind::Arrival),
                ),
                2 => (
                    Some(scheduler::EntityRef::Signal(rng.gen_range(0..100usize))),
                    scheduler::EventKind::Random(scheduler::RandomEventKind::SignalChange),
                ),
                _ => (
                    Some(scheduler::EntityRef::Train(rng.gen_range(0..n_trains))),
                    scheduler::EventKind::Random(scheduler::RandomEventKind::SpeedChange {
                        new_speed_kmh: rng.gen_range(0.0f64..120.0),
                    }),
                ),
            };
            queue.push(t, target, kind);
        }
        println!("Event queue seeded: 1 initial physics tick (self-scheduling) + {n_random} random placeholder events");
    }

    // -----------------------------------------------------------------------
    // Event-driven simulation loop
    // -----------------------------------------------------------------------

    // Last simulation time at which each train's state was computed.
    // Random events advance a single train from here to the event time.
    let mut last_times: Vec<f64> = vec![0.0; trains.len()];

    let mut total_rows: usize = 0;

    let pb = ProgressBar::new(steps as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} steps ({percent}%) ETA: {eta}"
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏ "),
    );

    while let Some(event) = queue.pop() {
        // Hard time bound: discard any event (including stray ticks) past the
        // simulation end. This is the primary stop condition for the loop.
        if event.time > duration { break; }

        match event.kind {
            scheduler::EventKind::PhysicsTick => {
                let t = event.time;

                // Schedule the next tick only if it is still within the simulation
                // window AND there is meaningful work left: either a train is still
                // moving, or pending events could change the state (e.g. a future
                // SpeedChange waking a stopped train).
                let any_moving = states.iter().any(|s| match s {
                    SimState::Physics(p) => p.speed > 0.0,
                    SimState::Timing(_)  => true, // timing traces may still have data
                });
                if t + dt <= duration && (any_moving || !queue.is_empty()) {
                    queue.push(t + dt, None, scheduler::EventKind::PhysicsTick);
                }

                // Each train advances from its last-known time to t (may differ if
                // random events have advanced individual trains in between).
                let step_results: Vec<(Option<f64>, Option<f64>, Option<f64>)> =
                    trains.par_iter()
                        .zip(states.par_iter_mut())
                        .zip(last_times.par_iter())
                        .map(|((cfg, state), &lt)| {
                            let dt_i = t - lt;
                            match (cfg, state) {
                                (TrainConfig::Physics { train, environment, driver, .. }, SimState::Physics(s)) => {
                                    if dt_i > 0.0 {
                                        *s = advance_train(s, train, driver, environment, AdvanceTarget::Time(dt_i));
                                    }
                                    (Some(s.position.x), Some(s.speed * 3.6), Some(s.acceleration))
                                }
                                (TrainConfig::Timing { .. }, SimState::Timing(trace)) => {
                                    (trace.position_at(t), None, None)
                                }
                                _ => unreachable!(),
                            }
                        })
                        .collect();

                last_times.iter_mut().for_each(|lt| *lt = t);

                for (i, (pos, spd, acc)) in step_results.into_iter().enumerate() {
                    train_id_data.push(train_id_cache[i]);
                    event_kind_data.push("physics_tick");
                    time_s_data.push(t);
                    position_m_data.push(pos);
                    speed_kmh_data.push(spd);
                    accel_mss_data.push(acc);
                }

                pb.inc(1);
            }

            scheduler::EventKind::Random(kind) => {
                let t = event.time;
                let kind_str: &'static str = match kind {
                    scheduler::RandomEventKind::Departure          => "departure",
                    scheduler::RandomEventKind::Arrival            => "arrival",
                    scheduler::RandomEventKind::SignalChange       => "signal_change",
                    scheduler::RandomEventKind::SpeedChange { .. } => "speed_change",
                };
                // Advance the targeted train to this event's time and record its state.
                if let Some(scheduler::EntityRef::Train(i)) = event.target {
                    if i < trains.len() {
                        let dt_i = t - last_times[i];
                        if dt_i > 0.0 {
                            let (pos, spd, acc) = match (&trains[i], &mut states[i]) {
                                (TrainConfig::Physics { train, environment, driver, .. }, SimState::Physics(s)) => {
                                    *s = advance_train(s, train, driver, environment, AdvanceTarget::Time(dt_i));
                                    (Some(s.position.x), Some(s.speed * 3.6), Some(s.acceleration))
                                }
                                (TrainConfig::Timing { .. }, SimState::Timing(trace)) => {
                                    (trace.position_at(t), None, None)
                                }
                                _ => unreachable!(),
                            };
                            last_times[i] = t;
                            train_id_data.push(train_id_cache[i]);
                            event_kind_data.push(kind_str);
                            time_s_data.push(t);
                            position_m_data.push(pos);
                            speed_kmh_data.push(spd);
                            accel_mss_data.push(acc);
                        }
                    }
                }
                // Signal events don't produce rows yet.
            }
        }

        if time_s_data.len() >= flush_rows {
            let n = time_s_data.len();
            let batch = build_batch(&train_id_data, &event_kind_data, &time_s_data, &position_m_data, &speed_kmh_data, &accel_mss_data);
            writer.write_batch(&batch)
                .unwrap_or_else(|e| { eprintln!("Write error: {e}"); std::process::exit(1) });
            total_rows += n;
            pb.println(format!("  flushed {n} rows (total: {total_rows})"));
            train_id_data.clear();
            event_kind_data.clear();
            time_s_data.clear();
            position_m_data.clear();
            speed_kmh_data.clear();
            accel_mss_data.clear();
        }
    }

    // Final flush: covers both normal queue exhaustion and the time-limit break.
    if !time_s_data.is_empty() {
        let n = time_s_data.len();
        let batch = build_batch(&train_id_data, &event_kind_data, &time_s_data, &position_m_data, &speed_kmh_data, &accel_mss_data);
        writer.write_batch(&batch)
            .unwrap_or_else(|e| { eprintln!("Write error (final flush): {e}"); std::process::exit(1) });
        total_rows += n;
        pb.println(format!("  flushed {n} rows (total: {total_rows})"));
    }

    pb.finish_and_clear();
    writer.finish()
        .unwrap_or_else(|e| { eprintln!("Failed to finalise Parquet file: {e}"); std::process::exit(1) });
    println!("Written {total_rows} rows to '{}'", output.display());
}
