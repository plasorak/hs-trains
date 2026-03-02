mod physics;
mod model;
mod timing;

use model::{SimulatedState, TrainDescription, Environment, DriverInput, Position};
use physics::step_trains;
use timing::TimingTrace;
use polars::prelude::*;
use clap::Parser;

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

fn default_flush_steps() -> usize { 10_000 }

#[derive(serde::Deserialize)]
struct SimulationConfig {
    time_step_s: f64,
    duration_s: f64,
    trains: Vec<TrainConfig>,
    /// Number of time steps between Parquet row-group flushes.
    /// Smaller values reduce peak memory use at the cost of more I/O.
    /// Defaults to 10 000.
    #[serde(default = "default_flush_steps")]
    flush_steps: usize,
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
        "Running simulation: {} train(s), dt={}s, duration={}s, flush every {} steps",
        sim.trains.len(), sim.time_step_s, sim.duration_s, sim.flush_steps
    );

    run_simulation(&sim.trains, sim.time_step_s, sim.duration_s, output_path, sim.flush_steps);
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
/// In-memory buffers hold at most `flush_steps * n_trains` rows at a time;
/// they are written as a row group and cleared on every flush boundary.
///
/// Output columns:
/// - `train_id`         — string identifier
/// - `time_s`           — elapsed simulation time in seconds
/// - `position_m`       — position along route in metres (null if timing train
///                        is outside its data range)
/// - `speed_kmh`        — speed in km/h (null for timing trains)
/// - `acceleration_mss` — acceleration in m/s² (null for timing trains)
fn run_simulation(trains: &[TrainConfig], dt: f64, duration: f64, output: &std::path::Path, flush_steps: usize) {
    let steps = (duration / dt).round() as usize;
    let buf_cap = flush_steps * trains.len();

    let mut train_id_data   = Vec::<String>::with_capacity(buf_cap);
    let mut time_s_data     = Vec::<f64>::with_capacity(buf_cap);
    let mut position_m_data = Vec::<Option<f64>>::with_capacity(buf_cap);
    let mut speed_kmh_data  = Vec::<Option<f64>>::with_capacity(buf_cap);
    let mut accel_mss_data  = Vec::<Option<f64>>::with_capacity(buf_cap);

    // Fixed output schema — must match the Series built in flush_batch().
    let schema = Schema::from_iter([
        Field::new("train_id".into(),         DataType::String),
        Field::new("time_s".into(),           DataType::Float64),
        Field::new("position_m".into(),       DataType::Float64),
        Field::new("speed_kmh".into(),        DataType::Float64),
        Field::new("acceleration_mss".into(), DataType::Float64),
    ]);

    let file = std::fs::File::create(output)
        .unwrap_or_else(|e| { eprintln!("Cannot create '{}': {e}", output.display()); std::process::exit(1) });
    let mut writer = ParquetWriter::new(file)
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

    let mut total_rows: usize = 0;

    for step in 0..steps {
        let t = (step + 1) as f64 * dt;

        for (cfg, state) in trains.iter().zip(states.iter_mut()) {
            train_id_data.push(cfg.id().to_string());
            time_s_data.push(t);

            match (cfg, state) {
                (TrainConfig::Physics { train, environment, driver, .. }, SimState::Physics(s)) => {
                    *s = step_trains(s, train, driver, environment, dt);
                    position_m_data.push(Some(s.position.x));
                    speed_kmh_data.push(Some(s.speed * 3.6));
                    accel_mss_data.push(Some(s.acceleration));
                }
                (TrainConfig::Timing { .. }, SimState::Timing(trace)) => {
                    position_m_data.push(trace.position_at(t));
                    speed_kmh_data.push(None);
                    accel_mss_data.push(None);
                }
                _ => unreachable!(),
            }
        }

        if (step + 1) % flush_steps == 0 || step + 1 == steps {
            let batch = DataFrame::new(
                train_id_data.len(),
                vec![
                    Series::new("train_id".into(),         &train_id_data).into(),
                    Series::new("time_s".into(),            &time_s_data).into(),
                    Series::new("position_m".into(),        &position_m_data).into(),
                    Series::new("speed_kmh".into(),         &speed_kmh_data).into(),
                    Series::new("acceleration_mss".into(),  &accel_mss_data).into(),
                ],
            ).unwrap();

            let n = batch.height();
            writer.write_batch(&batch)
                .unwrap_or_else(|e| { eprintln!("Write error at step {step}: {e}"); std::process::exit(1) });
            total_rows += n;
            println!("  flushed {n} rows (total: {total_rows})");

            train_id_data.clear();
            time_s_data.clear();
            position_m_data.clear();
            speed_kmh_data.clear();
            accel_mss_data.clear();
        }
    }

    writer.finish()
        .unwrap_or_else(|e| { eprintln!("Failed to finalise Parquet file: {e}"); std::process::exit(1) });
    println!("Written {total_rows} rows to '{}'", output.display());
}
