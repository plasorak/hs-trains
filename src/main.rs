mod physics;
mod model;
mod timing;

use model::{SimulatedState, TrainDescription, Environment, DriverInput, Position, TrainState};
use physics::step_trains;
use polars::prelude::*;
use clap::Parser;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Train network simulator.
///
/// Runs either a physics-based simulation or replays real timing data,
/// writing the result to a Parquet file.
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

#[derive(serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SimulationConfig {
    Physics {
        train: TrainDescription,
        environment: Environment,
        driver: DriverInput,
        time_step_s: f64,
        duration_s: f64,
    },
    Timing {
        parquet_file: String,
        train_id: String,
    },
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

    let mut df = match config.simulation {
        SimulationConfig::Physics { train, environment, driver, time_step_s, duration_s } => {
            println!("Running physics simulation (dt={time_step_s}s, duration={duration_s}s)");
            run_physics(&train, &environment, &driver, time_step_s, duration_s)
        }
        SimulationConfig::Timing { parquet_file, train_id } => {
            println!("Loading timing data for train '{train_id}' from '{parquet_file}'");
            run_timing(&parquet_file, &train_id)
        }
    };

    let file = std::fs::File::create(output_path)
        .unwrap_or_else(|e| { eprintln!("Cannot create '{}': {e}", output_path.display()); std::process::exit(1) });
    ParquetWriter::new(file).finish(&mut df).unwrap();
    println!("Written {} rows to '{}'", df.height(), output_path.display());
}

// ---------------------------------------------------------------------------
// Simulation runners
// ---------------------------------------------------------------------------

fn run_physics(
    train: &TrainDescription,
    env: &Environment,
    driver: &DriverInput,
    dt: f64,
    duration: f64,
) -> DataFrame {
    let steps = (duration / dt).round() as usize;
    let mut time_s_data       = Vec::with_capacity(steps);
    let mut position_m_data   = Vec::with_capacity(steps);
    let mut speed_kmh_data    = Vec::with_capacity(steps);
    let mut accel_mss_data    = Vec::with_capacity(steps);

    let mut state = SimulatedState {
        position: Position { x: 0.0, y: 0.0, z: 0.0 },
        speed: 0.0,
        acceleration: 0.0,
    };

    for step in 0..steps {
        state = step_trains(&state, train, driver, env, dt);
        time_s_data.push((step + 1) as f64 * dt);
        position_m_data.push(state.position.x);
        speed_kmh_data.push(state.speed * 3.6);
        accel_mss_data.push(state.acceleration);
    }

    DataFrame::new(
        time_s_data.len(),
        vec![
            Series::new("time_s".into(),          &time_s_data).into(),
            Series::new("position_m".into(),       &position_m_data).into(),
            Series::new("speed_kmh".into(),        &speed_kmh_data).into(),
            Series::new("acceleration_mss".into(), &accel_mss_data).into(),
        ],
    ).unwrap()
}

fn run_timing(parquet_file: &str, train_id: &str) -> DataFrame {
    let states = timing::load_timing_from_parquet(
        std::path::Path::new(parquet_file),
        train_id,
    ).unwrap_or_else(|e| { eprintln!("Error loading timing data: {e}"); std::process::exit(1) });

    let mut timestamp_ms_data = Vec::with_capacity(states.len());
    let mut position_m_data   = Vec::with_capacity(states.len());

    for state in &states {
        if let TrainState::Observed(o) = state {
            timestamp_ms_data.push(o.timestamp_ms);
            position_m_data.push(o.position.x);
        }
    }

    DataFrame::new(
        timestamp_ms_data.len(),
        vec![
            Series::new("timestamp_ms".into(), &timestamp_ms_data).into(),
            Series::new("position_m".into(),   &position_m_data).into(),
        ],
    ).unwrap()
}
