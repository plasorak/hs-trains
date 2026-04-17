//! Two-section analysis tool: timing → physics for a single train.
//!
//! Section 1 uses the timing trace to capture the train's behaviour from rest
//! (where the Davis equation is hardest to calibrate). Two speed estimates are
//! recorded side-by-side:
//!
//! * **Differential** — finite difference of the timing-trace position:
//!   `v_diff(t) = (pos(t) − pos(t−dt)) / dt`
//!   `a_diff(t) = (v_diff(t) − v_diff(t−dt)) / dt`
//!
//! * **Integral** — Davis-equation ODE integrated forward from rest in parallel
//!   (not synced to timing):
//!   `F_net = F_traction − F_gravity − (A + B·v + C·v²) − F_braking`
//!   `v_integ(t+dt) = v_integ(t) + (F_net/m)·dt`
//!
//! At the timing → physics boundary the physics state is re-seeded to
//! `(pos_timing, v_diff, a=0)` so section 2 starts from the last observed state.
//!
//! ## Output schema
//!
//! | column                   | description                                              |
//! |--------------------------|----------------------------------------------------------|
//! | `time_s`                 | elapsed simulation time (s)                              |
//! | `section`                | 1 = timing, 2 = physics                                  |
//! | `position_m`             | timing position in §1; physics position in §2            |
//! | `speed_differential_kmh` | Δpos/Δt speed estimate — §1 only, null elsewhere         |
//! | `accel_differential_mss` | Δspeed/Δt accel estimate — §1 only, null elsewhere       |
//! | `speed_integral_kmh`     | Davis-ODE speed (physics engine, all sections)           |
//! | `accel_integral_mss`     | Davis-ODE acceleration (physics engine, all sections)    |

use clap::Parser;
use hs_trains::core::model::{DriverInput, Environment, Position, SimulatedState};
use hs_trains::core::physics::{AdvanceTarget, advance_train};
use hs_trains::io::railml_rollingstock;
use hs_trains::io::timing::TimingTrace;
use polars::prelude::*;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "davis-analysis",
    about = "Timing / physics two-section analysis"
)]
struct Cli {
    /// RailML 3.3 file containing the formation
    railml_file: PathBuf,
    /// Formation ID to load from the RailML file
    formation_id: String,
    /// Parquet berth-timing file used for section 1
    timing_file: PathBuf,
    /// Train ID to read from the timing file
    timing_train_id: String,
    /// Output Parquet file
    output: PathBuf,

    /// Duration of section 1 (timing section) in seconds
    #[arg(long, default_value_t = 300.0)]
    timing_s: f64,

    /// Duration of section 2 (physics section) in seconds
    #[arg(long, default_value_t = 300.0)]
    physics_s: f64,

    /// Integration / output time step in seconds
    #[arg(long, default_value_t = 1.0)]
    dt: f64,

    /// Shift into the timing trace: t_trace = t_sim + timing_offset_s.
    /// Use this to pick which part of the trace aligns with the start of the run.
    #[arg(long, default_value_t = 0.0)]
    timing_offset_s: f64,

    /// Driver throttle setting (0–1)
    #[arg(long, default_value_t = 0.8)]
    power_ratio: f64,

    /// Track gradient (rise/run, positive = uphill, e.g. 0.01 = 1%)
    #[arg(long, default_value_t = 0.0)]
    gradient: f64,

    /// Head-wind speed (m/s, positive = head-wind)
    #[arg(long, default_value_t = 0.0)]
    wind_speed: f64,
}

fn main() {
    let cli = Cli::parse();

    let train =
        railml_rollingstock::load_formation(&cli.railml_file, &cli.formation_id).unwrap_or_else(|e| {
            eprintln!("Error loading rollingstock: {e}");
            std::process::exit(1)
        });

    let trace = TimingTrace::load(&cli.timing_file, &cli.timing_train_id).unwrap_or_else(|e| {
        eprintln!("Error loading timing data: {e}");
        std::process::exit(1)
    });

    let env = Environment {
        gradient: cli.gradient,
        wind_speed: cli.wind_speed,
    };
    let driver = DriverInput {
        power_ratio: cli.power_ratio,
        brake_ratio: 0.0,
    };

    let s1 = (cli.timing_s / cli.dt).round() as usize;
    let s2 = (cli.physics_s / cli.dt).round() as usize;
    let total = s1 + s2;

    let mut times: Vec<f64> = Vec::with_capacity(total);
    let mut sections: Vec<i32> = Vec::with_capacity(total);
    let mut pos_out: Vec<f64> = Vec::with_capacity(total);
    let mut spd_diff: Vec<Option<f64>> = Vec::with_capacity(total);
    let mut acc_diff: Vec<Option<f64>> = Vec::with_capacity(total);
    let mut spd_integ: Vec<f64> = Vec::with_capacity(total);
    let mut acc_integ: Vec<f64> = Vec::with_capacity(total);

    // Both sections share a common origin: the train starts from rest at position 0.
    let mut state = SimulatedState {
        position: Position { x: 0.0, y: 0.0, z: 0.0 },
        speed: 0.0,
        acceleration: 0.0,
    };

    // -----------------------------------------------------------------------
    // Section 1 — timing trace (differential) + physics in parallel (integral)
    // Both start from rest, so no position offset is needed.
    // -----------------------------------------------------------------------
    let mut prev_timing_pos: Option<f64> = None;
    let mut prev_diff_speed_ms: Option<f64> = None;

    for step in 0..s1 {
        let t = step as f64 * cli.dt;
        let t_trace = t + cli.timing_offset_s;

        let timing_pos = trace.position_at(t_trace);

        // v_diff = Δpos / Δt
        let diff_speed_ms: Option<f64> = match (timing_pos, prev_timing_pos) {
            (Some(p), Some(pp)) => Some((p - pp) / cli.dt),
            _ => None,
        };

        // a_diff = Δv_diff / Δt
        let diff_accel: Option<f64> = match (diff_speed_ms, prev_diff_speed_ms) {
            (Some(v), Some(pv)) => Some((v - pv) / cli.dt),
            _ => None,
        };

        // Physics advances in parallel — not synced to timing here.
        state = advance_train(&state, &train, &driver, &env, AdvanceTarget::Time(cli.dt));

        times.push(t);
        sections.push(1);
        // Use timing position when available; fall back to physics if trace has no data.
        pos_out.push(timing_pos.unwrap_or(state.position.x));
        spd_diff.push(diff_speed_ms.map(|v| v * 3.6));
        acc_diff.push(diff_accel);
        spd_integ.push(state.speed * 3.6);
        acc_integ.push(state.acceleration);

        prev_timing_pos = timing_pos;
        prev_diff_speed_ms = diff_speed_ms;
    }

    // -----------------------------------------------------------------------
    // Section 1 → 2 boundary: re-seed physics from last timing observation.
    // Assume a = 0 at the berth boundary; physics will compute the correct
    // acceleration from forces at the very next step.
    // -----------------------------------------------------------------------
    if let Some(pos) = prev_timing_pos {
        state.position.x = pos;
    }
    if let Some(v) = prev_diff_speed_ms {
        state.speed = v;
    }
    state.acceleration = 0.0;

    // -----------------------------------------------------------------------
    // Section 2 — physics from synced state
    // -----------------------------------------------------------------------
    for step in 0..s2 {
        let t = (s1 + step) as f64 * cli.dt;
        state = advance_train(&state, &train, &driver, &env, AdvanceTarget::Time(cli.dt));
        times.push(t);
        sections.push(2);
        pos_out.push(state.position.x);
        spd_diff.push(None);
        acc_diff.push(None);
        spd_integ.push(state.speed * 3.6);
        acc_integ.push(state.acceleration);
    }

    // -----------------------------------------------------------------------
    // Write Parquet
    // -----------------------------------------------------------------------
    let n = times.len();
    let mut df = DataFrame::new(
        n,
        vec![
            Series::new("time_s".into(), &times).into(),
            Series::new("section".into(), &sections).into(),
            Series::new("position_m".into(), &pos_out).into(),
            Series::new("speed_differential_kmh".into(), &spd_diff).into(),
            Series::new("accel_differential_mss".into(), &acc_diff).into(),
            Series::new("speed_integral_kmh".into(), &spd_integ).into(),
            Series::new("accel_integral_mss".into(), &acc_integ).into(),
        ],
    )
    .unwrap();

    let file = std::fs::File::create(&cli.output).unwrap_or_else(|e| {
        eprintln!("Cannot create '{}': {e}", cli.output.display());
        std::process::exit(1)
    });
    ParquetWriter::new(file)
        .with_compression(ParquetCompression::Lz4Raw)
        .finish(&mut df)
        .unwrap_or_else(|e| {
            eprintln!("Parquet write error: {e}");
            std::process::exit(1)
        });

    println!("Written {n} rows to '{}'", cli.output.display());
    println!(
        "  §1 timing  : {:>5} rows   (t = 0 … {:.0} s)",
        s1, cli.timing_s
    );
    println!(
        "  §2 physics : {:>5} rows   (t = {:.0} … {:.0} s)",
        s2,
        cli.timing_s,
        cli.timing_s + cli.physics_s
    );
}
