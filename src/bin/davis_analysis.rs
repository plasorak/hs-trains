//! Three-section analysis tool: physics → timing → physics for a single train.
//!
//! In the timing section (section 2) two speed estimates are recorded side-by-side:
//!
//! * **Differential** — finite difference of the timing-trace position:
//!   `v_diff(t) = (pos(t) − pos(t−dt)) / dt`
//!   `a_diff(t) = (v_diff(t) − v_diff(t−dt)) / dt`
//!
//! * **Integral** — Davis-equation ODE integrated forward (physics engine running
//!   in parallel, not synced to timing):
//!   `F_net = F_traction − F_gravity − (A + B·v + C·v²) − F_braking`
//!   `v_integ(t+dt) = v_integ(t) + (F_net/m)·dt`
//!
//! At the section-2 → section-3 boundary the physics state is re-seeded to
//! `(pos_timing, v_diff, a=0)` so section-3 physics starts from a sensible state.
//!
//! ## Output schema
//!
//! | column                   | description                                              |
//! |--------------------------|----------------------------------------------------------|
//! | `time_s`                 | elapsed simulation time (s)                              |
//! | `section`                | 1 = physics, 2 = timing, 3 = physics                     |
//! | `position_m`             | timing position in §2; physics position in §1 and §3     |
//! | `speed_differential_kmh` | Δpos/Δt speed estimate — §2 only, null elsewhere         |
//! | `accel_differential_mss` | Δspeed/Δt accel estimate — §2 only, null elsewhere       |
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
    about = "Physics / timing / physics three-section analysis"
)]
struct Cli {
    /// RailML 3.3 file containing the formation
    railml_file: PathBuf,
    /// Formation ID to load from the RailML file
    formation_id: String,
    /// Parquet berth-timing file used for section 2
    timing_file: PathBuf,
    /// Train ID to read from the timing file
    timing_train_id: String,
    /// Output Parquet file
    output: PathBuf,

    /// Duration of section 1 (first physics section) in seconds
    #[arg(long, default_value_t = 300.0)]
    section1_s: f64,

    /// Duration of section 2 (timing section) in seconds
    #[arg(long, default_value_t = 300.0)]
    section2_s: f64,

    /// Duration of section 3 (second physics section) in seconds
    #[arg(long, default_value_t = 300.0)]
    section3_s: f64,

    /// Integration / output time step in seconds
    #[arg(long, default_value_t = 1.0)]
    dt: f64,

    /// Shift into the timing trace: t_trace = t_sim − section1_s + timing_offset_s.
    /// Use this to pick which part of the trace lines up with section 2.
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

    let s1 = (cli.section1_s / cli.dt).round() as usize;
    let s2 = (cli.section2_s / cli.dt).round() as usize;
    let s3 = (cli.section3_s / cli.dt).round() as usize;
    let total = s1 + s2 + s3;

    let mut times: Vec<f64> = Vec::with_capacity(total);
    let mut sections: Vec<i32> = Vec::with_capacity(total);
    let mut pos_out: Vec<f64> = Vec::with_capacity(total);
    let mut spd_diff: Vec<Option<f64>> = Vec::with_capacity(total);
    let mut acc_diff: Vec<Option<f64>> = Vec::with_capacity(total);
    let mut spd_integ: Vec<f64> = Vec::with_capacity(total);
    let mut acc_integ: Vec<f64> = Vec::with_capacity(total);

    let mut state = SimulatedState {
        position: Position {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        speed: 0.0,
        acceleration: 0.0,
    };

    // -----------------------------------------------------------------------
    // Section 1 — pure physics
    // -----------------------------------------------------------------------
    for step in 0..s1 {
        let t = step as f64 * cli.dt;
        state = advance_train(&state, &train, &driver, &env, AdvanceTarget::Time(cli.dt));
        times.push(t);
        sections.push(1);
        pos_out.push(state.position.x);
        spd_diff.push(None);
        acc_diff.push(None);
        spd_integ.push(state.speed * 3.6);
        acc_integ.push(state.acceleration);
    }

    // -----------------------------------------------------------------------
    // Section 2 — timing trace (differential) + physics in parallel (integral)
    // -----------------------------------------------------------------------
    let mut prev_timing_pos: Option<f64> = None;
    let mut prev_diff_speed_ms: Option<f64> = None;

    // Shift timing-trace positions so §2 starts where §1 left off, giving all
    // three sections a common spatial reference for position-based plots.
    let timing_position_offset = state.position.x;

    for step in 0..s2 {
        let t = (s1 + step) as f64 * cli.dt;
        let t_trace = t - cli.section1_s + cli.timing_offset_s;

        let timing_pos = trace.position_at(t_trace).map(|p| p + timing_position_offset);

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
        sections.push(2);
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
    // Section 2 → 3 boundary: re-seed physics from last timing observation.
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
    // Section 3 — physics from synced state
    // -----------------------------------------------------------------------
    for step in 0..s3 {
        let t = (s1 + s2 + step) as f64 * cli.dt;
        state = advance_train(&state, &train, &driver, &env, AdvanceTarget::Time(cli.dt));
        times.push(t);
        sections.push(3);
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
        "  §1 physics : {:>5} rows   (t = 0 … {:.0} s)",
        s1, cli.section1_s
    );
    println!(
        "  §2 timing  : {:>5} rows   (t = {:.0} … {:.0} s)",
        s2,
        cli.section1_s,
        cli.section1_s + cli.section2_s
    );
    println!(
        "  §3 physics : {:>5} rows   (t = {:.0} … {:.0} s)",
        s3,
        cli.section1_s + cli.section2_s,
        cli.section1_s + cli.section2_s + cli.section3_s
    );
}
