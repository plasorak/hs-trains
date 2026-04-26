use clap::Parser;
use hs_trains::core::model::{DriverInput, Environment, Position, Route, SimulatedState, TrainDescription};
use hs_trains::core::physics::{AdvanceTarget, advance_train};
use hs_trains::io::timing::TimingTrace;
use hs_trains::{core::scheduler, io::railml_rollingstock, io::railml_infrastructure, io::railml_timetable};
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Train network simulator.
///
/// Runs a unified simulation where physics and timing trains coexist.
/// Time is always the outer loop: all trains advance together each step.
/// Results are flushed to Parquet in row-group batches to bound memory use.
#[derive(Parser)]
#[command(name = "hs-trains", version)]
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

/// Per-train configuration as written in the YAML file.
///
/// Every train is `kind: railml`: rolling-stock parameters come from a RailML 3.3
/// formation element.  If `timing_file` is supplied the train's position is taken
/// from the berth-timing trace whenever the trace has data at the current time;
/// physics simulation is used as a fallback for any gaps or out-of-range periods.
/// `driver` and `environment` are therefore always required — they govern the
/// physics fallback even for timing-traced trains.
///
/// If the simulation has an `infrastructure_file` and this train has a
/// `timetable_train_id`, the train's route is resolved from the RailML timetable
/// and the train will stop at the end of that route.
#[derive(serde::Deserialize)]
#[serde(tag = "kind")]
enum TrainConfigYaml {
    #[serde(rename = "railml")]
    RailML {
        id: String,
        railml_file: String,
        formation_id: String,
        /// Path to a Parquet berth-timing file. Optional; if absent, pure physics.
        #[serde(default)]
        timing_file: Option<String>,
        /// Train ID to filter from the timing Parquet file.
        /// Defaults to the train's `id` field when omitted.
        #[serde(default)]
        timing_train_id: Option<String>,
        /// Operational train ID in the RailML timetable.
        /// When set (and `infrastructure_file` is provided at the simulation level)
        /// the train follows the resolved route and stops at its end.
        #[serde(default)]
        timetable_train_id: Option<String>,
        environment: Environment,
        driver: DriverInput,
    },
}

/// Resolved per-train configuration used at runtime.
struct TrainConfig {
    id: String,
    train: TrainDescription,
    environment: Environment,
    driver: DriverInput,
    /// Pre-loaded timing trace, if a `timing_file` was specified.
    timing: Option<TimingTrace>,
    /// Resolved route from the RailML timetable, if `timetable_train_id` was set.
    route: Option<Route>,
}

fn resolve_train(yaml: TrainConfigYaml, routes: &HashMap<String, Route>) -> TrainConfig {
    match yaml {
        TrainConfigYaml::RailML {
            id,
            railml_file,
            formation_id,
            timing_file,
            timing_train_id,
            timetable_train_id,
            environment,
            driver,
        } => {
            let train =
                railml_rollingstock::load_formation(std::path::Path::new(&railml_file), &formation_id)
                    .unwrap_or_else(|e| {
                        eprintln!("Error loading rollingstock for train '{id}': {e}");
                        std::process::exit(1)
                    });
            let timing = timing_file.map(|tf| {
                let tid = timing_train_id.as_deref().unwrap_or(&id);
                TimingTrace::load(std::path::Path::new(&tf), tid).unwrap_or_else(|e| {
                    eprintln!("Error loading timing data for train '{id}': {e}");
                    std::process::exit(1)
                })
            });
            let route = timetable_train_id.as_deref().and_then(|tt_id| {
                match routes.get(tt_id) {
                    Some(r) => Some(r.clone()),
                    None => {
                        eprintln!(
                            "Warning: timetable_train_id '{tt_id}' for train '{id}' not found in timetable — no route assigned"
                        );
                        None
                    }
                }
            });
            TrainConfig { id, train, environment, driver, timing, route }
        }
    }
}

fn default_flush_rows() -> usize {
    1_000_000
}

#[derive(serde::Deserialize)]
struct SimulationConfig {
    time_step_s: f64,
    duration_s: f64,
    trains: Vec<TrainConfigYaml>,
    /// Maximum number of rows to buffer before flushing a Parquet row group.
    /// Row-based (not step-based) so the buffer size stays bounded regardless
    /// of the number of trains. Defaults to 1 000 000.
    #[serde(default = "default_flush_rows")]
    flush_rows: usize,
    /// Path to a RailML 3.3 file containing `<infrastructure>` and optionally
    /// `<timetable>` sections.  Required when any train specifies a
    /// `timetable_train_id`.
    #[serde(default)]
    infrastructure_file: Option<String>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    let config_path = &cli.config;
    let output_path = &cli.output;

    let config_str = std::fs::read_to_string(config_path).unwrap_or_else(|e| {
        eprintln!("Cannot read config '{}': {e}", config_path.display());
        std::process::exit(1)
    });
    let config: Config = serde_yaml_ng::from_str(&config_str).unwrap_or_else(|e| {
        eprintln!("Invalid config: {e}");
        std::process::exit(1)
    });

    let sim = config.simulation;

    // Load infrastructure + timetable routes if an infrastructure file is given.
    let routes: HashMap<String, Route> = match &sim.infrastructure_file {
        None => HashMap::new(),
        Some(path_str) => {
            let path = std::path::Path::new(path_str);
            let infra = railml_infrastructure::load_infrastructure(path).unwrap_or_else(|e| {
                eprintln!("Error loading infrastructure: {e}");
                std::process::exit(1)
            });
            railml_timetable::load_routes(path, &infra).unwrap_or_else(|e| {
                eprintln!("Error loading timetable routes: {e}");
                std::process::exit(1)
            })
        }
    };

    let trains: Vec<TrainConfig> = sim.trains.into_iter().map(|y| resolve_train(y, &routes)).collect();
    println!(
        "Running simulation: {} train(s), dt={}s, duration={}s, flush every {} rows",
        trains.len(),
        sim.time_step_s,
        sim.duration_s,
        sim.flush_rows
    );

    run_simulation(
        &trains,
        sim.time_step_s,
        sim.duration_s,
        output_path,
        sim.flush_rows,
    );
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

/// Run the simulation, streaming results to `output` as Parquet row groups.
///
/// Every train has a physics state that is advanced each step.  Trains with a
/// timing trace use the trace position when data is available at time `t` and
/// fall back to physics when outside the trace's range.
///
/// Trains with a route stop at the end of that route: once
/// `state.position.x >= route.total_length_m` the train is frozen and no
/// further physics is applied.
///
/// Per-step physics is parallelised with Rayon across all trains.
///
/// In-memory buffers hold at most `flush_rows` rows at a time; they are
/// written as a row group and cleared on every flush boundary.
///
/// Output columns:
/// - `train_id`          — string identifier
/// - `time_s`            — elapsed simulation time in seconds
/// - `position_m`        — position in metres (null only if timing trace returns
///                         null AND physics is also unavailable — should not occur)
/// - `speed_kmh`         — speed in km/h (null when timing position is used)
/// - `acceleration_mss`  — acceleration in m/s² (null when timing position is used)
/// - `track_id`          — RailML track id at the current position (null for trains
///                         without a route)
/// - `element_offset_m`  — distance from the start of the current track element
///                         in metres (null for trains without a route)

fn build_batch<'a>(
    train_id_data: &[&'a str],
    event_kind_data: &[&'a str],
    time_s_data: &[f64],
    position_m_data: &[Option<f64>],
    speed_kmh_data: &[Option<f64>],
    accel_mss_data: &[Option<f64>],
    track_id_data: &[Option<String>],
    element_offset_m_data: &[Option<f64>],
) -> DataFrame {
    let n = time_s_data.len();
    // Polars needs &[Option<&str>] for nullable string series.
    let track_id_refs: Vec<Option<&str>> =
        track_id_data.iter().map(|o| o.as_deref()).collect();
    DataFrame::new(
        n,
        vec![
            Series::new("train_id".into(), train_id_data).into(),
            Series::new("event_kind".into(), event_kind_data).into(),
            Series::new("time_s".into(), time_s_data).into(),
            Series::new("position_m".into(), position_m_data).into(),
            Series::new("speed_kmh".into(), speed_kmh_data).into(),
            Series::new("acceleration_mss".into(), accel_mss_data).into(),
            Series::new("track_id".into(), track_id_refs.as_slice()).into(),
            Series::new("element_offset_m".into(), element_offset_m_data).into(),
        ],
    )
    .unwrap()
}

/// Write buffered rows to the Parquet file and clear the buffers.
/// Extracted to avoid duplicating the flush logic between the mid-loop check and the final flush.
#[allow(clippy::too_many_arguments)]
fn flush_batch<W: std::io::Write>(
    writer: &mut polars::io::parquet::write::BatchedWriter<W>,
    train_id_data: &mut Vec<&str>,
    event_kind_data: &mut Vec<&'static str>,
    time_s_data: &mut Vec<f64>,
    position_m_data: &mut Vec<Option<f64>>,
    speed_kmh_data: &mut Vec<Option<f64>>,
    accel_mss_data: &mut Vec<Option<f64>>,
    track_id_data: &mut Vec<Option<String>>,
    element_offset_m_data: &mut Vec<Option<f64>>,
    total_rows: &mut usize,
    pb: &ProgressBar,
    label: &str,
) {
    let n = time_s_data.len();
    let batch = build_batch(
        train_id_data,
        event_kind_data,
        time_s_data,
        position_m_data,
        speed_kmh_data,
        accel_mss_data,
        track_id_data,
        element_offset_m_data,
    );
    writer.write_batch(&batch).unwrap_or_else(|e| {
        eprintln!("Write error ({label}): {e}");
        std::process::exit(1)
    });
    *total_rows += n;
    pb.println(format!("  flushed {n} rows (total: {total_rows})"));
    train_id_data.clear();
    event_kind_data.clear();
    time_s_data.clear();
    position_m_data.clear();
    speed_kmh_data.clear();
    accel_mss_data.clear();
    track_id_data.clear();
    element_offset_m_data.clear();
}

/// Output columns produced for one train at one time step.
struct StepRow {
    position_m: Option<f64>,
    speed_kmh: Option<f64>,
    accel_mss: Option<f64>,
    track_id: Option<String>,
    element_offset_m: Option<f64>,
}

/// Advance `state` by `dt_i` seconds, honouring the route end if one is set.
///
/// If the train has already reached the end of its route the state is left
/// unchanged.  If the physics step overshoots the route end, position is
/// clamped and the train is brought to a halt.
fn advance_with_route(state: &mut SimulatedState, cfg: &TrainConfig, dt_i: f64) {
    let at_end = cfg.route.as_ref().map_or(false, |r| state.position.x >= r.total_length_m);
    if at_end {
        return;
    }
    *state = advance_train(state, &cfg.train, &cfg.driver, &cfg.environment, AdvanceTarget::Time(dt_i));
    // Clamp to route end if the step overshot.
    if let Some(route) = &cfg.route {
        if state.position.x >= route.total_length_m {
            state.position.x = route.total_length_m;
            state.speed = 0.0;
            state.acceleration = 0.0;
        }
    }
}

/// Build a `StepRow` for one train, using the timing trace position when
/// available and the physics state otherwise.  The network position
/// (track_id / element_offset_m) is always derived from whichever position
/// is reported so that the two columns stay consistent with position_m.
fn make_step_row(cfg: &TrainConfig, state: &SimulatedState, t: f64) -> StepRow {
    match cfg.timing.as_ref().and_then(|tr| tr.position_at(t)) {
        Some(pos) => {
            let net_pos = cfg.route.as_ref().and_then(|r| r.locate(pos));
            let (track_id, element_offset_m) = match net_pos {
                Some(np) => (Some(np.track_id), Some(np.offset_m)),
                None => (None, None),
            };
            StepRow { position_m: Some(pos), speed_kmh: None, accel_mss: None, track_id, element_offset_m }
        }
        None => {
            let net_pos = cfg.route.as_ref().and_then(|r| r.locate(state.position.x));
            let (track_id, element_offset_m) = match net_pos {
                Some(np) => (Some(np.track_id), Some(np.offset_m)),
                None => (None, None),
            };
            StepRow {
                position_m: Some(state.position.x),
                speed_kmh: Some(state.speed * 3.6),
                accel_mss: Some(state.acceleration),
                track_id,
                element_offset_m,
            }
        }
    }
}

fn run_simulation(
    trains: &[TrainConfig],
    dt: f64,
    duration: f64,
    output: &std::path::Path,
    flush_rows: usize,
) {
    let steps = (duration / dt).round() as usize;
    let buf_cap = flush_rows.min(steps * trains.len());

    // Pre-cache IDs as &str slices — avoids per-step String allocations in the hot loop.
    let train_id_cache: Vec<&str> = trains.iter().map(|c| c.id.as_str()).collect();

    let mut train_id_data = Vec::<&str>::with_capacity(buf_cap);
    let mut event_kind_data = Vec::<&'static str>::with_capacity(buf_cap);
    let mut time_s_data = Vec::<f64>::with_capacity(buf_cap);
    let mut position_m_data = Vec::<Option<f64>>::with_capacity(buf_cap);
    let mut speed_kmh_data = Vec::<Option<f64>>::with_capacity(buf_cap);
    let mut accel_mss_data = Vec::<Option<f64>>::with_capacity(buf_cap);
    let mut track_id_data = Vec::<Option<String>>::with_capacity(buf_cap);
    let mut element_offset_m_data = Vec::<Option<f64>>::with_capacity(buf_cap);

    // Fixed output schema — must match the Series built at flush time.
    let schema = Schema::from_iter([
        Field::new("train_id".into(), DataType::String),
        Field::new("event_kind".into(), DataType::String),
        Field::new("time_s".into(), DataType::Float64),
        Field::new("position_m".into(), DataType::Float64),
        Field::new("speed_kmh".into(), DataType::Float64),
        Field::new("acceleration_mss".into(), DataType::Float64),
        Field::new("track_id".into(), DataType::String),
        Field::new("element_offset_m".into(), DataType::Float64),
    ]);

    let file = std::fs::File::create(output).unwrap_or_else(|e| {
        eprintln!("Cannot create '{}': {e}", output.display());
        std::process::exit(1)
    });
    let mut writer = ParquetWriter::new(file)
        .with_compression(ParquetCompression::Lz4Raw)
        .set_parallel(true)
        .batched(&schema)
        .unwrap_or_else(|e| {
            eprintln!("Cannot initialise Parquet writer: {e}");
            std::process::exit(1)
        });

    // Every train starts from rest at position 0.  The physics state is always
    // maintained; trains with a timing trace overlay it on top of physics.
    let mut states: Vec<SimulatedState> = trains
        .iter()
        .map(|_| SimulatedState {
            position: Position { x: 0.0, y: 0.0, z: 0.0 },
            speed: 0.0,
            acceleration: 0.0,
        })
        .collect();

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
        println!(
            "Event queue seeded: 1 initial physics tick (self-scheduling) + {n_random} random placeholder events"
        );
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
        if event.time > duration {
            break;
        }

        match event.kind {
            scheduler::EventKind::PhysicsTick => {
                let t = event.time;

                // Schedule the next tick only if it is still within the simulation
                // window AND there is meaningful work left.
                let any_active = trains.iter().zip(states.iter()).any(|(cfg, s)| {
                    cfg.timing.is_some()
                        || s.speed > 0.0
                        || cfg
                            .route
                            .as_ref()
                            .map_or(false, |r| s.position.x < r.total_length_m)
                });
                if t + dt <= duration && (any_active || !queue.is_empty()) {
                    queue.push(t + dt, None, scheduler::EventKind::PhysicsTick);
                }

                // Each train advances from its last-known time to t (may differ if
                // random events have advanced individual trains in between).
                let step_results: Vec<StepRow> = trains
                    .par_iter()
                    .zip(states.par_iter_mut())
                    .zip(last_times.par_iter())
                    .map(|((cfg, state), &lt)| {
                        let dt_i = t - lt;
                        if dt_i > 0.0 {
                            advance_with_route(state, cfg, dt_i);
                        }
                        make_step_row(cfg, state, t)
                    })
                    .collect();

                last_times.iter_mut().for_each(|lt| *lt = t);

                for (i, row) in step_results.into_iter().enumerate() {
                    train_id_data.push(train_id_cache[i]);
                    event_kind_data.push("physics_tick");
                    time_s_data.push(t);
                    position_m_data.push(row.position_m);
                    speed_kmh_data.push(row.speed_kmh);
                    accel_mss_data.push(row.accel_mss);
                    track_id_data.push(row.track_id);
                    element_offset_m_data.push(row.element_offset_m);
                }

                pb.inc(1);
            }

            scheduler::EventKind::Random(kind) => {
                let t = event.time;
                let kind_str: &'static str = match kind {
                    scheduler::RandomEventKind::Departure => "departure",
                    scheduler::RandomEventKind::Arrival => "arrival",
                    scheduler::RandomEventKind::SignalChange => "signal_change",
                    scheduler::RandomEventKind::SpeedChange { .. } => "speed_change",
                };
                // Advance the targeted train to this event's time and record its state.
                if let Some(scheduler::EntityRef::Train(i)) = event.target {
                    if i < trains.len() {
                        let dt_i = t - last_times[i];
                        if dt_i > 0.0 {
                            let cfg = &trains[i];
                            let s = &mut states[i];
                            advance_with_route(s, cfg, dt_i);
                            let row = make_step_row(cfg, s, t);
                            last_times[i] = t;
                            train_id_data.push(train_id_cache[i]);
                            event_kind_data.push(kind_str);
                            time_s_data.push(t);
                            position_m_data.push(row.position_m);
                            speed_kmh_data.push(row.speed_kmh);
                            accel_mss_data.push(row.accel_mss);
                            track_id_data.push(row.track_id);
                            element_offset_m_data.push(row.element_offset_m);
                        }
                    }
                }
                // Signal events don't produce rows yet.
            }
        }

        if time_s_data.len() >= flush_rows {
            flush_batch(
                &mut writer,
                &mut train_id_data,
                &mut event_kind_data,
                &mut time_s_data,
                &mut position_m_data,
                &mut speed_kmh_data,
                &mut accel_mss_data,
                &mut track_id_data,
                &mut element_offset_m_data,
                &mut total_rows,
                &pb,
                "mid-loop",
            );
        }
    }

    // Final flush: covers both normal queue exhaustion and the time-limit break.
    if !time_s_data.is_empty() {
        flush_batch(
            &mut writer,
            &mut train_id_data,
            &mut event_kind_data,
            &mut time_s_data,
            &mut position_m_data,
            &mut speed_kmh_data,
            &mut accel_mss_data,
            &mut track_id_data,
            &mut element_offset_m_data,
            &mut total_rows,
            &pb,
            "final flush",
        );
    }

    pb.finish_and_clear();
    writer.finish().unwrap_or_else(|e| {
        eprintln!("Failed to finalise Parquet file: {e}");
        std::process::exit(1)
    });
    println!("Written {total_rows} rows to '{}'", output.display());
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Config deserialization ----------------------------------------------

    #[test]
    fn test_config_defaults_no_infrastructure() {
        let yaml = r#"
simulation:
  time_step_s: 0.5
  duration_s: 100.0
  trains: []
"#;
        let config: Config = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(config.simulation.time_step_s, 0.5);
        assert_eq!(config.simulation.duration_s, 100.0);
        assert!(config.simulation.infrastructure_file.is_none());
        assert_eq!(config.simulation.flush_rows, 1_000_000);
    }

    #[test]
    fn test_config_infrastructure_file_round_trips() {
        let yaml = r#"
simulation:
  time_step_s: 1.0
  duration_s: 60.0
  infrastructure_file: network.xml
  trains: []
"#;
        let config: Config = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(
            config.simulation.infrastructure_file.as_deref(),
            Some("network.xml")
        );
    }

    #[test]
    fn test_config_timetable_train_id_round_trips() {
        let yaml = r#"
simulation:
  time_step_s: 1.0
  duration_s: 60.0
  trains:
    - id: express
      kind: railml
      railml_file: rs.xml
      formation_id: f1
      timetable_train_id: OT_Express
      environment:
        gradient: 0.0
        wind_speed: 0.0
      driver:
        power_ratio: 0.8
        brake_ratio: 0.0
"#;
        let config: Config = serde_yaml_ng::from_str(yaml).unwrap();
        let TrainConfigYaml::RailML { timetable_train_id, .. } = &config.simulation.trains[0];
        assert_eq!(timetable_train_id.as_deref(), Some("OT_Express"));
    }

    #[test]
    fn test_config_optional_train_fields_default_to_none() {
        let yaml = r#"
simulation:
  time_step_s: 1.0
  duration_s: 60.0
  trains:
    - id: plain
      kind: railml
      railml_file: rs.xml
      formation_id: f1
      environment:
        gradient: 0.0
        wind_speed: 0.0
      driver:
        power_ratio: 1.0
        brake_ratio: 0.0
"#;
        let config: Config = serde_yaml_ng::from_str(yaml).unwrap();
        let TrainConfigYaml::RailML {
            timing_file,
            timing_train_id,
            timetable_train_id,
            ..
        } = &config.simulation.trains[0];
        assert!(timing_file.is_none());
        assert!(timing_train_id.is_none());
        assert!(timetable_train_id.is_none());
    }

    #[test]
    fn test_config_flush_rows_explicit() {
        let yaml = r#"
simulation:
  time_step_s: 1.0
  duration_s: 10.0
  flush_rows: 500
  trains: []
"#;
        let config: Config = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(config.simulation.flush_rows, 500);
    }

    // --- build_batch ---------------------------------------------------------

    #[test]
    fn test_build_batch_column_names_and_count() {
        let df = build_batch(
            &["t1"],
            &["physics_tick"],
            &[1.0],
            &[Some(100.0)],
            &[Some(50.0)],
            &[Some(0.5)],
            &[Some("track_A".to_string())],
            &[Some(25.0)],
        );

        assert_eq!(df.height(), 1);
        assert_eq!(df.width(), 8);

        // get_column_names() returns Vec<&PlSmallStr>; compare via as_str().
        let names: Vec<&str> = df.get_column_names().into_iter().map(|s| s.as_str()).collect();
        assert!(names.contains(&"train_id"));
        assert!(names.contains(&"event_kind"));
        assert!(names.contains(&"time_s"));
        assert!(names.contains(&"position_m"));
        assert!(names.contains(&"speed_kmh"));
        assert!(names.contains(&"acceleration_mss"));
        assert!(names.contains(&"track_id"));
        assert!(names.contains(&"element_offset_m"));
    }

    #[test]
    fn test_build_batch_nullable_columns() {
        // Row 0: routed train — track_id and element_offset_m are populated.
        // Row 1: unrouted train — both are null.
        let df = build_batch(
            &["routed", "free"],
            &["physics_tick", "physics_tick"],
            &[1.0, 1.0],
            &[Some(500.0), Some(200.0)],
            &[Some(30.0), Some(80.0)],
            &[Some(0.1), Some(0.2)],
            &[Some("track_X".to_string()), None],
            &[Some(50.0), None],
        );

        // is_null() returns BooleanChunked; use .get(row) to check individual rows.
        let track_null = df.column("track_id").unwrap().is_null();
        assert_eq!(track_null.get(0), Some(false)); // routed row has a value
        assert_eq!(track_null.get(1), Some(true));  // free runner is null

        let offset_null = df.column("element_offset_m").unwrap().is_null();
        assert_eq!(offset_null.get(0), Some(false));
        assert_eq!(offset_null.get(1), Some(true));
    }

    #[test]
    fn test_build_batch_multiple_rows() {
        let n = 5_usize;
        let train_ids: Vec<&str> = vec!["t"; n];
        let event_kinds: Vec<&str> = vec!["physics_tick"; n];
        let times: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let positions: Vec<Option<f64>> = (0..n).map(|i| Some(i as f64 * 10.0)).collect();
        let speeds: Vec<Option<f64>> = vec![None; n];
        let accels: Vec<Option<f64>> = vec![None; n];
        let track_ids: Vec<Option<String>> = vec![None; n];
        let offsets: Vec<Option<f64>> = vec![None; n];

        let df = build_batch(
            &train_ids,
            &event_kinds,
            &times,
            &positions,
            &speeds,
            &accels,
            &track_ids,
            &offsets,
        );

        assert_eq!(df.height(), n);
    }
}
