use clap::Parser;
use hs_trains::core::model::{DriverInput, Environment, Position, Route, SimulatedState, TrainDescription};
use hs_trains::core::physics::{AdvanceTarget, advance_train, traction_power_kw};
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
/// formation element and every train must have a `timetable_train_id` that resolves
/// to a route in the simulation's `infrastructure_file`.  The train stops at the end
/// of that route.
///
/// If `timing_file` is also supplied the train's position is taken from the
/// berth-timing trace whenever the trace has data at the current time; physics is
/// used as a fallback for gaps or out-of-range periods.  `driver` and `environment`
/// are always required — they govern the physics fallback even for timing-traced trains.
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
        /// Operational train ID in the RailML timetable. Required.
        /// The train follows this route and stops at its end.
        timetable_train_id: String,
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
    /// Resolved route from the RailML timetable.
    route: Route,
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
            let route = match routes.get(&timetable_train_id) {
                Some(r) => r.clone(),
                None => {
                    eprintln!(
                        "Error: timetable_train_id '{timetable_train_id}' for train '{id}' not found in timetable"
                    );
                    std::process::exit(1)
                }
            };
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
    /// Path to a RailML 3.3 file containing `<infrastructure>` and `<timetable>`
    /// sections.  Every train's `timetable_train_id` is resolved against this file.
    infrastructure_file: String,
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

    // Load infrastructure and timetable routes.
    let path = std::path::Path::new(&sim.infrastructure_file);
    let infra = railml_infrastructure::load_infrastructure(path).unwrap_or_else(|e| {
        eprintln!("Error loading infrastructure: {e}");
        std::process::exit(1)
    });
    let routes = railml_timetable::load_routes(path, &infra).unwrap_or_else(|e| {
        eprintln!("Error loading timetable routes: {e}");
        std::process::exit(1)
    });

    // Build geographic lookup: track_id → (WGS84 coord list, element length in m).
    // Used by make_step_row to interpolate lon/lat from position along each track.
    let track_geo: HashMap<String, (Vec<(f64, f64)>, f64)> = infra
        .tracks
        .values()
        .filter_map(|t| {
            let length_m = infra.net_elements.get(&t.net_element_id)?.length_m;
            let coords = infra.track_coords.get(&t.id).cloned().unwrap_or_default();
            Some((t.id.clone(), (coords, length_m)))
        })
        .collect();

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
        &track_geo,
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
/// - `track_id`          — RailML track id at the current position
/// - `element_offset_m`  — distance from the start of the current track element
///                         in metres

/// Accumulates all output columns for a batch of rows before writing to Parquet.
struct BatchBuffers<'a> {
    train_id: Vec<&'a str>,
    event_kind: Vec<&'static str>,
    time_s: Vec<f64>,
    position_m: Vec<Option<f64>>,
    speed_kmh: Vec<Option<f64>>,
    accel_mss: Vec<Option<f64>>,
    track_id: Vec<Option<String>>,
    element_offset_m: Vec<Option<f64>>,
    lon_deg: Vec<Option<f64>>,
    lat_deg: Vec<Option<f64>>,
    power_kw: Vec<Option<f64>>,
}

impl<'a> BatchBuffers<'a> {
    fn with_capacity(cap: usize) -> Self {
        BatchBuffers {
            train_id: Vec::with_capacity(cap),
            event_kind: Vec::with_capacity(cap),
            time_s: Vec::with_capacity(cap),
            position_m: Vec::with_capacity(cap),
            speed_kmh: Vec::with_capacity(cap),
            accel_mss: Vec::with_capacity(cap),
            track_id: Vec::with_capacity(cap),
            element_offset_m: Vec::with_capacity(cap),
            lon_deg: Vec::with_capacity(cap),
            lat_deg: Vec::with_capacity(cap),
            power_kw: Vec::with_capacity(cap),
        }
    }

    fn len(&self) -> usize {
        self.time_s.len()
    }

    fn is_empty(&self) -> bool {
        self.time_s.is_empty()
    }

    fn push(&mut self, train_id: &'a str, event_kind: &'static str, t: f64, row: StepRow) {
        self.train_id.push(train_id);
        self.event_kind.push(event_kind);
        self.time_s.push(t);
        self.position_m.push(row.position_m);
        self.speed_kmh.push(row.speed_kmh);
        self.accel_mss.push(row.accel_mss);
        self.track_id.push(row.track_id);
        self.element_offset_m.push(row.element_offset_m);
        self.lon_deg.push(row.lon_deg);
        self.lat_deg.push(row.lat_deg);
        self.power_kw.push(row.power_kw);
    }

    fn clear(&mut self) {
        self.train_id.clear();
        self.event_kind.clear();
        self.time_s.clear();
        self.position_m.clear();
        self.speed_kmh.clear();
        self.accel_mss.clear();
        self.track_id.clear();
        self.element_offset_m.clear();
        self.lon_deg.clear();
        self.lat_deg.clear();
        self.power_kw.clear();
    }

    fn build_dataframe(&self) -> DataFrame {
        // Polars needs &[Option<&str>] for nullable string series.
        let track_id_refs: Vec<Option<&str>> =
            self.track_id.iter().map(|o| o.as_deref()).collect();
        DataFrame::new(
            self.time_s.len(),
            vec![
                Series::new("train_id".into(), self.train_id.as_slice()).into(),
                Series::new("event_kind".into(), self.event_kind.as_slice()).into(),
                Series::new("time_s".into(), self.time_s.as_slice()).into(),
                Series::new("position_m".into(), self.position_m.as_slice()).into(),
                Series::new("speed_kmh".into(), self.speed_kmh.as_slice()).into(),
                Series::new("acceleration_mss".into(), self.accel_mss.as_slice()).into(),
                Series::new("track_id".into(), track_id_refs.as_slice()).into(),
                Series::new("element_offset_m".into(), self.element_offset_m.as_slice()).into(),
                Series::new("lon_deg".into(), self.lon_deg.as_slice()).into(),
                Series::new("lat_deg".into(), self.lat_deg.as_slice()).into(),
                Series::new("power_kw".into(), self.power_kw.as_slice()).into(),
            ],
        )
        .unwrap()
    }
}

/// Write buffered rows to the Parquet file and clear the buffers.
/// Extracted to avoid duplicating the flush logic between the mid-loop check and the final flush.
fn flush_batch<W: std::io::Write>(
    writer: &mut polars::io::parquet::write::BatchedWriter<W>,
    buf: &mut BatchBuffers<'_>,
    total_rows: &mut usize,
    pb: &ProgressBar,
    label: &str,
) {
    let n = buf.len();
    let batch = buf.build_dataframe();
    writer.write_batch(&batch).unwrap_or_else(|e| {
        eprintln!("Write error ({label}): {e}");
        std::process::exit(1)
    });
    *total_rows += n;
    pb.println(format!("  flushed {n} rows (total: {total_rows})"));
    buf.clear();
}

/// Output columns produced for one train at one time step.
struct StepRow {
    position_m: Option<f64>,
    speed_kmh: Option<f64>,
    accel_mss: Option<f64>,
    track_id: Option<String>,
    element_offset_m: Option<f64>,
    lon_deg: Option<f64>,
    lat_deg: Option<f64>,
    power_kw: Option<f64>,
}

/// Advance `state` by `dt_i` seconds, stopping at the route end.
///
/// If the train has already reached the end of its route the state is left
/// unchanged.  If the physics step overshoots the route end, position is
/// clamped and the train is brought to a halt.
fn advance_with_route(state: &mut SimulatedState, cfg: &TrainConfig, dt_i: f64) {
    if state.position.x >= cfg.route.total_length_m {
        return;
    }
    *state = advance_train(state, &cfg.train, &cfg.driver, &cfg.environment, AdvanceTarget::Time(dt_i));
    if state.position.x >= cfg.route.total_length_m {
        state.position.x = cfg.route.total_length_m;
        state.speed = 0.0;
        state.acceleration = 0.0;
    }
}

/// Linearly interpolate a (lon, lat) coordinate along a polyline.
///
/// `t` is in `[0.0, 1.0]`; 0.0 returns the first vertex, 1.0 returns the last.
fn interpolate_coord(coords: &[(f64, f64)], t: f64) -> (f64, f64) {
    let n = coords.len();
    debug_assert!(n >= 2);
    let fi = t * (n - 1) as f64;
    let i = (fi as usize).min(n - 2);
    let frac = fi - i as f64;
    let (x0, y0) = coords[i];
    let (x1, y1) = coords[i + 1];
    (x0 + frac * (x1 - x0), y0 + frac * (y1 - y0))
}

/// Build a `StepRow` for one train, using the timing trace position when
/// available and the physics state otherwise.  The network position
/// (track_id / element_offset_m) is always derived from whichever position
/// is reported so that the two columns stay consistent with position_m.
fn make_step_row(
    cfg: &TrainConfig,
    state: &SimulatedState,
    t: f64,
    track_geo: &HashMap<String, (Vec<(f64, f64)>, f64)>,
) -> StepRow {
    let locate = |pos: f64| {
        // route.locate() returns None only for empty routes; routes are always
        // non-empty at this point (rejected at load time if empty).
        cfg.route.locate(pos).map(|np| (np.track_id, np.offset_m))
    };

    let geo_coords = |track_id: &Option<String>, offset_m: Option<f64>| -> (Option<f64>, Option<f64>) {
        let (tid, off) = match (track_id.as_deref(), offset_m) {
            (Some(tid), Some(off)) => (tid, off),
            _ => return (None, None),
        };
        let (coords, length_m) = match track_geo.get(tid) {
            Some(v) => v,
            None => return (None, None),
        };
        if coords.len() < 2 || *length_m <= 0.0 {
            return (None, None);
        }
        let t_frac = (off / length_m).clamp(0.0, 1.0);
        let (lon, lat) = interpolate_coord(coords, t_frac);
        (Some(lon), Some(lat))
    };

    match cfg.timing.as_ref().and_then(|tr| tr.position_at(t)) {
        Some(pos) => {
            let (track_id, element_offset_m) = locate(pos).unzip();
            let (lon_deg, lat_deg) = geo_coords(&track_id, element_offset_m);
            StepRow {
                position_m: Some(pos),
                speed_kmh: None,
                accel_mss: None,
                track_id,
                element_offset_m,
                lon_deg,
                lat_deg,
                power_kw: None,
            }
        }
        None => {
            let (track_id, element_offset_m) = locate(state.position.x).unzip();
            let (lon_deg, lat_deg) = geo_coords(&track_id, element_offset_m);
            StepRow {
                position_m: Some(state.position.x),
                speed_kmh: Some(state.speed * 3.6),
                accel_mss: Some(state.acceleration),
                track_id,
                element_offset_m,
                lon_deg,
                lat_deg,
                power_kw: Some(traction_power_kw(state.speed, &cfg.train, &cfg.driver)),
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
    track_geo: &HashMap<String, (Vec<(f64, f64)>, f64)>,
) {
    let steps = (duration / dt).round() as usize;
    let buf_cap = flush_rows.min(steps * trains.len());

    // Pre-cache IDs as &str slices — avoids per-step String allocations in the hot loop.
    let train_id_cache: Vec<&str> = trains.iter().map(|c| c.id.as_str()).collect();

    let mut buf = BatchBuffers::with_capacity(buf_cap);

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
        Field::new("lon_deg".into(), DataType::Float64),
        Field::new("lat_deg".into(), DataType::Float64),
        Field::new("power_kw".into(), DataType::Float64),
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
                    cfg.timing.is_some() || s.position.x < cfg.route.total_length_m
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
                        make_step_row(cfg, state, t, track_geo)
                    })
                    .collect();

                last_times.iter_mut().for_each(|lt| *lt = t);

                for (i, row) in step_results.into_iter().enumerate() {
                    buf.push(train_id_cache[i], "physics_tick", t, row);
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
                            let row = make_step_row(cfg, s, t, track_geo);
                            last_times[i] = t;
                            buf.push(train_id_cache[i], kind_str, t, row);
                        }
                    }
                }
                // Signal events don't produce rows yet.
            }
        }

        if buf.len() >= flush_rows {
            flush_batch(&mut writer, &mut buf, &mut total_rows, &pb, "mid-loop");
        }
    }

    // Final flush: covers both normal queue exhaustion and the time-limit break.
    if !buf.is_empty() {
        flush_batch(&mut writer, &mut buf, &mut total_rows, &pb, "final flush");
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
    fn test_config_missing_infrastructure_file_fails() {
        // infrastructure_file is required; omitting it must fail to deserialize.
        let yaml = r#"
simulation:
  time_step_s: 0.5
  duration_s: 100.0
  trains: []
"#;
        assert!(serde_yaml_ng::from_str::<Config>(yaml).is_err());
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
        assert_eq!(config.simulation.infrastructure_file, "network.xml");
        assert_eq!(config.simulation.flush_rows, 1_000_000);
    }

    #[test]
    fn test_config_timetable_train_id_round_trips() {
        let yaml = r#"
simulation:
  time_step_s: 1.0
  duration_s: 60.0
  infrastructure_file: network.xml
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
        assert_eq!(timetable_train_id, "OT_Express");
    }

    #[test]
    fn test_config_missing_timetable_train_id_fails() {
        // timetable_train_id is required; omitting it must fail to deserialize.
        let yaml = r#"
simulation:
  time_step_s: 1.0
  duration_s: 60.0
  infrastructure_file: network.xml
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
        assert!(serde_yaml_ng::from_str::<Config>(yaml).is_err());
    }

    #[test]
    fn test_config_optional_timing_fields_default_to_none() {
        // timing_file and timing_train_id are still optional.
        let yaml = r#"
simulation:
  time_step_s: 1.0
  duration_s: 60.0
  infrastructure_file: network.xml
  trains:
    - id: plain
      kind: railml
      railml_file: rs.xml
      formation_id: f1
      timetable_train_id: OT_Plain
      environment:
        gradient: 0.0
        wind_speed: 0.0
      driver:
        power_ratio: 1.0
        brake_ratio: 0.0
"#;
        let config: Config = serde_yaml_ng::from_str(yaml).unwrap();
        let TrainConfigYaml::RailML { timing_file, timing_train_id, .. } = &config.simulation.trains[0];
        assert!(timing_file.is_none());
        assert!(timing_train_id.is_none());
    }

    #[test]
    fn test_config_flush_rows_explicit() {
        let yaml = r#"
simulation:
  time_step_s: 1.0
  duration_s: 10.0
  flush_rows: 500
  infrastructure_file: network.xml
  trains: []
"#;
        let config: Config = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(config.simulation.flush_rows, 500);
    }

    // --- BatchBuffers / build_dataframe --------------------------------------

    fn push_row(buf: &mut BatchBuffers<'_>, train_id: &'static str, pos: Option<f64>, track: Option<&str>) {
        buf.push(train_id, "physics_tick", 1.0, StepRow {
            position_m: pos,
            speed_kmh: None,
            accel_mss: None,
            track_id: track.map(str::to_string),
            element_offset_m: track.map(|_| 0.0),
            lon_deg: None,
            lat_deg: None,
            power_kw: None,
        });
    }

    #[test]
    fn test_batch_buffers_column_names_and_count() {
        let mut buf = BatchBuffers::with_capacity(4);
        push_row(&mut buf, "t1", Some(100.0), Some("track_A"));
        let df = buf.build_dataframe();

        assert_eq!(df.height(), 1);
        assert_eq!(df.width(), 11);

        let names: Vec<&str> = df.get_column_names().into_iter().map(|s| s.as_str()).collect();
        assert!(names.contains(&"train_id"));
        assert!(names.contains(&"event_kind"));
        assert!(names.contains(&"time_s"));
        assert!(names.contains(&"position_m"));
        assert!(names.contains(&"speed_kmh"));
        assert!(names.contains(&"acceleration_mss"));
        assert!(names.contains(&"track_id"));
        assert!(names.contains(&"element_offset_m"));
        assert!(names.contains(&"lon_deg"));
        assert!(names.contains(&"lat_deg"));
        assert!(names.contains(&"power_kw"));
    }

    #[test]
    fn test_batch_buffers_nullable_columns() {
        // Row 0: routed train — track_id populated.
        // Row 1: no track — both null.
        let mut buf = BatchBuffers::with_capacity(4);
        push_row(&mut buf, "routed", Some(500.0), Some("track_X"));
        push_row(&mut buf, "free", Some(200.0), None);
        let df = buf.build_dataframe();

        let track_null = df.column("track_id").unwrap().is_null();
        assert_eq!(track_null.get(0), Some(false));
        assert_eq!(track_null.get(1), Some(true));

        let offset_null = df.column("element_offset_m").unwrap().is_null();
        assert_eq!(offset_null.get(0), Some(false));
        assert_eq!(offset_null.get(1), Some(true));
    }

    #[test]
    fn test_batch_buffers_multiple_rows_and_clear() {
        let mut buf = BatchBuffers::with_capacity(8);
        for i in 0..5 {
            buf.push("t", "physics_tick", i as f64, StepRow {
                position_m: Some(i as f64 * 10.0),
                speed_kmh: None,
                accel_mss: None,
                track_id: None,
                element_offset_m: None,
                lon_deg: None,
                lat_deg: None,
                power_kw: None,
            });
        }
        assert_eq!(buf.len(), 5);
        assert_eq!(buf.build_dataframe().height(), 5);

        buf.clear();
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    // --- advance_with_route --------------------------------------------------

    fn make_minimal_train() -> TrainDescription {
        TrainDescription {
            power: 2_000_000.0,
            traction_force_at_standstill: 300_000.0,
            max_speed: 200.0,
            mass: 100_000.0,
            davis_a: 1_500.0,
            davis_b: 0.0,
            drag_coeff: 0.0,
            braking_force: 500_000.0,
        }
    }

    fn make_route_cfg(length_m: f64) -> TrainConfig {
        use hs_trains::core::model::RouteElement;
        let elements = vec![RouteElement {
            track_id: "track_0".to_string(),
            net_element_id: "ne_0".to_string(),
            length_m,
        }];
        TrainConfig {
            id: "test".to_string(),
            train: make_minimal_train(),
            environment: Environment { gradient: 0.0, wind_speed: 0.0 },
            driver: DriverInput { power_ratio: 1.0, brake_ratio: 0.0 },
            timing: None,
            route: Route::new(elements),
        }
    }

    fn make_state(x: f64, speed: f64) -> SimulatedState {
        SimulatedState { position: Position { x, y: 0.0, z: 0.0 }, speed, acceleration: 0.0 }
    }

    #[test]
    fn test_advance_with_route_already_at_end_is_noop() {
        let cfg = make_route_cfg(100.0);
        let mut state = make_state(100.0, 10.0); // already at route end
        advance_with_route(&mut state, &cfg, 5.0);
        // State must not change.
        assert!((state.position.x - 100.0).abs() < 1e-9);
        assert!((state.speed - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_advance_with_route_clamps_and_halts_at_end() {
        let cfg = make_route_cfg(1.0); // very short route so any tick overshoots
        let mut state = make_state(0.0, 0.0);
        // A large dt will push the train past 1 m.
        advance_with_route(&mut state, &cfg, 1000.0);
        assert!((state.position.x - 1.0).abs() < 1e-9, "position should be clamped to route end");
        assert_eq!(state.speed, 0.0, "speed should be zeroed at route end");
        assert_eq!(state.acceleration, 0.0, "acceleration should be zeroed at route end");
    }

    #[test]
    fn test_advance_with_route_within_route_moves_forward() {
        let cfg = make_route_cfg(1_000_000.0); // very long so we never reach the end
        let mut state = make_state(0.0, 0.0);
        advance_with_route(&mut state, &cfg, 10.0);
        assert!(state.position.x > 0.0, "train should have moved");
        assert!(state.position.x < 1_000_000.0, "train should not have reached route end");
    }

    // --- make_step_row -------------------------------------------------------

    #[test]
    fn test_make_step_row_physics_path_populates_speed_and_track() {
        let cfg = make_route_cfg(1000.0);
        let state = make_state(50.0, 20.0); // 20 m/s = 72 km/h
        let track_geo = HashMap::new(); // no geometry — lon/lat will be None
        let row = make_step_row(&cfg, &state, 0.0, &track_geo);

        assert!((row.position_m.unwrap() - 50.0).abs() < 1e-9);
        assert!((row.speed_kmh.unwrap() - 72.0).abs() < 1e-6);
        assert!(row.accel_mss.is_some());
        assert_eq!(row.track_id.as_deref(), Some("track_0"));
        assert!((row.element_offset_m.unwrap() - 50.0).abs() < 1e-9);
        assert!(row.lon_deg.is_none());
        assert!(row.lat_deg.is_none());
        assert!(row.power_kw.is_some());
    }
}
