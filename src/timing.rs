use polars::prelude::*;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use crate::model::{ObservedState, Position, TrainState};

/// Read berth timing data from a Parquet file and return a time-ordered list
/// of [`TrainState::Observed`] values for the given `train_id`.
///
/// # Parquet file format
///
/// Each row represents a single berth step event — the moment a train's
/// description stepped into the named berth on the signalling panel.
///
/// | Column         | Parquet type | Description                                           |
/// |----------------|--------------|-------------------------------------------------------|
/// | `train_id`     | UTF8         | Train identifier / headcode, e.g. `"1A23"`            |
/// | `berth_id`     | UTF8         | Berth name, matching `BerthDescription::name`         |
/// | `timestamp_ms` | INT64        | Unix epoch timestamp in **milliseconds**              |
/// | `position_m`   | DOUBLE       | Along-track distance from route origin, in **metres** |
///
/// Rows do not need to be pre-sorted; the loader sorts by `timestamp_ms`.
/// Rows with null values in `timestamp_ms` or `position_m` are skipped.
///
/// # Limitations
///
/// Speed and acceleration cannot be derived from berth timing data alone.
/// All returned states are [`TrainState::Observed`]; call `.speed()` or
/// `.acceleration()` on them to get `None`.
pub fn load_timing_from_parquet(
    path: &Path,
    train_id: &str,
) -> PolarsResult<Vec<TrainState>> {
    let file = File::open(path)
        .map_err(|e| PolarsError::IO { error: Arc::new(e), msg: None })?;
    let df = ParquetReader::new(file).finish()?;

    // Filter rows matching the requested train_id.
    let train_id_col = df.column("train_id")?.str()?;
    let mask: BooleanChunked = train_id_col
        .into_iter()
        .map(|v| v == Some(train_id))
        .collect();
    let df = df.filter(&mask)?;

    // Sort chronologically.
    let df = df.sort(["timestamp_ms"], SortMultipleOptions::default())?;

    let timestamps = df.column("timestamp_ms")?.i64()?;
    let positions  = df.column("position_m")?.f64()?;

    let mut result = Vec::with_capacity(df.height());
    for i in 0..df.height() {
        let Some(ts)  = timestamps.get(i) else { continue };
        let Some(pos) = positions.get(i)  else { continue };
        result.push(TrainState::Observed(ObservedState {
            position: Position { x: pos, y: 0.0, z: 0.0 },
            timestamp_ms: ts,
        }));
    }

    Ok(result)
}

/// A normalized, interpolatable position trace for a single timing train.
///
/// Timestamps are converted to seconds elapsed from the first record, so
/// the trace always starts at `t = 0` regardless of the original Unix epoch.
pub struct TimingTrace {
    times_s: Vec<f64>,
    positions_m: Vec<f64>,
}

impl TimingTrace {
    /// Load and normalize timing data from a Parquet file for `train_id`.
    pub fn load(path: &Path, train_id: &str) -> PolarsResult<Self> {
        let states = load_timing_from_parquet(path, train_id)?;

        let mut times_ms: Vec<i64> = Vec::with_capacity(states.len());
        let mut positions_m: Vec<f64> = Vec::with_capacity(states.len());
        for s in &states {
            if let TrainState::Observed(o) = s {
                times_ms.push(o.timestamp_ms);
                positions_m.push(o.position.x);
            }
        }

        if times_ms.is_empty() {
            return Ok(Self { times_s: vec![], positions_m: vec![] });
        }

        let t0 = times_ms[0];
        let times_s = times_ms.iter().map(|&t| (t - t0) as f64 / 1000.0).collect();
        Ok(Self { times_s, positions_m })
    }

    /// Linearly interpolate position at `t` seconds from trace start.
    /// Returns `None` if `t` is outside the trace time range or the trace is empty.
    pub fn position_at(&self, t: f64) -> Option<f64> {
        if self.times_s.is_empty() { return None; }
        let t_last = *self.times_s.last().unwrap();
        if t < self.times_s[0] || t > t_last { return None; }

        let idx = self.times_s.partition_point(|&ts| ts <= t);
        if idx == 0 { return Some(self.positions_m[0]); }
        if idx >= self.times_s.len() { return Some(*self.positions_m.last().unwrap()); }

        let t0 = self.times_s[idx - 1];
        let t1 = self.times_s[idx];
        let p0 = self.positions_m[idx - 1];
        let p1 = self.positions_m[idx];
        let frac = (t - t0) / (t1 - t0);
        Some(p0 + frac * (p1 - p0))
    }
}
