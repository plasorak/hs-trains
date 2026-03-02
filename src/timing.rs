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
