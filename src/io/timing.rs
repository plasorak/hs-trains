use polars::prelude::*;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

/// An interpolatable position trace for a single timing train, indexed by
/// sectional running time (seconds elapsed from the train's journey start).
pub struct TimingTrace {
    times_s: Vec<f64>,
    positions_m: Vec<f64>,
}

impl TimingTrace {
    /// Load timing data from a Parquet file for `train_id`.
    ///
    /// # Parquet file format
    ///
    /// Each row represents a single berth step event — the moment a train's
    /// description stepped into the named berth on the signalling panel.
    ///
    /// | Column      | Parquet type | Description                                           |
    /// |-------------|--------------|-------------------------------------------------------|
    /// | `train_id`  | UTF8         | Train identifier / headcode, e.g. `"1A23"`            |
    /// | `berth_id`  | UTF8         | Berth name, matching `BerthDescription::name`         |
    /// | `elapsed_s` | DOUBLE       | Travel time for **this berth** in **seconds**  |
    /// | `length_m`  | DOUBLE       | Length of **this berth** in **metres**         |
    ///
    /// Rows must be in berth order (sequential along the route).
    /// The loader accumulates both columns to build running-time and position axes.
    /// Rows with null values in `elapsed_s` or `length_m` are skipped.
    ///
    /// # Limitations
    ///
    /// Speed and acceleration cannot be derived from berth timing data alone.
    pub fn load(path: &Path, train_id: &str) -> PolarsResult<Self> {
        let file = File::open(path).map_err(|e| PolarsError::IO {
            error: Arc::new(e),
            msg: None,
        })?;
        let df = ParquetReader::new(file).finish()?;

        // Filter rows matching the requested train_id.
        let train_id_col = df.column("train_id")?.str()?;
        let mask: BooleanChunked = train_id_col
            .into_iter()
            .map(|v| v == Some(train_id))
            .collect();
        let df = df.filter(&mask)?;

        let elapsed = df.column("elapsed_s")?.f64()?;
        let positions = df.column("length_m")?.f64()?;

        let mut times_s: Vec<f64> = Vec::with_capacity(df.height());
        let mut positions_m: Vec<f64> = Vec::with_capacity(df.height());
        let mut running_time = 0.0_f64;
        let mut running_pos = 0.0_f64;
        for i in 0..df.height() {
            let Some(berth_time) = elapsed.get(i) else {
                continue;
            };
            let Some(berth_len) = positions.get(i) else {
                continue;
            };
            times_s.push(running_time);
            positions_m.push(running_pos);
            running_time += berth_time;
            running_pos += berth_len;
        }

        Ok(Self {
            times_s,
            positions_m,
        })
    }

    /// Linearly interpolate position at `t` seconds from trace start.
    /// Returns `None` if `t` is outside the trace time range or the trace is empty.
    pub fn position_at(&self, t: f64) -> Option<f64> {
        if self.times_s.is_empty() {
            return None;
        }
        let t_last = *self.times_s.last().unwrap();
        if t < self.times_s[0] || t > t_last {
            return None;
        }

        let idx = self.times_s.partition_point(|&ts| ts <= t);
        if idx == 0 {
            return Some(self.positions_m[0]);
        }
        if idx >= self.times_s.len() {
            return Some(*self.positions_m.last().unwrap());
        }

        let t0 = self.times_s[idx - 1];
        let t1 = self.times_s[idx];
        let p0 = self.positions_m[idx - 1];
        let p1 = self.positions_m[idx];
        let frac = (t - t0) / (t1 - t0);
        Some(p0 + frac * (p1 - p0))
    }
}
