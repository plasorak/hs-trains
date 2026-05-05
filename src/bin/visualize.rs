/// Post-simulation GPUI visualizer.
///
/// Usage: visualize <output.parquet> [gpkg_path]
///
/// Reads a simulation Parquet file and the NWR GeoPackage, then opens a
/// GPUI window showing each train moving along its real geographic track
/// geometry (British National Grid coordinates projected to screen pixels).
///
/// The Parquet `track_id` column has the form "track_<ASSETID>" which maps
/// directly to the `ASSETID` column in the GeoPackage `NWR_GTCL` table.
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use gpui::{
    App, BorderStyle, Bounds, Context, FocusHandle, Hsla,
    IntoElement, PathBuilder, Pixels, Render, WeakEntity, Window, WindowBounds,
    WindowOptions, canvas, div, point, prelude::*, px, quad, size, transparent_black,
};
use gpui_platform::application;
use polars::prelude::*;
use rusqlite::Connection;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct TrainPosition {
    train_id: String,
    track_id: String,
    element_offset_m: f64,
}

#[derive(Clone)]
struct SimFrame {
    time_s: f64,
    trains: Vec<TrainPosition>,
}

#[derive(Clone, Copy, Debug)]
struct Bbox {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

struct VisData {
    geometries: HashMap<String, Vec<(f64, f64)>>,
    arc_lengths: HashMap<String, Vec<f64>>,
    route_polylines: Vec<Vec<(f64, f64)>>,
    frames: Vec<SimFrame>,
    bbox: Bbox,
    train_hues: HashMap<String, f32>,
}

// ---------------------------------------------------------------------------
// GeoPackage / WKB parsing
// ---------------------------------------------------------------------------

/// Decode a GeoPackage Binary blob into BNG (easting, northing) pairs.
///
/// GPB header layout:
///   [0-1]  'GP' magic
///   [2]    version
///   [3]    flags: bits 1-3 = envelope indicator (0=none,1=4d,2|3=6d,4=8d)
///   [4-7]  SRS id
///   [8+]   optional envelope (n doubles), then WKB geometry
fn parse_gpb_linestring(blob: &[u8]) -> Option<Vec<(f64, f64)>> {
    if blob.len() < 9 || &blob[0..2] != b"GP" {
        return None;
    }
    let env_doubles: usize = match (blob[3] >> 1) & 0x07 {
        0 => 0,
        1 => 4,
        2 | 3 => 6,
        4 => 8,
        _ => return None,
    };
    parse_wkb_linestring(blob.get(8 + env_doubles * 8..)?)
}

/// Parse a little-endian WKB LineString (geometry type 2).
fn parse_wkb_linestring(wkb: &[u8]) -> Option<Vec<(f64, f64)>> {
    if wkb.len() < 9 || wkb[0] != 1 {
        return None;
    }
    let geom_type = u32::from_le_bytes(wkb[1..5].try_into().ok()?);
    // Accept plain/Z/M/ZM variants: type % 1000 == 2.
    let coord_size = if geom_type >= 1000 { 24 } else { 16 };
    if geom_type % 1000 != 2 {
        return None;
    }
    let n = u32::from_le_bytes(wkb[5..9].try_into().ok()?) as usize;
    if wkb.len() < 9 + n * coord_size {
        return None;
    }
    let mut pts = Vec::with_capacity(n);
    let mut off = 9_usize;
    for _ in 0..n {
        let x = f64::from_le_bytes(wkb[off..off + 8].try_into().ok()?);
        let y = f64::from_le_bytes(wkb[off + 8..off + 16].try_into().ok()?);
        pts.push((x, y));
        off += coord_size;
    }
    Some(pts)
}

/// Cumulative arc-length array: entry `i` = distance from start to point `i`.
fn build_arc_lengths(pts: &[(f64, f64)]) -> Vec<f64> {
    let mut lengths = Vec::with_capacity(pts.len());
    lengths.push(0.0_f64);
    for w in pts.windows(2) {
        let dx = w[1].0 - w[0].0;
        let dy = w[1].1 - w[0].1;
        lengths.push(lengths.last().unwrap() + (dx * dx + dy * dy).sqrt());
    }
    lengths
}

/// Interpolate a BNG position along a polyline at `offset_m` metres from start.
fn interpolate(pts: &[(f64, f64)], lengths: &[f64], offset_m: f64) -> (f64, f64) {
    if pts.is_empty() {
        return (0.0, 0.0);
    }
    let t = offset_m.clamp(0.0, *lengths.last().unwrap());
    let idx = lengths.partition_point(|&l| l <= t).saturating_sub(1).min(pts.len() - 1);
    if idx + 1 >= pts.len() {
        return *pts.last().unwrap();
    }
    let seg_len = lengths[idx + 1] - lengths[idx];
    let frac = if seg_len > 0.0 { (t - lengths[idx]) / seg_len } else { 0.0 };
    (
        pts[idx].0 + frac * (pts[idx + 1].0 - pts[idx].0),
        pts[idx].1 + frac * (pts[idx + 1].1 - pts[idx].1),
    )
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

fn load_frames(path: &PathBuf) -> (Vec<SimFrame>, Vec<String>) {
    let file = std::fs::File::open(path).expect("cannot open parquet");
    let df = ParquetReader::new(file)
        .finish()
        .expect("cannot read parquet");
    let df = df.sort(["time_s"], SortMultipleOptions::default()).expect("cannot sort parquet");

    let times = df.column("time_s").unwrap().f64().unwrap();
    let train_ids = df.column("train_id").unwrap().str().unwrap();
    let track_ids = df.column("track_id").unwrap().str().unwrap();
    let offsets = df.column("element_offset_m").unwrap().f64().unwrap();

    let n = df.height();
    let mut frames: Vec<SimFrame> = Vec::new();
    let mut unique_tracks: Vec<String> = Vec::new();
    let mut seen_tracks = std::collections::HashSet::new();
    let mut i = 0_usize;

    while i < n {
        let Some(t) = times.get(i) else { i += 1; continue };
        let mut trains = Vec::new();
        while i < n && times.get(i).map_or(false, |tt| (tt - t).abs() < 1e-6) {
            if let (Some(tid), Some(tkid), Some(off)) =
                (train_ids.get(i), track_ids.get(i), offsets.get(i))
            {
                let track_id = tkid.to_string();
                if seen_tracks.insert(track_id.clone()) {
                    unique_tracks.push(track_id.clone());
                }
                trains.push(TrainPosition {
                    train_id: tid.to_string(),
                    track_id,
                    element_offset_m: off,
                });
            }
            i += 1;
        }
        if !trains.is_empty() {
            frames.push(SimFrame { time_s: t, trains });
        }
    }
    (frames, unique_tracks)
}

fn load_track_geometries(
    track_ids: &[String],
    gpkg_path: &PathBuf,
) -> (HashMap<String, Vec<(f64, f64)>>, HashMap<String, Vec<f64>>) {
    let conn = Connection::open(gpkg_path).expect("cannot open GeoPackage");
    let mut geoms: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
    let mut alens: HashMap<String, Vec<f64>> = HashMap::new();
    let mut not_found = 0_usize;

    for track_id in track_ids {
        let asset_id = track_id.strip_prefix("track_").unwrap_or(track_id.as_str());
        match conn.query_row(
            "SELECT geom FROM NWR_GTCL WHERE ASSETID = ?1 LIMIT 1",
            rusqlite::params![asset_id],
            |row| row.get::<_, Vec<u8>>(0),
        ) {
            Ok(blob) => {
                if let Some(pts) = parse_gpb_linestring(&blob) {
                    alens.insert(track_id.clone(), build_arc_lengths(&pts));
                    geoms.insert(track_id.clone(), pts);
                }
            }
            Err(_) => not_found += 1,
        }
    }
    if not_found > 0 {
        eprintln!("Warning: {not_found} track(s) not found in GeoPackage");
    }
    (geoms, alens)
}

fn compute_bbox(polys: &[Vec<(f64, f64)>]) -> Bbox {
    let (mut min_x, mut min_y) = (f64::MAX, f64::MAX);
    let (mut max_x, mut max_y) = (f64::MIN, f64::MIN);
    for poly in polys {
        for &(x, y) in poly {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    let px = (max_x - min_x) * 0.05;
    let py = (max_y - min_y) * 0.05;
    Bbox { min_x: min_x - px, min_y: min_y - py, max_x: max_x + px, max_y: max_y + py }
}

// ---------------------------------------------------------------------------
// Coordinate projection: BNG → screen pixels
// ---------------------------------------------------------------------------

/// Maps a BNG (easting, northing) coordinate to a screen point inside
/// `bounds`, maintaining aspect ratio and adding a uniform margin.
fn bng_to_screen(bx: f64, by: f64, bbox: Bbox, bounds: Bounds<Pixels>) -> (f32, f32) {
    let margin = 20.0_f32;
    let w = bounds.size.width.as_f32() - margin * 2.0;
    let h = bounds.size.height.as_f32() - margin * 2.0;
    let bng_w = (bbox.max_x - bbox.min_x) as f32;
    let bng_h = (bbox.max_y - bbox.min_y) as f32;
    let scale = (w / bng_w).min(h / bng_h);
    // Centre the map within the available area.
    let map_w = bng_w * scale;
    let map_h = bng_h * scale;
    let off_x = (w - map_w) * 0.5 + margin;
    let off_y = (h - map_h) * 0.5 + margin;
    let sx = off_x + (bx - bbox.min_x) as f32 * scale;
    let sy = off_y + (bbox.max_y - by) as f32 * scale; // flip Y
    (
        bounds.origin.x.as_f32() + sx,
        bounds.origin.y.as_f32() + sy,
    )
}

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------

/// Draw a line segment as a thin filled parallelogram via `PathBuilder`.
fn paint_segment(
    x0: f32, y0: f32, x1: f32, y1: f32,
    thickness: f32, color: Hsla,
    window: &mut Window,
) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 0.5 {
        return;
    }
    let nx = -dy / len * thickness * 0.5;
    let ny = dx / len * thickness * 0.5;
    let mut builder = PathBuilder::fill();
    builder.move_to(point(px(x0 + nx), px(y0 + ny)));
    builder.line_to(point(px(x0 - nx), px(y0 - ny)));
    builder.line_to(point(px(x1 - nx), px(y1 - ny)));
    builder.line_to(point(px(x1 + nx), px(y1 + ny)));
    if let Ok(path) = builder.build() {
        window.paint_path(path, color);
    }
}

/// Draw a filled circle via `quad` with full corner rounding.
fn paint_circle(cx: f32, cy: f32, r: f32, color: Hsla, window: &mut Window) {
    window.paint_quad(quad(
        Bounds {
            origin: point(px(cx - r), px(cy - r)),
            size: size(px(r * 2.0), px(r * 2.0)),
        },
        px(r),           // corner_radii → Corners<Pixels>
        color,           // background
        0.0_f32,         // border_widths → Edges<Pixels>
        transparent_black(),
        BorderStyle::default(),
    ));
}

// ---------------------------------------------------------------------------
// GPUI view
// ---------------------------------------------------------------------------

struct TrainViz {
    data: Arc<VisData>,
    frame_idx: usize,
    playing: bool,
    focus_handle: FocusHandle,
}

impl TrainViz {
    fn new(data: Arc<VisData>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus_handle = cx.focus_handle();

        // Animation timer: advance one frame every 100 ms.
        cx.spawn_in(window, async move |weak: WeakEntity<Self>, async_cx: &mut gpui::AsyncWindowContext| {
            loop {
                async_cx.background_executor().timer(Duration::from_millis(100)).await;
                weak.update_in(async_cx, |view, _window, cx| {
                    if view.playing && !view.data.frames.is_empty() {
                        view.frame_idx = (view.frame_idx + 1) % view.data.frames.len();
                        cx.notify();
                    }
                })
                .ok();
            }
        })
        .detach();

        TrainViz { data, frame_idx: 0, playing: true, focus_handle }
    }
}

impl Render for TrainViz {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let data = self.data.clone();
        let frame_idx = self.frame_idx;
        let time_s = data.frames.get(frame_idx).map(|f| f.time_s).unwrap_or(0.0);
        let time_label = format!("t = {:.0} s  ({:.1} min)", time_s, time_s / 60.0);
        let n_frames = data.frames.len();
        let playing = self.playing;

        div()
            .relative()
            .size_full()
            .bg(gpui::black())
            .track_focus(&self.focus_handle)
            .on_key_down(cx.listener(|view, event: &gpui::KeyDownEvent, _w, cx| {
                match event.keystroke.key.as_str() {
                    " " => { view.playing = !view.playing; cx.notify(); }
                    "right" => {
                        let max = view.data.frames.len().saturating_sub(1);
                        view.frame_idx = (view.frame_idx + 10).min(max);
                        cx.notify();
                    }
                    "left" => {
                        view.frame_idx = view.frame_idx.saturating_sub(10);
                        cx.notify();
                    }
                    _ => {}
                }
            }))
            .child(
                canvas(
                    |_bounds, _window, _cx| {},
                    move |bounds, (), window, _cx| {
                        let track_color = Hsla { h: 0.0, s: 0.0, l: 0.28, a: 1.0 };

                        // Draw all route track polylines.
                        for poly in &data.route_polylines {
                            for w in poly.windows(2) {
                                let (x0, y0) = bng_to_screen(w[0].0, w[0].1, data.bbox, bounds);
                                let (x1, y1) = bng_to_screen(w[1].0, w[1].1, data.bbox, bounds);
                                paint_segment(x0, y0, x1, y1, 2.0, track_color, window);
                            }
                        }

                        // Draw trains.
                        if let Some(frame) = data.frames.get(frame_idx) {
                            for tp in &frame.trains {
                                let Some(pts) = data.geometries.get(&tp.track_id) else { continue };
                                let Some(al) = data.arc_lengths.get(&tp.track_id) else { continue };
                                let (bx, by) = interpolate(pts, al, tp.element_offset_m);
                                let (sx, sy) = bng_to_screen(bx, by, data.bbox, bounds);
                                let hue = *data.train_hues.get(&tp.train_id).unwrap_or(&0.0);
                                let color = Hsla { h: hue, s: 0.9, l: 0.6, a: 1.0 };
                                paint_circle(sx, sy, 7.0, gpui::white(), window);
                                paint_circle(sx, sy, 5.5, color, window);
                            }
                        }
                    },
                )
                .size_full(),
            )
            // Time / status overlay.
            .child(
                div()
                    .absolute()
                    .top(px(10.0))
                    .left(px(14.0))
                    .text_color(gpui::white())
                    .text_size(px(14.0))
                    .child(time_label),
            )
            .child(
                div()
                    .absolute()
                    .top(px(30.0))
                    .left(px(14.0))
                    .text_color(Hsla { h: 0.0, s: 0.0, l: 0.55, a: 1.0 })
                    .text_size(px(11.0))
                    .child(format!(
                        "frame {}/{} | {} | Space pause · ← → skip",
                        frame_idx + 1,
                        n_frames,
                        if playing { "▶ playing" } else { "⏸ paused" },
                    )),
            )
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: visualize <output.parquet> [gpkg_path]");
        std::process::exit(1);
    }
    let parquet_path = PathBuf::from(&args[1]);
    let gpkg_path = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("assets/NWR_GTCL20260309.gpkg"));

    eprintln!("Loading simulation frames from {} …", parquet_path.display());
    let (frames, unique_tracks) = load_frames(&parquet_path);
    eprintln!("  {} frames, {} unique tracks", frames.len(), unique_tracks.len());

    eprintln!("Loading track geometry from {} …", gpkg_path.display());
    let (geometries, arc_lengths_map) = load_track_geometries(&unique_tracks, &gpkg_path);
    eprintln!("  {} tracks resolved", geometries.len());

    let route_polylines: Vec<Vec<(f64, f64)>> = unique_tracks
        .iter()
        .filter_map(|id| geometries.get(id).cloned())
        .collect();

    let bbox = compute_bbox(&route_polylines);
    eprintln!(
        "  BNG bbox: ({:.0}, {:.0}) – ({:.0}, {:.0})",
        bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y,
    );

    let train_ids: Vec<String> = {
        let mut seen = std::collections::HashSet::new();
        frames
            .iter()
            .flat_map(|f| f.trains.iter().map(|tp| tp.train_id.clone()))
            .filter(|id| seen.insert(id.clone()))
            .collect()
    };
    let train_hues: HashMap<String, f32> = train_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.clone(), i as f32 / train_ids.len().max(1) as f32))
        .collect();

    let data = Arc::new(VisData {
        geometries,
        arc_lengths: arc_lengths_map,
        route_polylines,
        frames,
        bbox,
        train_hues,
    });

    application().run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(1200.0), px(800.0)), cx);
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            titlebar: Some(gpui::TitlebarOptions {
                title: Some("Rusty Trains Visualizer".into()),
                appears_transparent: false,
                traffic_light_position: None,
            }),
            focus: true,
            show: true,
            ..WindowOptions::default()
        };
        let data = data.clone();
        cx.open_window(options, move |window, cx| {
            let entity = cx.new(|cx| TrainViz::new(data, window, cx));
            let fh = entity.read(cx).focus_handle.clone();
            window.focus(&fh, cx);
            entity
        })
        .unwrap();
        cx.activate(true);
    });
}
