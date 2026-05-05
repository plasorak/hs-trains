/// Post-simulation GPUI visualizer.
///
/// Usage: visualize <output.parquet> [gpkg_path]
///
/// Reads a simulation Parquet file and the NWR GeoPackage, then opens a
/// GPUI window showing each train moving along its real geographic track
/// geometry (British National Grid coordinates projected to screen pixels).
///
/// OSM raster tiles are fetched at startup (zoom 9) and cached in the OS
/// temp directory so subsequent runs are instant.
use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use gpui::{
    App, Bounds, Context, Corners, FocusHandle, Hsla,
    IntoElement, PathBuilder, Pixels, Render, RenderImage, WeakEntity, Window, WindowBounds,
    WindowOptions, canvas, div, point, prelude::*, px, size,
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

/// Tile grid laid out in uniform screen-space cells.
///
/// Rather than computing each tile's BNG bbox independently (which causes gaps
/// because constant-longitude lines have slightly varying easting in TM), we
/// record the overall grid corners once and divide the projected rectangle
/// evenly at render time. Adjacent tiles then share exact screen boundaries.
struct TileGrid {
    /// Row-major storage: `rows[row][col]`, where row 0 is the northernmost.
    rows: Vec<Vec<Option<Arc<RenderImage>>>>,
    n_rows: u32,
    n_cols: u32,
    /// BNG (easting, northing) of the NW corner of the entire grid.
    nw: (f64, f64),
    /// BNG (easting, northing) of the SE corner of the entire grid.
    se: (f64, f64),
}

struct VisData {
    geometries: HashMap<String, Vec<(f64, f64)>>,
    arc_lengths: HashMap<String, Vec<f64>>,
    route_polylines: Vec<Vec<(f64, f64)>>,
    frames: Vec<SimFrame>,
    bbox: Bbox,
    train_hues: HashMap<String, f32>,
    tile_grid: TileGrid,
    start_time: f64,
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
    // Accept XY / XYZ / XYM / XYZM variants; type % 1000 == 2 means LineString.
    if geom_type % 1000 != 2 {
        return None;
    }
    // Bytes per coordinate point: XY=16, XYZ or XYM=24, XYZM=32.
    let coord_size: usize = match geom_type / 1000 {
        0 => 16,
        1 | 2 => 24,
        3 => 32,
        _ => return None,
    };
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
// Coordinate transformations: BNG ↔ WGS84
//
// Uses OS OSGB36 Transverse Mercator projection (Airy 1830 ellipsoid) plus a
// simplified 7-parameter Helmert transform between OSGB36 and WGS84.
// Accuracy is ~5 m, more than enough for raster tile placement.
// ---------------------------------------------------------------------------

const PI: f64 = std::f64::consts::PI;
const AIRY_A: f64 = 6_377_563.396;
const AIRY_B: f64 = 6_356_256.909;
const GRS80_A: f64 = 6_378_137.0;
const GRS80_B: f64 = 6_356_752.314_2;
const BNG_F0: f64 = 0.999_601_271_7;
const BNG_LAT0: f64 = 49.0 * PI / 180.0;
const BNG_LON0: f64 = -2.0 * PI / 180.0;
const BNG_E0: f64 = 400_000.0;
const BNG_N0: f64 = -100_000.0;

/// Convert BNG (easting, northing) → WGS84 (latitude°, longitude°).
fn bng_to_wgs84(e: f64, n: f64) -> (f64, f64) {
    let e2 = 1.0 - (AIRY_B / AIRY_A).powi(2);
    let nv = (AIRY_A - AIRY_B) / (AIRY_A + AIRY_B);
    let (nv2, nv3) = (nv * nv, nv * nv * nv);

    // Iterate to find latitude from northing (OS method).
    let mut lat = (n - BNG_N0) / (AIRY_A * BNG_F0) + BNG_LAT0;
    loop {
        let m = AIRY_B * BNG_F0 * (
            (1.0 + nv + 1.25 * nv2 + 1.25 * nv3) * (lat - BNG_LAT0)
            - (3.0 * nv + 3.0 * nv2 + 2.625 * nv3) * (lat - BNG_LAT0).sin() * (lat + BNG_LAT0).cos()
            + (1.875 * nv2 + 1.875 * nv3) * (2.0 * (lat - BNG_LAT0)).sin() * (2.0 * (lat + BNG_LAT0)).cos()
            - (35.0 / 24.0 * nv3) * (3.0 * (lat - BNG_LAT0)).sin() * (3.0 * (lat + BNG_LAT0)).cos()
        );
        let prev = lat;
        lat += (n - BNG_N0 - m) / (AIRY_A * BNG_F0);
        if (lat - prev).abs() < 1e-11 { break; }
    }

    let nu = AIRY_A * BNG_F0 / (1.0 - e2 * lat.sin().powi(2)).sqrt();
    let rho = AIRY_A * BNG_F0 * (1.0 - e2) / (1.0 - e2 * lat.sin().powi(2)).powf(1.5);
    let eta2 = nu / rho - 1.0;
    let tan_l = lat.tan();
    let de = e - BNG_E0;

    let vii = tan_l / (2.0 * rho * nu);
    let viii = tan_l / (24.0 * rho * nu.powi(3)) * (5.0 + 3.0 * tan_l.powi(2) + eta2 - 9.0 * tan_l.powi(2) * eta2);
    let ix = tan_l / (720.0 * rho * nu.powi(5)) * (61.0 + 90.0 * tan_l.powi(2) + 45.0 * tan_l.powi(4));
    let x_c = 1.0 / (lat.cos() * nu);
    let xi = 1.0 / (lat.cos() * 6.0 * nu.powi(3)) * (nu / rho + 2.0 * tan_l.powi(2));
    let xii = 1.0 / (lat.cos() * 120.0 * nu.powi(5)) * (5.0 + 28.0 * tan_l.powi(2) + 24.0 * tan_l.powi(4));
    let xiia = 1.0 / (lat.cos() * 5040.0 * nu.powi(7)) * (61.0 + 662.0 * tan_l.powi(2) + 1320.0 * tan_l.powi(4) + 720.0 * tan_l.powi(6));

    let lat_o = lat - vii * de.powi(2) + viii * de.powi(4) - ix * de.powi(6);
    let lon_o = BNG_LON0 + x_c * de - xi * de.powi(3) + xii * de.powi(5) - xiia * de.powi(7);

    // OSGB36 ellipsoidal → Cartesian (Airy 1830).
    let nu_h = AIRY_A / (1.0 - e2 * lat_o.sin().powi(2)).sqrt();
    let (xo, yo, zo) = (
        nu_h * lat_o.cos() * lon_o.cos(),
        nu_h * lat_o.cos() * lon_o.sin(),
        nu_h * (1.0 - e2) * lat_o.sin(),
    );

    // Simplified Helmert OSGB36 → WGS84.
    let (tx, ty, tz) = (446.448, -125.157, 542.060);
    let arcsec = PI / (180.0 * 3600.0);
    let (rx, ry, rz) = (0.1502 * arcsec, 0.2470 * arcsec, 0.8421 * arcsec);
    let s = 1.0 - 20.4894e-6;
    let (xw, yw, zw) = (
        tx + s * (xo - rz * yo + ry * zo),
        ty + s * (rz * xo + yo - rx * zo),
        tz + s * (-ry * xo + rx * yo + zo),
    );

    // WGS84 Cartesian → lat/lon (iterate on ellipsoid).
    let e2w = 1.0 - (GRS80_B / GRS80_A).powi(2);
    let p = (xw * xw + yw * yw).sqrt();
    let mut lat_w = (zw / (p * (1.0 - e2w))).atan();
    loop {
        let nu_w = GRS80_A / (1.0 - e2w * lat_w.sin().powi(2)).sqrt();
        let prev = lat_w;
        lat_w = ((zw + e2w * nu_w * lat_w.sin()) / p).atan();
        if (lat_w - prev).abs() < 1e-11 { break; }
    }
    (lat_w.to_degrees(), yw.atan2(xw).to_degrees())
}

/// Convert WGS84 (latitude°, longitude°) → BNG (easting, northing).
fn wgs84_to_bng(lat_deg: f64, lon_deg: f64) -> (f64, f64) {
    let (lat, lon) = (lat_deg.to_radians(), lon_deg.to_radians());

    // WGS84 lat/lon → Cartesian (GRS80).
    let e2w = 1.0 - (GRS80_B / GRS80_A).powi(2);
    let nu_w = GRS80_A / (1.0 - e2w * lat.sin().powi(2)).sqrt();
    let (xw, yw, zw) = (
        nu_w * lat.cos() * lon.cos(),
        nu_w * lat.cos() * lon.sin(),
        nu_w * (1.0 - e2w) * lat.sin(),
    );

    // Reverse Helmert WGS84 → OSGB36.
    let (tx, ty, tz) = (-446.448, 125.157, -542.060);
    let arcsec = PI / (180.0 * 3600.0);
    let (rx, ry, rz) = (-0.1502 * arcsec, -0.2470 * arcsec, -0.8421 * arcsec);
    let s = 1.0 + 20.4894e-6;
    let (xo, yo, zo) = (
        tx + s * (xw - rz * yw + ry * zw),
        ty + s * (rz * xw + yw - rx * zw),
        tz + s * (-ry * xw + rx * yw + zw),
    );

    // OSGB36 Cartesian → lat/lon (Airy 1830, iterate).
    let e2 = 1.0 - (AIRY_B / AIRY_A).powi(2);
    let p = (xo * xo + yo * yo).sqrt();
    let mut lat_o = (zo / (p * (1.0 - e2))).atan();
    loop {
        let nu = AIRY_A / (1.0 - e2 * lat_o.sin().powi(2)).sqrt();
        let prev = lat_o;
        lat_o = ((zo + e2 * nu * lat_o.sin()) / p).atan();
        if (lat_o - prev).abs() < 1e-11 { break; }
    }
    let lon_o = yo.atan2(xo);

    // OSGB36 lat/lon → BNG (forward TM projection).
    let nv = (AIRY_A - AIRY_B) / (AIRY_A + AIRY_B);
    let (nv2, nv3) = (nv * nv, nv * nv * nv);
    let (sl, cl, tl) = (lat_o.sin(), lat_o.cos(), lat_o.tan());
    let nu_b = AIRY_A * BNG_F0 / (1.0 - e2 * sl.powi(2)).sqrt();
    let rho = AIRY_A * BNG_F0 * (1.0 - e2) / (1.0 - e2 * sl.powi(2)).powf(1.5);
    let eta2 = nu_b / rho - 1.0;

    let m = AIRY_B * BNG_F0 * (
        (1.0 + nv + 1.25 * nv2 + 1.25 * nv3) * (lat_o - BNG_LAT0)
        - (3.0 * nv + 3.0 * nv2 + 2.625 * nv3) * (lat_o - BNG_LAT0).sin() * (lat_o + BNG_LAT0).cos()
        + (1.875 * nv2 + 1.875 * nv3) * (2.0 * (lat_o - BNG_LAT0)).sin() * (2.0 * (lat_o + BNG_LAT0)).cos()
        - (35.0 / 24.0 * nv3) * (3.0 * (lat_o - BNG_LAT0)).sin() * (3.0 * (lat_o + BNG_LAT0)).cos()
    );
    let dl = lon_o - BNG_LON0;
    let north = BNG_N0 + m + (nu_b / 2.0) * sl * cl * dl.powi(2)
        + (nu_b / 24.0) * sl * cl.powi(3) * (5.0 - tl.powi(2) + 9.0 * eta2) * dl.powi(4)
        + (nu_b / 720.0) * sl * cl.powi(5) * (61.0 - 58.0 * tl.powi(2) + tl.powi(4)) * dl.powi(6);
    let east = BNG_E0 + nu_b * cl * dl
        + (nu_b / 6.0) * cl.powi(3) * (nu_b / rho - tl.powi(2)) * dl.powi(3)
        + (nu_b / 120.0) * cl.powi(5) * (5.0 - 18.0 * tl.powi(2) + tl.powi(4) + 14.0 * eta2 - 58.0 * tl.powi(2) * eta2) * dl.powi(5);
    (east, north)
}

// ---------------------------------------------------------------------------
// OSM tile helpers (Web Mercator / EPSG:3857)
// ---------------------------------------------------------------------------

const TILE_ZOOM: u32 = 9;

/// WGS84 lat/lon → OSM tile (x, y) at the given zoom level.
fn lat_lon_to_tile(lat: f64, lon: f64, zoom: u32) -> (u32, u32) {
    let n = 2_f64.powi(zoom as i32);
    let x = ((lon + 180.0) / 360.0 * n).floor() as u32;
    let lat_r = lat.to_radians();
    let y = ((1.0 - (lat_r.tan() + 1.0 / lat_r.cos()).ln() / PI) / 2.0 * n).floor() as u32;
    (x, y)
}

/// WGS84 lat/lon of the NW corner of tile (tx, ty) at the given zoom level.
fn tile_nw_corner(tx: u32, ty: u32, zoom: u32) -> (f64, f64) {
    let n = 2_f64.powi(zoom as i32);
    let lon = tx as f64 / n * 360.0 - 180.0;
    let lat = (PI * (1.0 - 2.0 * ty as f64 / n)).sinh().atan().to_degrees();
    (lat, lon)
}

/// Fetch one PNG tile as raw bytes, using a disk cache in the OS temp dir.
fn fetch_tile_bytes(tx: u32, ty: u32, zoom: u32) -> Option<Vec<u8>> {
    let cache = std::env::temp_dir()
        .join("rusty-trains-tiles")
        .join(zoom.to_string());
    std::fs::create_dir_all(&cache).ok();
    let path = cache.join(format!("{tx}_{ty}.png"));

    if let Ok(bytes) = std::fs::read(&path) {
        return Some(bytes);
    }

    let url = format!("https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png");
    let resp = ureq::get(&url)
        .set("User-Agent", "rusty-trains-visualizer/0.1 (educational)")
        .call()
        .ok()?;
    if resp.status() != 200 {
        return None;
    }
    let mut buf = Vec::new();
    resp.into_reader().read_to_end(&mut buf).ok()?;
    std::fs::write(&path, &buf).ok();
    Some(buf)
}

/// Fetch all OSM tiles covering `bbox` (BNG) at `TILE_ZOOM` and return a
/// `TileGrid` with a uniform screen-space layout.
///
/// Each tile's screen position is derived by dividing the overall grid's
/// projected BNG rectangle evenly. This guarantees zero gaps between adjacent
/// tiles regardless of projection distortion.
fn fetch_tiles(bbox: Bbox) -> TileGrid {
    // Expand the bbox slightly so tiles cover the full visible map.
    let margin = (bbox.max_x - bbox.min_x) * 0.05;
    let (lat_sw, lon_sw) = bng_to_wgs84(bbox.min_x - margin, bbox.min_y - margin);
    let (lat_ne, lon_ne) = bng_to_wgs84(bbox.max_x + margin, bbox.max_y + margin);

    // Tile y increases southward (flip vs latitude).
    let (tx_min, ty_max) = lat_lon_to_tile(lat_sw, lon_sw, TILE_ZOOM);
    let (tx_max, ty_min) = lat_lon_to_tile(lat_ne, lon_ne, TILE_ZOOM);
    let n_cols = tx_max - tx_min + 1;
    let n_rows = ty_max - ty_min + 1;

    let total = n_cols * n_rows;
    eprintln!("  Fetching {total} map tiles (zoom {TILE_ZOOM}, cached in temp dir) …");

    // Pre-allocate the row-major grid with None cells.
    let mut rows: Vec<Vec<Option<Arc<RenderImage>>>> =
        (0..n_rows).map(|_| vec![None; n_cols as usize]).collect();

    for ty in ty_min..=ty_max {
        for tx in tx_min..=tx_max {
            let Some(bytes) = fetch_tile_bytes(tx, ty, TILE_ZOOM) else { continue };
            let Ok(img) = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png) else { continue };
            let frame = image::Frame::new(img.into_rgba8());
            let render_img = Arc::new(RenderImage::new(vec![frame]));
            let row = (ty - ty_min) as usize;
            let col = (tx - tx_min) as usize;
            rows[row][col] = Some(render_img);
        }
    }

    // BNG coords of the overall grid corners: NW = top-left tile's NW corner,
    // SE = bottom-right tile's SE corner.  Using a single consistent pair of
    // reference points means tiles are placed in a perfectly flush uniform grid.
    let (lat_grid_nw, lon_grid_nw) = tile_nw_corner(tx_min, ty_min, TILE_ZOOM);
    let (lat_grid_se, lon_grid_se) = tile_nw_corner(tx_max + 1, ty_max + 1, TILE_ZOOM);
    let nw = wgs84_to_bng(lat_grid_nw, lon_grid_nw);
    let se = wgs84_to_bng(lat_grid_se, lon_grid_se);

    let loaded = rows.iter().flatten().filter(|c| c.is_some()).count();
    eprintln!("  {loaded}/{total} tiles ready");
    TileGrid { rows, n_rows, n_cols, nw, se }
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
            // Discard coordinates outside valid UK BNG bounds — guards against
            // garbage values produced by malformed WKB (e.g. wrong coord_size).
            if !(0.0..=700_000.0).contains(&x) || !(0.0..=1_300_000.0).contains(&y) {
                continue;
            }
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
    let map_w = bng_w * scale;
    let map_h = bng_h * scale;
    let off_x = (w - map_w) * 0.5 + margin;
    let off_y = (h - map_h) * 0.5 + margin;
    let sx = off_x + (bx - bbox.min_x) as f32 * scale;
    let sy = off_y + (bbox.max_y - by) as f32 * scale; // flip Y
    (bounds.origin.x.as_f32() + sx, bounds.origin.y.as_f32() + sy)
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

/// Draw a filled circle as a path polygon.
///
/// Using PathBuilder (not paint_quad) keeps train dots in the same primitive
/// type as track lines. Within a canvas paint closure GPUI renders all Paths
/// in insertion order, so dots drawn after tracks appear on top of them.
/// paint_quad produces Quad primitives which GPUI renders *before* Paths,
/// which would put train dots behind track lines.
fn paint_circle(cx: f32, cy: f32, r: f32, color: Hsla, window: &mut Window) {
    const SEGS: usize = 20;
    let mut builder = PathBuilder::fill();
    builder.move_to(point(px(cx + r), px(cy)));
    for i in 1..=SEGS {
        let a = 2.0 * std::f32::consts::PI * i as f32 / SEGS as f32;
        builder.line_to(point(px(cx + r * a.cos()), px(cy + r * a.sin())));
    }
    if let Ok(path) = builder.build() {
        window.paint_path(path, color);
    }
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
        let data_tiles = self.data.clone();
        let data_draw = self.data.clone();
        let frame_idx = self.frame_idx;
        let time_s = self.data.frames.get(frame_idx).map(|f| f.time_s).unwrap_or(self.data.start_time);
        let elapsed = time_s - self.data.start_time;
        let n_frames = self.data.frames.len();
        let playing = self.playing;

        let abs_h = (time_s / 3600.0) as u32;
        let abs_m = ((time_s % 3600.0) / 60.0) as u32;
        let abs_s = (time_s % 60.0) as u32;
        let time_label = format!("t = {:02}:{:02}:{:02}", abs_h, abs_m, abs_s);

        let el_h = (elapsed / 3600.0) as u32;
        let el_m = ((elapsed % 3600.0) / 60.0) as u32;
        let el_s = (elapsed % 60.0) as u32;
        let elapsed_label = format!("+{:02}:{:02}:{:02} elapsed", el_h, el_m, el_s);

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
            // Layer 1: map tiles (PolychromeSprites).
            // Kept in its own canvas so that track paths — which GPUI always
            // renders before PolychromeSprites within the same scene layer —
            // can live in a later sibling canvas and thus appear on top.
            .child(
                canvas(
                    |_b, _w, _cx| {},
                    move |bounds, (), window, _cx| {
                        let tg = &data_tiles.tile_grid;
                        let (gx0, gy0) = bng_to_screen(tg.nw.0, tg.nw.1, data_tiles.bbox, bounds);
                        let (gx1, gy1) = bng_to_screen(tg.se.0, tg.se.1, data_tiles.bbox, bounds);
                        let tw = (gx1 - gx0) / tg.n_cols as f32;
                        let th = (gy1 - gy0) / tg.n_rows as f32;
                        for (row_idx, row) in tg.rows.iter().enumerate() {
                            for (col_idx, cell) in row.iter().enumerate() {
                                let Some(img) = cell else { continue };
                                let tile_bounds = Bounds {
                                    origin: point(
                                        px(gx0 + col_idx as f32 * tw),
                                        px(gy0 + row_idx as f32 * th),
                                    ),
                                    size: size(px(tw), px(th)),
                                };
                                window
                                    .paint_image(tile_bounds, Corners::all(px(0.0)), img.clone(), 0, false)
                                    .ok();
                            }
                        }
                    },
                )
                .absolute()
                .size_full(),
            )
            // Layer 2: tracks and trains — all Paths, drawn in insertion order
            // so trains (drawn last) appear on top of track lines.
            .child(
                canvas(
                    |_b, _w, _cx| {},
                    move |bounds, (), window, _cx| {
                        let track_outer = Hsla { h: 0.0,  s: 0.0,  l: 0.08, a: 1.0 };
                        let track_inner = Hsla { h: 0.08, s: 0.95, l: 0.65, a: 1.0 };

                        for poly in &data_draw.route_polylines {
                            for w in poly.windows(2) {
                                let (x0, y0) = bng_to_screen(w[0].0, w[0].1, data_draw.bbox, bounds);
                                let (x1, y1) = bng_to_screen(w[1].0, w[1].1, data_draw.bbox, bounds);
                                paint_segment(x0, y0, x1, y1, 4.5, track_outer, window);
                                paint_segment(x0, y0, x1, y1, 2.5, track_inner, window);
                            }
                        }

                        if let Some(frame) = data_draw.frames.get(frame_idx) {
                            for tp in &frame.trains {
                                let Some(pts) = data_draw.geometries.get(&tp.track_id) else { continue };
                                let Some(al)  = data_draw.arc_lengths.get(&tp.track_id) else { continue };
                                let (bx, by) = interpolate(pts, al, tp.element_offset_m);
                                let (sx, sy) = bng_to_screen(bx, by, data_draw.bbox, bounds);
                                let hue = *data_draw.train_hues.get(&tp.train_id).unwrap_or(&0.0);
                                let color = Hsla { h: hue, s: 0.9, l: 0.6, a: 1.0 };
                                paint_circle(sx, sy, 7.0, gpui::white(), window);
                                paint_circle(sx, sy, 5.5, color, window);
                            }
                        }
                    },
                )
                .absolute()
                .size_full(),
            )
            // HUD — last child so it renders above both canvas layers.
            // Explicit w() ensures the absolutely-positioned div has a computed
            // size so GPUI lays out and paints its background and text.
            .child(
                div()
                    .absolute()
                    .top(px(10.0))
                    .left(px(14.0))
                    .w(px(260.0))
                    .flex()
                    .flex_col()
                    .gap(px(3.0))
                    .p(px(10.0))
                    .rounded_lg()
                    .bg(Hsla { h: 0.0, s: 0.0, l: 0.0, a: 0.68 })
                    .child(
                        div()
                            .text_color(gpui::white())
                            .text_size(px(17.0))
                            .child(time_label),
                    )
                    .child(
                        div()
                            .text_color(Hsla { h: 0.13, s: 1.0, l: 0.72, a: 1.0 })
                            .text_size(px(15.0))
                            .child(elapsed_label),
                    )
                    .child(
                        div()
                            .text_color(Hsla { h: 0.0, s: 0.0, l: 0.55, a: 1.0 })
                            .text_size(px(11.0))
                            .child(format!(
                                "frame {}/{} | {}  Space ←→",
                                frame_idx + 1,
                                n_frames,
                                if playing { "▶" } else { "⏸" },
                            )),
                    ),
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

    eprintln!("Loading map tiles …");
    let tiles = fetch_tiles(bbox);

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

    let start_time = frames.first().map(|f| f.time_s).unwrap_or(0.0);

    let data = Arc::new(VisData {
        geometries,
        arc_lengths: arc_lengths_map,
        route_polylines,
        frames,
        bbox,
        train_hues,
        tile_grid: tiles,
        start_time,
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
