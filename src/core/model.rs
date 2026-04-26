#![allow(dead_code)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Infrastructure
// ---------------------------------------------------------------------------

/// A directed track section in the RailML topology layer.
#[derive(Debug, Clone)]
pub struct NetElement {
    pub id: String,
    pub length_m: f64,
}

/// A functional track that maps 1-to-1 onto a NetElement.
#[derive(Debug, Clone)]
pub struct Track {
    pub id: String,
    pub net_element_id: String,
}

/// A named stopping/passing location on the network.
#[derive(Debug, Clone)]
pub struct OperationalPoint {
    pub id: String,
    pub name: Option<String>,
}

/// Parsed infrastructure from a RailML `<infrastructure>` section.
#[derive(Debug, Clone)]
pub struct Infrastructure {
    pub net_elements: HashMap<String, NetElement>,
    /// track id → Track (which carries the net_element_id back-link)
    pub tracks: HashMap<String, Track>,
    pub ops: HashMap<String, OperationalPoint>,
    /// track id → ordered WGS84 (lon, lat) coordinates from gml:posList.
    /// Empty vec when the track has no embedded geometry.
    pub track_coords: HashMap<String, Vec<(f64, f64)>>,
}

// ---------------------------------------------------------------------------
// Route
// ---------------------------------------------------------------------------

/// One track section on a train's route, with a resolved length.
#[derive(Debug, Clone)]
pub struct RouteElement {
    pub track_id: String,
    pub net_element_id: String,
    pub length_m: f64,
}

/// The resolved path a physics train follows through the network.
///
/// Position along the route is the scalar `x` used by the physics engine.
/// `locate` maps that scalar back to a specific track and offset.
#[derive(Debug, Clone)]
pub struct Route {
    pub elements: Vec<RouteElement>,
    /// Prefix-sum of element lengths: `cumulative_lengths[i]` is the start of
    /// `elements[i]`.  Length is `elements.len() + 1`; the first entry is 0.
    pub cumulative_lengths: Vec<f64>,
    pub total_length_m: f64,
}

impl Route {
    pub fn new(elements: Vec<RouteElement>) -> Self {
        let mut cumulative_lengths = Vec::with_capacity(elements.len() + 1);
        cumulative_lengths.push(0.0);
        for el in &elements {
            let prev = *cumulative_lengths.last().unwrap();
            cumulative_lengths.push(prev + el.length_m);
        }
        let total_length_m = *cumulative_lengths.last().unwrap_or(&0.0);
        Route { elements, cumulative_lengths, total_length_m }
    }

    /// Map a scalar distance along the route to the containing track and offset.
    ///
    /// Returns `None` only when `elements` is empty.
    /// Clamps to the last element's end when `distance_m >= total_length_m`.
    pub fn locate(&self, distance_m: f64) -> Option<NetworkPosition> {
        if self.elements.is_empty() {
            return None;
        }
        let clamped = distance_m.min(self.total_length_m);
        // partition_point returns first index where cumulative_lengths[i] > clamped.
        // Subtract 1 to get the element whose start <= clamped.
        let idx = self.cumulative_lengths
            .partition_point(|&cl| cl <= clamped)
            .saturating_sub(1)
            .min(self.elements.len() - 1);
        let el = &self.elements[idx];
        let offset_m = clamped - self.cumulative_lengths[idx];
        Some(NetworkPosition {
            track_id: el.track_id.clone(),
            net_element_id: el.net_element_id.clone(),
            offset_m,
        })
    }
}

/// Position within a specific track element, expressed as an offset from its start.
#[derive(Debug, Clone)]
pub struct NetworkPosition {
    pub track_id: String,
    pub net_element_id: String,
    pub offset_m: f64,
}

// ---------------------------------------------------------------------------
// Existing types
// ---------------------------------------------------------------------------

#[cfg(test)]
mod route_tests {
    use super::*;

    fn make_route(lengths: &[f64]) -> Route {
        let elements: Vec<RouteElement> = lengths
            .iter()
            .enumerate()
            .map(|(i, &l)| RouteElement {
                track_id: format!("track_{i}"),
                net_element_id: format!("ne_{i}"),
                length_m: l,
            })
            .collect();
        Route::new(elements)
    }

    #[test]
    fn test_locate_at_start() {
        let r = make_route(&[100.0, 200.0, 150.0]);
        let p = r.locate(0.0).unwrap();
        assert_eq!(p.track_id, "track_0");
        assert_eq!(p.offset_m, 0.0);
    }

    #[test]
    fn test_locate_mid_first_element() {
        let r = make_route(&[100.0, 200.0]);
        let p = r.locate(50.0).unwrap();
        assert_eq!(p.track_id, "track_0");
        assert_eq!(p.offset_m, 50.0);
    }

    #[test]
    fn test_locate_at_element_boundary() {
        // exactly at 100 m — start of second element
        let r = make_route(&[100.0, 200.0]);
        let p = r.locate(100.0).unwrap();
        assert_eq!(p.track_id, "track_1");
        assert_eq!(p.offset_m, 0.0);
    }

    #[test]
    fn test_locate_at_route_end() {
        let r = make_route(&[100.0, 200.0]);
        let p = r.locate(300.0).unwrap();
        assert_eq!(p.track_id, "track_1");
        assert!((p.offset_m - 200.0).abs() < 1e-9);
    }

    #[test]
    fn test_locate_beyond_route_clamped() {
        // Should clamp to route end, not return None.
        let r = make_route(&[100.0]);
        let p = r.locate(999.0).unwrap();
        assert_eq!(p.track_id, "track_0");
        assert!((p.offset_m - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_locate_empty_route() {
        let r = Route::new(vec![]);
        assert!(r.locate(0.0).is_none());
    }

    #[test]
    fn test_total_length() {
        let r = make_route(&[100.0, 200.0, 50.0]);
        assert!((r.total_length_m - 350.0).abs() < 1e-9);
    }
}

#[derive(Debug, Clone)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone)]
pub struct Trajectory {
    pub points: Vec<Position>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DriverInput {
    pub brake_ratio: f64,
    pub power_ratio: f64,
}

/// Train state produced by the physics engine. All kinematic fields are always
/// known, so they are stored as plain `f64`.
#[derive(Debug, Clone)]
pub struct SimulatedState {
    pub position: Position, // metres
    pub speed: f64,         // m/s
    pub acceleration: f64,  // m/s²
}

/// Train state derived from berth timing data. Speed and acceleration are not
/// available from timing records, so only position and timestamp are stored.
#[derive(Debug, Clone)]
pub struct ObservedState {
    pub position: Position, // metres along route
    pub timestamp_ms: i64,  // Unix epoch, milliseconds
}

/// A train's state, which is either physics-simulated or timing-observed.
#[derive(Debug, Clone)]
pub enum TrainState {
    Simulated(SimulatedState),
    Observed(ObservedState),
}

impl TrainState {
    pub fn position(&self) -> &Position {
        match self {
            TrainState::Simulated(s) => &s.position,
            TrainState::Observed(o) => &o.position,
        }
    }

    pub fn speed(&self) -> Option<f64> {
        match self {
            TrainState::Simulated(s) => Some(s.speed),
            TrainState::Observed(_) => None,
        }
    }

    pub fn acceleration(&self) -> Option<f64> {
        match self {
            TrainState::Simulated(s) => Some(s.acceleration),
            TrainState::Observed(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainDescription {
    pub power: f64,                        // Max Watts
    pub traction_force_at_standstill: f64, // N
    pub max_speed: f64,                    // km/h
    pub mass: f64,                         // kg
    pub davis_a: f64,                      // Davis constant term A (N)
    pub davis_b: f64,                      // Davis linear term B, converted to SI (N·s/m)
    pub drag_coeff: f64,                   // Davis quadratic term C, converted to SI (kg/m)
    pub braking_force: f64,                // N, maximum braking force
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Environment {
    pub wind_speed: f64,
    pub gradient: f64, // rise/run (e.g. 0.01 = 1% grade)
}

#[derive(Debug, Clone)]
pub struct SignalDescription {
    pub position: Position,
    pub sighting_position: Position,
}

#[derive(Debug, Clone)]
pub struct OverlapDescription {
    pub position: Position,
}

#[derive(Debug, Clone)]
pub struct BerthDescription {
    pub name: std::string::String,
    pub trajectory: Trajectory,
    pub entering_signal: SignalDescription,
    pub entering_overlap: OverlapDescription,
    pub exiting_overlap: OverlapDescription,
}

#[derive(Debug, Clone)]
pub struct BerthState {
    pub signal_aspect: std::string::String,
}
