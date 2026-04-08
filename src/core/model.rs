#![allow(dead_code)]

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
