
#[derive(Debug,Clone)]
pub struct Position {
    pub x:f64,
    pub y:f64,
    pub z:f64,
}

#[derive(Debug,Clone)]
pub struct Trajectory {
    pub points: Vec<Position>,
}

#[derive(Debug,Clone)]
pub struct DriverInput {
    pub break_ratio: f64,
    pub power_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct TrainState {
    pub position: Position,  // meters
    pub speed: f64,     // m/s
    pub acceleration: f64,
}

#[derive(Debug, Clone)]
pub struct TrainDescription {
    pub power: f64,       // Max Watts
    pub traction_force_at_standstill: f64, // N
    pub max_speed: f64,
    pub mass: f64,        // kg
    pub drag_coeff: f64,  // aerodynamic drag coefficient (kg/m), tune as needed
    pub braking_force: f64, // Newtons, maximum braking force
}

#[derive(Debug, Clone)]
pub struct Environment {
    pub wind_speed: f64,
    pub gradient: f64,    // rise/run (e.g. 0.01 = 1% grade)
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
