use crate::model::{DriverInput, TrainState, TrainDescription, Environment, Position};


const G: f64 = 9.81; // m/s²

pub fn step_trains(state: &TrainState, params: &TrainDescription, driver: &DriverInput, env: &Environment, dt: f64) -> TrainState {
    let speed = state.speed;
    let breaking = driver.break_ratio>0.0;

    let low_speed_force = params.traction_force_at_standstill * driver.power_ratio;
    let high_speed_force = if speed > 0.1 { params.power * driver.power_ratio / speed } else { low_speed_force };

    let traction_force = if !breaking {
        f64::min(low_speed_force, high_speed_force)
    } else {
        0.0
    };

    let braking_force = if breaking { params.braking_force * driver.break_ratio } else { 0.0 };
    // Gravity component along track (positive = uphill resistance)
    let gravity_force = params.mass * G * env.gradient;

    // Aerodynamic drag: F = c * v²
    let drag_force = params.drag_coeff * (speed+env.wind_speed) * (speed+env.wind_speed);

    // Rolling resistance (Davis equation simplified: ~0.002 * mass * g)
    let rolling_resistance = 0.002 * params.mass * G;

    // Net force
    let net_force = traction_force - gravity_force - drag_force - rolling_resistance - braking_force;

    // --- Kinematics (Euler integration) ---
    let acceleration = net_force / params.mass;
    println!("traction force: {}, acceleration: {}", traction_force, acceleration);

    let mut new_speed = (speed + acceleration * dt).max(0.0); // can't go negative
    let max_speed_m_s = params.max_speed / 3.6;
    if new_speed>max_speed_m_s {
        new_speed = max_speed_m_s;
    }

    let new_position = state.position.x + speed * dt; // use old speed for position update

    TrainState {
        position: Position{x:new_position,y:0., z:0.},
        speed: new_speed,
        acceleration: acceleration,
    }
}