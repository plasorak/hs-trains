mod physics;
mod model;

use model::{DriverInput, TrainState, TrainDescription, Environment, Position};
use physics::step_trains;
use polars::prelude::*;

fn main() {

    let mut time_data: Vec<f64> = Vec::new();
    let mut speed_data: Vec<f64> = Vec::new();
    let mut position_data: Vec<f64> = Vec::new();
    let mut acceleration_data: Vec<f64> = Vec::new();

    let mut state = TrainState {
        position: Position{x: 0., y: 0., z: 0.},
        speed: 0.0,
        acceleration: 0.0,
    };

    let params = TrainDescription {
        power: 2_460_000.0,
        traction_force_at_standstill: 409_000.,
        max_speed: 120.,
        mass: 2_000_000.0,
        drag_coeff: 10.0,
        braking_force: 800_000.0,
    };

    let driver_input = DriverInput {
        power_ratio: 0.8,
        break_ratio: 0.0,
    };

    let env = Environment {
        wind_speed: 0.,
        gradient: 0.01,
    };

    let dt = 0.1; // 0.1 second time step

    for step in 0..20_000 {
        state = step_trains(&state, &params, &driver_input, &env, dt);
        time_data.push(step as f64 * dt);
        speed_data.push(state.speed * 3.6);
        position_data.push(state.position.x);
        acceleration_data.push(state.acceleration);
        println!(
            "t={:>4}s  pos={:>8.1}m  speed={:.2} m/s ({:.1} km/h)",
            step as f64*dt,
            state.position.x,
            state.speed,
            state.speed * 3.6
        );
    }

    let mut df = DataFrame::new(
        time_data.len(),
        vec![
            Series::new("time_s".into(), &time_data).into(),
            Series::new("speed_kmh".into(), &speed_data).into(),
            Series::new("position_m".into(), &position_data).into(),
            Series::new("acceleration".into(), &acceleration_data).into(),
        ]
    ).unwrap();

    let file = std::fs::File::create("simulation.parquet").unwrap();

    ParquetWriter::new(file).finish(&mut df).unwrap();
}