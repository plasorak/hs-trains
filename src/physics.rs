use crate::model::{DriverInput, Environment, Position, SimulatedState, TrainDescription};

const G: f64 = 9.81; // m/s²

#[allow(dead_code)]
pub enum AdvanceTarget {
    Time(f64),     // seconds to advance
    Distance(f64), // metres to travel
}

fn net_force_at_speed(
    v: f64,
    params: &TrainDescription,
    driver: &DriverInput,
    env: &Environment,
) -> f64 {
    let braking = driver.brake_ratio > 0.0;

    let low_speed_force = params.traction_force_at_standstill * driver.power_ratio;
    let high_speed_force = if v > 0.1 {
        params.power * driver.power_ratio / v
    } else {
        low_speed_force
    };

    let traction_force = if !braking {
        f64::min(low_speed_force, high_speed_force)
    } else {
        0.0
    };
    let braking_force = if braking {
        params.braking_force * driver.brake_ratio
    } else {
        0.0
    };

    let gravity_force = params.mass * G * env.gradient;
    let drag_force = params.drag_coeff * (v + env.wind_speed).powi(2);
    let rolling_resistance = params.davis_a + params.davis_b * v;

    traction_force - gravity_force - drag_force - rolling_resistance - braking_force
}

#[allow(dead_code)]
fn compute_acceleration(
    state: &SimulatedState,
    params: &TrainDescription,
    driver: &DriverInput,
    env: &Environment,
) -> f64 {
    net_force_at_speed(state.speed, params, driver, env) / params.mass
}

/// Find the equilibrium speed (where net force = 0) in [v_lo, v_hi] via bisection.
/// Returns None if no zero crossing exists in that interval.
#[allow(dead_code)]
fn terminal_speed(
    v_lo: f64,
    v_hi: f64,
    params: &TrainDescription,
    driver: &DriverInput,
    env: &Environment,
) -> Option<f64> {
    let f_lo = net_force_at_speed(v_lo, params, driver, env);
    let f_hi = net_force_at_speed(v_hi, params, driver, env);
    if f_lo * f_hi >= 0.0 {
        return None;
    }
    // Keep lo on the positive-force side so bisection is consistent.
    let (mut lo, mut hi) = if f_lo > 0.0 {
        (v_lo, v_hi)
    } else {
        (v_hi, v_lo)
    };
    for _ in 0..52 {
        let mid = 0.5 * (lo + hi);
        if net_force_at_speed(mid, params, driver, env) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

/// Compute the train state after advancing by a fixed time or a fixed distance,
/// using constant-acceleration kinematics (one force evaluation, no iteration).
///
/// Terminal-velocity capping: the train cannot overshoot (or undershoot) the
/// equilibrium speed where net force = 0.  When the projected speed would cross
/// that point the motion is split into two phases — accelerate/decelerate to
/// equilibrium, then cruise — exactly as is done for the track speed limit.
#[allow(dead_code)]
pub fn advance_train(
    state: &SimulatedState,
    params: &TrainDescription,
    driver: &DriverInput,
    env: &Environment,
    target: AdvanceTarget,
) -> SimulatedState {
    let a = compute_acceleration(state, params, driver, env);
    let v0 = state.speed;
    let x0 = state.position.x;
    let vmax = params.max_speed / 3.6;

    let (new_speed, new_position) = match target {
        AdvanceTarget::Time(dt) => {
            if a > 0.0 {
                // Cap at terminal velocity (tighter than vmax; prevents overshoot past equilibrium).
                let v_eq = terminal_speed(v0, vmax, params, driver, env).unwrap_or(vmax);
                let t_to_eq = (v_eq - v0) / a;
                if dt <= t_to_eq {
                    (v0 + a * dt, x0 + v0 * dt + 0.5 * a * dt * dt)
                } else {
                    // Phase 1: accelerate to v_eq, then cruise.
                    let x1 = x0 + v0 * t_to_eq + 0.5 * a * t_to_eq * t_to_eq;
                    (v_eq, x1 + v_eq * (dt - t_to_eq))
                }
            } else if a < 0.0 {
                // When coasting above equilibrium (no brakes), cap deceleration at v_eq
                // to prevent undershooting.  When braking there is no equilibrium above 0.
                let v_floor = if driver.brake_ratio == 0.0 {
                    terminal_speed(0.0, v0, params, driver, env).unwrap_or(0.0)
                } else {
                    0.0
                };
                let t_to_floor = (v_floor - v0) / a; // positive: a<0, v_floor<v0
                if dt <= t_to_floor {
                    let ns = (v0 + a * dt).max(0.0);
                    (ns, x0 + v0 * dt + 0.5 * a * dt * dt)
                } else {
                    // Phase 1: decelerate to v_floor, then cruise.
                    let x1 = x0 + v0 * t_to_floor + 0.5 * a * t_to_floor * t_to_floor;
                    (v_floor, x1 + v_floor * (dt - t_to_floor))
                }
            } else {
                (v0, x0 + v0 * dt)
            }
        }
        AdvanceTarget::Distance(dx) => {
            if v0 == 0.0 && a <= 0.0 {
                return state.clone();
            }
            let v_sq = v0 * v0 + 2.0 * a * dx;
            if v_sq <= 0.0 {
                // Decelerating — train stops before covering dx.
                let stop_dist = v0 * v0 / (2.0 * a.abs());
                return SimulatedState {
                    position: Position {
                        x: x0 + stop_dist,
                        y: 0.0,
                        z: 0.0,
                    },
                    speed: 0.0,
                    acceleration: a,
                };
            }
            let v_cap = if a > 0.0 {
                terminal_speed(v0, vmax, params, driver, env).unwrap_or(vmax)
            } else {
                vmax
            };
            (v_sq.sqrt().min(v_cap), x0 + dx)
        }
    };

    SimulatedState {
        position: Position {
            x: new_position,
            y: 0.0,
            z: 0.0,
        },
        speed: new_speed,
        acceleration: a,
    }
}

pub fn step_trains(
    state: &SimulatedState,
    params: &TrainDescription,
    driver: &DriverInput,
    env: &Environment,
    dt: f64,
) -> SimulatedState {
    let net_force = net_force_at_speed(state.speed, params, driver, env);

    // --- Kinematics (Euler integration) ---
    let acceleration = net_force / params.mass;
    let mut new_speed = (state.speed + acceleration * dt).max(0.0); // can't go negative
    let max_speed_m_s = params.max_speed / 3.6;
    if new_speed > max_speed_m_s {
        new_speed = max_speed_m_s;
    }

    let new_position = state.position.x + state.speed * dt; // use old speed for position update

    SimulatedState {
        position: Position {
            x: new_position,
            y: 0.0,
            z: 0.0,
        },
        speed: new_speed,
        acceleration,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DriverInput, Environment, Position, SimulatedState, TrainDescription};

    fn test_params() -> TrainDescription {
        TrainDescription {
            power: 2_460_000.0,
            traction_force_at_standstill: 409_000.0,
            max_speed: 120.0,
            mass: 2_000_000.0,
            davis_a: 39_240.0, // ≈ 0.002 × mass × g — equivalent to the old hardcoded value
            davis_b: 0.0,
            drag_coeff: 10.0,
            braking_force: 800_000.0,
        }
    }

    fn initial_state(speed: f64) -> SimulatedState {
        SimulatedState {
            position: Position {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            speed,
            acceleration: 0.0,
        }
    }

    /// Reference: run step_trains with fine 0.1 s steps for total_time seconds.
    fn step_reference(
        s0: &SimulatedState,
        p: &TrainDescription,
        d: &DriverInput,
        e: &Environment,
        total_time: f64,
    ) -> SimulatedState {
        let dt = 0.1;
        let n = (total_time / dt).round() as usize;
        let mut state = s0.clone();
        for _ in 0..n {
            state = step_trains(&state, p, d, e, dt);
        }
        state
    }

    fn assert_within(label: &str, result: &SimulatedState, reference: &SimulatedState) {
        let speed_tol = 1.0_f64; // m/s
        let pos_tol = 5.0_f64; // m
        let dv = (result.speed - reference.speed).abs();
        let dx = (result.position.x - reference.position.x).abs();
        assert!(
            dv < speed_tol,
            "{label}: speed {:.4} vs ref {:.4} (diff {dv:.4} m/s)",
            result.speed,
            reference.speed
        );
        assert!(
            dx < pos_tol,
            "{label}: position {:.2} vs ref {:.2} (diff {dx:.2} m)",
            result.position.x,
            reference.position.x
        );
    }

    #[test]
    fn test_accelerating_from_standstill() {
        let p = test_params();
        let d = DriverInput {
            power_ratio: 0.8,
            brake_ratio: 0.0,
        };
        let e = Environment {
            gradient: 0.0,
            wind_speed: 0.0,
        };
        let s0 = initial_state(0.0);
        assert_within(
            "standstill",
            &advance_train(&s0, &p, &d, &e, AdvanceTarget::Time(10.0)),
            &step_reference(&s0, &p, &d, &e, 10.0),
        );
    }

    #[test]
    fn test_accelerating_from_standstill_distance() {
        let p = test_params();
        let d = DriverInput {
            power_ratio: 0.8,
            brake_ratio: 0.0,
        };
        let e = Environment {
            gradient: 0.0,
            wind_speed: 0.0,
        };
        let s0 = initial_state(0.0);
        let reference = step_reference(&s0, &p, &d, &e, 10.0);
        assert_within(
            "standstill distance",
            &advance_train(
                &s0,
                &p,
                &d,
                &e,
                AdvanceTarget::Distance(reference.position.x),
            ),
            &reference,
        );
    }

    #[test]
    fn test_braking() {
        let p = test_params();
        let d = DriverInput {
            power_ratio: 0.0,
            brake_ratio: 0.5,
        };
        let e = Environment {
            gradient: 0.0,
            wind_speed: 0.0,
        };
        let s0 = initial_state(20.0); // 72 km/h
        assert_within(
            "braking",
            &advance_train(&s0, &p, &d, &e, AdvanceTarget::Time(10.0)),
            &step_reference(&s0, &p, &d, &e, 10.0),
        );
    }

    #[test]
    fn test_braking_distance() {
        let p = test_params();
        let d = DriverInput {
            power_ratio: 0.0,
            brake_ratio: 0.5,
        };
        let e = Environment {
            gradient: 0.0,
            wind_speed: 0.0,
        };
        let s0 = initial_state(20.0);
        let reference = step_reference(&s0, &p, &d, &e, 10.0);
        assert_within(
            "braking distance",
            &advance_train(
                &s0,
                &p,
                &d,
                &e,
                AdvanceTarget::Distance(reference.position.x),
            ),
            &reference,
        );
    }

    #[test]
    fn test_positive_gradient() {
        let p = test_params();
        let d = DriverInput {
            power_ratio: 0.8,
            brake_ratio: 0.0,
        };
        let e = Environment {
            gradient: 0.02,
            wind_speed: 0.0,
        }; // 2% uphill
        let s0 = initial_state(10.0);
        assert_within(
            "positive gradient",
            &advance_train(&s0, &p, &d, &e, AdvanceTarget::Time(10.0)),
            &step_reference(&s0, &p, &d, &e, 10.0),
        );
    }

    #[test]
    fn test_positive_gradient_distance() {
        let p = test_params();
        let d = DriverInput {
            power_ratio: 0.8,
            brake_ratio: 0.0,
        };
        let e = Environment {
            gradient: 0.02,
            wind_speed: 0.0,
        };
        let s0 = initial_state(10.0);
        let reference = step_reference(&s0, &p, &d, &e, 10.0);
        assert_within(
            "positive gradient distance",
            &advance_train(
                &s0,
                &p,
                &d,
                &e,
                AdvanceTarget::Distance(reference.position.x),
            ),
            &reference,
        );
    }

    #[test]
    fn test_negative_gradient() {
        let p = test_params();
        let d = DriverInput {
            power_ratio: 0.8,
            brake_ratio: 0.0,
        };
        let e = Environment {
            gradient: -0.02,
            wind_speed: 0.0,
        }; // 2% downhill
        let s0 = initial_state(10.0);
        assert_within(
            "negative gradient",
            &advance_train(&s0, &p, &d, &e, AdvanceTarget::Time(10.0)),
            &step_reference(&s0, &p, &d, &e, 10.0),
        );
    }

    #[test]
    fn test_negative_gradient_distance() {
        let p = test_params();
        let d = DriverInput {
            power_ratio: 0.8,
            brake_ratio: 0.0,
        };
        let e = Environment {
            gradient: -0.02,
            wind_speed: 0.0,
        };
        let s0 = initial_state(10.0);
        let reference = step_reference(&s0, &p, &d, &e, 10.0);
        assert_within(
            "negative gradient distance",
            &advance_train(
                &s0,
                &p,
                &d,
                &e,
                AdvanceTarget::Distance(reference.position.x),
            ),
            &reference,
        );
    }
}
