# Physics Engine

This document describes the force model and numerical integration used by the physics-based train simulator.

## The Davis Equation

Train resistance is modelled using the **Davis equation**, originally published by W.J. Davis Jr. in 1926 [1].
In its general form it expresses the total resistance force as a polynomial in speed:

```
R(v) = A + B·v + C·v²
```

| Term | Name | Unit | Field in code |
|------|------|------|---------------|
| A    | Constant resistance (journal/bearing friction, unsprung mass, flange contact) | N | `davis_a` |
| B    | Linear term (flanging, hunting, miscellaneous speed-proportional loss) | N·s/m | `davis_b` |
| C    | Aerodynamic drag coefficient | kg/m | `drag_coeff` |

### Aerodynamic drag (C term)

Wind speed is folded into the quadratic term so that a head-wind increases drag and a tail-wind reduces it:

```
F_drag = drag_coeff · (v + wind_speed)²
```

where `wind_speed` is positive for a head-wind (same direction as train motion) and negative for a tail-wind.
This follows the relative-velocity convention described in UIC 779-1 [2].

### Rolling resistance (A + B terms)

```
F_rolling = davis_a + davis_b · v
```

For many heavy freight and passenger vehicles the B term is small and is sometimes set to zero.
The A term is typically proportional to vehicle weight; a common rule of thumb is
`A ≈ 0.002 · m · g` (approximately 2 N per kN of weight), which matches the test train in the unit tests.

## Gravity

On a gradient *g_r* (dimensionless rise/run, positive = uphill):

```
F_gravity = m · g · g_r
```

For the small angles used in railway engineering (|g_r| ≤ 0.05) this is a valid small-angle approximation
of `m · g · sin(θ)`.  See EN 13803 [3] for gradient conventions.

## Traction force

The traction envelope follows the classical **tractive-effort–speed hyperbola** [4]:

```
             ┌ traction_force_at_standstill · power_ratio    (adhesion-limited, low speed)
F_traction = │
             └ power · power_ratio / v                       (power-limited, high speed)

F_traction = min(adhesion-limited, power-limited)
```

Setting `power_ratio = 0` disables traction entirely.  When `brake_ratio > 0` the traction force
is forced to zero regardless of `power_ratio`.

## Braking force

```
F_braking = braking_force · brake_ratio    (when brake_ratio > 0)
F_braking = 0                              (otherwise)
```

`braking_force` is the maximum retarding force the train can apply (e.g. regenerative + friction brakes combined).

## Net force

Combining all terms:

```
F_net = F_traction − F_gravity − F_drag − F_rolling − F_braking
```

Acceleration follows Newton's second law:

```
a = F_net / m
```

## Terminal (equilibrium) speed

Terminal speed *v_eq* is the speed at which `F_net = 0`.
It is found by **bisection** over the interval `[v_lo, v_hi]` in 52 iterations
(relative error < 2⁻⁵² ≈ 2 × 10⁻¹⁶, well within floating-point precision).
The bisection always brackets the root because the net-force function is monotonically decreasing
in speed within the traction-limited and power-limited regimes.

## Numerical integration

Two integrators are provided.

### Euler integrator — `step_trains`

Simple forward-Euler step used for fine-grained reference simulation:

```
a  = F_net(v) / m
v' = clamp(v + a·Δt, 0, v_max)
x' = x + v·Δt           ← uses speed at start of step (semi-implicit)
```

Step size is typically 0.1 s.  Accuracy degrades for larger steps (first-order method).

### Closed-form integrator — `advance_train`

Used for event-driven advancement by an arbitrary time Δt or distance Δx.
Acceleration is evaluated once at the current speed (constant-*a* assumption), then the motion is
split into at most two phases to prevent overshoot past the equilibrium speed:

**Time-advance (AdvanceTarget::Time)**

- If `a > 0`: accelerate towards *v_eq*.  If Δt would overshoot, cruise at *v_eq* for the remainder.
- If `a < 0`: decelerate towards *v_eq* (or 0 if braking).  If Δt would undershoot, cruise at the floor.
- If `a = 0`: pure cruise.

**Distance-advance (AdvanceTarget::Distance)**

Uses the kinematic identity `v² = v₀² + 2·a·Δx` to find the exit speed.
If the train would stop before covering Δx, it stops at the braking distance `v₀²/(2|a|)`.
Terminal-velocity capping is applied when accelerating.

The one-force-evaluation approach gives exact constant-acceleration kinematics and avoids the
accumulation of Euler truncation error for large steps.

## Parameters

All parameters live in `TrainDescription` ([src/core/model.rs](../src/core/model.rs)):

| Field | Unit | Description |
|-------|------|-------------|
| `power` | W | Continuous installed traction power |
| `traction_force_at_standstill` | N | Maximum tractive effort at zero speed (adhesion limit) |
| `max_speed` | km/h | Top speed (hard cap applied after integration) |
| `mass` | kg | Total train mass (tare + payload) |
| `davis_a` | N | Davis constant resistance term |
| `davis_b` | N·s/m | Davis linear resistance term |
| `drag_coeff` | kg/m | Davis quadratic (aerodynamic) coefficient |
| `braking_force` | N | Maximum braking force |

Environmental inputs live in `Environment`:

| Field | Unit | Description |
|-------|------|-------------|
| `gradient` | — | Track gradient, rise/run (positive = uphill) |
| `wind_speed` | m/s | Head-wind component along track (positive = opposing motion) |

Driver commands live in `DriverInput`:

| Field | Range | Description |
|-------|-------|-------------|
| `power_ratio` | 0–1 | Throttle demand as fraction of full power |
| `brake_ratio` | 0–1 | Brake demand as fraction of maximum braking force |

## References

[1] Davis, W.J. Jr., "The Tractive Resistance of Electric Locomotives and Cars," *General Electric Review*, vol. 29, no. 10, pp. 685–708, 1926.

[2] UIC Code 779-1, *Resistance of Trains in Tunnels*, Union Internationale des Chemins de Fer, 2005.

[3] EN 13803:2017, *Railway Applications — Track Alignment Design Parameters*, CEN, 2017.

[4] Iwnicki, S. (ed.), *Handbook of Railway Vehicle Dynamics*, CRC Press, 2006, ch. 4 ("Traction and Braking").
