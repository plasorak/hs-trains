# RailML 3.3 Format Reference

This document is a deep dive into the RailML 3.3 rollingstock sub-schema as used by hs-trains. It covers element semantics, attribute units, curve conventions, and the mapping from RailML fields to the internal physics parameters.

For the Python tooling that generates and validates RailML files, see [rollingstock.md](rollingstock.md).

---

## What is RailML?

[RailML](https://www.railml.org/) is an open XML standard for exchanging railway data between systems. Version 3.3 divides the domain into independent sub-schemas: `infrastructure`, `timetable`, `rollingstock`, `interlocking`, and others.

hs-trains uses only the **rollingstock** sub-schema, which describes the physical and operational properties of rail vehicles and the consists (formations) they form.

The official XSD schema is bundled at `railml/railML-3.3-SR1/`. A small stub (`dcterms_stub.xsd`) replaces the unavailable Dublin Core Terms namespace referenced by `common3.xsd`.

---

## Namespace and Document Root

All elements live in the namespace `https://www.railml.org/schemas/3.3`, conventionally prefixed `rail3:`.

```xml
<?xml version='1.0' encoding='utf-8'?>
<rail3:railML xmlns:rail3="https://www.railml.org/schemas/3.3" version="3.3">
  <rail3:rollingstock>
    <rail3:vehicles> … </rail3:vehicles>
    <rail3:formations> … </rail3:formations>
  </rail3:rollingstock>
</rail3:railML>
```

The `version` attribute is fixed at `"3.3"`.

---

## Document Structure

```
railML
└── rollingstock
    ├── vehicles
    │   └── vehicle [1..*]
    └── formations
        └── formation [1..*]
```

A **vehicle** is a single rolling-stock unit (locomotive, coach, wagon). A **formation** is an ordered sequence of vehicles coupled as an operational train consist. The simulation reads formations, not individual vehicles.

---

## Curve Primitives

Speed-dependent data (tractive effort, deceleration, driving resistance) is encoded as tables rather than analytical functions. Three nested elements form the pattern:

### `valueTable`

The outer container. Declares axis names and units.

| Attribute | Type | Description |
|---|---|---|
| `xValueName` | string | Name of the X axis (always `"speed"` in hs-trains) |
| `xValueUnit` | string | Unit of the X axis (always `"km/h"`) |
| `yValueName` | string | Name of the Y axis (e.g. `"tractiveEffort"`, `"deceleration"`) |
| `yValueUnit` | string | Unit of the Y axis (e.g. `"N"`, `"m/s/s"`) |

### `valueLine`

One row in the table, keyed by a single X value.

| Attribute | Type | Description |
|---|---|---|
| `xValue` | decimal | Speed in km/h |

### `value`

One Y datum on a `valueLine`.

| Attribute | Type | Description |
|---|---|---|
| `yValue` | decimal | The measured or computed quantity at the given speed |

**Example — tractive effort table:**

```xml
<rail3:valueTable xValueName="speed" xValueUnit="km/h"
                  yValueName="tractiveEffort" yValueUnit="N">
  <rail3:valueLine xValue="0">
    <rail3:value yValue="270000" />
  </rail3:valueLine>
  <rail3:valueLine xValue="20">
    <rail3:value yValue="270000" />
  </rail3:valueLine>
  <rail3:valueLine xValue="40">
    <rail3:value yValue="200000" />
  </rail3:valueLine>
  <rail3:valueLine xValue="120">
    <rail3:value yValue="66000" />
  </rail3:valueLine>
</rail3:valueTable>
```

Linear interpolation is used between rows; values outside the table range are not extrapolated.

---

## `vehicle` Element

Represents a single vehicle class or individual unit.

### Top-level attributes

| Attribute | Type | Unit | Description |
|---|---|---|---|
| `id` | string | — | Unique identifier; referenced by `trainOrder/@vehicleRef` |
| `speed` | decimal | km/h | Maximum permissible speed |
| `bruttoWeight` | decimal | t | Gross mass (tare + maximum payload + full passenger load at ~75 kg/person) |
| `tareWeight` | decimal | t | Empty operational mass (no payload, no passengers, ready to run). **Used as vehicle mass in physics calculations.** |
| `nettoWeight` | decimal | t | Maximum payload |
| `length` | decimal | m | Overall length over buffers/couplings; used to compute formation length |
| `numberOfDrivenAxles` | int | — | Must be > 0 for any vehicle with an `engine` element |
| `numberOfNonDrivenAxles` | int | — | Non-driven axle count; contributes to rolling resistance |
| `adhesionWeight` | decimal | t | Tare weight usable for traction (adhesion limit estimation) |
| `rotatingMassFactor` | decimal | — | Multiplier on static mass to account for rotating inertia. Typical range 1.05–1.25. |
| `maximumAxleLoad` | decimal | t | Maximum static load per axle |
| `towingSpeed` | decimal | km/h | Maximum speed when being towed |

### `designator` (0..*)

Provides identifiers from named registers.

| Attribute | Description |
|---|---|
| `register` | Register name, e.g. `"UIC"`, `"operator"` |
| `entry` | The identifier within that register, e.g. `"92 70 0 066 001-1"` |

**Example:**
```xml
<rail3:designator register="UIC" entry="92 70 0 066 001-1" />
<rail3:designator register="operator" entry="Class66-001" />
```

### `vehiclePart` (0..*)

Physical sections of the vehicle body.

| Attribute | Type | Description |
|---|---|---|
| `id` | string | Unique ID |
| `partOrder` | int | Position in the vehicle, starting at 1 |
| `category` | enum | `locomotive`, `motorCoach`, `passengerCoach`, `freightWagon`, `cabCoach`, `booster` |

### `engine`

Contains one or more `powerMode` children.

#### `powerMode`

| Attribute | Type | Description |
|---|---|---|
| `mode` | enum | `diesel`, `electric`, `battery` |
| `isPrimaryMode` | boolean | `"true"` if this mode is the main propulsion source |

Contains a `tractionData` element.

##### `tractionData`

| Child | Description |
|---|---|
| `info` | Scalar summary: `maxTractiveEffort` (N), `tractivePower` (W) |
| `details` | Speed-dependent curve via `tractiveEffort` > `valueTable` |

The `info` element is the primary source for physics extraction. `details` holds the full tractive-effort curve (force vs speed in km/h).

**Example — Class 66 traction:**
```xml
<rail3:engine>
  <rail3:powerMode mode="diesel" isPrimaryMode="true">
    <rail3:tractionData>
      <rail3:info maxTractiveEffort="270000" tractivePower="2420000" />
      <rail3:details>
        <rail3:tractiveEffort>
          <rail3:valueTable xValueName="speed" xValueUnit="km/h"
                            yValueName="tractiveEffort" yValueUnit="N">
            <rail3:valueLine xValue="0"><rail3:value yValue="270000" /></rail3:valueLine>
            <rail3:valueLine xValue="20"><rail3:value yValue="270000" /></rail3:valueLine>
            <rail3:valueLine xValue="40"><rail3:value yValue="200000" /></rail3:valueLine>
            <rail3:valueLine xValue="60"><rail3:value yValue="133000" /></rail3:valueLine>
            <rail3:valueLine xValue="80"><rail3:value yValue="100000" /></rail3:valueLine>
            <rail3:valueLine xValue="100"><rail3:value yValue="80000" /></rail3:valueLine>
            <rail3:valueLine xValue="120"><rail3:value yValue="66000" /></rail3:valueLine>
          </rail3:valueTable>
        </rail3:tractiveEffort>
      </rail3:details>
    </rail3:tractionData>
  </rail3:powerMode>
</rail3:engine>
```

### `brakes`

| Child | Description |
|---|---|
| `vehicleBrakes` (0..*) | Brake system configuration block (see attributes below) |
| `brakeEffort` | Speed-dependent brake force curve (N vs km/h) |
| `decelerationTable` | Speed-dependent deceleration curve (m/s² vs km/h) |

#### `vehicleBrakes` attributes

| Attribute | Type | Unit | Description |
|---|---|---|---|
| `brakeType` | string | — | Technology: vacuum, compressed air, hand brake, etc. |
| `meanDeceleration` | decimal | m/s² | Mean deceleration over a complete braking operation |
| `maxDeceleration` | decimal | m/s² | Maximum instantaneous deceleration |
| `regularBrakePercentage` | decimal | % | Brake percentage for normal operations |
| `emergencyBrakePercentage` | decimal | % | Brake percentage for emergency braking |

**Example — deceleration table:**
```xml
<rail3:brakes>
  <rail3:decelerationTable>
    <rail3:valueTable xValueName="speed" xValueUnit="km/h"
                      yValueName="deceleration" yValueUnit="m/s/s">
      <rail3:valueLine xValue="0"><rail3:value yValue="0.90" /></rail3:valueLine>
      <rail3:valueLine xValue="40"><rail3:value yValue="0.85" /></rail3:valueLine>
      <rail3:valueLine xValue="80"><rail3:value yValue="0.75" /></rail3:valueLine>
      <rail3:valueLine xValue="120"><rail3:value yValue="0.65" /></rail3:valueLine>
    </rail3:valueTable>
  </rail3:decelerationTable>
</rail3:brakes>
```

### `drivingResistance`

Describes the resistance forces acting on the vehicle when moving.

| Attribute | Type | Description |
|---|---|---|
| `tunnelFactor` | decimal | Multiplier on resistance inside a tunnel (typically 1.5–2.0) |

| Child | Description |
|---|---|
| `info` | Scalar parameters: `airDragCoefficient`, `crossSectionArea` (m²), `rollingResistance` (N/kN) |
| `details` | Speed-dependent resistance curve |

**Example:**
```xml
<rail3:drivingResistance tunnelFactor="1.5">
  <rail3:info airDragCoefficient="0.80"
              crossSectionArea="9.5"
              rollingResistance="1.5" />
</rail3:drivingResistance>
```

---

## `formation` Element

A formation is the operational consist: an ordered list of vehicles coupled together. This is the primary unit read by the simulator.

### Attributes

| Attribute | Type | Unit | Description |
|---|---|---|---|
| `id` | string | — | Unique identifier; referenced in the YAML config |
| `speed` | decimal | km/h | Formation maximum speed (minimum of all vehicles) |
| `bruttoWeight` | decimal | t | Gross mass of the complete consist |
| `tareWeight` | decimal | t | Empty operational mass of the complete consist. **Used as formation mass in physics calculations.** |
| `nettoWeight` | decimal | t | Total payload capacity |
| `length` | decimal | m | Overall formation length |
| `numberOfAxles` | int | — | Total axle count |
| `numberOfWagons` | int | — | Number of vehicles in the consist |

### `trainOrder` (1..*)

Lists which vehicles form the consist and in what order.

| Attribute | Type | Description |
|---|---|---|
| `orderNumber` | int | Position in the consist, starting at 1 |
| `vehicleRef` | IDREF | References the `id` of a `vehicle` element |
| `orientation` | enum | `"normal"` (default) or `"reverse"` |

### `trainEngine`

Aggregated traction properties for the formation.

| Attribute | Type | Unit | Description |
|---|---|---|---|
| `maxAcceleration` | decimal | m/s² | Maximum achievable acceleration |
| `meanAcceleration` | decimal | m/s² | Mean acceleration over a departure manoeuvre |

Contains a `tractionMode` child (same structure as `powerMode` on a vehicle).

### `trainBrakes` (0..*)

Formation-level brake system. Same attributes as vehicle `vehicleBrakes`.

The key attribute for the physics engine is `meanDeceleration` (m/s²). The braking force is derived as:

```
F_braking = meanDeceleration × formation_mass_kg
```

**Example:**
```xml
<rail3:trainBrakes meanDeceleration="0.9" />
```

### `trainResistance`

Formation-level driving resistance. Extends the vehicle-level element with a **Davies formula** block.

#### `daviesFormulaFactors`

The Davis equation expresses rolling resistance as a polynomial in speed:

```
R(v) = A + B·v + C·v²
```

where `v` is in **km/h** and `R` is in **N**.

| Attribute | Type | Unit | Description |
|---|---|---|---|
| `constantFactorA` | decimal | N | Constant (speed-independent) term — bearing friction, track irregularity |
| `speedDependentFactorB` | decimal | N/(km/h) | Linear term — flange friction, lateral oscillation |
| `squareSpeedDependentFactorC` | decimal | N/(km/h)² | Aerodynamic drag term |
| `massDependent` | xs:boolean | — | `"true"` if A and B scale with mass |

**Example:**
```xml
<rail3:trainResistance>
  <rail3:daviesFormulaFactors
    constantFactorA="3800"
    speedDependentFactorB="45"
    squareSpeedDependentFactorC="2.5"
    massDependent="false" />
</rail3:trainResistance>
```

---

## Unit Conversions

The physics engine (`src/physics.rs`) works in SI units (m, s, kg, N). RailML uses mixed units that require conversion.

### Mass

RailML weights are in **tonnes**. Conversion:

```
mass_kg = tareWeight_t × 1000
```

### Davis C coefficient (aerodynamic drag)

`squareSpeedDependentFactorC` is in N/(km/h)². The drag force in N is:

```
F_aero = C × v_kmh²
```

The physics engine computes drag as `drag_coeff × v_ms²` (v in m/s). Converting:

```
v_kmh = v_ms × 3.6
F_aero = C × (v_ms × 3.6)² = C × 12.96 × v_ms²
```

Therefore:

```
drag_coeff [kg/m] = C [N/(km/h)²] × 12.96
```

### Summary table

| RailML field | RailML unit | Physics engine field | SI unit | Conversion |
|---|---|---|---|---|
| `formation/@tareWeight` | t | `mass` | kg | × 1000 |
| `tractionData/info/@tractivePower` | W | `power` | W | × 1 |
| `tractionData/info/@maxTractiveEffort` | N | `traction_force_at_standstill` | N | × 1 |
| `formation/@speed` | km/h | `max_speed` | km/h | × 1 |
| `daviesFormulaFactors/@squareSpeedDependentFactorC` | N/(km/h)² | `drag_coeff` | kg/m | × 12.96 |
| `trainBrakes/@meanDeceleration` | m/s² | `braking_force` | N | × mass_kg |

---

## Physics Mapping

When the simulator loads a formation, it extracts a `TrainDescription` struct (defined in `src/model.rs`):

```rust
TrainDescription {
    power:                        f64,  // W
    traction_force_at_standstill: f64,  // N
    max_speed:                    f64,  // km/h
    mass:                         f64,  // kg
    drag_coeff:                   f64,  // kg/m  (aerodynamic only)
    braking_force:                f64,  // N
}
```

The extraction path through the XML:

| `TrainDescription` field | XPath from `formation` |
|---|---|
| `max_speed` | `@speed` |
| `mass` | `@tareWeight` × 1000 |
| `power` | `trainEngine/tractionMode[@isPrimaryMode='true']/tractionData/info/@tractivePower` |
| `traction_force_at_standstill` | `trainEngine/tractionMode[@isPrimaryMode='true']/tractionData/info/@maxTractiveEffort` |
| `drag_coeff` | `trainResistance/daviesFormulaFactors/@squareSpeedDependentFactorC` × 12.96 |
| `braking_force` | `trainBrakes/@meanDeceleration` × mass_kg |

The `constantFactorA` and `speedDependentFactorB` Davis terms are not currently used — rolling resistance is approximated as a fixed 2 N/kN (`0.002 × mass × g`). The aerodynamic C term is used through `drag_coeff`.

---

## Complete Annotated Example

Below is the `output.xml` generated by `uv run make-railml-rollingstock`, trimmed to one coach for readability.

```xml
<?xml version='1.0' encoding='utf-8'?>
<rail3:railML xmlns:rail3="https://www.railml.org/schemas/3.3" version="3.3">
  <rail3:rollingstock>

    <!-- ── VEHICLES ─────────────────────────────────────────────────── -->
    <rail3:vehicles>

      <!-- Class 66 diesel-electric locomotive -->
      <rail3:vehicle
          id="vehicle_class66"
          speed="120"           <!-- max speed: 120 km/h -->
          bruttoWeight="130"    <!-- gross mass: 130 t -->
          tareWeight="130"      <!-- tare mass: 130 t (same, no payload) -->
          length="21.34"        <!-- 21.34 m over buffers -->
          numberOfDrivenAxles="6"
          numberOfNonDrivenAxles="0"
          adhesionWeight="130"
          rotatingMassFactor="1.15">

        <!-- UIC number -->
        <rail3:designator register="UIC" entry="92 70 0 066 001-1" />

        <!-- This vehicle is a locomotive -->
        <rail3:vehiclePart id="vp_class66_body" partOrder="1" category="locomotive" />

        <!-- Diesel-electric propulsion -->
        <rail3:engine>
          <rail3:powerMode mode="diesel" isPrimaryMode="true">
            <rail3:tractionData>
              <!-- Scalar summary used by the physics engine -->
              <rail3:info maxTractiveEffort="270000" tractivePower="2420000" />
              <!-- Speed-dependent curve (informational) -->
              <rail3:details>
                <rail3:tractiveEffort>
                  <rail3:valueTable xValueName="speed" xValueUnit="km/h"
                                    yValueName="tractiveEffort" yValueUnit="N">
                    <rail3:valueLine xValue="0"><rail3:value yValue="270000" /></rail3:valueLine>
                    <rail3:valueLine xValue="120"><rail3:value yValue="66000" /></rail3:valueLine>
                  </rail3:valueTable>
                </rail3:tractiveEffort>
              </rail3:details>
            </rail3:tractionData>
          </rail3:powerMode>
        </rail3:engine>

        <!-- Vehicle-level braking (deceleration curve) -->
        <rail3:brakes>
          <rail3:decelerationTable>
            <rail3:valueTable xValueName="speed" xValueUnit="km/h"
                              yValueName="deceleration" yValueUnit="m/s/s">
              <rail3:valueLine xValue="0"><rail3:value yValue="0.90" /></rail3:valueLine>
              <rail3:valueLine xValue="120"><rail3:value yValue="0.65" /></rail3:valueLine>
            </rail3:valueTable>
          </rail3:decelerationTable>
        </rail3:brakes>

        <!-- Aerodynamic properties -->
        <rail3:drivingResistance tunnelFactor="1.5">
          <rail3:info
              airDragCoefficient="0.80"  <!-- Cd -->
              crossSectionArea="9.5"     <!-- m² -->
              rollingResistance="1.5" /> <!-- N/kN -->
        </rail3:drivingResistance>

      </rail3:vehicle>

      <!-- Mk3 passenger coach (one of several) -->
      <rail3:vehicle
          id="vehicle_mk3_01"
          speed="200"
          bruttoWeight="48"   <!-- 33 t tare + 15 t passengers -->
          tareWeight="33"
          length="23.0"
          numberOfDrivenAxles="0"
          numberOfNonDrivenAxles="4">
        <rail3:designator register="operator" entry="Mk3-001" />
        <rail3:vehiclePart id="vp_mk3_01_body" partOrder="1" category="passengerCoach" />
        <rail3:brakes>
          <rail3:decelerationTable>
            <rail3:valueTable xValueName="speed" xValueUnit="km/h"
                              yValueName="deceleration" yValueUnit="m/s/s">
              <rail3:valueLine xValue="0"><rail3:value yValue="0.80" /></rail3:valueLine>
              <rail3:valueLine xValue="120"><rail3:value yValue="0.65" /></rail3:valueLine>
            </rail3:valueTable>
          </rail3:decelerationTable>
        </rail3:brakes>
        <rail3:drivingResistance>
          <rail3:info airDragCoefficient="0.60" crossSectionArea="9.0" rollingResistance="1.2" />
        </rail3:drivingResistance>
      </rail3:vehicle>

    </rail3:vehicles>

    <!-- ── FORMATIONS ────────────────────────────────────────────────── -->
    <rail3:formations>

      <rail3:formation
          id="formation_class66_mk3"  <!-- referenced in the YAML config -->
          speed="120"                 <!-- limited by loco, not coaches -->
          bruttoWeight="226"          <!-- 130 + 48 × 4 (example) -->
          tareWeight="163"            <!-- 130 + 33 × ... -->
          length="147.34">

        <rail3:designator register="operator" entry="1A23-consist" />

        <!-- Vehicle order in the consist -->
        <rail3:trainOrder orderNumber="1" vehicleRef="vehicle_class66" />
        <rail3:trainOrder orderNumber="2" vehicleRef="vehicle_mk3_01" />
        <!-- … more coaches … -->

        <!-- Formation-level traction (aggregated from loco) -->
        <rail3:trainEngine maxAcceleration="0.40" meanAcceleration="0.25">
          <rail3:tractionMode mode="diesel" isPrimaryMode="true">
            <rail3:tractionData>
              <rail3:info maxTractiveEffort="270000" tractivePower="2420000" />
            </rail3:tractionData>
          </rail3:tractionMode>
        </rail3:trainEngine>

        <!-- Formation-level braking — meanDeceleration is used by the physics engine -->
        <rail3:trainBrakes meanDeceleration="0.9" />

        <!-- Davis equation for the complete consist -->
        <rail3:trainResistance tunnelFactor="1.8">
          <rail3:daviesFormulaFactors
              constantFactorA="3800"             <!-- N -->
              speedDependentFactorB="45"         <!-- N/(km/h) -->
              squareSpeedDependentFactorC="2.5"  <!-- N/(km/h)² → ×12.96 → kg/m -->
              massDependent="false" />
        </rail3:trainResistance>

      </rail3:formation>

    </rail3:formations>
  </rail3:rollingstock>
</rail3:railML>
```

---

## Generating and Validating RailML Files

```bash
# Generate a Class 66 + 5 × Mk3 consist, write to output.xml, validate against XSD:
uv run make-railml-rollingstock output.xml

# Change the number of coaches:
uv run make-railml-rollingstock output.xml --coaches 8
```

Validation uses `xmlschema` against `railml/railML-3.3-SR1/source/schema/railml3.xsd`. The Dublin Core stub is loaded automatically.

---

## Known Limitations

- Only the **rollingstock** sub-schema is implemented. Infrastructure, timetable, interlocking, and other sub-schemas are not parsed.
- Vehicle-level driving resistance (`drivingResistance/info`) is not summed into formation resistance; only `formation/trainResistance/daviesFormulaFactors/@squareSpeedDependentFactorC` drives the `drag_coeff`.
- The Davis A (constant) and B (linear) terms are not wired to the physics engine; rolling resistance is approximated as 0.002 × mass × g.
- The speed-dependent tractive-effort curve in `details` is stored in the XML but the physics engine uses only the scalar `info/@maxTractiveEffort` and `info/@tractivePower`.
