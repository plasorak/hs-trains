use crate::model::TrainDescription;
use std::path::Path;

const NS: &str = "https://www.railml.org/schemas/3.3";

/// Parse a RailML 3.3 file and extract a [`TrainDescription`] for the named formation.
///
/// ## Field mapping
///
/// | `TrainDescription` field        | RailML source                                                                 |
/// |---------------------------------|-------------------------------------------------------------------------------|
/// | `max_speed`                     | `formation/@speed`  (km/h)                                                    |
/// | `mass`                          | `formation/@tareWeight` (tonnes) × 1 000 → kg                                |
/// | `power`                         | `formation/trainEngine/tractionMode[@isPrimaryMode]/tractionData/info/@tractivePower` (W) |
/// | `traction_force_at_standstill`  | …/`info/@maxTractiveEffort` (N)                                               |
/// | `drag_coeff`                    | `formation/trainResistance/daviesFormulaFactors/@squareSpeedDependentFactorC` × 12.96 (converts N/(km/h)² → kg/m) |
/// | `braking_force`                 | `formation/trainBrakes/@meanDeceleration` × mass (N)                         |
pub fn load_formation(path: &Path, formation_id: &str) -> Result<TrainDescription, String> {
    let xml = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read '{}': {e}", path.display()))?;
    let doc = roxmltree::Document::parse(&xml)
        .map_err(|e| format!("XML parse error in '{}': {e}", path.display()))?;

    let formation = doc
        .descendants()
        .find(|n| n.has_tag_name((NS, "formation")) && n.attribute("id") == Some(formation_id))
        .ok_or_else(|| {
            format!(
                "formation '{}' not found in '{}'",
                formation_id,
                path.display()
            )
        })?;

    // Closure that prefixes every error with formation / file context.
    let err = |msg: &str| {
        format!(
            "formation '{}' in '{}': {msg}",
            formation_id,
            path.display()
        )
    };

    // --- Basic formation attributes ---

    let max_speed: f64 = formation
        .attribute("speed")
        .ok_or_else(|| err("missing 'speed' attribute"))?
        .parse()
        .map_err(|_| err("invalid 'speed' value"))?;

    let tare_t: f64 = formation
        .attribute("tareWeight")
        .ok_or_else(|| err("missing 'tareWeight' attribute"))?
        .parse()
        .map_err(|_| err("invalid 'tareWeight' value"))?;
    let mass = tare_t * 1_000.0; // tonnes → kg

    // --- Traction: trainEngine > tractionMode[primary] > tractionData > info ---

    let train_engine = formation
        .children()
        .find(|n| n.has_tag_name((NS, "trainEngine")))
        .ok_or_else(|| err("no <trainEngine> element"))?;

    let traction_mode = train_engine
        .children()
        .find(|n| {
            n.has_tag_name((NS, "tractionMode"))
                && n.attribute("isPrimaryMode").map_or(true, |v| v == "true")
        })
        .ok_or_else(|| err("no primary <tractionMode> in <trainEngine>"))?;

    let traction_data = traction_mode
        .children()
        .find(|n| n.has_tag_name((NS, "tractionData")))
        .ok_or_else(|| err("no <tractionData> in <tractionMode>"))?;

    let traction_info = traction_data
        .children()
        .find(|n| n.has_tag_name((NS, "info")))
        .ok_or_else(|| err("no <info> in <tractionData>"))?;

    let power: f64 = traction_info
        .attribute("tractivePower")
        .ok_or_else(|| err("<info> missing 'tractivePower'"))?
        .parse()
        .map_err(|_| err("invalid 'tractivePower' value"))?;

    let traction_force_at_standstill: f64 = traction_info
        .attribute("maxTractiveEffort")
        .ok_or_else(|| err("<info> missing 'maxTractiveEffort'"))?
        .parse()
        .map_err(|_| err("invalid 'maxTractiveEffort' value"))?;

    // --- Driving resistance: trainResistance > daviesFormulaFactors ---
    //
    // The Davis C coefficient is in N/(km/h)².  The physics engine's drag_coeff
    // is in kg/m (= N·s²/m²).  Since v_kmh = v_ms × 3.6:
    //   C_davis × v_kmh² = (C_davis × 3.6²) × v_ms²
    // so  drag_coeff [kg/m] = C_davis [N/(km/h)²] × 12.96

    let davies = formation
        .descendants()
        .find(|n| n.has_tag_name((NS, "daviesFormulaFactors")))
        .ok_or_else(|| err("no <daviesFormulaFactors> inside <trainResistance>"))?;

    // Davis A [N] — constant mechanical resistance; stored as-is.
    let davis_a: f64 = davies
        .attribute("constantFactorA")
        .ok_or_else(|| err("<daviesFormulaFactors> missing 'constantFactorA'"))?
        .parse()
        .map_err(|_| err("invalid 'constantFactorA' value"))?;

    // Davis B [N/(km/h)] — linear speed term.  Convert to SI (N·s/m):
    //   B_davis × v_kmh = B_davis × (v_ms × 3.6)
    // so  davis_b [N·s/m] = B_davis × 3.6
    let b_davis: f64 = davies
        .attribute("speedDependentFactorB")
        .ok_or_else(|| err("<daviesFormulaFactors> missing 'speedDependentFactorB'"))?
        .parse()
        .map_err(|_| err("invalid 'speedDependentFactorB' value"))?;
    let davis_b = b_davis * 3.6;

    // Davis C [N/(km/h)²] — aerodynamic drag.  Convert to kg/m:
    //   C_davis × v_kmh² = (C_davis × 3.6²) × v_ms²
    // so  drag_coeff [kg/m] = C_davis × 12.96
    let c_davis: f64 = davies
        .attribute("squareSpeedDependentFactorC")
        .ok_or_else(|| err("<daviesFormulaFactors> missing 'squareSpeedDependentFactorC'"))?
        .parse()
        .map_err(|_| err("invalid 'squareSpeedDependentFactorC' value"))?;
    let drag_coeff = c_davis * 12.96;

    // --- Braking: trainBrakes[@meanDeceleration] × mass ---

    let mean_decel: f64 = formation
        .children()
        .find(|n| n.has_tag_name((NS, "trainBrakes")))
        .and_then(|tb| tb.attribute("meanDeceleration"))
        .ok_or_else(|| err("no <trainBrakes meanDeceleration=…> element"))?
        .parse()
        .map_err(|_| err("invalid 'meanDeceleration' value"))?;
    let braking_force = mean_decel * mass;

    Ok(TrainDescription {
        power,
        traction_force_at_standstill,
        max_speed,
        mass,
        davis_a,
        davis_b,
        drag_coeff,
        braking_force,
    })
}
