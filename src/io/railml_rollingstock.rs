use crate::core::model::TrainDescription;
use crate::io::xml_util::{parse_attr, required_child, required_descendant};
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
/// | `davis_a`                       | `formation/trainResistance/daviesFormulaFactors/@constantFactorA` (N)         |
/// | `davis_b`                       | `formation/trainResistance/daviesFormulaFactors/@speedDependentFactorB` × 3.6 (N·s/m) |
/// | `drag_coeff`                    | `formation/trainResistance/daviesFormulaFactors/@squareSpeedDependentFactorC` × 12.96 (N/(km/h)² → kg/m) |
/// | `braking_force`                 | `formation/trainBrakes/@meanDeceleration` × mass (N)                         |
pub fn load_formation(path: &Path, formation_id: &str) -> Result<TrainDescription, String> {
    let xml = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read '{}': {e}", path.display()))?;
    let doc = roxmltree::Document::parse(&xml)
        .map_err(|e| format!("XML parse error in '{}': {e}", path.display()))?;

    let formation = doc
        .descendants()
        .find(|n| n.has_tag_name((NS, "formation")) && n.attribute("id") == Some(formation_id))
        .ok_or_else(|| format!("formation '{formation_id}' not found in '{}'", path.display()))?;

    let ctx = format!("formation '{formation_id}' in '{}'", path.display());

    // --- Basic formation attributes ---

    let max_speed: f64 = parse_attr(formation, "speed", &ctx)?;
    let tare_t: f64 = parse_attr(formation, "tareWeight", &ctx)?;
    let mass = tare_t * 1_000.0; // tonnes → kg

    // --- Traction: trainEngine > tractionMode[primary] > tractionData > info ---

    let train_engine = required_child(formation, NS, "trainEngine", &ctx)?;
    let traction_mode = train_engine
        .children()
        .find(|n| {
            n.has_tag_name((NS, "tractionMode"))
                && n.attribute("isPrimaryMode").map_or(true, |v| v == "true")
        })
        .ok_or_else(|| format!("{ctx}: no primary <tractionMode> in <trainEngine>"))?;
    let traction_data = required_child(traction_mode, NS, "tractionData", &ctx)?;
    let traction_info = required_child(traction_data, NS, "info", &ctx)?;

    let power: f64 = parse_attr(traction_info, "tractivePower", &ctx)?;
    let traction_force_at_standstill: f64 = parse_attr(traction_info, "maxTractiveEffort", &ctx)?;

    // --- Driving resistance: trainResistance > daviesFormulaFactors ---
    //
    // Davis B [N/(km/h)] → N·s/m:  B × v_kmh = B × (v_ms × 3.6)  ⟹  davis_b = B × 3.6
    // Davis C [N/(km/h)²] → kg/m:  C × v_kmh² = C × (v_ms × 3.6)²  ⟹  drag_coeff = C × 12.96

    let davies = required_descendant(formation, NS, "daviesFormulaFactors", &ctx)?;
    let davis_a: f64 = parse_attr(davies, "constantFactorA", &ctx)?;
    let davis_b: f64 = parse_attr::<f64>(davies, "speedDependentFactorB", &ctx)? * 3.6;
    let drag_coeff: f64 = parse_attr::<f64>(davies, "squareSpeedDependentFactorC", &ctx)? * 12.96;

    // --- Braking: trainBrakes[@meanDeceleration] × mass ---

    let train_brakes = required_child(formation, NS, "trainBrakes", &ctx)?;
    let mean_decel: f64 = parse_attr(train_brakes, "meanDeceleration", &ctx)?;
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
