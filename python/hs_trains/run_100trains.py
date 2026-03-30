#!/usr/bin/env python3
"""Generate a multi-train RailML rollingstock file and a matching simulation config.

Each train becomes a formation in the RailML file.  Physics parameters are sampled
randomly within realistic fleet ranges; the YAML config references those formations
via ``kind: railml``.  The old inline ``kind: physics`` config format is no longer
used.

Field mapping (physics param → RailML location):
  power_w           → formation/trainEngine/tractionMode/tractionData/info/@tractivePower
  traction_n        → …/info/@maxTractiveEffort
  max_speed_kmh     → formation/@speed
  mass_kg           → formation/@tareWeight  (÷ 1 000 → tonnes)
  drag_coeff [kg/m] → formation/trainResistance/daviesFormulaFactors/@squareSpeedDependentFactorC
                        (÷ 12.96 converts kg/m → N/(km/h)²)
  braking_force_n   → formation/trainBrakes/@meanDeceleration  (÷ mass_kg → m/s²)

Usage
-----
    uv run scripts/run_100trains.py
    uv run scripts/run_100trains.py --trains 50 --dt 0.5 --output my_output.parquet
"""

import random
import xml.etree.ElementTree as ET
from decimal import Decimal
from pathlib import Path
from typing import Annotated
from xml.etree.ElementTree import ElementTree, fromstring, indent

import typer
import yaml

from hs_trains.model.rollingstock import (
    DaviesFormula,
    Formation,
    FormationBrakeSystem,
    Formations,
    RailML,
    Rollingstock,
    TractionData,
    TractionInfo,
    TrainDrivingResistance,
    TrainEngine,
    TrainTractionMode,
)

SEED = 42

# Realistic baseline ranges for a mixed fleet.
FLEET_PARAMS = {
    "power_w":         (1_500_000, 3_000_000),  # W
    "traction_n":      (200_000,   600_000),     # N at standstill
    "max_speed_kmh":   (80,        160),         # km/h
    "mass_kg":         (500_000, 3_000_000),     # kg
    "drag_coeff":      (5.0,       15.0),        # kg/m
    "braking_force_n": (300_000, 1_000_000),     # N
    "gradient":        (-0.02,     0.02),        # rise/run
    "wind_speed_ms":   (-5.0,      5.0),         # m/s (head-wind positive)
    "power_ratio":     (0.5,       1.0),         # throttle
}

NS = "https://www.railml.org/schemas/3.3"


def make_formation(index: int, rng: random.Random) -> tuple[Formation, dict]:
    """Return (RailML formation, YAML train-config dict) for one randomly sampled train."""
    r = rng.uniform
    p = FLEET_PARAMS

    # Sample physics params — same order as the old YAML generator to preserve
    # reproducibility with the same seed.
    power_w = round(r(*p["power_w"]))
    traction_n = round(r(*p["traction_n"]))
    max_speed_kmh = round(r(*p["max_speed_kmh"]), 1)
    mass_kg = round(r(*p["mass_kg"]))
    drag_coeff = round(r(*p["drag_coeff"]), 2)
    braking_force_n = round(r(*p["braking_force_n"]))
    gradient = round(r(*p["gradient"]), 4)
    wind_speed = round(r(*p["wind_speed_ms"]), 2)
    power_ratio = round(r(*p["power_ratio"]), 2)

    formation_id = f"formation_train_{index:03d}"

    # Davis C coefficient: drag_coeff [kg/m] = C [N/(km/h)²] × 12.96
    c_davis = round(drag_coeff / 12.96, 6)
    mean_decel = round(braking_force_n / mass_kg, 6)  # m/s²

    formation = Formation(
        id=formation_id,
        speed=Decimal(str(max_speed_kmh)),
        tare_weight=Decimal(str(round(mass_kg / 1_000, 3))),
        train_engines=[
            TrainEngine(
                traction_mode=TrainTractionMode(
                    mode="diesel",
                    is_primary_mode=True,
                    traction_data=TractionData(
                        info=TractionInfo(
                            max_tractive_effort=Decimal(str(traction_n)),
                            tractive_power=Decimal(str(power_w)),
                        )
                    ),
                )
            )
        ],
        train_brakes=[FormationBrakeSystem(mean_deceleration=Decimal(str(mean_decel)))],
        train_resistance=TrainDrivingResistance(
            davies_formula_factors=DaviesFormula(
                # A and B terms are zero; the Rust engine handles rolling resistance
                # internally (0.002 × mass × g), so only the aerodynamic C term is needed.
                constant_factor_a=Decimal("0"),
                speed_dependent_factor_b=Decimal("0"),
                square_speed_dependent_factor_c=Decimal(str(c_davis)),
            )
        ),
    )

    train_config = {
        "id": f"train_{index:03d}",
        "kind": "railml",
        "formation_id": formation_id,
        "environment": {"gradient": gradient, "wind_speed": wind_speed},
        "driver": {"power_ratio": power_ratio, "brake_ratio": 0.0},
    }

    return formation, train_config


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="Config YAML file to write")] = Path("config/config_100trains.yaml"),
    output: Annotated[Path, typer.Option(help="Output Parquet file")] = Path("output_100trains.parquet"),
    railml: Annotated[Path, typer.Option(help="RailML rollingstock file to write")] = Path("config/rollingstock_100trains.xml"),
    trains: Annotated[int, typer.Option(help="Number of trains")] = 100,
    dt: Annotated[float, typer.Option(help="Time step in seconds")] = 1.0,
    duration: Annotated[float, typer.Option(help="Simulation duration in hours")] = 1.0,
    seed: Annotated[int, typer.Option(help="Random seed for reproducibility")] = SEED,
    flush_rows: Annotated[int, typer.Option(help="Max rows buffered before a Parquet flush")] = 1_000_000,
) -> None:
    """Generate a RailML rollingstock file and physics config for N trains."""
    rng = random.Random(seed)

    formations = []
    train_configs = []
    for i in range(1, trains + 1):
        formation, train_cfg = make_formation(i, rng)
        formations.append(formation)
        train_cfg["railml_file"] = str(railml)
        train_configs.append(train_cfg)

    # Write RailML file.
    railml_doc = RailML(
        rollingstock=Rollingstock(formations=Formations(formations=formations))
    )
    ET.register_namespace("rail3", NS)
    xml_bytes = railml_doc.to_xml(encoding="unicode", exclude_none=True)
    root = fromstring(xml_bytes)
    indent(root, space="  ")
    railml.parent.mkdir(parents=True, exist_ok=True)
    ElementTree(root).write(str(railml), encoding="unicode", xml_declaration=True)

    # Write YAML simulation config.
    simulation = {
        "simulation": {
            "time_step_s": dt,
            "duration_s": duration * 3_600,
            "flush_rows": flush_rows,
            "trains": train_configs,
        }
    }
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.dump(simulation, default_flow_style=False, sort_keys=False))

    expected_rows = int(duration * 3_600 / dt) * trains
    typer.echo(
        f"Wrote {railml}  ({trains} formations)"
    )
    typer.echo(
        f"Wrote {config}  "
        f"({trains} trains, {duration}h @ {dt}s steps, ~{expected_rows:,} output rows, "
        f"flush every {flush_rows:,} rows)"
    )
    typer.echo("\nTo run the simulation:")
    typer.echo("  cargo build --release")
    typer.echo(f"  ./target/release/hs-trains {config} {output}")


if __name__ == "__main__":
    app()
