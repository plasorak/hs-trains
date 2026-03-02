#!/usr/bin/env python3
"""Generate a 100-train physics config and print the command to run it.

Usage
-----
    uv run scripts/run_100trains.py
    uv run scripts/run_100trains.py --trains 50 --dt 0.5 --output my_output.parquet
"""

import random
from pathlib import Path
from typing import Annotated

import typer
import yaml


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


def make_train(index: int, rng: random.Random) -> dict:
    r = rng.uniform
    p = FLEET_PARAMS
    return {
        "id": f"train_{index:03d}",
        "kind": "physics",
        "train": {
            "power":                        round(r(*p["power_w"])),
            "traction_force_at_standstill": round(r(*p["traction_n"])),
            "max_speed":                    round(r(*p["max_speed_kmh"]), 1),
            "mass":                         round(r(*p["mass_kg"])),
            "drag_coeff":                   round(r(*p["drag_coeff"]), 2),
            "braking_force":                round(r(*p["braking_force_n"])),
        },
        "environment": {
            "gradient":   round(r(*p["gradient"]), 4),
            "wind_speed": round(r(*p["wind_speed_ms"]), 2),
        },
        "driver": {
            "power_ratio": round(r(*p["power_ratio"]), 2),
            "break_ratio": 0.0,
        },
    }


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="Config YAML file to write")] = Path("config/config_100trains.yaml"),
    output: Annotated[Path, typer.Option(help="Output Parquet file")] = Path("output_100trains.parquet"),
    trains: Annotated[int, typer.Option(help="Number of trains")] = 100,
    dt: Annotated[float, typer.Option(help="Time step in seconds")] = 1.0,
    duration: Annotated[float, typer.Option(help="Simulation duration in hours")] = 1.0,
    seed: Annotated[int, typer.Option(help="Random seed for reproducibility")] = SEED,
) -> None:
    """Generate a physics config for N trains and print the command to run it."""
    rng = random.Random(seed)

    simulation = {
        "simulation": {
            "time_step_s": dt,
            "duration_s": duration * 3_600,
            "trains": [make_train(i, rng) for i in range(1, trains + 1)],
        }
    }

    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.dump(simulation, default_flow_style=False, sort_keys=False))
    expected_rows = int(duration * 3_600 / dt) * trains
    typer.echo(
        f"Wrote {config}  "
        f"({trains} trains, {duration}h @ {dt}s steps, ~{expected_rows:,} output rows)"
    )

    typer.echo("\nTo run the simulation:")
    typer.echo(f"  cargo build --release")
    typer.echo(f"  ./target/release/rusty-trains {config} {output}")


if __name__ == "__main__":
    app()
