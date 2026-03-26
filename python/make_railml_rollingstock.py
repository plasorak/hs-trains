#!/usr/bin/env python3
"""Generate a valid RailML 3.3 rollingstock XML file using Pydantic-XML models.

Models a diesel locomotive (Class 66 style) hauling Mk3 passenger coaches,
then serialises the result to a schema-conformant railML 3.3 XML file.

Usage
-----
    uv run make-railml-rollingstock output.xml
    uv run make-railml-rollingstock output.xml --coaches 8
"""

import argparse
import xml.etree.ElementTree as ET
from decimal import Decimal
from pathlib import Path
from xml.etree.ElementTree import indent, fromstring, ElementTree

import xmlschema
from model.rollingstock import (
    BrakeEffortCurve,
    Brakes,
    DaviesFormula,
    DecelerationCurve,
    Designator,
    DrivingResistance,
    DrivingResistanceInfo,
    Engine,
    Formation,
    Formations,
    PowerMode,
    RailML,
    Rollingstock,
    TractiveEffortCurve,
    TractionData,
    TractionDetails,
    TractionInfo,
    TrainDrivingResistance,
    TrainEngine,
    TrainOrder,
    TrainTractionMode,
    Value,
    ValueLine,
    ValueTable,
    Vehicle,
    VehiclePart,
    Vehicles,
)

# Root XSD for railML 3.3 (includes all sub-schemas)
_RAILML_SCHEMA_RELATIVE = Path("railml/railML-3.3-SR1/source/schema/railml3.xsd")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _speed_curve(
    curve_class: type,
    y_name: str,
    y_unit: str,
    *points: tuple[float, float],
) -> object:
    """Build a typed curve from (speed_km_h, y) pairs."""
    return curve_class(
        value_table=ValueTable(
            x_value_name="speed",
            x_value_unit="km/h",
            y_value_name=y_name,
            y_value_unit=y_unit,
            value_lines=[
                ValueLine(x_value=Decimal(str(x)), values=[Value(y_value=Decimal(str(y)))])
                for x, y in points
            ],
        )
    )


# ---------------------------------------------------------------------------
# Sample data — Class 66 locomotive + BR Mk3 coaches
# ---------------------------------------------------------------------------


def make_class66() -> Vehicle:
    """Class 66 diesel-electric freight/passenger loco (approximate values)."""
    te_curve = _speed_curve(
        TractiveEffortCurve,
        "tractiveEffort", "N",
        (0, 270_000),
        (20, 270_000),
        (40, 200_000),
        (60, 133_000),
        (80, 100_000),
        (100, 80_000),
        (120, 66_000),
    )
    decel_curve = _speed_curve(
        DecelerationCurve,
        "deceleration", "m/s/s",
        (0, 0.90),
        (40, 0.85),
        (80, 0.75),
        (120, 0.65),
    )
    return Vehicle(
        id="vehicle_class66",
        speed=Decimal("120"),
        brutto_weight=Decimal("130"),
        tare_weight=Decimal("130"),
        length=Decimal("21.34"),
        number_of_driven_axles=6,
        number_of_non_driven_axles=0,
        adhesion_weight=Decimal("130"),
        rotating_mass_factor=Decimal("1.15"),
        designators=[
            Designator(register_name="UIC", entry="92 70 0 066 001-1"),
            Designator(register_name="operator", entry="Class66-001"),
        ],
        vehicle_parts=[VehiclePart(id="vp_class66_body", part_order=1, category="locomotive")],
        engines=[
            Engine(
                power_modes=[
                    PowerMode(
                        mode="diesel",
                        is_primary_mode=True,
                        traction_data=TractionData(
                            info=TractionInfo(
                                max_tractive_effort=Decimal("270000"),
                                tractive_power=Decimal("2420000"),
                            ),
                            details=TractionDetails(tractive_effort=te_curve),
                        ),
                    )
                ]
            )
        ],
        brakes=[Brakes(deceleration_table=decel_curve)],
        driving_resistance=DrivingResistance(
            info=DrivingResistanceInfo(
                air_drag_coefficient=Decimal("0.80"),
                cross_section_area=Decimal("9.5"),
                rolling_resistance=Decimal("1.5"),
            ),
            tunnel_factor=Decimal("1.5"),
        ),
    )


def make_mk3_coach(n: int) -> Vehicle:
    """BR Mk3 passenger coach."""
    decel_curve = _speed_curve(
        DecelerationCurve,
        "deceleration", "m/s/s",
        (0, 0.80),
        (60, 0.75),
        (120, 0.65),
        (200, 0.55),
    )
    return Vehicle(
        id=f"vehicle_mk3_{n:02d}",
        speed=Decimal("200"),
        brutto_weight=Decimal("48"),
        tare_weight=Decimal("33"),
        length=Decimal("23.0"),
        number_of_driven_axles=0,
        number_of_non_driven_axles=4,
        designators=[Designator(register_name="operator", entry=f"Mk3-{n:03d}")],
        vehicle_parts=[
            VehiclePart(id=f"vp_mk3_{n:02d}_body", part_order=1, category="passengerCoach")
        ],
        brakes=[Brakes(deceleration_table=decel_curve)],
        driving_resistance=DrivingResistance(
            info=DrivingResistanceInfo(
                air_drag_coefficient=Decimal("0.60"),
                cross_section_area=Decimal("9.0"),
                rolling_resistance=Decimal("1.2"),
            )
        ),
    )


def make_formation(loco: Vehicle, coaches: list[Vehicle]) -> Formation:
    """Loco-hauled formation: locomotive at position 1, coaches following."""
    train_orders = [TrainOrder(order_number=1, vehicle_ref=loco.id)] + [
        TrainOrder(order_number=i + 2, vehicle_ref=c.id) for i, c in enumerate(coaches)
    ]

    def _sum(vehicles: list[Vehicle], attr: str) -> Decimal:
        return sum((getattr(v, attr) or Decimal("0")) for v in vehicles)

    all_vehicles = [loco] + coaches
    return Formation(
        id="formation_class66_mk3",
        brutto_weight=_sum(all_vehicles, "brutto_weight"),
        tare_weight=_sum(all_vehicles, "tare_weight"),
        length=_sum(all_vehicles, "length"),
        speed=Decimal("120"),
        designators=[Designator(register_name="operator", entry="1A23-consist")],
        train_orders=train_orders,
        train_engines=[
            TrainEngine(
                max_acceleration=Decimal("0.40"),
                mean_acceleration=Decimal("0.25"),
                traction_mode=TrainTractionMode(mode="diesel", is_primary_mode=True),
            )
        ],
        train_resistance=TrainDrivingResistance(
            davies_formula_factors=DaviesFormula(
                constant_factor_a=Decimal("3800"),
                speed_dependent_factor_b=Decimal("45"),
                square_speed_dependent_factor_c=Decimal("2.5"),
                mass_dependent=False,
            ),
            tunnel_factor=Decimal("1.8"),
        ),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a RailML 3.3 rollingstock XML file."
    )
    parser.add_argument("output", help="Output XML file path")
    parser.add_argument(
        "--coaches",
        type=int,
        default=5,
        metavar="N",
        help="Number of Mk3 coaches in the formation (default: 5)",
    )
    args = parser.parse_args()

    loco = make_class66()
    coaches = [make_mk3_coach(i + 1) for i in range(args.coaches)]
    formation = make_formation(loco, coaches)

    railml = RailML(
        rollingstock=Rollingstock(
            vehicles=Vehicles(vehicles=[loco, *coaches]),
            formations=Formations(formations=[formation]),
        )
    )

    NS = "https://www.railml.org/schemas/3.3"
    ET.register_namespace("rail3", NS)

    xml_bytes = railml.to_xml(encoding="unicode", exclude_none=True)
    root = fromstring(xml_bytes)
    indent(root, space="  ")
    ElementTree(root).write(args.output, encoding="unicode", xml_declaration=True)

    print(f"Written RailML 3.3 rollingstock to '{args.output}'")
    print(f"  Vehicles : 1 Class 66 + {args.coaches} Mk3 coaches")
    print(f"  Formation: {formation.id}")
    print(f"  Brutto weight : {formation.brutto_weight} t")
    print(f"  Total length  : {formation.length} m")
    print(f"  Max speed     : {formation.speed} km/h")

    schema_path = Path.cwd() / _RAILML_SCHEMA_RELATIVE
    dcterms_stub = str(schema_path.parent / "dcterms_stub.xsd")
    print(f"\nValidating against {schema_path.name} ...")
    xs = xmlschema.XMLSchema(
        str(schema_path),
        locations={"http://purl.org/dc/terms/": dcterms_stub},
    )
    xs.validate(args.output)
    print("  OK — document is schema-valid.")


if __name__ == "__main__":
    main()
