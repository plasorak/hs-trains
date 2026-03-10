"""Pydantic-XML models for RailML 3.3 rollingstock sub-schema.

Covers vehicles, formations, engines, brakes, driving resistance, and the
curve/value-table primitives used throughout the rollingstock schema.
"""

import uuid
from decimal import Decimal
from typing import Annotated, Literal, Optional

from pydantic import PlainSerializer
from pydantic_xml import BaseXmlModel, attr, element

NS = "https://www.railml.org/schemas/3.3"
_NS = "rail3"  # prefix alias used throughout
_NSMAP = {_NS: NS}


class _Base(BaseXmlModel, nsmap=_NSMAP):
    """Base class that propagates the railML namespace map to all submodels."""

# xs:boolean must be lowercase "true"/"false"
XmlBool = Annotated[bool, PlainSerializer(lambda v: "true" if v else "false", return_type=str)]


def _make_id() -> str:
    return f"id_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Curve primitives  (common3.xsd: ValueTable, ValueLine, Value, Curve)
# ---------------------------------------------------------------------------


class Value(_Base, tag="value", ns=_NS):
    y_value: Decimal = attr(name="yValue")


class ValueLine(_Base, tag="valueLine", ns=_NS):
    x_value: Decimal = attr(name="xValue")
    values: list[Value] = element(tag="value", ns=_NS, default_factory=list)


class ValueTable(_Base, tag="valueTable", ns=_NS):
    x_value_name: str = attr(name="xValueName")
    x_value_unit: str = attr(name="xValueUnit")
    y_value_name: str = attr(name="yValueName")
    y_value_unit: str = attr(name="yValueUnit")
    value_lines: list[ValueLine] = element(tag="valueLine", ns=_NS, default_factory=list)


# ---------------------------------------------------------------------------
# Common identifier  (generic3.xsd: Designator)
# ---------------------------------------------------------------------------


class Designator(_Base, tag="designator", ns=_NS):
    register_name: str = attr(name="register")
    entry: str = attr(name="entry")
    description: Optional[str] = attr(name="description", default=None)


# ---------------------------------------------------------------------------
# Driving resistance  (rollingstock3.xsd)
# ---------------------------------------------------------------------------


class DaviesFormula(_Base, tag="daviesFormulaFactors", ns=_NS):
    """Davis equation coefficients: R(N) = A + B·v + C·v²  (v in km/h)."""

    constant_factor_a: Decimal = attr(name="constantFactorA")
    speed_dependent_factor_b: Decimal = attr(name="speedDependentFactorB")
    square_speed_dependent_factor_c: Decimal = attr(name="squareSpeedDependentFactorC")
    mass_dependent: Optional[XmlBool] = attr(name="massDependent", default=None)


class DrivingResistanceInfo(_Base, tag="info", ns=_NS):
    air_drag_coefficient: Decimal = attr(name="airDragCoefficient")
    cross_section_area: Decimal = attr(name="crossSectionArea")
    rolling_resistance: Decimal = attr(name="rollingResistance")


class DrivingResistance(_Base, tag="drivingResistance", ns=_NS):
    tunnel_factor: Optional[Decimal] = attr(name="tunnelFactor", default=None)
    info: Optional[DrivingResistanceInfo] = element(tag="info", ns=_NS, default=None)


class TrainDrivingResistance(_Base, tag="trainResistance", ns=_NS):
    """Formation-level driving resistance, adds Davies formula factors."""

    tunnel_factor: Optional[Decimal] = attr(name="tunnelFactor", default=None)
    info: Optional[DrivingResistanceInfo] = element(tag="info", ns=_NS, default=None)
    davies_formula_factors: Optional[DaviesFormula] = element(
        tag="daviesFormulaFactors", ns=_NS, default=None
    )


# ---------------------------------------------------------------------------
# Engine / traction  (rollingstock3.xsd)
# ---------------------------------------------------------------------------


class TractionInfo(_Base, tag="info", ns=_NS):
    max_tractive_effort: Decimal = attr(name="maxTractiveEffort")
    tractive_power: Decimal = attr(name="tractivePower")


class TractiveEffortCurve(_Base, tag="tractiveEffort", ns=_NS):
    value_table: ValueTable = element(tag="valueTable", ns=_NS)


class TractionDetails(_Base, tag="details", ns=_NS):
    tractive_effort: TractiveEffortCurve = element(tag="tractiveEffort", ns=_NS)


class TractionData(_Base, tag="tractionData", ns=_NS):
    info: Optional[TractionInfo] = element(tag="info", ns=_NS, default=None)
    details: Optional[TractionDetails] = element(tag="details", ns=_NS, default=None)


class PowerMode(_Base, tag="powerMode", ns=_NS):
    mode: Literal["diesel", "electric", "battery"] = attr(name="mode")
    is_primary_mode: XmlBool = attr(name="isPrimaryMode", default=True)
    traction_data: Optional[TractionData] = element(tag="tractionData", ns=_NS, default=None)


class TrainTractionMode(_Base, tag="tractionMode", ns=_NS):
    mode: Literal["diesel", "electric", "battery"] = attr(name="mode")
    is_primary_mode: XmlBool = attr(name="isPrimaryMode", default=True)
    traction_data: Optional[TractionData] = element(tag="tractionData", ns=_NS, default=None)


class Engine(_Base, tag="engine", ns=_NS):
    power_modes: list[PowerMode] = element(tag="powerMode", ns=_NS, default_factory=list)


# ---------------------------------------------------------------------------
# Brakes  (rollingstock3.xsd)
# ---------------------------------------------------------------------------


class BrakeEffortCurve(_Base, tag="brakeEffort", ns=_NS):
    value_table: ValueTable = element(tag="valueTable", ns=_NS)


class DecelerationCurve(_Base, tag="decelerationTable", ns=_NS):
    value_table: ValueTable = element(tag="valueTable", ns=_NS)


class Brakes(_Base, tag="brakes", ns=_NS):
    brake_effort: Optional[BrakeEffortCurve] = element(tag="brakeEffort", ns=_NS, default=None)
    deceleration_table: Optional[DecelerationCurve] = element(
        tag="decelerationTable", ns=_NS, default=None
    )


# ---------------------------------------------------------------------------
# Vehicle  (rollingstock3.xsd)
# ---------------------------------------------------------------------------


class VehiclePart(_Base, tag="vehiclePart", ns=_NS):
    id: str = attr(name="id", default_factory=_make_id)
    part_order: int = attr(name="partOrder")
    category: Optional[
        Literal[
            "locomotive",
            "motorCoach",
            "passengerCoach",
            "freightWagon",
            "cabCoach",
            "booster",
        ]
    ] = attr(name="category", default=None)


class Vehicle(_Base, tag="vehicle", ns=_NS):
    id: str = attr(name="id", default_factory=_make_id)
    speed: Optional[Decimal] = attr(name="speed", default=None)
    brutto_weight: Optional[Decimal] = attr(name="bruttoWeight", default=None)
    tare_weight: Optional[Decimal] = attr(name="tareWeight", default=None)
    length: Optional[Decimal] = attr(name="length", default=None)
    number_of_driven_axles: Optional[int] = attr(name="numberOfDrivenAxles", default=None)
    number_of_non_driven_axles: Optional[int] = attr(name="numberOfNonDrivenAxles", default=None)
    adhesion_weight: Optional[Decimal] = attr(name="adhesionWeight", default=None)
    rotating_mass_factor: Optional[Decimal] = attr(name="rotatingMassFactor", default=None)
    designators: list[Designator] = element(tag="designator", ns=_NS, default_factory=list)
    vehicle_parts: list[VehiclePart] = element(tag="vehiclePart", ns=_NS, default_factory=list)
    engines: list[Engine] = element(tag="engine", ns=_NS, default_factory=list)
    brakes: list[Brakes] = element(tag="brakes", ns=_NS, default_factory=list)
    driving_resistance: Optional[DrivingResistance] = element(
        tag="drivingResistance", ns=_NS, default=None
    )


# ---------------------------------------------------------------------------
# Formation  (rollingstock3.xsd)
# ---------------------------------------------------------------------------


class TrainOrder(_Base, tag="trainOrder", ns=_NS):
    order_number: int = attr(name="orderNumber")
    vehicle_ref: str = attr(name="vehicleRef")
    orientation: Literal["normal", "reverse"] = attr(name="orientation", default="normal")


class TrainEngine(_Base, tag="trainEngine", ns=_NS):
    max_acceleration: Optional[Decimal] = attr(name="maxAcceleration", default=None)
    mean_acceleration: Optional[Decimal] = attr(name="meanAcceleration", default=None)
    traction_mode: Optional[TrainTractionMode] = element(
        tag="tractionMode", ns=_NS, default=None
    )


class Formation(_Base, tag="formation", ns=_NS):
    id: str = attr(name="id", default_factory=_make_id)
    brutto_weight: Optional[Decimal] = attr(name="bruttoWeight", default=None)
    tare_weight: Optional[Decimal] = attr(name="tareWeight", default=None)
    length: Optional[Decimal] = attr(name="length", default=None)
    speed: Optional[Decimal] = attr(name="speed", default=None)
    designators: list[Designator] = element(tag="designator", ns=_NS, default_factory=list)
    train_orders: list[TrainOrder] = element(tag="trainOrder", ns=_NS, default_factory=list)
    train_engines: list[TrainEngine] = element(tag="trainEngine", ns=_NS, default_factory=list)
    train_resistance: Optional[TrainDrivingResistance] = element(
        tag="trainResistance", ns=_NS, default=None
    )


# ---------------------------------------------------------------------------
# Top-level containers
# ---------------------------------------------------------------------------


class Vehicles(_Base, tag="vehicles", ns=_NS):
    vehicles: list[Vehicle] = element(tag="vehicle", ns=_NS, default_factory=list)


class Formations(_Base, tag="formations", ns=_NS):
    formations: list[Formation] = element(tag="formation", ns=_NS, default_factory=list)


class Rollingstock(_Base, tag="rollingstock", ns=_NS):
    vehicles: Optional[Vehicles] = element(tag="vehicles", ns=_NS, default=None)
    formations: Optional[Formations] = element(tag="formations", ns=_NS, default=None)


class RailML(_Base, tag="railML", ns=_NS, nsmap=_NSMAP):
    version: str = attr(name="version", default="3.3")
    rollingstock: Optional[Rollingstock] = element(tag="rollingstock", ns=_NS, default=None)
