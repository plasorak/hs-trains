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
# Common identifier  (generic3.xsd: Designator, Name)
# ---------------------------------------------------------------------------


class Designator(_Base, tag="designator", ns=_NS):
    register_name: str = attr(name="register")
    entry: str = attr(name="entry")
    description: Optional[str] = attr(name="description", default=None)


class Name(_Base, tag="name", ns=_NS):
    name: str = attr(name="name")
    language: str = attr(name="language")
    description: Optional[str] = attr(name="description", default=None)


# ---------------------------------------------------------------------------
# Tilting  (generic3.xsd: TiltingSpecification)
# ---------------------------------------------------------------------------


class TiltingSpecification(_Base, tag="isSupportingTilting", ns=_NS):
    actuation: Optional[str] = attr(name="actuation", default=None)
    max_tilting_angle: Optional[Decimal] = attr(name="maxTiltingAngle", default=None)
    max_tilting_speed: Optional[Decimal] = attr(name="maxTiltingSpeed", default=None)


# ---------------------------------------------------------------------------
# Brake system  (generic3.xsd: tAuxiliaryBrakes, tBrakeSystem)
# ---------------------------------------------------------------------------


class AuxiliaryBrakes(_Base, tag="auxiliaryBrakes", ns=_NS):
    """Flags indicating which auxiliary brakes are active in this brake setting."""

    brake_use: Optional[str] = attr(name="brakeUse", default=None)
    E: Optional[XmlBool] = attr(name="E", default=None)   # electro-dynamic / rheostatic
    ep: Optional[XmlBool] = attr(name="ep", default=None)  # electro-pneumatic
    H: Optional[XmlBool] = attr(name="H", default=None)   # hydro-dynamic
    Mg: Optional[XmlBool] = attr(name="Mg", default=None)  # magnetic shoe
    Wb: Optional[XmlBool] = attr(name="Wb", default=None)  # eddy current


class BrakeSystem(_Base, tag="vehicleBrakes", ns=_NS):
    """One brake-system configuration (vehicle or formation level)."""

    auxiliary_brakes: list[AuxiliaryBrakes] = element(
        tag="auxiliaryBrakes", ns=_NS, default_factory=list
    )
    air_brake_application_position: Optional[str] = attr(
        name="airBrakeApplicationPosition", default=None
    )
    brake_type: Optional[str] = attr(name="brakeType", default=None)
    emergency_brake_mass: Optional[Decimal] = attr(name="emergencyBrakeMass", default=None)
    emergency_brake_percentage: Optional[Decimal] = attr(
        name="emergencyBrakePercentage", default=None
    )
    load_switch: Optional[str] = attr(name="loadSwitch", default=None)
    max_deceleration: Optional[Decimal] = attr(name="maxDeceleration", default=None)
    mean_deceleration: Optional[Decimal] = attr(name="meanDeceleration", default=None)
    regular_brake_mass: Optional[Decimal] = attr(name="regularBrakeMass", default=None)
    regular_brake_percentage: Optional[Decimal] = attr(
        name="regularBrakePercentage", default=None
    )


# ---------------------------------------------------------------------------
# Passenger / freight facilities  (generic3.xsd)
# ---------------------------------------------------------------------------


class Places(_Base, tag="places", ns=_NS):
    designators: list[Designator] = element(tag="designator", ns=_NS, default_factory=list)
    category: Optional[str] = attr(name="category", default=None)
    travel_class: Optional[str] = attr(name="class", default=None)
    compartment_size: Optional[int] = attr(name="compartmentSize", default=None)
    count: Optional[int] = attr(name="count", default=None)
    is_usable_for_wheel_chair: Optional[XmlBool] = attr(
        name="isUsableForWheelChair", default=None
    )
    place_related_service: Optional[str] = attr(name="placeRelatedService", default=None)
    standing_area: Optional[float] = attr(name="standingArea", default=None)


class Service(_Base, tag="service", ns=_NS):
    designators: list[Designator] = element(tag="designator", ns=_NS, default_factory=list)
    category: Optional[str] = attr(name="category", default=None)
    count: Optional[int] = attr(name="count", default=None)


class PassengerFacilities(_Base, tag="passengerFacilities", ns=_NS):
    places: list[Places] = element(tag="places", ns=_NS, default_factory=list)
    services: list[Service] = element(tag="service", ns=_NS, default_factory=list)
    has_emergency_brake_override: Optional[XmlBool] = attr(
        name="hasEmergencyBrakeOverride", default=None
    )


class FreightFacilities(_Base, tag="freightFacilities", ns=_NS):
    designators: list[Designator] = element(tag="designator", ns=_NS, default_factory=list)
    count: Optional[int] = attr(name="count", default=None)
    freight_type: Optional[str] = attr(name="freightType", default=None)
    has_weather_protection: Optional[XmlBool] = attr(
        name="hasWeatherProtection", default=None
    )
    load: Optional[Decimal] = attr(name="load", default=None)
    load_access: Optional[str] = attr(name="loadAccess", default=None)
    load_area: Optional[Decimal] = attr(name="loadArea", default=None)
    load_volume: Optional[Decimal] = attr(name="loadVolume", default=None)
    self_discharge: Optional[str] = attr(name="selfDischarge", default=None)


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


class DrivingResistanceDetails(_Base, tag="details", ns=_NS):
    """Speed-dependent driving resistance curve."""

    mass_dependent: Optional[XmlBool] = attr(name="massDependent", default=None)
    value_table: Optional[ValueTable] = element(tag="valueTable", ns=_NS, default=None)


class DrivingResistance(_Base, tag="drivingResistance", ns=_NS):
    tunnel_factor: Optional[Decimal] = attr(name="tunnelFactor", default=None)
    info: Optional[DrivingResistanceInfo] = element(tag="info", ns=_NS, default=None)
    details: Optional[DrivingResistanceDetails] = element(tag="details", ns=_NS, default=None)


class TrainDrivingResistance(_Base, tag="trainResistance", ns=_NS):
    """Formation-level driving resistance, adds Davies formula factors."""

    tunnel_factor: Optional[Decimal] = attr(name="tunnelFactor", default=None)
    info: Optional[DrivingResistanceInfo] = element(tag="info", ns=_NS, default=None)
    details: Optional[DrivingResistanceDetails] = element(tag="details", ns=_NS, default=None)
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
    vehicle_brakes: list[BrakeSystem] = element(
        tag="vehicleBrakes", ns=_NS, default_factory=list
    )
    brake_effort: Optional[BrakeEffortCurve] = element(tag="brakeEffort", ns=_NS, default=None)
    deceleration_table: Optional[DecelerationCurve] = element(
        tag="decelerationTable", ns=_NS, default=None
    )


# ---------------------------------------------------------------------------
# Administrative data  (rollingstock3.xsd)
# ---------------------------------------------------------------------------


class VehicleAdministration(_Base):
    """Abstract base for manufacturer/owner/operator/keeper administrative data."""

    designators: list[Designator] = element(tag="designator", ns=_NS, default_factory=list)
    names: list[Name] = element(tag="name", ns=_NS, default_factory=list)
    vehicle_class: Optional[str] = attr(name="class", default=None)
    refers_to: str = attr(name="refersTo")


class VehicleManufacturerRS(VehicleAdministration, tag="manufacturer", ns=_NS):
    pass


class VehicleOwnerRS(VehicleAdministration, tag="owner", ns=_NS):
    pass


class VehicleOperatorRS(VehicleAdministration, tag="operator", ns=_NS):
    pass


class VehicleKeeperRS(VehicleAdministration, tag="keeper", ns=_NS):
    pass


class AdministrativeData(_Base, tag="administrativeData", ns=_NS):
    manufacturer: Optional[VehicleManufacturerRS] = element(
        tag="manufacturer", ns=_NS, default=None
    )
    owner: Optional[VehicleOwnerRS] = element(tag="owner", ns=_NS, default=None)
    operator: Optional[VehicleOperatorRS] = element(tag="operator", ns=_NS, default=None)
    keeper: Optional[VehicleKeeperRS] = element(tag="keeper", ns=_NS, default=None)


# ---------------------------------------------------------------------------
# Speed profile and track gauge references  (rollingstock3.xsd)
# ---------------------------------------------------------------------------


class SpeedProfileRef(_Base, tag="speedProfileRef", ns=_NS):
    ref: str = attr(name="ref")


class TrackGaugeRS(_Base, tag="supportedTrackGauge", ns=_NS):
    value: Optional[Decimal] = attr(name="value", default=None)


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
    air_tightness: Optional[XmlBool] = attr(name="airTightness", default=None)
    emergency_brake_override: Optional[XmlBool] = attr(
        name="emergencyBrakeOverride", default=None
    )
    maximum_cant_deficiency: Optional[Decimal] = attr(
        name="maximumCantDeficiency", default=None
    )
    passenger_facilities: list[PassengerFacilities] = element(
        tag="passengerFacilities", ns=_NS, default_factory=list
    )
    freight_facilities: list[FreightFacilities] = element(
        tag="freightFacilities", ns=_NS, default_factory=list
    )
    is_supporting_tilting: Optional[TiltingSpecification] = element(
        tag="isSupportingTilting", ns=_NS, default=None
    )


class Vehicle(_Base, tag="vehicle", ns=_NS):
    id: str = attr(name="id", default_factory=_make_id)
    speed: Optional[Decimal] = attr(name="speed", default=None)
    brutto_weight: Optional[Decimal] = attr(name="bruttoWeight", default=None)
    tare_weight: Optional[Decimal] = attr(name="tareWeight", default=None)
    netto_weight: Optional[Decimal] = attr(name="nettoWeight", default=None)
    timetable_weight: Optional[Decimal] = attr(name="timetableWeight", default=None)
    maximum_weight: Optional[Decimal] = attr(name="maximumWeight", default=None)
    maximum_axle_load: Optional[Decimal] = attr(name="maximumAxleLoad", default=None)
    length: Optional[Decimal] = attr(name="length", default=None)
    number_of_driven_axles: Optional[int] = attr(name="numberOfDrivenAxles", default=None)
    number_of_non_driven_axles: Optional[int] = attr(
        name="numberOfNonDrivenAxles", default=None
    )
    adhesion_weight: Optional[Decimal] = attr(name="adhesionWeight", default=None)
    rotating_mass_factor: Optional[Decimal] = attr(name="rotatingMassFactor", default=None)
    effective_axle_distance: Optional[Decimal] = attr(
        name="effectiveAxleDistance", default=None
    )
    towing_speed: Optional[Decimal] = attr(name="towingSpeed", default=None)
    based_on_template: Optional[str] = attr(name="basedOnTemplate", default=None)
    designators: list[Designator] = element(tag="designator", ns=_NS, default_factory=list)
    vehicle_parts: list[VehiclePart] = element(tag="vehiclePart", ns=_NS, default_factory=list)
    engines: list[Engine] = element(tag="engine", ns=_NS, default_factory=list)
    brakes: list[Brakes] = element(tag="brakes", ns=_NS, default_factory=list)
    administrative_data: Optional[AdministrativeData] = element(
        tag="administrativeData", ns=_NS, default=None
    )
    driving_resistance: Optional[DrivingResistance] = element(
        tag="drivingResistance", ns=_NS, default=None
    )
    speed_profile_refs: list[SpeedProfileRef] = element(
        tag="speedProfileRef", ns=_NS, default_factory=list
    )
    supported_track_gauges: list[TrackGaugeRS] = element(
        tag="supportedTrackGauge", ns=_NS, default_factory=list
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
    min_time_hold_speed: Optional[str] = attr(name="minTimeHoldSpeed", default=None)
    traction_mode: Optional[TrainTractionMode] = element(
        tag="tractionMode", ns=_NS, default=None
    )


class FormationDecelerationCurve(_Base, tag="decelerationTable", ns=_NS):
    value_table: ValueTable = element(tag="valueTable", ns=_NS)


class FormationBrakeSystem(_Base, tag="trainBrakes", ns=_NS):
    """Formation-level brake system configuration (same fields as BrakeSystem)."""

    auxiliary_brakes: list[AuxiliaryBrakes] = element(
        tag="auxiliaryBrakes", ns=_NS, default_factory=list
    )
    air_brake_application_position: Optional[str] = attr(
        name="airBrakeApplicationPosition", default=None
    )
    brake_type: Optional[str] = attr(name="brakeType", default=None)
    emergency_brake_mass: Optional[Decimal] = attr(name="emergencyBrakeMass", default=None)
    emergency_brake_percentage: Optional[Decimal] = attr(
        name="emergencyBrakePercentage", default=None
    )
    load_switch: Optional[str] = attr(name="loadSwitch", default=None)
    max_deceleration: Optional[Decimal] = attr(name="maxDeceleration", default=None)
    mean_deceleration: Optional[Decimal] = attr(name="meanDeceleration", default=None)
    regular_brake_mass: Optional[Decimal] = attr(name="regularBrakeMass", default=None)
    regular_brake_percentage: Optional[Decimal] = attr(
        name="regularBrakePercentage", default=None
    )


class Formation(_Base, tag="formation", ns=_NS):
    id: str = attr(name="id", default_factory=_make_id)
    brutto_weight: Optional[Decimal] = attr(name="bruttoWeight", default=None)
    tare_weight: Optional[Decimal] = attr(name="tareWeight", default=None)
    netto_weight: Optional[Decimal] = attr(name="nettoWeight", default=None)
    timetable_weight: Optional[Decimal] = attr(name="timetableWeight", default=None)
    hauling_weight: Optional[Decimal] = attr(name="haulingWeight", default=None)
    length: Optional[Decimal] = attr(name="length", default=None)
    speed: Optional[Decimal] = attr(name="speed", default=None)
    maximum_axle_load: Optional[Decimal] = attr(name="maximumAxleLoad", default=None)
    maximum_cant_deficiency: Optional[Decimal] = attr(
        name="maximumCantDeficiency", default=None
    )
    number_of_axles: Optional[int] = attr(name="numberOfAxles", default=None)
    number_of_wagons: Optional[int] = attr(name="numberOfWagons", default=None)
    designators: list[Designator] = element(tag="designator", ns=_NS, default_factory=list)
    train_orders: list[TrainOrder] = element(tag="trainOrder", ns=_NS, default_factory=list)
    train_engines: list[TrainEngine] = element(tag="trainEngine", ns=_NS, default_factory=list)
    train_brakes: list[FormationBrakeSystem] = element(
        tag="trainBrakes", ns=_NS, default_factory=list
    )
    is_supporting_tilting: Optional[TiltingSpecification] = element(
        tag="isSupportingTilting", ns=_NS, default=None
    )
    train_resistance: Optional[TrainDrivingResistance] = element(
        tag="trainResistance", ns=_NS, default=None
    )
    deceleration_table: Optional[FormationDecelerationCurve] = element(
        tag="decelerationTable", ns=_NS, default=None
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
