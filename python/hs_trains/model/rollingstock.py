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


class _BrakeSystemBase(_Base):
    """Shared fields for vehicle- and formation-level brake system configurations."""

    auxiliary_brakes: list[AuxiliaryBrakes] = element(
        tag="auxiliaryBrakes", ns=_NS, default_factory=list
    )
    air_brake_application_position: Optional[str] = attr(
        name="airBrakeApplicationPosition", default=None
    )
    brake_type: Optional[str] = attr(name="brakeType", default=None,
        description="Technical type of brake system: vacuum or compressed air brake, hand brake, parking brake, etc.")
    emergency_brake_mass: Optional[Decimal] = attr(name="emergencyBrakeMass", default=None)
    emergency_brake_percentage: Optional[Decimal] = attr(
        name="emergencyBrakePercentage", default=None,
        description="Brake percentage for emergency brake operations; may differ from regular due to auxiliary brakes.")
    load_switch: Optional[str] = attr(name="loadSwitch", default=None)
    max_deceleration: Optional[Decimal] = attr(name="maxDeceleration", default=None,
        description="Maximum possible momentary deceleration in m/s².")
    mean_deceleration: Optional[Decimal] = attr(name="meanDeceleration", default=None,
        description="Effective mean deceleration over a whole brake operation in m/s². Does not necessarily equal the mean of the deceleration table.")
    regular_brake_mass: Optional[Decimal] = attr(name="regularBrakeMass", default=None)
    regular_brake_percentage: Optional[Decimal] = attr(
        name="regularBrakePercentage", default=None,
        description="Brake percentage for normal brake operations."
    )


class BrakeSystem(_BrakeSystemBase, tag="vehicleBrakes", ns=_NS):
    """Vehicle-level brake system configuration (tag: vehicleBrakes)."""


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

    constant_factor_a: Decimal = attr(name="constantFactorA",
        description="Constant (speed-independent) term A of the Davis formula R(N) = A + B·v + C·v².")
    speed_dependent_factor_b: Decimal = attr(name="speedDependentFactorB",
        description="Speed-linear term B of the Davis formula R(N) = A + B·v + C·v²  (v in km/h).")
    square_speed_dependent_factor_c: Decimal = attr(name="squareSpeedDependentFactorC",
        description="Aerodynamic drag term C of the Davis formula R(N) = A + B·v + C·v²  (v in km/h).")
    mass_dependent: Optional[XmlBool] = attr(name="massDependent", default=None)


class DrivingResistanceInfo(_Base, tag="info", ns=_NS):
    """Key values for calculating driving resistance when no curve is given."""

    air_drag_coefficient: Decimal = attr(name="airDragCoefficient",
        description="Air drag coefficient (Cd). Used together with crossSectionArea and rollingResistance to describe running resistance.")
    cross_section_area: Decimal = attr(name="crossSectionArea",
        description="Cross-section area in m². Used together with airDragCoefficient and rollingResistance to describe running resistance.")
    rolling_resistance: Decimal = attr(name="rollingResistance",
        description="Rolling resistance in N/kN. Used together with airDragCoefficient and crossSectionArea to describe running resistance.")


class DrivingResistanceDetails(_Base, tag="details", ns=_NS):
    """Speed-dependent driving resistance curve."""

    mass_dependent: Optional[XmlBool] = attr(name="massDependent", default=None)
    value_table: Optional[ValueTable] = element(tag="valueTable", ns=_NS, default=None)


class _DrivingResistanceBase(_Base):
    """Shared fields for vehicle- and formation-level driving resistance."""

    tunnel_factor: Optional[Decimal] = attr(name="tunnelFactor", default=None,
        description="Multiplier applied to driving resistance when travelling through a tunnel.")
    info: Optional[DrivingResistanceInfo] = element(tag="info", ns=_NS, default=None)
    details: Optional[DrivingResistanceDetails] = element(tag="details", ns=_NS, default=None)


class DrivingResistance(_DrivingResistanceBase, tag="drivingResistance", ns=_NS):
    """Sum of resistances a vehicle must overcome to travel at constant or accelerated speed."""


class TrainDrivingResistance(_DrivingResistanceBase, tag="trainResistance", ns=_NS):
    """Formation-level driving resistance, adds Davies formula factors."""

    davies_formula_factors: Optional[DaviesFormula] = element(
        tag="daviesFormulaFactors", ns=_NS, default=None
    )


# ---------------------------------------------------------------------------
# Engine / traction  (rollingstock3.xsd)
# ---------------------------------------------------------------------------


class TractionInfo(_Base, tag="info", ns=_NS):
    max_tractive_effort: Decimal = attr(name="maxTractiveEffort",
        description="Maximum tractive effort in N.")
    tractive_power: Decimal = attr(name="tractivePower",
        description="Maximum traction power in W.")


class TractiveEffortCurve(_Base, tag="tractiveEffort", ns=_NS):
    value_table: ValueTable = element(tag="valueTable", ns=_NS)


class TractionDetails(_Base, tag="details", ns=_NS):
    tractive_effort: TractiveEffortCurve = element(tag="tractiveEffort", ns=_NS)


class TractionData(_Base, tag="tractionData", ns=_NS):
    info: Optional[TractionInfo] = element(tag="info", ns=_NS, default=None)
    details: Optional[TractionDetails] = element(tag="details", ns=_NS, default=None)


class _TractionModeBase(_Base):
    """Shared fields for vehicle- and formation-level traction modes."""

    mode: Literal["diesel", "electric", "battery"] = attr(name="mode")
    is_primary_mode: XmlBool = attr(name="isPrimaryMode", default=True)
    traction_data: Optional[TractionData] = element(tag="tractionData", ns=_NS, default=None)


class PowerMode(_TractionModeBase, tag="powerMode", ns=_NS):
    """Traction mode for a single vehicle."""


class TrainTractionMode(_TractionModeBase, tag="tractionMode", ns=_NS):
    """Traction mode for a formation."""


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
    speed: Optional[Decimal] = attr(name="speed", default=None,
        description="Maximum permissible speed for the vehicle in km/h.")
    brutto_weight: Optional[Decimal] = attr(name="bruttoWeight", default=None,
        description="Mass ready to run incl. operating fluids; for passenger vehicles all seats taken at ~75 kg/person; for freight with full payload. In tonnes.")
    tare_weight: Optional[Decimal] = attr(name="tareWeight", default=None,
        description="Mass ready to run incl. operating fluids, no passengers, no payload, in tonnes. Used as vehicle mass in run-time calculations.")
    netto_weight: Optional[Decimal] = attr(name="nettoWeight", default=None,
        description="Maximum payload of the vehicle in tonnes.")
    timetable_weight: Optional[Decimal] = attr(name="timetableWeight", default=None)
    maximum_weight: Optional[Decimal] = attr(name="maximumWeight", default=None)
    maximum_axle_load: Optional[Decimal] = attr(name="maximumAxleLoad", default=None)
    length: Optional[Decimal] = attr(name="length", default=None,
        description="Overall length of the vehicle over buffers or couplings in metres; used to calculate train length.")
    number_of_driven_axles: Optional[int] = attr(name="numberOfDrivenAxles", default=None,
        description="Number of driven axles. Must be > 0 for all vehicles with an engine element.")
    number_of_non_driven_axles: Optional[int] = attr(
        name="numberOfNonDrivenAxles", default=None,
        description="Number of non-driven axles of non-driven as well as self-driven vehicles.")
    adhesion_weight: Optional[Decimal] = attr(name="adhesionWeight", default=None,
        description="Weight in tonnes usable for traction (tare adhesion weight). Used to estimate the adhesion limit in run-time calculations.")
    rotating_mass_factor: Optional[Decimal] = attr(name="rotatingMassFactor", default=None,
        description="Factor applied to static mass to account for energy needed to accelerate/decelerate rotating masses. Typically 1.05–1.25.")
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
    max_acceleration: Optional[Decimal] = attr(name="maxAcceleration", default=None,
        description="Maximum acceleration in m/s² possible with the traction system of the formation.")
    mean_acceleration: Optional[Decimal] = attr(name="meanAcceleration", default=None,
        description="Mean acceleration in m/s² possible with the traction system of the formation.")
    min_time_hold_speed: Optional[str] = attr(name="minTimeHoldSpeed", default=None)
    traction_mode: Optional[TrainTractionMode] = element(
        tag="tractionMode", ns=_NS, default=None
    )


class FormationDecelerationCurve(_Base, tag="decelerationTable", ns=_NS):
    value_table: ValueTable = element(tag="valueTable", ns=_NS)


class FormationBrakeSystem(_BrakeSystemBase, tag="trainBrakes", ns=_NS):
    """Formation-level brake system configuration (same fields as BrakeSystem)."""


class Formation(_Base, tag="formation", ns=_NS):
    """Collection of vehicles coupled as a train for operation."""

    id: str = attr(name="id", default_factory=_make_id)
    brutto_weight: Optional[Decimal] = attr(name="bruttoWeight", default=None,
        description="Maximum overall weight (= tareWeight + nettoWeight) of the formation in tonnes.")
    tare_weight: Optional[Decimal] = attr(name="tareWeight", default=None,
        description="Empty weight (without payload, ready to run, incl. operating fluids) of the complete formation in tonnes.")
    netto_weight: Optional[Decimal] = attr(name="nettoWeight", default=None,
        description="Weight of maximum payload for the complete formation in tonnes.")
    timetable_weight: Optional[Decimal] = attr(name="timetableWeight", default=None)
    hauling_weight: Optional[Decimal] = attr(name="haulingWeight", default=None)
    length: Optional[Decimal] = attr(name="length", default=None,
        description="Overall length of the entire formation in metres.")
    speed: Optional[Decimal] = attr(name="speed", default=None,
        description="Maximum permissible speed in km/h of the entire formation, limited by the lowest maximum of contained vehicles.")
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
