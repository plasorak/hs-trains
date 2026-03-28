"""Unit tests for the RailML 3.3 rollingstock pydantic-xml models and UI helpers."""

from decimal import Decimal
from xml.etree.ElementTree import fromstring

import pytest

from rusty_trains.model.rollingstock import (
    NS,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xml(model, **kwargs):
    """Serialise a model to an Element, excluding None attributes."""
    return fromstring(model.to_xml(encoding="unicode", exclude_none=True, **kwargs))


def _clark(name: str) -> str:
    return f"{{{NS}}}{name}"


def _speed_curve(cls, *points):
    return cls(
        value_table=ValueTable(
            x_value_name="speed",
            x_value_unit="km/h",
            y_value_name="test",
            y_value_unit="N",
            value_lines=[
                ValueLine(x_value=Decimal(str(x)), values=[Value(y_value=Decimal(str(y)))])
                for x, y in points
            ],
        )
    )


# ---------------------------------------------------------------------------
# ValueTable / ValueLine / Value
# ---------------------------------------------------------------------------


class TestValueTable:
    def test_attributes_serialised(self):
        vt = ValueTable(
            x_value_name="speed",
            x_value_unit="km/h",
            y_value_name="tractiveEffort",
            y_value_unit="N",
            value_lines=[ValueLine(x_value=Decimal("0"), values=[Value(y_value=Decimal("100"))])],
        )
        el = _xml(vt)
        assert el.get("xValueName") == "speed"
        assert el.get("xValueUnit") == "km/h"
        assert el.get("yValueName") == "tractiveEffort"
        assert el.get("yValueUnit") == "N"

    def test_value_lines_children(self):
        vt = ValueTable(
            x_value_name="speed",
            x_value_unit="km/h",
            y_value_name="force",
            y_value_unit="N",
            value_lines=[
                ValueLine(x_value=Decimal("0"), values=[Value(y_value=Decimal("200"))]),
                ValueLine(x_value=Decimal("60"), values=[Value(y_value=Decimal("100"))]),
            ],
        )
        el = _xml(vt)
        lines = el.findall(_clark("valueLine"))
        assert len(lines) == 2
        assert lines[0].get("xValue") == "0"
        assert lines[1].get("xValue") == "60"

    def test_value_y_value(self):
        vl = ValueLine(
            x_value=Decimal("50"),
            values=[Value(y_value=Decimal("99.5"))],
        )
        el = _xml(vl)
        v_el = el.find(_clark("value"))
        assert v_el is not None
        assert v_el.get("yValue") == "99.5"

    def test_empty_value_lines_allowed(self):
        # No min_length constraint in pydantic-xml element()
        vt = ValueTable(
            x_value_name="speed",
            x_value_unit="km/h",
            y_value_name="force",
            y_value_unit="N",
            value_lines=[],
        )
        assert vt.value_lines == []


# ---------------------------------------------------------------------------
# Designator
# ---------------------------------------------------------------------------


class TestDesignator:
    def test_register_attribute(self):
        d = Designator(register_name="UIC", entry="92 70 0 066 001-1")
        el = _xml(d)
        assert el.get("register") == "UIC"
        assert el.get("entry") == "92 70 0 066 001-1"

    def test_optional_description_omitted_when_none(self):
        d = Designator(register_name="op", entry="X1")
        el = _xml(d)
        assert el.get("description") is None

    def test_description_present_when_set(self):
        d = Designator(register_name="op", entry="X1", description="a loco")
        el = _xml(d)
        assert el.get("description") == "a loco"


# ---------------------------------------------------------------------------
# DaviesFormula — XmlBool serialisation
# ---------------------------------------------------------------------------


class TestDaviesFormula:
    def test_mass_dependent_false_serialised_lowercase(self):
        df = DaviesFormula(
            constant_factor_a=Decimal("100"),
            speed_dependent_factor_b=Decimal("2"),
            square_speed_dependent_factor_c=Decimal("0.1"),
            mass_dependent=False,
        )
        el = _xml(df)
        assert el.get("massDependent") == "false"

    def test_mass_dependent_true_serialised_lowercase(self):
        df = DaviesFormula(
            constant_factor_a=Decimal("100"),
            speed_dependent_factor_b=Decimal("2"),
            square_speed_dependent_factor_c=Decimal("0.1"),
            mass_dependent=True,
        )
        el = _xml(df)
        assert el.get("massDependent") == "true"

    def test_mass_dependent_omitted_when_none(self):
        df = DaviesFormula(
            constant_factor_a=Decimal("100"),
            speed_dependent_factor_b=Decimal("2"),
            square_speed_dependent_factor_c=Decimal("0.1"),
        )
        el = _xml(df)
        assert el.get("massDependent") is None


# ---------------------------------------------------------------------------
# DrivingResistance
# ---------------------------------------------------------------------------


class TestDrivingResistance:
    def test_tunnel_factor_omitted_when_none(self):
        dr = DrivingResistance(
            info=DrivingResistanceInfo(
                air_drag_coefficient=Decimal("0.8"),
                cross_section_area=Decimal("9.5"),
                rolling_resistance=Decimal("1.5"),
            )
        )
        el = _xml(dr)
        assert el.get("tunnelFactor") is None

    def test_tunnel_factor_present(self):
        dr = DrivingResistance(
            tunnel_factor=Decimal("1.5"),
        )
        el = _xml(dr)
        assert el.get("tunnelFactor") == "1.5"

    def test_info_child_element(self):
        dr = DrivingResistance(
            info=DrivingResistanceInfo(
                air_drag_coefficient=Decimal("0.8"),
                cross_section_area=Decimal("9.0"),
                rolling_resistance=Decimal("1.2"),
            )
        )
        el = _xml(dr)
        info = el.find(_clark("info"))
        assert info is not None
        assert info.get("airDragCoefficient") == "0.8"
        assert info.get("crossSectionArea") == "9.0"
        assert info.get("rollingResistance") == "1.2"


# ---------------------------------------------------------------------------
# Vehicle
# ---------------------------------------------------------------------------


class TestVehicle:
    def _make_vehicle(self) -> Vehicle:
        return Vehicle(
            id="v1",
            speed=Decimal("120"),
            brutto_weight=Decimal("130"),
            tare_weight=Decimal("130"),
            length=Decimal("21.34"),
            number_of_driven_axles=6,
            number_of_non_driven_axles=0,
        )

    def test_basic_attributes(self):
        el = _xml(self._make_vehicle())
        assert el.get("id") == "v1"
        assert el.get("speed") == "120"
        assert el.get("bruttoWeight") == "130"
        assert el.get("length") == "21.34"
        assert el.get("numberOfDrivenAxles") == "6"
        assert el.get("numberOfNonDrivenAxles") == "0"

    def test_optional_attributes_omitted(self):
        v = Vehicle(id="v2")
        el = _xml(v)
        for attr in ("speed", "bruttoWeight", "tareWeight", "length",
                     "adhesionWeight", "rotatingMassFactor"):
            assert el.get(attr) is None, f"{attr} should be absent"

    def test_designators_serialised(self):
        v = Vehicle(
            id="v3",
            designators=[Designator(register_name="UIC", entry="123")],
        )
        el = _xml(v)
        desigs = el.findall(_clark("designator"))
        assert len(desigs) == 1
        assert desigs[0].get("register") == "UIC"

    def test_vehicle_part_category(self):
        v = Vehicle(
            id="v4",
            vehicle_parts=[VehiclePart(id="vp1", part_order=1, category="locomotive")],
        )
        el = _xml(v)
        vp = el.find(_clark("vehiclePart"))
        assert vp is not None
        assert vp.get("category") == "locomotive"

    def test_engine_with_power_mode(self):
        te = _speed_curve(TractiveEffortCurve, (0, 270_000), (120, 66_000))
        v = Vehicle(
            id="v5",
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
                                details=TractionDetails(tractive_effort=te),
                            ),
                        )
                    ]
                )
            ],
        )
        el = _xml(v)
        engine = el.find(_clark("engine"))
        assert engine is not None
        pm = engine.find(_clark("powerMode"))
        assert pm is not None
        assert pm.get("mode") == "diesel"
        assert pm.get("isPrimaryMode") == "true"

    def test_brakes_deceleration_table(self):
        dc = _speed_curve(DecelerationCurve, (0, 0.9), (120, 0.65))
        v = Vehicle(id="v6", brakes=[Brakes(deceleration_table=dc)])
        el = _xml(v)
        brakes = el.find(_clark("brakes"))
        assert brakes is not None
        dt = brakes.find(_clark("decelerationTable"))
        assert dt is not None
        vt = dt.find(_clark("valueTable"))
        assert vt is not None
        assert vt.get("yValueName") == "test"


# ---------------------------------------------------------------------------
# Formation
# ---------------------------------------------------------------------------


class TestFormation:
    def test_train_order_attributes(self):
        to = TrainOrder(order_number=1, vehicle_ref="v1")
        el = _xml(to)
        assert el.get("orderNumber") == "1"
        assert el.get("vehicleRef") == "v1"
        assert el.get("orientation") == "normal"

    def test_formation_basic(self):
        f = Formation(
            id="f1",
            speed=Decimal("120"),
            train_orders=[TrainOrder(order_number=1, vehicle_ref="v1")],
        )
        el = _xml(f)
        assert el.get("id") == "f1"
        assert el.get("speed") == "120"
        orders = el.findall(_clark("trainOrder"))
        assert len(orders) == 1

    def test_train_engine_traction_mode(self):
        te = TrainEngine(
            max_acceleration=Decimal("0.4"),
            traction_mode=TrainTractionMode(mode="diesel", is_primary_mode=True),
        )
        el = _xml(te)
        assert el.get("maxAcceleration") == "0.4"
        tm = el.find(_clark("tractionMode"))
        assert tm is not None
        assert tm.get("mode") == "diesel"

    def test_train_resistance_davies_formula(self):
        tr = TrainDrivingResistance(
            tunnel_factor=Decimal("1.8"),
            davies_formula_factors=DaviesFormula(
                constant_factor_a=Decimal("3800"),
                speed_dependent_factor_b=Decimal("45"),
                square_speed_dependent_factor_c=Decimal("2.5"),
                mass_dependent=False,
            ),
        )
        el = _xml(tr)
        assert el.get("tunnelFactor") == "1.8"
        df = el.find(_clark("daviesFormulaFactors"))
        assert df is not None
        assert df.get("constantFactorA") == "3800"
        assert df.get("massDependent") == "false"


# ---------------------------------------------------------------------------
# Namespace
# ---------------------------------------------------------------------------


class TestNamespace:
    def test_railml_root_has_namespace(self):
        root = RailML()
        el = _xml(root)
        assert el.tag == _clark("railML"), f"Expected namespaced tag, got {el.tag}"

    def test_all_children_in_namespace(self):
        v = Vehicle(id="nv1")
        railml = RailML(
            rollingstock=Rollingstock(
                vehicles=Vehicles(vehicles=[v]),
            )
        )
        el = _xml(railml)
        for child in el.iter():
            assert child.tag.startswith("{"), f"Element without namespace: {child.tag}"
            assert child.tag.startswith(f"{{{NS}}}"), (
                f"Element in wrong namespace: {child.tag}"
            )

    def test_namespace_uri(self):
        assert NS == "https://www.railml.org/schemas/3.3"


# ---------------------------------------------------------------------------
# Round-trip: to_xml / from_xml
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_value_line_round_trip(self):
        original = ValueLine(
            x_value=Decimal("42"),
            values=[Value(y_value=Decimal("99.9"))],
        )
        xml_str = original.to_xml(encoding="unicode", exclude_none=True)
        restored = ValueLine.from_xml(xml_str)
        assert restored.x_value == original.x_value
        assert restored.values[0].y_value == original.values[0].y_value

    def test_designator_round_trip(self):
        original = Designator(register_name="UIC", entry="001")
        xml_str = original.to_xml(encoding="unicode", exclude_none=True)
        restored = Designator.from_xml(xml_str)
        assert restored.register_name == "UIC"
        assert restored.entry == "001"

    def test_vehicle_round_trip(self):
        original = Vehicle(
            id="rt1",
            speed=Decimal("160"),
            brutto_weight=Decimal("90"),
            designators=[Designator(register_name="op", entry="X1")],
        )
        xml_str = original.to_xml(encoding="unicode", exclude_none=True)
        restored = Vehicle.from_xml(xml_str)
        assert restored.id == "rt1"
        assert restored.speed == Decimal("160")
        assert len(restored.designators) == 1
        assert restored.designators[0].register_name == "op"


# ---------------------------------------------------------------------------
# validate_xml
# ---------------------------------------------------------------------------


class TestValidateXml:
    def test_raises_when_schema_missing(self, tmp_path, monkeypatch):
        import rusty_trains.utils as utils_mod
        monkeypatch.setattr(utils_mod, "_SCHEMA_PATH", tmp_path / "nonexistent.xsd")
        # also clear the lru_cache so the patched path is used
        utils_mod._load_schema.cache_clear()
        with pytest.raises(FileNotFoundError, match="RailML XSD not found"):
            utils_mod.validate_xml("<railML/>")


# ---------------------------------------------------------------------------
# UI helpers: _dec, _pos
# ---------------------------------------------------------------------------


class TestDecPos:
    def test_dec_none(self):
        from rusty_trains.rollingstock_ui import _dec
        assert _dec(None) is None

    def test_dec_valid(self):
        from rusty_trains.rollingstock_ui import _dec
        assert _dec("1.5") == Decimal("1.5")
        assert _dec(42) == Decimal("42")

    def test_dec_invalid_returns_none(self):
        from rusty_trains.rollingstock_ui import _dec
        assert _dec("not-a-number") is None

    def test_pos_zero_returns_none(self):
        from rusty_trains.rollingstock_ui import _pos
        assert _pos(0) is None
        assert _pos("0") is None

    def test_pos_negative_returns_none(self):
        from rusty_trains.rollingstock_ui import _pos
        assert _pos(-1) is None

    def test_pos_positive(self):
        from rusty_trains.rollingstock_ui import _pos
        assert _pos("3.14") == Decimal("3.14")


# ---------------------------------------------------------------------------
# _VehicleDraft.build
# ---------------------------------------------------------------------------


class TestVehicleDraftBuild:
    def test_default_build_produces_vehicle(self):
        from rusty_trains.rollingstock_ui import _VehicleDraft
        vd = _VehicleDraft()
        v = vd.build()
        assert isinstance(v, Vehicle)

    def test_build_sets_id(self):
        from rusty_trains.rollingstock_ui import _VehicleDraft
        vd = _VehicleDraft()
        vd.id = "my_loco"
        v = vd.build()
        assert v.id == "my_loco"

    def test_build_no_engine_when_flag_false(self):
        from rusty_trains.rollingstock_ui import _VehicleDraft
        vd = _VehicleDraft()
        vd.has_engine = False
        v = vd.build()
        assert v.engines == []

    def test_build_with_engine(self):
        from rusty_trains.rollingstock_ui import _VehicleDraft
        vd = _VehicleDraft()
        vd.has_engine = True
        vd.max_te = 200_000.0
        vd.power = 1_000_000.0
        vd.mode = "electric"
        v = vd.build()
        assert len(v.engines) == 1
        assert v.engines[0].power_modes[0].mode == "electric"

    def test_build_no_resistance_when_flag_false(self):
        from rusty_trains.rollingstock_ui import _VehicleDraft
        vd = _VehicleDraft()
        vd.has_res = False
        v = vd.build()
        assert v.driving_resistance is None

    def test_reset_from_existing_vehicle(self):
        from rusty_trains.rollingstock_ui import _VehicleDraft
        original = Vehicle(id="v_orig", speed=Decimal("200"), tare_weight=Decimal("55"))
        vd = _VehicleDraft()
        vd.reset(original, 0)
        assert vd.id == "v_orig"
        assert vd.speed == 200.0
        assert vd.tare == 55.0


# ---------------------------------------------------------------------------
# _FormationDraft.build
# ---------------------------------------------------------------------------


class TestFormationDraftBuild:
    def test_default_build_produces_formation(self):
        from rusty_trains.rollingstock_ui import _FormationDraft
        fd = _FormationDraft()
        f = fd.build()
        assert isinstance(f, Formation)

    def test_build_sets_id(self):
        from rusty_trains.rollingstock_ui import _FormationDraft
        fd = _FormationDraft()
        fd.id = "my_formation"
        f = fd.build()
        assert f.id == "my_formation"

    def test_build_no_engine_when_flag_false(self):
        from rusty_trains.rollingstock_ui import _FormationDraft
        fd = _FormationDraft()
        fd.has_engine = False
        f = fd.build()
        assert f.train_engines == []

    def test_build_with_davies(self):
        from rusty_trains.rollingstock_ui import _FormationDraft
        fd = _FormationDraft()
        fd.has_davies = True
        fd.a_val = 4000.0
        fd.b_val = 50.0
        fd.c_val = 3.0
        f = fd.build()
        assert f.train_resistance is not None
        df = f.train_resistance.davies_formula_factors
        assert df.constant_factor_a == Decimal("4000")

    def test_build_no_davies_when_flag_false(self):
        from rusty_trains.rollingstock_ui import _FormationDraft
        fd = _FormationDraft()
        fd.has_davies = False
        f = fd.build()
        assert f.train_resistance is None
