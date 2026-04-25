"""Unit tests for the RailML 3.3 infrastructure pydantic-xml models."""

from decimal import Decimal

import pytest

from hs_trains.model.common import Designator
from hs_trains.model.infrastructure import (
    GmlLineString,
    GmlLocation,
    GmlPoint,
    GmlPos,
    GmlPosList,
    Length,
    NetElement,
)


def test_net_element_roundtrip():
    ne = NetElement(
        id="ne_001",
        length=Length(quantity=Decimal("100")),
        designators=[Designator(register_name="NR-ELR", entry="ECM1")],
    )
    xml = ne.to_xml(encoding="unicode", exclude_none=True)
    ne2 = NetElement.from_xml(xml)
    assert ne2.id == "ne_001"
    assert ne2.length is not None
    assert ne2.length.quantity == Decimal("100")
    assert len(ne2.designators) == 1
    assert ne2.designators[0].entry == "ECM1"


def test_gml_location_line_string():
    loc = GmlLocation(
        line_string=GmlLineString(pos_list=GmlPosList(root="-0.1 51.5 -0.2 51.6"))
    )
    assert loc.line_string is not None
    assert loc.point is None


def test_gml_location_point():
    loc = GmlLocation(point=GmlPoint(pos=GmlPos(root="-0.1 51.5")))
    assert loc.point is not None
    assert loc.line_string is None


def test_gml_location_neither_raises():
    with pytest.raises(ValueError, match="exactly one"):
        GmlLocation()


def test_gml_location_both_raises():
    with pytest.raises(ValueError, match="exactly one"):
        GmlLocation(
            line_string=GmlLineString(pos_list=GmlPosList(root="-0.1 51.5 -0.2 51.6")),
            point=GmlPoint(pos=GmlPos(root="-0.1 51.5")),
        )
