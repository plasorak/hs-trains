"""Unit tests for python/hs_trains/tps.py.

All tests use synthetic in-memory XML so the real TPS asset file is not
required.  The helpers under test are:
  - load_tps / load_tps_stations — stream-parse TPS XML elements
  - build_operational_points     — TpsStation → OperationalPoint
  - build_tps_line_networks      — TpsLine    → Network
  - build_tps_signals            — TpsSignal  → Signal
  - WaymarkIndex                 — ELR chainage → BNG interpolation
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from hs_trains.tps import (
    TpsLine,
    TpsSignal,
    TpsStation,
    WaymarkIndex,
    _ELRWaymarks,
    build_operational_points,
    build_tps_line_networks,
    build_tps_signals,
    build_waymark_index,
    load_tps,
    load_tps_stations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_xml(tmp_path: Path, stations: list[dict]) -> Path:
    """Serialise a list of station-attr dicts into a minimal TPS XML file."""
    root = ET.Element("tps_data")
    for attrs in stations:
        ET.SubElement(root, "station", attrs)
    p = tmp_path / "tps.xml"
    ET.ElementTree(root).write(str(p), encoding="unicode", xml_declaration=True)
    return p


def _write_xml_with_positions(
    tmp_path: Path,
    stations: list[tuple[dict, dict | None]],
) -> Path:
    """Write stations where each entry is (station_attrs, stationposition_attrs | None)."""
    root = ET.Element("tps_data")
    for station_attrs, pos_attrs in stations:
        s_elem = ET.SubElement(root, "station", station_attrs)
        if pos_attrs is not None:
            ET.SubElement(s_elem, "stationposition", pos_attrs)
    p = tmp_path / "tps_pos.xml"
    ET.ElementTree(root).write(str(p), encoding="unicode", xml_declaration=True)
    return p


# ---------------------------------------------------------------------------
# load_tps_stations
# ---------------------------------------------------------------------------

def test_load_basic_station(tmp_path):
    p = _write_xml(tmp_path, [
        {"stationid": "1", "abbrev": "GLGC", "stanox": "89805", "crscode": "GLC",
         "longname": "Glasgow Central", "easting": "259300", "northing": "665000"},
    ])
    stations = load_tps_stations(p)
    assert len(stations) == 1
    s = stations[0]
    assert s.station_id == "1"
    assert s.tiploc == "GLGC"
    assert s.stanox == "89805"
    assert s.crs == "GLC"
    assert s.name == "Glasgow Central"
    assert s.easting_m == 259300
    assert s.northing_m == 665000


def test_load_skips_placeholder(tmp_path):
    """The null station with abbrev '- - -' must be excluded."""
    p = _write_xml(tmp_path, [
        {"stationid": "0", "abbrev": "- - -", "stanox": "", "crscode": "",
         "longname": "", "easting": "0", "northing": "0"},
        {"stationid": "1", "abbrev": "ASCOT", "stanox": "84601", "crscode": "ACT",
         "longname": "Ascot", "easting": "0", "northing": "0"},
    ])
    stations = load_tps_stations(p)
    assert len(stations) == 1
    assert stations[0].tiploc == "ASCOT"


def test_load_skips_empty_abbrev(tmp_path):
    p = _write_xml(tmp_path, [
        {"stationid": "5", "abbrev": "", "stanox": "", "crscode": "",
         "longname": "Ghost", "easting": "0", "northing": "0"},
    ])
    assert load_tps_stations(p) == []


def test_load_missing_coordinates_default_to_zero(tmp_path):
    p = _write_xml(tmp_path, [
        {"stationid": "2", "abbrev": "BAGSHOT", "stanox": "84610", "crscode": "BAG",
         "longname": "Bagshot"},
    ])
    s = load_tps_stations(p)[0]
    assert s.easting_m == 0
    assert s.northing_m == 0
    assert not s.has_coordinates


def test_load_zero_coordinates_not_treated_as_real(tmp_path):
    p = _write_xml(tmp_path, [
        {"stationid": "3", "abbrev": "FRIMLEY", "stanox": "84612", "crscode": "FML",
         "longname": "Frimley", "easting": "0", "northing": "0"},
    ])
    s = load_tps_stations(p)[0]
    assert not s.has_coordinates


def test_load_multiple_stations_order_preserved(tmp_path):
    p = _write_xml(tmp_path, [
        {"stationid": "10", "abbrev": "AAAA", "stanox": "", "crscode": "", "longname": "A"},
        {"stationid": "20", "abbrev": "BBBB", "stanox": "", "crscode": "", "longname": "B"},
        {"stationid": "30", "abbrev": "CCCC", "stanox": "", "crscode": "", "longname": "C"},
    ])
    stations = load_tps_stations(p)
    assert [s.tiploc for s in stations] == ["AAAA", "BBBB", "CCCC"]


def test_load_ignores_non_station_elements(tmp_path):
    """Other TPS element types must not be returned as stations."""
    root = ET.Element("tps_data")
    ET.SubElement(root, "node", {"businessnumber": "1", "netx": "100", "nety": "200",
                                  "kmregionid": "1", "kmvalue": "0"})
    ET.SubElement(root, "station", {"stationid": "1", "abbrev": "REDHL",
                                     "stanox": "12345", "crscode": "RHL",
                                     "longname": "Redhill", "easting": "0", "northing": "0"})
    ET.SubElement(root, "edge", {"id": "9", "length": "500"})
    p = tmp_path / "tps.xml"
    ET.ElementTree(root).write(str(p), encoding="unicode", xml_declaration=True)
    stations = load_tps_stations(p)
    assert len(stations) == 1
    assert stations[0].tiploc == "REDHL"


# ---------------------------------------------------------------------------
# TpsStation.has_coordinates
# ---------------------------------------------------------------------------

def test_has_coordinates_true():
    s = TpsStation("1", "GLGC", "89805", "GLC", "Glasgow Central",
                   easting_m=259300, northing_m=665000)
    assert s.has_coordinates


def test_has_coordinates_false_both_zero():
    s = TpsStation("2", "ASCOT", "84601", "ACT", "Ascot")
    assert not s.has_coordinates


def test_has_coordinates_false_only_easting():
    s = TpsStation("3", "X", "", "", "", easting_m=100000, northing_m=0)
    assert not s.has_coordinates


# ---------------------------------------------------------------------------
# build_operational_points
# ---------------------------------------------------------------------------

def test_build_op_id_uses_tiploc():
    s = TpsStation("1", "GLGC", "89805", "GLC", "Glasgow Central")
    ops = build_operational_points([s])
    assert ops[0].id == "op_GLGC"


def test_build_op_name():
    s = TpsStation("1", "GLGC", "89805", "GLC", "Glasgow Central")
    ops = build_operational_points([s])
    assert ops[0].name == "Glasgow Central"


def test_build_op_name_falls_back_to_tiploc_when_empty():
    s = TpsStation("1", "GLGC", "89805", "GLC", "")
    ops = build_operational_points([s])
    assert ops[0].name == "GLGC"


def test_build_op_tiploc_designator_always_present():
    s = TpsStation("1", "BAGSHOT", "", "", "Bagshot")
    ops = build_operational_points([s])
    registers = {d.register_name for d in ops[0].designators}
    assert "NR-TIPLOC" in registers
    tiploc_desig = next(d for d in ops[0].designators if d.register_name == "NR-TIPLOC")
    assert tiploc_desig.entry == "BAGSHOT"


def test_build_op_stanox_designator_when_present():
    s = TpsStation("1", "ASCOT", "84601", "", "Ascot")
    ops = build_operational_points([s])
    registers = {d.register_name for d in ops[0].designators}
    assert "NR-STANOX" in registers


def test_build_op_stanox_designator_omitted_when_empty():
    s = TpsStation("1", "ASCOT", "", "", "Ascot")
    ops = build_operational_points([s])
    registers = {d.register_name for d in ops[0].designators}
    assert "NR-STANOX" not in registers


def test_build_op_crs_designator_when_present():
    s = TpsStation("1", "ASCOT", "", "ACT", "Ascot")
    ops = build_operational_points([s])
    registers = {d.register_name for d in ops[0].designators}
    assert "NR-CRS" in registers


def test_build_op_crs_designator_omitted_when_empty():
    s = TpsStation("1", "ASCOT", "", "", "Ascot")
    ops = build_operational_points([s])
    registers = {d.register_name for d in ops[0].designators}
    assert "NR-CRS" not in registers


def test_build_op_no_gml_without_coordinates():
    s = TpsStation("1", "ASCOT", "84601", "ACT", "Ascot")
    ops = build_operational_points([s])
    assert ops[0].gml_locations == []


def test_build_op_gml_point_with_coordinates():
    # Glasgow Central approx BNG: 259300 E, 665000 N
    s = TpsStation("1", "GLGC", "89805", "GLC", "Glasgow Central",
                   easting_m=259300, northing_m=665000)
    ops = build_operational_points([s])
    assert len(ops[0].gml_locations) == 1
    loc = ops[0].gml_locations[0]
    assert loc.point is not None
    assert loc.line_string is None
    # Rough sanity check: Glasgow is around -4.25 lon, 55.86 lat
    parts = loc.point.pos.root.split()
    lon, lat = float(parts[0]), float(parts[1])
    assert -5.0 < lon < -3.5
    assert 55.0 < lat < 57.0


def test_build_op_count_matches_input():
    stations = [
        TpsStation(str(i), f"S{i:04d}", "", "", f"Station {i}")
        for i in range(50)
    ]
    ops = build_operational_points(stations)
    assert len(ops) == 50


def test_build_op_empty_input():
    assert build_operational_points([]) == []


# ---------------------------------------------------------------------------
# load_tps — single-pass loader for all element types
# ---------------------------------------------------------------------------

def _write_full_xml(tmp_path: Path, *, stations=(), lines=(), signals=(), interlocking=(), kmregions=()) -> Path:
    """Write a TPS XML file containing a mix of element types."""
    root = ET.Element("tps_data")
    for attrs in kmregions:
        ET.SubElement(root, "kmregionmasterdesc", attrs)
    for attrs in interlocking:
        ET.SubElement(root, "interlockingsystem", attrs)
    for attrs in stations:
        ET.SubElement(root, "station", attrs)
    for attrs in lines:
        ET.SubElement(root, "line", attrs)
    for attrs in signals:
        ET.SubElement(root, "signal", attrs)
    p = tmp_path / "tps_full.xml"
    ET.ElementTree(root).write(str(p), encoding="unicode", xml_declaration=True)
    return p


def test_load_tps_returns_all_types(tmp_path):
    p = _write_full_xml(
        tmp_path,
        stations=[{"stationid": "1", "abbrev": "GLGC", "stanox": "89805",
                   "crscode": "GLC", "longname": "Glasgow Central",
                   "easting": "0", "northing": "0"}],
        lines=[{"lineid": "1", "desc": "Edinburgh-Inverness"}],
        signals=[{"id": "10", "name": "SIG: A1", "bumper": "false",
                  "interlockingsysid": "8"}],
        interlocking=[{"id": "8", "name": "4-Aspect Colour Light (IECC)"}],
    )
    data = load_tps(p)
    assert len(data.stations) == 1
    assert len(data.lines) == 1
    assert len(data.signals) == 1


def test_load_tps_elr_lookup(tmp_path):
    p = _write_full_xml(
        tmp_path,
        kmregions=[
            {"id": "1271", "vanillatext": "ECM1"},
            {"id": "1272", "vanillatext": "WEB"},
            {"id": "0", "vanillatext": ""},  # blank — should be excluded
        ],
    )
    data = load_tps(p)
    assert data.elr_lookup == {"1271": "ECM1", "1272": "WEB"}


def test_load_tps_signal_type_resolved(tmp_path):
    """Signal interlocking type must be resolved from the interlockingsystem lookup."""
    p = _write_full_xml(
        tmp_path,
        interlocking=[{"id": "8", "name": "4-Aspect Colour Light (IECC)"}],
        signals=[{"id": "5", "name": "SIG: X1", "bumper": "false",
                  "interlockingsysid": "8"}],
    )
    data = load_tps(p)
    assert data.signals[0].interlocking_type == "4-Aspect Colour Light (IECC)"


def test_load_tps_signal_bumper_flag(tmp_path):
    p = _write_full_xml(
        tmp_path,
        signals=[
            {"id": "1", "name": "BUF", "bumper": "true", "interlockingsysid": "0"},
            {"id": "2", "name": "SIG", "bumper": "false", "interlockingsysid": "0"},
        ],
    )
    data = load_tps(p)
    assert data.signals[0].is_bumper is True
    assert data.signals[1].is_bumper is False


def test_load_tps_lines_skips_empty_desc(tmp_path):
    p = _write_full_xml(
        tmp_path,
        lines=[
            {"lineid": "1", "desc": "Edinburgh-Inverness"},
            {"lineid": "2", "desc": ""},   # empty — should be excluded
        ],
    )
    data = load_tps(p)
    assert len(data.lines) == 1
    assert data.lines[0].description == "Edinburgh-Inverness"


def test_load_tps_stations_wrapper_matches_load_tps(tmp_path):
    """load_tps_stations() must return the same records as load_tps().stations."""
    p = _write_full_xml(
        tmp_path,
        stations=[
            {"stationid": "1", "abbrev": "ASCOT", "stanox": "", "crscode": "",
             "longname": "Ascot", "easting": "0", "northing": "0"},
        ],
    )
    assert load_tps_stations(p) == load_tps(p).stations


# ---------------------------------------------------------------------------
# build_tps_line_networks
# ---------------------------------------------------------------------------

def test_build_line_networks_id_and_name():
    lines = [TpsLine("1", "Edinburgh-Inverness")]
    networks = build_tps_line_networks(lines)
    assert len(networks) == 1
    n = networks[0]
    assert n.id == "net_tpsline_1"
    assert any(nm.name == "Edinburgh-Inverness" for nm in n.names)


def test_build_line_networks_designator():
    lines = [TpsLine("42", "Some Line")]
    networks = build_tps_line_networks(lines)
    registers = {d.register_name for d in networks[0].designators}
    assert "NR-TPS-Line" in registers
    entry = next(d.entry for d in networks[0].designators if d.register_name == "NR-TPS-Line")
    assert entry == "42"


def test_build_line_networks_no_net_element_refs():
    """Route→line link not populated in this export; networks must have no netElementRefs."""
    lines = [TpsLine("1", "A Line")]
    networks = build_tps_line_networks(lines)
    nr = networks[0].network_resource
    assert nr is None or nr.element_collection_unordered is None or \
           nr.element_collection_unordered.net_element_refs == []


def test_build_line_networks_count():
    lines = [TpsLine(str(i), f"Line {i}") for i in range(10)]
    assert len(build_tps_line_networks(lines)) == 10


def test_build_line_networks_empty():
    assert build_tps_line_networks([]) == []


# ---------------------------------------------------------------------------
# build_tps_signals
# ---------------------------------------------------------------------------

def test_build_signals_id_uses_signal_id():
    s = TpsSignal("99", "SIG: A1", False, "4-Aspect Colour Light (IECC)", "8")
    sigs = build_tps_signals([s])
    assert sigs[0].id == "sig_99"


def test_build_signals_name():
    s = TpsSignal("1", "SIG: D18", False, "3-Aspect Colour Light", "6")
    sigs = build_tps_signals([s])
    assert sigs[0].name == "SIG: D18"


def test_build_signals_bumper_flag():
    bumper = TpsSignal("2", "BUF", True, "Buffer Stop", "2")
    normal = TpsSignal("3", "SIG", False, "4-Aspect Colour Light (IECC)", "8")
    sigs = build_tps_signals([bumper, normal])
    assert sigs[0].is_bumper is True
    assert sigs[1].is_bumper is False


def test_build_signals_type_designator_present():
    s = TpsSignal("5", "SIG: X", False, "4-Aspect Colour Light (IECC)", "8")
    sigs = build_tps_signals([s])
    registers = {d.register_name for d in sigs[0].designators}
    assert "NR-Signal-Type" in registers
    entry = next(d.entry for d in sigs[0].designators if d.register_name == "NR-Signal-Type")
    assert entry == "4-Aspect Colour Light (IECC)"


def test_build_signals_interlocking_designator_present():
    s = TpsSignal("5", "SIG: X", False, "3-Aspect Colour Light", "6")
    sigs = build_tps_signals([s])
    registers = {d.register_name for d in sigs[0].designators}
    assert "NR-Interlocking" in registers


def test_build_signals_empty_type_omits_designator():
    s = TpsSignal("5", "SIG: X", False, "", "")
    sigs = build_tps_signals([s])
    assert sigs[0].designators == []


def test_build_signals_count():
    signals = [TpsSignal(str(i), f"SIG:{i}", False, "3-Aspect", "6") for i in range(20)]
    assert len(build_tps_signals(signals)) == 20


def test_build_signals_empty():
    assert build_tps_signals([]) == []


# ---------------------------------------------------------------------------
# TpsStation.has_chainage and stationposition parsing
# ---------------------------------------------------------------------------

def test_has_chainage_true():
    s = TpsStation("1", "GLGC", "", "", "", kmregion_id="1271", km_value_m=256000)
    assert s.has_chainage


def test_has_chainage_false_missing_region():
    s = TpsStation("1", "GLGC", "", "", "", kmregion_id="", km_value_m=1000)
    assert not s.has_chainage


def test_has_chainage_false_zero_value():
    s = TpsStation("1", "GLGC", "", "", "", kmregion_id="1271", km_value_m=0)
    assert not s.has_chainage


def test_load_station_with_stationposition(tmp_path):
    """stationposition child element must populate kmregion_id and km_value_m."""
    p = _write_xml_with_positions(
        tmp_path,
        [
            (
                {"stationid": "1", "abbrev": "GLGC", "stanox": "", "crscode": "",
                 "longname": "Glasgow Central", "easting": "0", "northing": "0"},
                {"kmregionid": "1271", "kmvalue": "256000"},
            ),
        ],
    )
    stations = load_tps_stations(p)
    assert len(stations) == 1
    s = stations[0]
    assert s.kmregion_id == "1271"
    assert s.km_value_m == 256000
    assert s.has_chainage


def test_load_station_without_stationposition(tmp_path):
    """Stations with no stationposition child must have empty kmregion_id and km_value_m=0."""
    p = _write_xml(tmp_path, [
        {"stationid": "1", "abbrev": "ASCOT", "stanox": "", "crscode": "",
         "longname": "Ascot", "easting": "0", "northing": "0"},
    ])
    s = load_tps_stations(p)[0]
    assert s.kmregion_id == ""
    assert s.km_value_m == 0
    assert not s.has_chainage


def test_load_station_decimal_kmvalue(tmp_path):
    """A decimal kmvalue string (e.g. '256000.5') must not raise ValueError."""
    p = _write_xml_with_positions(
        tmp_path,
        [
            (
                {"stationid": "1", "abbrev": "GLGC", "stanox": "", "crscode": "",
                 "longname": "Glasgow Central", "easting": "0", "northing": "0"},
                {"kmregionid": "1271", "kmvalue": "256000.5"},
            ),
        ],
    )
    stations = load_tps_stations(p)
    assert stations[0].km_value_m == pytest.approx(256000.5)


def test_load_stationposition_does_not_leak_to_next_station(tmp_path):
    """A stationposition from one station must not bleed into the next."""
    p = _write_xml_with_positions(
        tmp_path,
        [
            (
                {"stationid": "1", "abbrev": "GLGC", "stanox": "", "crscode": "",
                 "longname": "Glasgow Central", "easting": "0", "northing": "0"},
                {"kmregionid": "1271", "kmvalue": "256000"},
            ),
            (
                {"stationid": "2", "abbrev": "ASCOT", "stanox": "", "crscode": "",
                 "longname": "Ascot", "easting": "0", "northing": "0"},
                None,
            ),
        ],
    )
    stations = load_tps_stations(p)
    assert stations[0].has_chainage
    assert not stations[1].has_chainage


# ---------------------------------------------------------------------------
# WaymarkIndex — construction and interpolation
# ---------------------------------------------------------------------------

def _make_index(elr: str, positions_m: list[float], xs: list[float], ys: list[float]) -> WaymarkIndex:
    """Build a WaymarkIndex with a single ELR entry from plain lists."""
    entry = _ELRWaymarks(
        position_m=np.array(positions_m, dtype=np.float64),
        easting=np.array(xs, dtype=np.float64),
        northing=np.array(ys, dtype=np.float64),
    )
    return WaymarkIndex({elr: entry})


def test_waymark_index_known_elr():
    idx = _make_index("ECM1", [0, 1000, 2000], [100, 200, 300], [500, 500, 500])
    assert "ECM1" in idx


def test_waymark_index_unknown_elr():
    idx = _make_index("ECM1", [0, 1000], [0, 100], [0, 0])
    assert "WEB" not in idx


def test_waymark_interpolate_at_start():
    idx = _make_index("ECM1", [0, 1000, 2000], [100, 200, 300], [500, 600, 700])
    result = idx.interpolate("ECM1", 0.0)
    assert result == pytest.approx((100.0, 500.0))


def test_waymark_interpolate_at_end():
    idx = _make_index("ECM1", [0, 1000, 2000], [100, 200, 300], [500, 600, 700])
    result = idx.interpolate("ECM1", 2000.0)
    assert result == pytest.approx((300.0, 700.0))


def test_waymark_interpolate_midpoint():
    idx = _make_index("ECM1", [0, 1000, 2000], [0, 1000, 2000], [0, 0, 0])
    result = idx.interpolate("ECM1", 500.0)
    assert result == pytest.approx((500.0, 0.0))


def test_waymark_interpolate_between_segments():
    idx = _make_index("ECM1", [0, 1000, 2000], [0, 100, 300], [0, 0, 0])
    # At 1500 m, between waymarks at 1000 (x=100) and 2000 (x=300) → x=200
    result = idx.interpolate("ECM1", 1500.0)
    assert result == pytest.approx((200.0, 0.0))


def test_waymark_interpolate_below_range_returns_none():
    idx = _make_index("ECM1", [1000, 2000], [100, 200], [0, 0])
    assert idx.interpolate("ECM1", 0.0) is None


def test_waymark_interpolate_above_range_returns_none():
    idx = _make_index("ECM1", [0, 1000], [0, 100], [0, 0])
    assert idx.interpolate("ECM1", 2000.0) is None


def test_waymark_interpolate_unknown_elr_returns_none():
    idx = _make_index("ECM1", [0, 1000], [0, 100], [0, 0])
    assert idx.interpolate("WEB", 500.0) is None


def test_waymark_elr_count():
    e = _ELRWaymarks(
        position_m=np.array([0.0, 1000.0]),
        easting=np.array([0.0, 100.0]),
        northing=np.array([0.0, 0.0]),
    )
    idx = WaymarkIndex({"ECM1": e, "WEB": e})
    assert idx.elr_count() == 2


def test_build_waymark_index_drops_unknown_unit(tmp_path, capsys):
    """Waymark rows with a UNIT that is neither 'M' nor 'K' must be dropped with a warning."""
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {
            "ELR": ["ECM1", "ECM1", "ECM1"],
            "VALUE": [0.0, 1.0, 2.0],
            "UNIT": ["M", "X", "M"],  # "X" is unknown
        },
        geometry=[Point(0, 0), Point(1609, 0), Point(3219, 0)],
        crs="EPSG:27700",
    )
    shp = tmp_path / "waymarks.shp"
    gdf.to_file(str(shp))

    idx = build_waymark_index(shp)
    captured = capsys.readouterr()
    assert "dropping 1 waymarks with unknown UNIT" in captured.out
    # Only the two 'M' rows survive; the 'X' row at VALUE=1.0 is gone.
    entry = idx._index["ECM1"]
    assert len(entry.position_m) == 2


# ---------------------------------------------------------------------------
# build_operational_points — waymark integration
# ---------------------------------------------------------------------------

def test_build_op_uses_waymark_position_when_available():
    """When waymark index covers the station ELR, position is waymark-derived."""
    elr_lookup = {"1271": "ECM1"}
    # Build a simple index: ECM1 from 0 to 300000 m, x linear 0→300000, y=0
    idx = _make_index("ECM1", [0, 300000], [0, 300000], [0, 0])

    s = TpsStation("1", "GLGC", "", "", "Glasgow Central",
                   kmregion_id="1271", km_value_m=150000)
    ops = build_operational_points([s], elr_lookup=elr_lookup, waymark_index=idx)
    assert len(ops[0].gml_locations) == 1


def test_build_op_waymark_falls_back_to_raw_bng_when_elr_missing():
    """When the ELR is not in the waymark index, raw BNG coords are used if present."""
    elr_lookup = {"9999": "NOEXI"}  # ELR not in our index
    idx = _make_index("ECM1", [0, 1000], [0, 100], [0, 0])

    s = TpsStation("1", "GLGC", "", "", "Glasgow Central",
                   easting_m=259300, northing_m=665000,
                   kmregion_id="9999", km_value_m=10000)
    ops = build_operational_points([s], elr_lookup=elr_lookup, waymark_index=idx)
    # Should have fallen back to raw BNG
    assert len(ops[0].gml_locations) == 1


def test_build_op_no_position_when_waymark_out_of_range_and_no_bng():
    """No GML location added when waymark range is missed and no raw coords."""
    elr_lookup = {"1271": "ECM1"}
    idx = _make_index("ECM1", [0, 1000], [0, 100], [0, 0])

    s = TpsStation("1", "X", "", "", "Far Away",
                   kmregion_id="1271", km_value_m=999999)
    ops = build_operational_points([s], elr_lookup=elr_lookup, waymark_index=idx)
    assert ops[0].gml_locations == []


def test_build_op_no_waymark_index_uses_raw_bng():
    """Passing waymark_index=None must still yield GML from raw BNG coords."""
    s = TpsStation("1", "GLGC", "", "", "Glasgow Central",
                   easting_m=259300, northing_m=665000)
    ops = build_operational_points([s], elr_lookup=None, waymark_index=None)
    assert len(ops[0].gml_locations) == 1
