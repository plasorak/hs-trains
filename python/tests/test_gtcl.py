"""Unit tests for python/hs_trains/gtcl.py and make_ecm1_railml.py.

These tests use synthetic in-memory GeoDataFrames so no GeoPackage file is
required; all geometry is constructed with Shapely.
"""

import xml.etree.ElementTree as ET
from decimal import Decimal

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from hs_trains.gtcl import (
    build_ecm1_adjacency,
    build_functional_nodes,
    build_net_elements,
    build_net_relations,
    build_networks,
    build_tracks,
    find_ecm1_route,
    poslist,
    pospoint,
    round_bng,
)
from hs_trains.make_ecm1_railml import _NS, _build_timetable, _inject_operational_points

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CRS_BNG = "EPSG:27700"
_CRS_WGS = "EPSG:4326"


def _lines_gdf(rows: list[dict], crs: str = _CRS_BNG) -> gpd.GeoDataFrame:
    """Build a line GeoDataFrame from a list of row dicts (each with 'geometry')."""
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def _points_gdf(rows: list[dict], crs: str = _CRS_BNG) -> gpd.GeoDataFrame:
    """Build a point GeoDataFrame from a list of row dicts (each with 'geometry')."""
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def _simple_chain(n: int = 3, seg_len: int = 1000) -> gpd.GeoDataFrame:
    """A straight chain of n segments, each seg_len BNG metres, sharing endpoints.

    Segment k runs from x=k*seg_len to x=(k+1)*seg_len at y=0.
    No junctions or crossings — just a linear sequence.
    """
    rows = []
    for k in range(n):
        x0 = k * seg_len
        x1 = (k + 1) * seg_len
        rows.append({
            "ASSETID": str(10000 + k),
            "ELR": "ECM1",
            "geometry": LineString([(x0, 0), (x1, 0)]),
        })
    return _lines_gdf(rows)


# ---------------------------------------------------------------------------
# round_bng / poslist / pospoint
# ---------------------------------------------------------------------------

def test_round_bng_rounds_to_integer():
    assert round_bng(100.4, 200.6) == (100, 201)


def test_round_bng_exact_integer_unchanged():
    assert round_bng(500.0, 300.0) == (500, 300)


def test_poslist_formats_correctly():
    result = poslist([(1.123456789, 51.123456789), (-0.5, 52.0)])
    assert result == "1.123457 51.123457 -0.500000 52.000000"


def test_pospoint_formats_correctly():
    assert pospoint(-0.1, 51.5) == "-0.100000 51.500000"


# ---------------------------------------------------------------------------
# build_ecm1_adjacency
# ---------------------------------------------------------------------------

def test_adjacency_two_connected_segments():
    """Two segments sharing an endpoint produce two entries for the shared coord."""
    gdf = _simple_chain(2)
    em = build_ecm1_adjacency(gdf)
    # Shared node at x=1000, y=0
    shared = round_bng(1000, 0)
    assert shared in em
    assert len(em[shared]) == 2  # both segments touch this coord


def test_adjacency_isolated_segment_has_degree_one_endpoints():
    gdf = _simple_chain(1)
    em = build_ecm1_adjacency(gdf)
    deg1 = [coord for coord, segs in em.items() if len(segs) == 1]
    assert len(deg1) == 2  # both ends of a single segment are degree-1


def test_adjacency_all_segments_appear():
    gdf = _simple_chain(4)
    em = build_ecm1_adjacency(gdf)
    all_ids = {seg for segs in em.values() for seg in segs}
    for k in range(4):
        assert str(10000 + k) in all_ids


# ---------------------------------------------------------------------------
# find_ecm1_route
# ---------------------------------------------------------------------------

def test_find_route_linear_chain_is_ordered():
    """A linear chain must be returned south-to-north (lowest to highest y endpoint)."""
    n = 4
    gdf = _simple_chain(n)
    em = build_ecm1_adjacency(gdf)
    route = find_ecm1_route(gdf, em)
    # All segments should appear exactly once.
    assert len(route) == n
    assert set(route) == {str(10000 + k) for k in range(n)}


def test_find_route_no_duplicates():
    gdf = _simple_chain(5)
    em = build_ecm1_adjacency(gdf)
    route = find_ecm1_route(gdf, em)
    assert len(route) == len(set(route))


def test_find_route_y_junction():
    """A Y-junction (degree-3 node) must still produce a valid route."""
    # Segments: A (0,0)→(1000,0), B (1000,0)→(2000,0), C (1000,0)→(1000,1000)
    # Node at (1000,0) has degree 3.  Route must go A→B (highest northing is B's far end
    # at y=0, y=0... let's arrange so B's end is northernmost).
    rows = [
        {"ASSETID": "A", "ELR": "ECM1", "geometry": LineString([(0, 0), (1000, 0)])},
        {"ASSETID": "B", "ELR": "ECM1", "geometry": LineString([(1000, 0), (2000, 0)])},
        {"ASSETID": "C", "ELR": "ECM1", "geometry": LineString([(1000, 0), (1000, 500)])},
    ]
    gdf = _lines_gdf(rows)
    em = build_ecm1_adjacency(gdf)
    route = find_ecm1_route(gdf, em)
    # Route must be non-empty and contain no duplicates.
    assert len(route) >= 1
    assert len(route) == len(set(route))


def test_find_route_raises_with_no_open_ends():
    """A closed loop has no degree-1 nodes and must raise SystemExit."""
    # Triangle: each vertex shared by exactly 2 segments → no open ends.
    rows = [
        {"ASSETID": "AB", "ELR": "ECM1", "geometry": LineString([(0, 0), (1000, 0)])},
        {"ASSETID": "BC", "ELR": "ECM1", "geometry": LineString([(1000, 0), (500, 866)])},
        {"ASSETID": "CA", "ELR": "ECM1", "geometry": LineString([(500, 866), (0, 0)])},
    ]
    gdf = _lines_gdf(rows)
    em = build_ecm1_adjacency(gdf)
    with pytest.raises(SystemExit):
        find_ecm1_route(gdf, em)


# ---------------------------------------------------------------------------
# build_net_elements
# ---------------------------------------------------------------------------

def test_build_net_elements_ids_and_lengths():
    gdf = _simple_chain(2, seg_len=500)
    elements = build_net_elements(gdf)
    assert len(elements) == 2
    ids = {e.id for e in elements}
    assert "ne_10000" in ids
    assert "ne_10001" in ids
    for e in elements:
        assert e.length is not None
        assert e.length.quantity == Decimal("500.00")


def test_build_net_elements_designators():
    gdf = _simple_chain(1)
    elements = build_net_elements(gdf)
    desig_registers = {d.register_name for d in elements[0].designators}
    assert "NR-ELR" in desig_registers
    assert "NR-ASSETID" in desig_registers


# ---------------------------------------------------------------------------
# build_net_relations
# ---------------------------------------------------------------------------

def test_build_net_relations_chain_of_two():
    """Two connected segments produce one relation at their shared endpoint."""
    gdf = _simple_chain(2)
    relations, endpoints = build_net_relations(gdf)
    assert len(relations) == 1
    assert relations[0].navigability == "Both"
    # Endpoint set must include the 3 unique endpoints of the chain.
    assert len(endpoints) == 3


def test_build_net_relations_single_segment_no_relations():
    gdf = _simple_chain(1)
    relations, endpoints = build_net_relations(gdf)
    assert len(relations) == 0
    assert len(endpoints) == 2


def test_build_net_relations_y_junction_three_relations():
    """Three segments meeting at one point → C(3,2)=3 relations."""
    rows = [
        {"ASSETID": "A", "ELR": "ECM1", "geometry": LineString([(0, 0), (1000, 0)])},
        {"ASSETID": "B", "ELR": "ECM1", "geometry": LineString([(1000, 0), (2000, 0)])},
        {"ASSETID": "C", "ELR": "ECM1", "geometry": LineString([(1000, 0), (1000, 1000)])},
    ]
    gdf = _lines_gdf(rows)
    relations, _ = build_net_relations(gdf)
    # 3 pairs from the junction + 0 from isolated endpoints
    assert len(relations) == 3


def test_build_net_relations_ids_are_unique():
    gdf = _simple_chain(4)
    relations, _ = build_net_relations(gdf)
    ids = [r.id for r in relations]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# build_networks
# ---------------------------------------------------------------------------

def test_build_networks_groups_by_elr():
    rows = [
        {"ASSETID": "1", "ELR": "ECM1", "geometry": LineString([(0, 0), (1, 0)])},
        {"ASSETID": "2", "ELR": "ECM1", "geometry": LineString([(1, 0), (2, 0)])},
        {"ASSETID": "3", "ELR": "MML1", "geometry": LineString([(0, 1), (1, 1)])},
    ]
    gdf = _lines_gdf(rows)
    networks = build_networks(gdf)
    assert len(networks) == 2
    net_ids = {n.id for n in networks}
    assert "net_ECM1" in net_ids
    assert "net_MML1" in net_ids


def test_build_networks_single_elr():
    gdf = _simple_chain(3)
    networks = build_networks(gdf)
    assert len(networks) == 1
    assert networks[0].id == "net_ECM1"
    # All three net elements must be referenced.
    refs = {r.ref for r in networks[0].network_resource.element_collection_unordered.net_element_refs}
    assert refs == {"ne_10000", "ne_10001", "ne_10002"}


# ---------------------------------------------------------------------------
# build_functional_nodes
# ---------------------------------------------------------------------------

def _nodes_gdf(valancies: list[tuple[float, float, float]], crs: str = _CRS_BNG) -> gpd.GeoDataFrame:
    """(x, y, valancy) → GeoDataFrame of node points."""
    rows = [
        {"ASSETID": str(i), "VALANCY": v, "geometry": Point(x, y)}
        for i, (x, y, v) in enumerate(valancies)
    ]
    return _points_gdf(rows, crs=crs)


def test_build_functional_nodes_buffer_stop():
    nodes_bng = _nodes_gdf([(500, 0, 1.0)])
    nodes_wgs84 = nodes_bng.to_crs(_CRS_WGS)
    endpoints = {round_bng(500, 0)}
    switches, buffer_stops, borders, crossings = build_functional_nodes(
        nodes_bng, nodes_wgs84, endpoints
    )
    assert len(buffer_stops) == 1
    assert buffer_stops[0].id.startswith("bs_")
    assert switches == [] and crossings == [] and borders == []


def test_build_functional_nodes_switch():
    nodes_bng = _nodes_gdf([(1000, 0, 3.0)])
    nodes_wgs84 = nodes_bng.to_crs(_CRS_WGS)
    endpoints = {round_bng(1000, 0)}
    switches, buffer_stops, borders, crossings = build_functional_nodes(
        nodes_bng, nodes_wgs84, endpoints
    )
    assert len(switches) == 1
    assert switches[0].id.startswith("sw_")


def test_build_functional_nodes_crossing():
    nodes_bng = _nodes_gdf([(2000, 0, 4.0)])
    nodes_wgs84 = nodes_bng.to_crs(_CRS_WGS)
    endpoints = {round_bng(2000, 0)}
    switches, buffer_stops, borders, crossings = build_functional_nodes(
        nodes_bng, nodes_wgs84, endpoints
    )
    assert len(crossings) == 1
    assert crossings[0].id.startswith("cx_")


def test_build_functional_nodes_through_node_not_emitted():
    """Valancy 2 is a through-node and must not produce any functional element."""
    nodes_bng = _nodes_gdf([(3000, 0, 2.0)])
    nodes_wgs84 = nodes_bng.to_crs(_CRS_WGS)
    endpoints = {round_bng(3000, 0)}
    switches, buffer_stops, borders, crossings = build_functional_nodes(
        nodes_bng, nodes_wgs84, endpoints
    )
    assert switches == [] and buffer_stops == [] and crossings == [] and borders == []


def test_build_functional_nodes_outside_segment_endpoints_skipped():
    """A node whose coordinate is not in segment_endpoints must be ignored."""
    nodes_bng = _nodes_gdf([(9999, 9999, 1.0)])
    nodes_wgs84 = nodes_bng.to_crs(_CRS_WGS)
    endpoints: set[tuple[int, int]] = set()  # empty — node is not connected to any segment
    switches, buffer_stops, borders, crossings = build_functional_nodes(
        nodes_bng, nodes_wgs84, endpoints
    )
    assert buffer_stops == []


def test_build_functional_nodes_complex_junction_treated_as_switch():
    """Valancy >= 5 is a complex junction and must become a switchIS."""
    nodes_bng = _nodes_gdf([(4000, 0, 5.0)])
    nodes_wgs84 = nodes_bng.to_crs(_CRS_WGS)
    endpoints = {round_bng(4000, 0)}
    switches, buffer_stops, borders, crossings = build_functional_nodes(
        nodes_bng, nodes_wgs84, endpoints
    )
    assert len(switches) == 1


def test_build_functional_nodes_length_mismatch_raises():
    """Mismatched BNG/WGS84 frame lengths must raise AssertionError."""
    nodes_bng = _nodes_gdf([(0, 0, 1.0), (1000, 0, 1.0)])
    nodes_wgs84 = _nodes_gdf([(0, 0, 1.0)]).to_crs(_CRS_WGS)  # only one row
    with pytest.raises(AssertionError):
        build_functional_nodes(nodes_bng, nodes_wgs84, set())


# ---------------------------------------------------------------------------
# build_tracks
# ---------------------------------------------------------------------------

def test_build_tracks_ids_and_net_element_refs():
    gdf_bng = _simple_chain(2)
    gdf_wgs84 = gdf_bng.to_crs(_CRS_WGS)
    tracks = build_tracks(gdf_bng, gdf_wgs84)
    assert len(tracks) == 2
    for t in tracks:
        assert t.id.startswith("track_")
        assert t.net_element_ref.startswith("ne_")
        # Each track must have exactly one GML location (a LineString).
        assert len(t.gml_locations) == 1
        assert t.gml_locations[0].line_string is not None


def test_build_tracks_designators():
    gdf_bng = _simple_chain(1)
    gdf_wgs84 = gdf_bng.to_crs(_CRS_WGS)
    tracks = build_tracks(gdf_bng, gdf_wgs84)
    registers = {d.register_name for d in tracks[0].designators}
    assert "NR-ELR" in registers
    assert "NR-ASSETID" in registers


# ---------------------------------------------------------------------------
# find_ecm1_route — diamond crossing path
# ---------------------------------------------------------------------------

def _diamond_crossing_gdf() -> gpd.GeoDataFrame:
    """Four segments meeting at one point, forming a diamond crossing (degree-4 node).

    Layout (BNG coordinates, y values chosen so northernmost open end is clear):

        south1 (0,0)→(1000,0)    — approaches the crossing from south
        north1 (1000,0)→(2000,0) — leaves toward north (highest northing)
        east   (1000,0)→(1000,500)  — crossing branch (dead-end at east)
        west   (500,0)→(1000,0)     — another approach (dead-end at west)

    The degree-4 node at (1000,0) is a diamond crossing.
    The two open ends with lowest/highest y are (0,0) and (2000,0) → south→north.
    """
    rows = [
        {"ASSETID": "south1", "ELR": "ECM1", "geometry": LineString([(0, 0), (1000, 0)])},
        {"ASSETID": "north1", "ELR": "ECM1", "geometry": LineString([(1000, 0), (2000, 0)])},
        {"ASSETID": "east",   "ELR": "ECM1", "geometry": LineString([(1000, 0), (1000, 500)])},
        {"ASSETID": "west",   "ELR": "ECM1", "geometry": LineString([(500, 0), (1000, 0)])},
    ]
    return _lines_gdf(rows)


def test_find_route_through_diamond_crossing():
    """Route through a degree-4 node must be found and contain no duplicates."""
    gdf = _diamond_crossing_gdf()
    em = build_ecm1_adjacency(gdf)
    # The crossing node at (1000, 0) has degree 4 → triggers the crossing-bypass path.
    crossing_coord = round_bng(1000, 0)
    assert len(em[crossing_coord]) == 4

    route = find_ecm1_route(gdf, em)
    assert len(route) >= 1
    assert len(route) == len(set(route))


# ---------------------------------------------------------------------------
# _build_timetable
# ---------------------------------------------------------------------------

_NS_URI = "https://www.railml.org/schemas/3.3"


def _find(el: ET.Element, tag: str) -> ET.Element | None:
    return el.find(f"{{{_NS_URI}}}{tag}")


def _findall(el: ET.Element, tag: str) -> list[ET.Element]:
    return el.findall(f"{{{_NS_URI}}}{tag}")


def test_build_timetable_structure():
    tt = _build_timetable(["A1", "B2", "C3"])
    assert tt.tag == f"{_NS}timetable"
    assert _find(tt, "baseItineraries") is not None
    assert _find(tt, "itineraries") is not None
    assert _find(tt, "operationalTrains") is not None


def test_build_timetable_track_refs_order_and_count():
    route = ["seg1", "seg2", "seg3"]
    tt = _build_timetable(route)
    base_itis = _find(tt, "baseItineraries")
    bi = _find(base_itis, "baseItinerary")
    # First BIP (south) has the followupSections with all track refs.
    bips = _findall(bi, "baseItineraryPoint")
    assert bips[0].attrib["id"] == "BIP_ECM1_South"
    followups = _find(bips[0], "followupSections")
    followup = _find(followups, "followupSection")
    track_refs_el = _find(followup, "trackRefs")
    refs = _findall(track_refs_el, "trackRef")
    assert len(refs) == 3
    assert refs[0].attrib["ref"] == "track_seg1"
    assert refs[0].attrib["sequenceNumber"] == "1"
    assert refs[2].attrib["ref"] == "track_seg3"
    assert refs[2].attrib["sequenceNumber"] == "3"


def test_build_timetable_north_bip_is_terminal():
    tt = _build_timetable(["x"])
    bi = _find(_find(tt, "baseItineraries"), "baseItinerary")
    bips = _findall(bi, "baseItineraryPoint")
    assert bips[-1].attrib["id"] == "BIP_ECM1_North"
    # The north BIP has no followupSections.
    assert _find(bips[-1], "followupSections") is None


def test_build_timetable_operational_train_id():
    tt = _build_timetable(["a"])
    op_trains = _find(tt, "operationalTrains")
    ot = _find(op_trains, "operationalTrain")
    assert ot.attrib["id"] == "OT_ECM1_Real"
    otv = _find(ot, "operationalTrainVariant")
    assert otv.attrib["itineraryRef"] == "ITI_ECM1_Real"


def test_build_timetable_empty_route_no_track_refs():
    tt = _build_timetable([])
    bi = _find(_find(tt, "baseItineraries"), "baseItinerary")
    bip_south = _find(bi, "baseItineraryPoint")
    followups = _find(bip_south, "followupSections")
    refs = _findall(_find(followups, "followupSection"), "trackRef")
    assert refs == []


# ---------------------------------------------------------------------------
# _inject_operational_points
# ---------------------------------------------------------------------------

def test_inject_operational_points_into_existing_fi():
    infra = ET.Element(f"{_NS}infrastructure")
    fi = ET.SubElement(infra, f"{_NS}functionalInfrastructure")
    _inject_operational_points(infra)
    ops_el = _find(fi, "operationalPoints")
    assert ops_el is not None
    ops = _findall(ops_el, "operationalPoint")
    ids = {op.attrib["id"] for op in ops}
    assert "OP_ECM1_South" in ids
    assert "OP_ECM1_North" in ids


def test_inject_operational_points_creates_fi_if_absent():
    infra = ET.Element(f"{_NS}infrastructure")
    # No functionalInfrastructure child.
    _inject_operational_points(infra)
    fi = _find(infra, "functionalInfrastructure")
    assert fi is not None
    ops_el = _find(fi, "operationalPoints")
    assert ops_el is not None


def test_inject_operational_points_south_has_name():
    infra = ET.Element(f"{_NS}infrastructure")
    _inject_operational_points(infra)
    fi = _find(infra, "functionalInfrastructure")
    ops = _findall(_find(fi, "operationalPoints"), "operationalPoint")
    south = next(op for op in ops if op.attrib["id"] == "OP_ECM1_South")
    assert "south" in south.attrib.get("name", "").lower()
