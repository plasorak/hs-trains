"""Convert the NWR_GTCL GeoPackage to a RailML 3.3 infrastructure topology XML.

What is produced
----------------
For every non-superseded row in the NWR_GTCL layer the converter creates:
  - One ``netElement``  in the topology layer (abstract graph edge)
  - One ``track``       in functionalInfrastructure (geometry carrier)

Connectivity (``netRelation``) is inferred by matching segment endpoints: two
segment ends that share a coordinate (within 1 m) are connected by a relation.
``network`` elements group all elements that share the same ELR.

Functional nodes are derived from NWR_GTCL_Nodes:
  - valancy 1  →  ``bufferStop``  (dead end)
  - valancy 3  →  ``switchIS``    (Y-junction)
  - valancy 4  →  ``crossingIS``  (diamond crossing)
  - valancy ≥ 5 → ``switchIS``   (complex junction)
  - valancy 2  →  through-node; no functional element generated

When ``--tps`` is given, the TPS XML (assets/XML_p.xml) is parsed in a single
streaming pass and three categories of data are added:

  operationalPoints
      One per TPS station, carrying NR-TIPLOC, NR-STANOX and NR-CRS
      designators.  These are the named locations referenced by PPTimetable
      journeys.  The 27 stations with a non-zero BNG coordinate also receive a
      GML Point geometry in WGS84.

  networks (TPS lines)
      263 named operational lines (e.g. "Edinburgh-Inverness") added alongside
      the ELR-based networks from GTCL.  In this TPS export the route→line
      link is not populated so these networks carry a NR-TPS-Line designator
      and name only (no netElementRefs).

  signals
      70 k signals from the TPS signal catalogue, each carrying a
      NR-Signal-Type designator (e.g. "4-Aspect Colour Light (IECC)") and the
      interlocking system id.  Position data is absent in this export;
      signals are positioned as a named catalogue only and will be placed on
      track once the per-ELR chainage alignment with GTCL is implemented.

Limitations: speed limits and gradients from TPS edges are not yet linked to
GTCL segments (requires per-ELR chainage alignment, a separate task).

Usage
-----
    uv run make-gtcl-railml output.xml
    uv run make-gtcl-railml output.xml --elr ECM1 MML1   # subset of lines
    uv run make-gtcl-railml output.xml --tps              # add all TPS data
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.etree.ElementTree import ElementTree, fromstring, indent

import geopandas as gpd

from hs_trains.gtcl import (
    GPKG,
    build_functional_nodes,
    build_net_elements,
    build_net_relations,
    build_networks,
    build_tracks,
)
from hs_trains.model.infrastructure import (
    FunctionalInfrastructure,
    Infrastructure,
    RailML,
    Topology,
)
from hs_trains.tps import (
    TPS_XML,
    _WAYMARKS_SHP,
    build_operational_points,
    build_tps_line_networks,
    build_tps_signals,
    build_waymark_index,
    load_tps,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NWR_GTCL GeoPackage to RailML 3.3 infrastructure topology."
    )
    parser.add_argument("output", help="Output XML file path")
    parser.add_argument(
        "--elr",
        nargs="+",
        metavar="ELR",
        help="Only convert segments belonging to these ELRs (e.g. --elr ECM1 MML1)."
        " Omit to convert the entire network (slow, large output).",
    )
    parser.add_argument(
        "--tps",
        action="store_true",
        help="Parse the TPS XML (single pass) and add: operationalPoints (stations),"
        " named line networks, and signal catalogue"
        f" (reads {TPS_XML}; takes ~60 s).",
    )
    args = parser.parse_args()

    if not GPKG.exists():
        raise SystemExit(
            f"GeoPackage not found: {GPKG}\nRun 'uv run network-map' first to verify the asset path."
        )
    if args.tps and not TPS_XML.exists():
        raise SystemExit(f"TPS XML not found: {TPS_XML}")
    if args.tps and not _WAYMARKS_SHP.exists():
        print(f"  [warn] NWR_Waymarks.shp not found at {_WAYMARKS_SHP}; station positions will use raw BNG coords only.")

    # ------------------------------------------------------------------
    # Load layers
    # ------------------------------------------------------------------
    print(f"Loading NWR_GTCL from {GPKG} …")
    lines_bng: gpd.GeoDataFrame = gpd.read_file(GPKG, layer="NWR_GTCL")
    lines_bng = lines_bng[lines_bng["SUPERCEDED"] != "YES"].reset_index(drop=True)

    if args.elr:
        lines_bng = lines_bng[lines_bng["ELR"].isin(args.elr)].reset_index(drop=True)
        if lines_bng.empty:
            raise SystemExit(f"No segments found for ELR(s): {args.elr}")

    print("Loading NWR_GTCL_Nodes …")
    nodes_bng: gpd.GeoDataFrame = gpd.read_file(GPKG, layer="NWR_GTCL_Nodes")
    nodes_bng = nodes_bng[nodes_bng["SUPERCEDED"] != "YES"].reset_index(drop=True)

    # Reproject once for all GML output
    lines_wgs84 = lines_bng.to_crs("EPSG:4326")
    nodes_wgs84 = nodes_bng.to_crs("EPSG:4326")

    n_segs = len(lines_bng)
    print(
        f"  {n_segs:,} segments, {len(nodes_bng):,} nodes"
        + (f"  (filtered to ELR: {args.elr})" if args.elr else "")
    )

    # ------------------------------------------------------------------
    # Build topology
    # ------------------------------------------------------------------
    print("Building net elements …")
    net_elements = build_net_elements(lines_bng)

    print("Building net relations …")
    net_relations, endpoints = build_net_relations(lines_bng)

    print("Building networks (ELR groupings) …")
    networks = build_networks(lines_bng)

    # ------------------------------------------------------------------
    # Build functional infrastructure
    # ------------------------------------------------------------------
    print("Building tracks …")
    tracks = build_tracks(lines_bng, lines_wgs84)

    print("Building functional nodes (switches / crossings / buffer stops) …")
    switches, buffer_stops, borders, crossings = build_functional_nodes(
        nodes_bng, nodes_wgs84, endpoints
    )

    # ------------------------------------------------------------------
    # Optionally load TPS data (single streaming parse)
    # ------------------------------------------------------------------
    operational_points = []
    tps_line_networks: list = []
    tps_signals: list = []
    if args.tps:
        print(f"Loading TPS data from {TPS_XML} …")
        tps = load_tps()

        waymark_index = None
        if _WAYMARKS_SHP.exists():
            print(f"Building waymark index from {_WAYMARKS_SHP} …")
            waymark_index = build_waymark_index()
            print(f"  {waymark_index.elr_count():,} ELRs indexed")

        operational_points = build_operational_points(
            tps.stations,
            elr_lookup=tps.elr_lookup,
            waymark_index=waymark_index,
        )
        tps_line_networks = build_tps_line_networks(tps.lines)
        tps_signals = build_tps_signals(tps.signals)
        coords_count = sum(1 for op in operational_points if op.gml_locations)
        print(f"  {len(operational_points):,} operational points  ({coords_count} with coordinates)")
        print(f"  {len(tps_line_networks):,} TPS line networks")
        print(f"  {len(tps_signals):,} signals")

    # ------------------------------------------------------------------
    # Assemble and serialise
    # ------------------------------------------------------------------
    print("Assembling RailML document …")
    railml = RailML(
        infrastructure=Infrastructure(
            topology=Topology(
                net_elements=net_elements,
                net_relations=net_relations,
                networks=networks + tps_line_networks,
            ),
            functional_infrastructure=FunctionalInfrastructure(
                tracks=tracks,
                switches=switches,
                crossings=crossings,
                buffer_stops=buffer_stops,
                borders=borders,
                signals=tps_signals,
                operational_points=operational_points,
            ),
        )
    )

    ET.register_namespace("rail3", "https://www.railml.org/schemas/3.3")
    ET.register_namespace("gml", "http://www.opengis.net/gml/3.2")

    xml_str = railml.to_xml(encoding="unicode", exclude_none=True)
    root = fromstring(xml_str)
    indent(root, space="  ")

    output = Path(args.output)
    ElementTree(root).write(str(output), encoding="unicode", xml_declaration=True)

    size_mb = output.stat().st_size / 1_000_000
    print(f"\nWrote {output}  ({size_mb:.1f} MB)")
    print(f"  netElements       : {len(net_elements):,}")
    print(f"  netRelations      : {len(net_relations):,}")
    print(f"  networks          : {len(networks):,}  (ELRs)")
    print(f"  tracks            : {len(tracks):,}")
    print(f"  switches          : {len(switches):,}")
    print(f"  crossings         : {len(crossings):,}")
    print(f"  bufferStops       : {len(buffer_stops):,}")
    if tps_line_networks:
        print(f"  TPS line networks : {len(tps_line_networks):,}")
    if tps_signals:
        print(f"  signals           : {len(tps_signals):,}")
    if operational_points:
        print(f"  operationalPoints : {len(operational_points):,}")


if __name__ == "__main__":
    main()
