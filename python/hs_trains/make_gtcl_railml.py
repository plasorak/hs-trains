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
  - valancy 1  →  ``bufferStop`` (dead end)
  - valancy ≥ 3 → ``switchIS``   (junction / crossing)
  - valancy 2  →  through-node; no functional element generated

Limitations: switch branch types, speed limits, signals, electrification,
platforms, and gradients are not present in the GTCL and are omitted.

Usage
-----
    uv run make-gtcl-railml output.xml
    uv run make-gtcl-railml output.xml --elr ECM MML1   # subset of lines
"""

import argparse
import uuid
import xml.etree.ElementTree as ET
from collections import defaultdict
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from typing import Literal
from xml.etree.ElementTree import ElementTree, fromstring, indent

import geopandas as gpd

from hs_trains.model.infrastructure import (
    Border,
    BufferStop,
    ElementA,
    ElementB,
    ElementCollectionUnordered,
    FunctionalInfrastructure,
    GmlLineString,
    GmlLocation,
    GmlPoint,
    GmlPosList,
    GmlPos,
    Infrastructure,
    NetElement,
    NetElementRef,
    NetRelation,
    Network,
    NetworkResource,
    RailML,
    SwitchIS,
    Topology,
    Track,
    Length,
)
from hs_trains.model.common import Designator, Name

GPKG = Path(__file__).parents[2] / "assets" / "NWR_GTCL20260309.gpkg"

# Coordinate rounding precision (metres in BNG) used to match segment endpoints.
# 1 m is generous enough to absorb floating-point noise while avoiding false matches.
_MATCH_PRECISION = 0


def _round_bng(x: float, y: float) -> tuple[int, int]:
    return (round(x, _MATCH_PRECISION), round(y, _MATCH_PRECISION))


def _poslist(coords: list[tuple[float, float]]) -> str:
    return " ".join(f"{lon:.6f} {lat:.6f}" for lon, lat in coords)


def _pospoint(x: float, y: float) -> str:
    return f"{x:.6f} {y:.6f}"


def _build_net_elements(lines_bng: gpd.GeoDataFrame) -> list[NetElement]:
    elements = []
    for _, row in lines_bng.iterrows():
        elements.append(
            NetElement(
                id=f"ne_{row['ASSETID']}",
                length=Length(quantity=Decimal(f"{row.geometry.length:.2f}")),
                designators=[
                    Designator(register_name="NR-ELR", entry=str(row["ELR"])),
                    Designator(register_name="NR-ASSETID", entry=str(row["ASSETID"])),
                ],
            )
        )
    return elements


def _build_net_relations(
    lines_bng: gpd.GeoDataFrame,
) -> tuple[list[NetRelation], set[tuple[int, int]]]:
    """Infer connectivity from shared segment endpoints.

    For each node coordinate, collect all segment ends that coincide there,
    then emit one NetRelation for every pair.  This is conservative: at a
    switch with three meeting segments it produces three relations (all
    traversal options), because GTCL carries no routing constraint.

    Also returns the set of all endpoint coordinates so callers can avoid a
    redundant pass over the geometry.
    """
    # endpoint_map[coord] = [(segment_id, position), ...]
    # position: "0" = start of segment, "1" = end of segment
    endpoint_map: dict[tuple[int, int], list[tuple[str, Literal["0", "1"]]]] = defaultdict(list)
    for _, row in lines_bng.iterrows():
        seg_id = f"ne_{row['ASSETID']}"
        coords = list(row.geometry.coords)
        endpoint_map[_round_bng(*coords[0])].append((seg_id, "0"))
        endpoint_map[_round_bng(*coords[-1])].append((seg_id, "1"))

    relations = []
    rel_idx = 0
    for endpoints in endpoint_map.values():
        if len(endpoints) < 2:
            continue
        for (id_a, pos_a), (id_b, pos_b) in combinations(endpoints, 2):
            relations.append(
                NetRelation(
                    id=f"nr_{rel_idx:07d}",
                    navigability="Both",
                    position_on_a=pos_a,
                    position_on_b=pos_b,
                    element_a=ElementA(ref=id_a),
                    element_b=ElementB(ref=id_b),
                )
            )
            rel_idx += 1
    return relations, set(endpoint_map.keys())


def _build_networks(lines_bng: gpd.GeoDataFrame) -> list[Network]:
    elr_to_ids: dict[str, list[str]] = defaultdict(list)
    for _, row in lines_bng.iterrows():
        elr_to_ids[str(row["ELR"])].append(f"ne_{row['ASSETID']}")

    return [
        Network(
            id=f"net_{elr}",
            names=[Name(name=elr, language="en")],
            designators=[Designator(register_name="NR-ELR", entry=elr)],
            network_resource=NetworkResource(
                element_collection_unordered=ElementCollectionUnordered(
                    net_element_refs=[NetElementRef(ref=i) for i in ids]
                )
            ),
        )
        for elr, ids in sorted(elr_to_ids.items())
    ]


def _build_tracks(
    lines_bng: gpd.GeoDataFrame, lines_wgs84: gpd.GeoDataFrame
) -> list[Track]:
    tracks = []
    for (_, row_bng), (_, row_wgs84) in zip(
        lines_bng.iterrows(), lines_wgs84.iterrows()
    ):
        asset_id = str(row_bng["ASSETID"])
        tracks.append(
            Track(
                id=f"track_{asset_id}",
                net_element_ref=f"ne_{asset_id}",
                designators=[
                    Designator(register_name="NR-ELR", entry=str(row_bng["ELR"])),
                    Designator(register_name="NR-ASSETID", entry=asset_id),
                ],
                gml_locations=[
                    GmlLocation(
                        line_string=GmlLineString(
                            pos_list=GmlPosList(
                                root=_poslist(list(row_wgs84.geometry.coords))
                            )
                        )
                    )
                ],
            )
        )
    return tracks


def _build_functional_nodes(
    nodes_bng: gpd.GeoDataFrame,
    nodes_wgs84: gpd.GeoDataFrame,
    segment_endpoints: set[tuple[int, int]],
) -> tuple[list[SwitchIS], list[BufferStop], list[Border]]:
    """Only emit functional nodes whose coordinates appear in segment_endpoints.

    This avoids including the entire network's node set when only a subset of
    segments (e.g. a single ELR) has been selected.
    """
    switches: list[SwitchIS] = []
    buffer_stops: list[BufferStop] = []
    borders: list[Border] = []

    for (_, row_bng), (_, row_wgs84) in zip(
        nodes_bng.iterrows(), nodes_wgs84.iterrows()
    ):
        coord = _round_bng(row_bng.geometry.x, row_bng.geometry.y)
        if coord not in segment_endpoints:
            continue

        valancy = row_bng.get("VALANCY")
        asset_id = str(row_bng.get("ASSETID") or uuid.uuid4())
        lon, lat = row_wgs84.geometry.x, row_wgs84.geometry.y
        gml = [GmlLocation(point=GmlPoint(pos=GmlPos(root=_pospoint(lon, lat))))]

        if valancy == 1.0:
            # Dead end — treat as buffer stop; could be a border for open-ended
            # networks but we have no way to distinguish from GTCL alone.
            buffer_stops.append(BufferStop(id=f"bs_{asset_id}", gml_locations=gml))
        elif valancy is not None and valancy >= 3.0:
            switches.append(SwitchIS(id=f"sw_{asset_id}", gml_locations=gml))
        # valancy 2 = simple through-node; no functional element needed

    return switches, buffer_stops, borders


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NWR_GTCL GeoPackage to RailML 3.3 infrastructure topology."
    )
    parser.add_argument("output", help="Output XML file path")
    parser.add_argument(
        "--elr",
        nargs="+",
        metavar="ELR",
        help="Only convert segments belonging to these ELRs (e.g. --elr ECM MML1)."
        " Omit to convert the entire network (slow, large output).",
    )
    args = parser.parse_args()

    if not GPKG.exists():
        raise SystemExit(
            f"GeoPackage not found: {GPKG}\nRun 'uv run network-map' first to verify the asset path."
        )

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
    net_elements = _build_net_elements(lines_bng)

    print("Building net relations …")
    net_relations, endpoints = _build_net_relations(lines_bng)

    print("Building networks (ELR groupings) …")
    networks = _build_networks(lines_bng)

    # ------------------------------------------------------------------
    # Build functional infrastructure
    # ------------------------------------------------------------------
    print("Building tracks …")
    tracks = _build_tracks(lines_bng, lines_wgs84)

    print("Building functional nodes (switches / buffer stops) …")
    switches, buffer_stops, borders = _build_functional_nodes(
        nodes_bng, nodes_wgs84, endpoints
    )

    # ------------------------------------------------------------------
    # Assemble and serialise
    # ------------------------------------------------------------------
    print("Assembling RailML document …")
    railml = RailML(
        infrastructure=Infrastructure(
            topology=Topology(
                net_elements=net_elements,
                net_relations=net_relations,
                networks=networks,
            ),
            functional_infrastructure=FunctionalInfrastructure(
                tracks=tracks,
                switches=switches,
                buffer_stops=buffer_stops,
                borders=borders,
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
    print(f"  netElements  : {len(net_elements):,}")
    print(f"  netRelations : {len(net_relations):,}")
    print(f"  networks     : {len(networks):,}  (ELRs)")
    print(f"  tracks       : {len(tracks):,}")
    print(f"  switches     : {len(switches):,}")
    print(f"  bufferStops  : {len(buffer_stops):,}")


if __name__ == "__main__":
    main()
