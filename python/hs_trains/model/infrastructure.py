"""Pydantic-XML models for the RailML 3.3 infrastructure sub-schema.

Covers the elements that can be derived from a GTCL GeoPackage:
  - GML 3.2 geometry — LineString and Point for track and node locations
  - topology         — netElement, netRelation, network
  - functionalInfrastructure (limited) — track, switchIS, bufferStop, border

Elements that require data not present in GTCL (signals, speed sections,
electrification, platforms, …) are intentionally omitted.
"""

from decimal import Decimal
from typing import Literal, Optional

from pydantic import model_validator
from pydantic_xml import BaseXmlModel, RootXmlModel, attr, element

from hs_trains.model.common import (
    NS,
    _NS,
    _NSMAP,
    _Base,
    _make_id,
    Designator,
    Name,
)

# GML 3.2 namespace — used for geometry elements embedded inside rail3 elements.
_GML_URI = "http://www.opengis.net/gml/3.2"
_GML = "gml"

# Extended namespace map: both rail3 and gml must be declared on any element
# that contains GML geometry so that serialised XML is valid.
_INFRA_NSMAP = {**_NSMAP, _GML: _GML_URI}


class _InfraBase(BaseXmlModel, nsmap=_INFRA_NSMAP):
    """Base for infrastructure elements that may embed GML geometry."""


# ---------------------------------------------------------------------------
# GML 3.2 geometry primitives
# ---------------------------------------------------------------------------


class GmlPosList(RootXmlModel[str], tag="posList", ns=_GML, nsmap={_GML: _GML_URI}):
    """Space-separated 'lon lat' pairs describing a line: lon1 lat1 lon2 lat2 …"""


class GmlLineString(BaseXmlModel, tag="LineString", ns=_GML, nsmap={_GML: _GML_URI}):
    srs_name: str = attr(name="srsName", default="urn:ogc:def:crs:EPSG::4326")
    pos_list: GmlPosList = element(tag="posList", ns=_GML)


class GmlPos(RootXmlModel[str], tag="pos", ns=_GML, nsmap={_GML: _GML_URI}):
    """Space-separated 'lon lat' pair for a single point."""


class GmlPoint(BaseXmlModel, tag="Point", ns=_GML, nsmap={_GML: _GML_URI}):
    srs_name: str = attr(name="srsName", default="urn:ogc:def:crs:EPSG::4326")
    pos: GmlPos = element(tag="pos", ns=_GML)


class GmlLocation(_InfraBase, tag="gmlLocation", ns=_NS):
    """Wrapper element holding either a LineString or a Point geometry."""

    line_string: Optional[GmlLineString] = element(tag="LineString", ns=_GML, default=None)
    point: Optional[GmlPoint] = element(tag="Point", ns=_GML, default=None)

    @model_validator(mode="after")
    def _exactly_one_geometry(self) -> "GmlLocation":
        if (self.line_string is None) == (self.point is None):
            raise ValueError("GmlLocation must have exactly one of line_string or point")
        return self


# ---------------------------------------------------------------------------
# Topology — netElement
# ---------------------------------------------------------------------------


class Length(_Base, tag="length", ns=_NS):
    """Declared length of a net element in metres."""

    quantity: Decimal = attr(name="quantity")


class NetElement(_Base, tag="netElement", ns=_NS):
    """Abstract directed edge in the network graph.

    In GTCL each row of NWR_GTCL maps to one NetElement.  The ELR and ASSETID
    are stored as Designators; the length is computed from the projected geometry.
    """

    id: str = attr(name="id", default_factory=_make_id)
    length: Optional[Length] = element(tag="length", ns=_NS, default=None)
    designators: list[Designator] = element(tag="designator", ns=_NS, default_factory=list)


# ---------------------------------------------------------------------------
# Topology — netRelation
# ---------------------------------------------------------------------------


class ElementA(_Base, tag="elementA", ns=_NS):
    """Reference to the first net element in a relation."""

    ref: str = attr(name="ref")


class ElementB(_Base, tag="elementB", ns=_NS):
    """Reference to the second net element in a relation."""

    ref: str = attr(name="ref")


class NetRelation(_Base, tag="netRelation", ns=_NS):
    """Connectivity between two net element ends.

    positionOnA/B: "0" = start of that element, "1" = end of that element.
    A valancy-2 node (simple through-node) produces one relation.
    A valancy-3 node (switch) produces C(3,2)=3 relations.
    """

    id: str = attr(name="id", default_factory=_make_id)
    navigability: Literal["AB", "BA", "Both", "None"] = attr(
        name="navigability", default="Both"
    )
    position_on_a: Literal["0", "1"] = attr(name="positionOnA")
    position_on_b: Literal["0", "1"] = attr(name="positionOnB")
    element_a: ElementA = element(tag="elementA", ns=_NS)
    element_b: ElementB = element(tag="elementB", ns=_NS)


# ---------------------------------------------------------------------------
# Topology — network (ELR groupings)
# ---------------------------------------------------------------------------


class NetElementRef(_Base, tag="netElementRef", ns=_NS):
    ref: str = attr(name="ref")


class ElementCollectionUnordered(_Base, tag="elementCollectionUnordered", ns=_NS):
    net_element_refs: list[NetElementRef] = element(
        tag="netElementRef", ns=_NS, default_factory=list
    )


class NetworkResource(_Base, tag="networkResource", ns=_NS):
    element_collection_unordered: Optional[ElementCollectionUnordered] = element(
        tag="elementCollectionUnordered", ns=_NS, default=None
    )


class Network(_Base, tag="network", ns=_NS):
    """Named grouping of net elements — one per ELR in the GTCL conversion."""

    id: str = attr(name="id", default_factory=_make_id)
    names: list[Name] = element(tag="name", ns=_NS, default_factory=list)
    designators: list[Designator] = element(tag="designator", ns=_NS, default_factory=list)
    network_resource: Optional[NetworkResource] = element(
        tag="networkResource", ns=_NS, default=None
    )


# ---------------------------------------------------------------------------
# Topology container
# ---------------------------------------------------------------------------


class Topology(_Base, tag="topology", ns=_NS):
    net_elements: list[NetElement] = element(tag="netElement", ns=_NS, default_factory=list)
    net_relations: list[NetRelation] = element(
        tag="netRelation", ns=_NS, default_factory=list
    )
    networks: list[Network] = element(tag="network", ns=_NS, default_factory=list)


# ---------------------------------------------------------------------------
# Functional infrastructure — tracks
# ---------------------------------------------------------------------------


class Track(_InfraBase, tag="track", ns=_NS):
    """Physical track section.  In GTCL each NWR_GTCL segment maps to one Track.

    The GML geometry gives the centre-line shape.  The net_element_ref links back
    to the corresponding NetElement in the topology layer.
    """

    id: str = attr(name="id", default_factory=_make_id)
    # TODO: In RailML 3.3 the Track→NetElement link belongs in a child
    # networkLocation element, not as a direct attribute.  This is a pragmatic
    # extension that works for our output but will fail schema validation.
    net_element_ref: Optional[str] = attr(name="netElementRef", default=None)
    designators: list[Designator] = element(tag="designator", ns=_NS, default_factory=list)
    gml_locations: list[GmlLocation] = element(
        tag="gmlLocation", ns=_NS, default_factory=list
    )


# ---------------------------------------------------------------------------
# Functional infrastructure — track nodes
# ---------------------------------------------------------------------------


class SwitchIS(_InfraBase, tag="switchIS", ns=_NS):
    """A junction where two or more routes diverge (valancy ≥ 3).

    Branch types and speeds are not available from GTCL — only position.
    """

    id: str = attr(name="id", default_factory=_make_id)
    gml_locations: list[GmlLocation] = element(
        tag="gmlLocation", ns=_NS, default_factory=list
    )


class BufferStop(_InfraBase, tag="bufferStop", ns=_NS):
    """Terminal end of a track (valancy = 1 in GTCL)."""

    id: str = attr(name="id", default_factory=_make_id)
    gml_locations: list[GmlLocation] = element(
        tag="gmlLocation", ns=_NS, default_factory=list
    )


class Border(_InfraBase, tag="border", ns=_NS):
    """Open network boundary — a valancy-1 node at the edge of the modelled area."""

    id: str = attr(name="id", default_factory=_make_id)
    # True when no further network is modelled beyond this point.
    is_open_end: bool = attr(name="isOpenEnd", default=True)
    gml_locations: list[GmlLocation] = element(
        tag="gmlLocation", ns=_NS, default_factory=list
    )


# ---------------------------------------------------------------------------
# Functional infrastructure container
# ---------------------------------------------------------------------------


class FunctionalInfrastructure(_Base, tag="functionalInfrastructure", ns=_NS):
    tracks: list[Track] = element(tag="track", ns=_NS, default_factory=list)
    switches: list[SwitchIS] = element(tag="switchIS", ns=_NS, default_factory=list)
    buffer_stops: list[BufferStop] = element(
        tag="bufferStop", ns=_NS, default_factory=list
    )
    borders: list[Border] = element(tag="border", ns=_NS, default_factory=list)


# ---------------------------------------------------------------------------
# Top-level containers
# ---------------------------------------------------------------------------


class Infrastructure(_Base, tag="infrastructure", ns=_NS):
    topology: Optional[Topology] = element(tag="topology", ns=_NS, default=None)
    functional_infrastructure: Optional[FunctionalInfrastructure] = element(
        tag="functionalInfrastructure", ns=_NS, default=None
    )


class RailML(_InfraBase, tag="railML", ns=_NS, nsmap=_INFRA_NSMAP):
    version: str = attr(name="version", default="3.3")
    infrastructure: Optional[Infrastructure] = element(
        tag="infrastructure", ns=_NS, default=None
    )
