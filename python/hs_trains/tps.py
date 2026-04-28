"""Helpers for loading and converting TPS (Train Planning System) data to RailML.

The TPS XML (assets/XML_p.xml, ~12 M lines) is always parsed with a single
streaming ``iterparse`` pass.  Call ``load_tps()`` to obtain a ``TpsData``
object containing all extracted records; the individual ``load_tps_*``
functions are thin wrappers kept for backward compatibility.

What is extracted
-----------------
stations (``TpsStation``)
    Named timing points keyed by TIPLOC.  Carry STANOX and CRS identifiers
    that link to PPTimetable journeys and NR CORPUS.  BNG coordinates are
    present for a small minority of records.  Where a ``<stationposition>``
    child element is present the (kmregionid, kmvalue) chainage is extracted
    and used to interpolate a BNG position via the NR Waymarks index.

lines (``TpsLine``)
    Named operational lines (e.g. "Edinburgh-Inverness") from TPS ``line``
    elements.  In this export the route→line link is not populated (all
    ``route.lineid=0``), so line membership cannot be inferred automatically.

signals (``TpsSignal``)
    Signal catalogue from TPS ``signal`` elements, enriched with the
    interlocking-system type name (2-aspect, 3-aspect, 4-aspect colour-light,
    buffer stop, ground position light, …).  Positional data is absent in
    this export (``kmRegionID=0``, empty ``directed`` child); signals are
    therefore added to RailML as a named catalogue only.

elr_lookup (``dict[str, str]``)
    Mapping of TPS km-region ID → ELR code (e.g. ``"1271"`` → ``"ECM1"``).
    Built from ``kmregionmasterdesc`` elements.  Used by callers that need to
    resolve a node's ELR from its ``kmregionid`` attribute.

Geographic coordinates
----------------------
TPS stations carry their position in one of two ways:

1. Direct BNG easting/northing attributes on the ``<station>`` element.
   Most records have 0/0 (unknown); only a small cluster carry real values.

2. A ``<stationposition>`` child element with ``kmregionid`` and ``kmvalue``
   (in metres along the ELR).  ~5,961 stations have this.  The position is
   resolved by interpolating between the two nearest NR Waymarks on the same
   ELR — see ``WaymarkIndex`` and ``build_waymark_index()``.  The waymarks
   file (NWR_Waymarks.shp) gives BNG points at known mileages along each ELR.

Preference: stationposition interpolation is preferred over raw BNG coords
because it covers far more stations and is geometrically consistent with the
GTCL segment network.  Raw BNG coords are used only when no stationposition
is present.

Chainage unit conversion
------------------------
TPS ``kmvalue`` is in metres.  NR Waymarks VALUE is in miles (UNIT='M') for
nearly all ELRs.  Conversion: ``miles = metres / 1609.344``.
A small number of ELRs use UNIT='K' (km); for these: ``km = metres / 1000``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from xml.etree.ElementTree import iterparse

import numpy as np
from pyproj import Transformer

from hs_trains.model.common import Designator
from hs_trains.model.infrastructure import (
    GmlLocation,
    GmlPoint,
    GmlPos,
    Network,
    OperationalPoint,
    Signal,
)
from hs_trains.model.common import Name

TPS_XML = Path(__file__).parents[2] / "assets" / "XML_p.xml"
WAYMARKS_SHP = (
    Path(__file__).parents[2]
    / "assets"
    / "NWR_TrackModel20250210 20250306"
    / "NWR_Waymarks.shp"
)

# Reusable BNG → WGS84 transformer (always_xy=True: input is easting/northing,
# output is lon/lat as required by GML srsName=EPSG:4326).
_BNG_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

_METRES_PER_MILE = 1609.344
_METRES_PER_KM = 1000.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TpsStation:
    """Parsed representation of a single TPS <station> element."""

    station_id: str
    tiploc: str       # abbrev — the NR TIPLOC code
    stanox: str       # 5-digit NR STANOX (may be empty)
    crs: str          # 3-letter CRS / NLC code (may be empty)
    name: str         # longname — human-readable station name
    # BNG OS grid coordinates in metres from the station element; 0 = unknown.
    easting_m: int = field(default=0)
    northing_m: int = field(default=0)
    # Chainage position from <stationposition> child element.
    # kmregion_id maps to an ELR via TpsData.elr_lookup.
    # km_value_m is the distance in metres along that ELR from km=0.
    # Empty string / 0 means absent.
    kmregion_id: str = field(default="")
    km_value_m: float = field(default=0.0)

    @property
    def has_coordinates(self) -> bool:
        return self.easting_m != 0 and self.northing_m != 0

    @property
    def has_chainage(self) -> bool:
        return bool(self.kmregion_id) and self.km_value_m != 0


@dataclass
class TpsLine:
    """Parsed representation of a single TPS <line> element."""

    line_id: str
    description: str  # e.g. "Edinburgh-Inverness"


@dataclass
class TpsSignal:
    """Parsed representation of a single TPS <signal> element.

    The interlocking_type is resolved from the <interlockingsystem> lookup
    during the single parse pass (e.g. "4-Aspect Colour Light (IECC)").
    Position data is not available in this TPS export.
    """

    signal_id: str        # internal TPS id attribute
    name: str             # human-readable signal name (e.g. "SIG: D18")
    is_bumper: bool
    interlocking_type: str   # resolved name from interlockingsystem
    interlocking_sys_id: str  # raw interlockingsysid for cross-reference


@dataclass
class TpsData:
    """All records extracted from a single streaming parse of the TPS XML."""

    stations: list[TpsStation] = field(default_factory=list)
    lines: list[TpsLine] = field(default_factory=list)
    signals: list[TpsSignal] = field(default_factory=list)
    # kmregionid (str) → ELR code (str), e.g. "1271" → "ECM1"
    elr_lookup: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Waymark index — ELR chainage → BNG position
# ---------------------------------------------------------------------------


@dataclass
class _ELRWaymarks:
    """Sorted waymark arrays for a single ELR."""

    # VALUE column converted to metres along the ELR.
    position_m: np.ndarray   # shape (N,), dtype float64, strictly ascending
    easting: np.ndarray      # BNG easting, shape (N,)
    northing: np.ndarray     # BNG northing, shape (N,)


class WaymarkIndex:
    """Per-ELR sorted chainage→BNG lookup built from NWR_Waymarks.shp.

    Construct via ``build_waymark_index()``.  Call ``interpolate(elr,
    km_value_m)`` to get a (easting, northing) BNG pair for any (ELR,
    chainage) that falls within the waymarks coverage.  Returns None when
    the ELR is unknown or the chainage is outside the waymarks range.
    """

    def __init__(self, index: dict[str, _ELRWaymarks]) -> None:
        self._index = index

    def interpolate(self, elr: str, km_value_m: float) -> tuple[float, float] | None:
        """Return (easting, northing) BNG for the given ELR + chainage (metres).

        Returns None if the ELR is not in the index or the chainage is
        outside the coverage of the waymarks for that ELR.
        """
        entry = self._index.get(elr)
        if entry is None:
            return None
        pos = entry.position_m
        if km_value_m < pos[0] or km_value_m > pos[-1]:
            return None
        idx = int(np.searchsorted(pos, km_value_m))
        # Clamp to valid interpolation range (searchsorted can return len(pos))
        if idx == 0:
            return float(entry.easting[0]), float(entry.northing[0])
        if idx >= len(pos):
            return float(entry.easting[-1]), float(entry.northing[-1])
        t = (km_value_m - pos[idx - 1]) / (pos[idx] - pos[idx - 1])
        e = entry.easting[idx - 1] + t * (entry.easting[idx] - entry.easting[idx - 1])
        n = entry.northing[idx - 1] + t * (entry.northing[idx] - entry.northing[idx - 1])
        return float(e), float(n)

    def __contains__(self, elr: str) -> bool:
        return elr in self._index

    def elr_count(self) -> int:
        return len(self._index)


def build_waymark_index(shp_path: Path = WAYMARKS_SHP) -> WaymarkIndex:
    """Load NWR_Waymarks.shp and build an ELR → sorted-arrays lookup.

    The VALUE column is in miles (UNIT='M') for almost all ELRs, and in km
    (UNIT='K') for a small minority.  Both are converted to metres so the
    index is always in the same unit as TPS ``kmvalue``.
    """
    import geopandas as gpd  # defer import — only needed when called

    wm = gpd.read_file(str(shp_path))

    # Convert VALUE to metres based on UNIT column.
    miles_mask = wm["UNIT"] == "M"
    km_mask = wm["UNIT"] == "K"
    wm = wm.copy()
    unknown_mask = ~miles_mask & ~km_mask
    if unknown_mask.any():
        print(f"  [warn] dropping {unknown_mask.sum()} waymarks with unknown UNIT")
        wm = wm[~unknown_mask].copy()
        miles_mask = wm["UNIT"] == "M"
        km_mask = wm["UNIT"] == "K"

    wm["position_m"] = 0.0
    wm.loc[miles_mask, "position_m"] = wm.loc[miles_mask, "VALUE"] * _METRES_PER_MILE
    wm.loc[km_mask, "position_m"] = wm.loc[km_mask, "VALUE"] * _METRES_PER_KM

    index: dict[str, _ELRWaymarks] = {}
    for elr, group in wm.groupby("ELR"):
        group = group.sort_values("position_m")
        # Drop duplicate positions (keep first) to ensure strictly ascending array.
        group = group.drop_duplicates(subset="position_m", keep="first")
        index[str(elr)] = _ELRWaymarks(
            position_m=group["position_m"].to_numpy(dtype=np.float64),
            easting=group.geometry.x.to_numpy(dtype=np.float64),
            northing=group.geometry.y.to_numpy(dtype=np.float64),
        )
    return WaymarkIndex(index)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_tps(xml_path: Path = TPS_XML) -> TpsData:
    """Stream-parse the TPS XML in a single pass and return all extracted data.

    Elements consumed (in document order):
      kmregionmasterdesc → elr_lookup
      interlockingsystem → internal signal-type lookup (not in TpsData)
      station            → TpsData.stations  (with stationposition child if present)
      line               → TpsData.lines
      signal             → TpsData.signals  (uses interlockingsystem lookup)

    The parser reads two depth levels:
      depth 2 — top-level records (station, signal, line, …)
      depth 3 — child elements of depth-2 records (stationposition)

    Elements deeper than 3 are cleared immediately.  depth-3 elements are
    cleared after being read.  depth-2 elements are cleared at their own
    end event after all children have been processed.
    """
    data = TpsData()
    interlocking_types: dict[str, str] = {}  # id → name, built before signals
    depth = 0

    # Accumulate stationposition data for the station currently being parsed.
    # This is populated at depth==3 "end" events and consumed at depth==2 "end".
    _pending_kmregion_id: str = ""
    _pending_km_value_m: float = 0.0

    for event, elem in iterparse(str(xml_path), events=("start", "end")):
        if event == "start":
            depth += 1
            # Reset pending chainage state at the start of each depth-2 element
            # so stale data from a previous element can't leak into the next.
            if depth == 2:
                _pending_kmregion_id = ""
                _pending_km_value_m = 0
        else:  # "end"
            if depth == 3:
                # Read interesting depth-3 children before clearing.
                if elem.tag == "stationposition":
                    _pending_kmregion_id = elem.attrib.get("kmregionid", "")
                    _pending_km_value_m = float(elem.attrib.get("kmvalue", 0) or 0)
                elem.clear()

            elif depth == 2:
                tag = elem.tag
                if tag == "kmregionmasterdesc":
                    elr = elem.attrib.get("vanillatext", "").strip()
                    if elr:
                        data.elr_lookup[elem.attrib.get("id", "")] = elr

                elif tag == "interlockingsystem":
                    interlocking_types[elem.attrib.get("id", "")] = (
                        elem.attrib.get("name", "").strip()
                    )

                elif tag == "station":
                    tiploc = elem.attrib.get("abbrev", "").strip()
                    if tiploc and tiploc != "- - -":
                        data.stations.append(
                            TpsStation(
                                station_id=elem.attrib.get("stationid", ""),
                                tiploc=tiploc,
                                stanox=elem.attrib.get("stanox", "").strip(),
                                crs=elem.attrib.get("crscode", "").strip(),
                                name=elem.attrib.get("longname", "").strip(),
                                easting_m=int(elem.attrib.get("easting", 0) or 0),
                                northing_m=int(elem.attrib.get("northing", 0) or 0),
                                kmregion_id=_pending_kmregion_id,
                                km_value_m=_pending_km_value_m,
                            )
                        )

                elif tag == "line":
                    desc = elem.attrib.get("desc", "").strip()
                    if desc:
                        data.lines.append(
                            TpsLine(
                                line_id=elem.attrib.get("lineid", ""),
                                description=desc,
                            )
                        )

                elif tag == "signal":
                    sys_id = elem.attrib.get("interlockingsysid", "")
                    data.signals.append(
                        TpsSignal(
                            signal_id=elem.attrib.get("id", ""),
                            name=elem.attrib.get("name", "").strip(),
                            is_bumper=elem.attrib.get("bumper", "false").lower() == "true",
                            interlocking_type=interlocking_types.get(sys_id, ""),
                            interlocking_sys_id=sys_id,
                        )
                    )

            elif depth > 3:
                elem.clear()

            depth -= 1

    return data


def load_tps_stations(xml_path: Path = TPS_XML) -> list[TpsStation]:
    """Convenience wrapper: load only the station records.

    For most callers, prefer ``load_tps()`` which performs a single streaming
    pass over the large XML file and returns all extracted data at once.
    """
    return load_tps(xml_path).stations


# ---------------------------------------------------------------------------
# RailML builders
# ---------------------------------------------------------------------------


def build_operational_points(
    stations: list[TpsStation],
    elr_lookup: dict[str, str] | None = None,
    waymark_index: WaymarkIndex | None = None,
) -> list[OperationalPoint]:
    """Convert TpsStation records to RailML OperationalPoint objects.

    Designators per point:
      NR-TIPLOC  (always — used by PPTimetable)
      NR-STANOX  (when non-empty)
      NR-CRS     (when non-empty — passenger-visible 3-letter code)

    GML Point geometry is added when a BNG position can be resolved.  Two
    sources are tried in preference order:

    1. Waymark interpolation — requires ``elr_lookup`` and ``waymark_index``.
       The station's ``kmregion_id`` is mapped to an ELR; ``km_value_m`` is
       interpolated against the waymarks for that ELR.  Covers ~5,961 stations.

    2. Direct BNG coordinates on the station element — only a small cluster
       have non-zero easting/northing (and those are geographically suspect).
       Used as a fallback when waymark interpolation is unavailable or fails.
    """
    ops: list[OperationalPoint] = []
    for s in stations:
        designators = [Designator(register_name="NR-TIPLOC", entry=s.tiploc)]
        if s.stanox:
            designators.append(Designator(register_name="NR-STANOX", entry=s.stanox))
        if s.crs:
            designators.append(Designator(register_name="NR-CRS", entry=s.crs))

        gml_locations: list[GmlLocation] = []
        bng: tuple[float, float] | None = None

        # Prefer waymark-derived position over raw BNG coords.
        if (
            waymark_index is not None
            and elr_lookup is not None
            and s.has_chainage
        ):
            elr = elr_lookup.get(s.kmregion_id, "")
            if elr:
                bng = waymark_index.interpolate(elr, s.km_value_m)

        if bng is None and s.has_coordinates:
            bng = (float(s.easting_m), float(s.northing_m))

        if bng is not None:
            lon, lat = _BNG_TO_WGS84.transform(bng[0], bng[1])
            gml_locations.append(
                GmlLocation(
                    point=GmlPoint(pos=GmlPos(root=f"{lon:.6f} {lat:.6f}"))
                )
            )

        ops.append(
            OperationalPoint(
                id=f"op_{s.tiploc}",
                name=s.name or s.tiploc,
                designators=designators,
                gml_locations=gml_locations,
            )
        )
    return ops


def build_tps_line_networks(lines: list[TpsLine]) -> list[Network]:
    """Convert TpsLine records to RailML Network objects.

    TPS lines (e.g. "Edinburgh-Inverness") group operational routes across
    multiple ELRs.  In this export the route→line link is not populated, so
    these networks carry only the line name and a NR-TPS-Line designator; no
    netElementRefs are added.
    """
    return [
        Network(
            id=f"net_tpsline_{line.line_id}",
            names=[Name(name=line.description, language="en")],
            designators=[
                Designator(register_name="NR-TPS-Line", entry=line.line_id),
            ],
        )
        for line in lines
    ]


def build_tps_signals(signals: list[TpsSignal]) -> list[Signal]:
    """Convert TpsSignal records to RailML Signal objects.

    Each signal carries two designators:
      NR-Signal-Type   human-readable interlocking system name
      NR-Interlocking  numeric interlocking system id

    Position is not available in this TPS export; signals are a named
    catalogue only and must be positioned via chainage alignment later.
    """
    result: list[Signal] = []
    for s in signals:
        designators: list[Designator] = []
        if s.interlocking_type:
            designators.append(
                Designator(register_name="NR-Signal-Type", entry=s.interlocking_type)
            )
        if s.interlocking_sys_id:
            designators.append(
                Designator(register_name="NR-Interlocking", entry=s.interlocking_sys_id)
            )
        result.append(
            Signal(
                id=f"sig_{s.signal_id}",
                name=s.name or None,
                is_bumper=s.is_bumper,
                designators=designators,
            )
        )
    return result
