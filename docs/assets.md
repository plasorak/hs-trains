# Network Assets — GeoPackage

The file `assets/NWR_GTCL20260309.gpkg` is the primary geographic data source for hs-trains.

## What it is

A [GeoPackage](https://www.geopackage.org/) (SQLite container, `.gpkg`) exported from the **Network Rail Geographic Track Centre Line (GTCL)** dataset, dated **9 March 2026** (encoded in the filename as `20260309`).

The GTCL is the canonical digital representation of the GB main-line rail network maintained by Network Rail. It captures the physical centre-line geometry of every track, together with the topological node structure that links track sections at junctions, crossings, and termini.

All coordinates are in **EPSG:27700 (British National Grid)**, unit: metres. Reprojection to WGS-84 (EPSG:4326) is required for map rendering — the `network_map.py` script does this automatically.

---

## Layers

### `NWR_GTCL` — track segments (48 078 rows)

The main geometry layer. Each row is a single directed track segment represented as a `LineString`.

| Column | Type | Description |
|---|---|---|
| `ASSETID` | string | Unique asset identifier within the Network Rail asset register. Nearly all values are unique; a handful of segments share an ID where one record supersedes another. |
| `ELR` | string | **Engineer's Line Reference** — the primary human-readable track identifier used across UK rail (e.g. `ECM1` = East Coast Main Line section 1, `MLN1` = Midland Main Line North). There are **1 403 distinct ELRs** in this extract. |
| `TRID` | string | Track Record ID — a finer-grained identifier below ELR level; 892 unique values, nullable in a small number of rows. |
| `SOURCE` | string | Data provenance: `"New Feature Extracted"` (79 % of rows) or a second value for manually digitised additions. |
| `SUPERCEDED` | string | `"YES"` / `"NO"` flag. Superseded segments are retained for audit purposes but should be filtered out for operational use — the majority (`"NO"`) are current. |
| `GEOMETRY_UPDATED` | string | Whether the geometry has been revised since initial capture (`"YES"` / `"NO"`). |
| `geometry` | LineString | 2-D centre-line geometry in British National Grid (EPSG:27700). |

### `NWR_GTCL_Nodes` — junctions (37 477 rows)

Topological nodes connecting the track segments — roughly one node per switch, crossing, or terminal.

| Column | Type | Description |
|---|---|---|
| `ASSETID` | string | Unique asset identifier. |
| `VALANCY` | float | Number of track segments meeting at this node (i.e. the graph degree). Mean ≈ 2.6; max = 4. A valancy of 1 is a buffer stop or network boundary; 3 is a typical Y-junction; 4 is a crossover or diamond crossing. |
| `SOURCE` | string | Always `"FE EXTRACTION"` in this layer. |
| `SUPERCEDED` | string | `"YES"` / `"NO"`, same semantics as above. |
| `GEOMETRY_UPDATED` | string | `"YES"` / `"NO"`. |
| `geometry` | Point | Node position in EPSG:27700. |

### `NWR_GTCL_NewLinks` — supplementary segments (3 257 rows)

A smaller set of additional LineString segments extracted in a separate pass, likely representing sidings, depots, or connections digitised after the main extract.

| Column | Type | Description |
|---|---|---|
| `SOURCE` | string | `"New Feature Extracted"` (99 %) or a second source. |
| `CLOSEST_ELR` | string | ELR of the nearest main-line segment — a proximity link rather than a strict topological membership. 243 distinct values. |
| `geometry` | LineString | Track centre-line in EPSG:27700. |

### `NWR_GTCL_NewNodes` — supplementary nodes (3 491 rows)

Nodes connecting the supplementary link segments.

| Column | Type | Description |
|---|---|---|
| `SOURCE` | string | Always `"FE EXTRACTION"`. |
| `VALANCY` | float | Graph degree at this node. Mean ≈ 1.9; max = 4. Lower mean than the main nodes layer reflects more terminal endpoints in the supplementary set. |
| `geometry` | Point | Node position in EPSG:27700. |

---

## Key domain concepts

**Engineer's Line Reference (ELR)** — the primary identifier for a track route in UK rail. An ELR covers the full length of a named line and is subdivided into mileage-based sections. The GTCL uses it as a grouping key: all segments belonging to a section of the East Coast Main Line share the same ELR (e.g. `ECM1`–`ECM9`), for example. ELRs are the bridge between this geometry file and timetable/operational data sources.

**Valancy** — the number of edges connected at a node. In a simple bidirectional track, interior nodes have valancy 2. Switches introduce valancy 3; diamond crossings introduce valancy 4. Valancy 1 marks a dead end (buffer stop or an open network boundary).

**Superseded records** — Network Rail retains superseded geometry rather than deleting it, so that audit and history queries remain possible. Filter `SUPERCEDED = 'NO'` to work with the current network.

---

## Usage in hs-trains

The file is not yet read by the Rust simulation engine. The intended integration point is `src/assets.rs`, which is currently a stub. The `BerthDescription` type in `src/core/model.rs` (trajectory, signals, overlaps) sketches the shape that parsed GTCL geometry would eventually populate.

The Python script `python/hs_trains/network_map.py` provides an immediate way to visualise the data:

```bash
uv run network-map                    # generates network_map.html and opens it in a browser
uv run network-map --simplify 0       # full resolution (larger output)
uv run network-map --no-open          # write the file without opening a browser
```
