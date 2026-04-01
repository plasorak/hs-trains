# RailML 3.3 — General Reference

[RailML](https://www.railml.org/) is an open XML standard for exchanging railway data. Version 3.3 divides the domain into independent sub-schemas, each covering a different aspect of the railway.

---

## Sub-schemas

| Sub-schema | Description | Docs |
|---|---|---|
| `rollingstock` | Vehicles, formations, traction, braking, driving resistance | [rollingstock.md](rollingstock.md) |
| `infrastructure` | Tracks, switches, signals, balises, platforms, electrification | [infrastructure.md](infrastructure.md) |
| `timetable` | Train runs, stopping times, connections | — |
| `interlocking` | Signal logic, routes, overlaps, track vacancy detection (berths) | [interlocking.md](interlocking.md) |
| `visualizations` | Map layers and display hints | — |

hs-trains currently parses only the **rollingstock** sub-schema. Infrastructure support is planned (stub: `src/assets.rs`).

The official XSD schema is bundled at `railml/railML-3.3-SR1/`. A small stub (`dcterms_stub.xsd`) replaces the unavailable Dublin Core Terms namespace referenced by `common3.xsd`.

---

## Namespace and Document Root

All elements share a single namespace regardless of sub-schema, conventionally prefixed `rail3:`.

```xml
<?xml version='1.0' encoding='utf-8'?>
<rail3:railML xmlns:rail3="https://www.railml.org/schemas/3.3" version="3.3">
  <rail3:rollingstock> … </rail3:rollingstock>
  <rail3:infrastructure> … </rail3:infrastructure>
</rail3:railML>
```

The `version` attribute is fixed at `"3.3"`. Sub-schemas can appear together in one document or in separate files.

---

## Common Primitives

These types appear across multiple sub-schemas.

### `designator`

An external identifier in a named register — used on vehicles, formations, signals, operational points, and more.

| Attribute | Description |
|---|---|
| `register` | Name of the register, e.g. `"UIC"`, `"operator"` |
| `entry` | The identifier within that register, e.g. `"92 70 0 066 001-1"` |
| `description` | Optional free-text description |

```xml
<rail3:designator register="UIC" entry="92 70 0 066 001-1" />
<rail3:designator register="operator" entry="Class66-001" />
```

### `valueTable` / `valueLine` / `value`

Speed-dependent quantities (tractive effort, deceleration, driving resistance) are encoded as lookup tables rather than analytical functions.

```
valueTable   — declares axis names and units
└── valueLine (one per speed point)
    └── value (the y-value at that speed)
```

| Element | Key attribute | Description |
|---|---|---|
| `valueTable` | `xValueName`, `xValueUnit`, `yValueName`, `yValueUnit` | Axis metadata |
| `valueLine` | `xValue` | Speed in km/h |
| `value` | `yValue` | The quantity at that speed |

Linear interpolation is used between rows; values outside the table range are not extrapolated.

```xml
<rail3:valueTable xValueName="speed" xValueUnit="km/h"
                  yValueName="tractiveEffort" yValueUnit="N">
  <rail3:valueLine xValue="0">   <rail3:value yValue="270000" /></rail3:valueLine>
  <rail3:valueLine xValue="20">  <rail3:value yValue="270000" /></rail3:valueLine>
  <rail3:valueLine xValue="120"> <rail3:value yValue="66000"  /></rail3:valueLine>
</rail3:valueTable>
```

### `xs:boolean`

RailML's XSD requires lowercase `"true"` / `"false"` for boolean attributes. Python's default `str(True)` produces `"True"` which fails validation — see the `XmlBool` type alias in [rollingstock.md](rollingstock.md#xmlbool).

---

## XSD Validation

The schema is bundled in the repository at `railml/railML-3.3-SR1/` — no separate download is required. The path below is relative to the repo root.

```python
import xmlschema
from pathlib import Path

schema_path = Path("railml/railML-3.3-SR1/source/schema/railml3.xsd")
dcterms_stub = str(schema_path.parent / "dcterms_stub.xsd")

xs = xmlschema.XMLSchema(
    str(schema_path),
    locations={"http://purl.org/dc/terms/": dcterms_stub},
)
xs.validate("output.xml")
```

The Dublin Core stub is needed because `common3.xsd` references `http://purl.org/dc/terms/` but that URL is unavailable. The stub at `railml/railML-3.3-SR1/source/schema/dcterms_stub.xsd` satisfies the import without fetching the upstream schema.
