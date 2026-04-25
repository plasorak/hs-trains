"""Interactive network map overlaid on OpenStreetMap.

Reads the Network Rail GTCL GeoPackage, reprojects to WGS-84, and produces a
self-contained HTML file using Plotly + the free OpenStreetMap tile layer.

Usage:
    uv run scripts/network_map.py [--output network_map.html]
    # or via entry-point:
    network-map [--output network_map.html]

The output file can be opened in any modern browser — no server required.
"""

from pathlib import Path

import geopandas as gpd
import plotly.graph_objects as go
import typer

ASSETS_DIR = Path(__file__).parents[2] / "assets"
GPKG = ASSETS_DIR / "NWR_GTCL20260309.gpkg"


def _load_layer(layer: str, simplify_m: float | None = None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(GPKG, layer=layer)
    if simplify_m is not None:
        # Simplify in the native CRS (metres) before reprojecting to keep accuracy.
        gdf["geometry"] = gdf.geometry.simplify(simplify_m, preserve_topology=True)
    return gdf.to_crs("EPSG:4326")


def _lines_trace(gdf: gpd.GeoDataFrame) -> go.Scattermapbox:
    """Pack all line segments into a single trace using None as segment separators."""
    lats: list[float | None] = []
    lons: list[float | None] = []
    hover: list[str | None] = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        # Flatten MultiLineString if present.
        parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        text = f"ELR: {row['ELR']}<br>ID: {row['ASSETID']}"
        for part in parts:
            coords = list(part.coords)
            lats += [c[1] for c in coords] + [None]
            lons += [c[0] for c in coords] + [None]
            # Attach hover to the midpoint of each segment so it appears on hover.
            hover += [text] * len(coords) + [None]

    return go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode="lines",
        line=dict(width=1.5, color="#1f6fca"),
        hoverinfo="text",
        hovertext=hover,
        name="Track segments",
    )


def _nodes_trace(gdf: gpd.GeoDataFrame) -> go.Scattermapbox:
    lats = gdf.geometry.y.tolist()
    lons = gdf.geometry.x.tolist()
    hover = [
        f"ID: {row['ASSETID']}<br>Valancy: {row['VALANCY']}"
        for _, row in gdf.iterrows()
    ]
    return go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode="markers",
        marker=dict(size=5, color="#e03030", opacity=0.8),
        hoverinfo="text",
        hovertext=hover,
        name="Junctions",
        # Hidden by default; click the legend entry to show.
        visible="legendonly",
    )


app = typer.Typer()


@app.command()
def main(
    output: Path = typer.Option(Path("network_map.html"), help="Output HTML path"),
    simplify: float = typer.Option(
        10.0,
        help="Geometry simplification tolerance in metres (0 = off). "
        "Reduces file size at the cost of fine detail.",
    ),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open in browser after saving"),
) -> None:
    typer.echo(f"Loading track segments from {GPKG} …")
    tol = simplify if simplify > 0 else None
    lines = _load_layer("NWR_GTCL", simplify_m=tol)

    typer.echo(f"Loading junction nodes …")
    nodes = _load_layer("NWR_GTCL_Nodes")

    typer.echo("Building figure …")
    fig = go.Figure(data=[_lines_trace(lines), _nodes_trace(nodes)])

    # Centre the view on the mean of all track coordinates.
    all_lats = lines.geometry.apply(lambda g: g.centroid.y)
    all_lons = lines.geometry.apply(lambda g: g.centroid.x)
    centre = dict(lat=float(all_lats.mean()), lon=float(all_lons.mean()))

    fig.update_layout(
        title="Network Rail Western Region – GTCL Network",
        mapbox=dict(
            style="open-street-map",
            center=centre,
            zoom=6,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="gray",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=900,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs="cdn")
    size_mb = output.stat().st_size / 1_000_000
    typer.echo(f"Saved {output} ({size_mb:.1f} MB)")

    if open_browser:
        fig.show()


if __name__ == "__main__":
    app()
