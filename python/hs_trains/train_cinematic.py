"""Interactive animated train visualisation on OpenStreetMap.

Reads a simulation Parquet file (produced by `hs-trains`) that must contain
`lon_deg`, `lat_deg`, and `power_kw` columns (generated when the infrastructure
XML includes GML track geometry).  Produces a self-contained HTML file with a
Plotly animation: trains move along the map, coloured by speed, sized by power.

CLI::

    train-cinematic simulation.parquet
    train-cinematic simulation.parquet -o out.html --frame-step 5 --no-open
"""

from pathlib import Path

import polars as pl
import plotly.graph_objects as go
import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    parquet: Path = typer.Argument(..., help="Simulation output Parquet file"),
    output: Path = typer.Option(Path("train_cinematic.html"), "-o", "--output", help="Output HTML path"),
    frame_step: int = typer.Option(1, "--frame-step", help="Keep every Nth time step as an animation frame (reduces file size)"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open output in browser after saving"),
) -> None:
    typer.echo(f"Loading {parquet} …")
    df = pl.read_parquet(parquet)

    if "event_kind" in df.schema:
        df = df.filter(pl.col("event_kind") == "physics_tick")

    if "lon_deg" not in df.schema or "lat_deg" not in df.schema:
        typer.echo(
            "Error: Parquet file is missing lon_deg / lat_deg columns.\n"
            "Re-run the simulation with an infrastructure file that contains GML track geometry.",
            err=True,
        )
        raise typer.Exit(1)

    df = df.drop_nulls(subset=["lon_deg", "lat_deg"])
    if df.is_empty():
        typer.echo("Error: no rows with geographic coordinates found.", err=True)
        raise typer.Exit(1)

    # Down-sample time steps to control animation frame count / file size.
    all_times = sorted(df["time_s"].unique().to_list())
    sampled_times = all_times[::frame_step]
    df = df.filter(pl.col("time_s").is_in(sampled_times))

    # String label used as Plotly animation_frame key.
    df = df.with_columns(pl.col("time_s").cast(pl.Utf8).alias("_frame"))

    n_frames = df["_frame"].n_unique()
    n_trains = df["train_id"].n_unique()
    typer.echo(f"  {n_trains} train(s), {n_frames} animation frames")

    max_speed = float(df["speed_kmh"].drop_nulls().max() or 120.0)
    max_power = float(df["power_kw"].drop_nulls().max() or 1.0)

    frame_labels = sorted(df["_frame"].unique().to_list(), key=float)

    typer.echo("Building animation frames …")
    frames = []
    for label in frame_labels:
        fd = df.filter(pl.col("_frame") == label)
        speeds = fd["speed_kmh"].fill_null(0.0).to_list()
        powers = fd["power_kw"].fill_null(0.0).to_list()
        # Dot size: 6 px minimum, grows with power fraction up to 24 px.
        sizes = [6.0 + 18.0 * (p / max_power) for p in powers]
        hover = [
            f"{tid}<br>t = {float(label):.0f} s<br>{s:.1f} km/h<br>{p:.0f} kW"
            for tid, s, p in zip(fd["train_id"].to_list(), speeds, powers)
        ]
        frames.append(go.Frame(
            name=label,
            data=[go.Scattermapbox(
                lat=fd["lat_deg"].to_list(),
                lon=fd["lon_deg"].to_list(),
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=speeds,
                    colorscale="RdYlGn",
                    cmin=0,
                    cmax=max_speed,
                    colorbar=dict(title="Speed (km/h)", x=1.0),
                    showscale=True,
                ),
                hoverinfo="text",
                hovertext=hover,
                name="Trains",
            )],
        ))

    centre = dict(
        lat=float(df["lat_deg"].mean()),
        lon=float(df["lon_deg"].mean()),
    )

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title="Train simulation — speed & power",
            mapbox=dict(style="open-street-map", center=centre, zoom=7),
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "y": 0,
                "x": 0.05,
                "xanchor": "right",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }],
            sliders=[{
                "active": 0,
                "steps": [
                    {
                        "args": [[lb], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": f"{float(lb):.0f} s",
                        "method": "animate",
                    }
                    for lb in frame_labels
                ],
                "currentvalue": {"prefix": "Time: ", "suffix": " s"},
                "pad": {"t": 50},
                "y": 0,
                "len": 0.9,
            }],
            height=900,
            margin=dict(l=0, r=0, t=40, b=60),
        ),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs="cdn")
    size_mb = output.stat().st_size / 1_000_000
    typer.echo(f"Saved {output} ({size_mb:.1f} MB)")

    if open_browser:
        fig.show()
