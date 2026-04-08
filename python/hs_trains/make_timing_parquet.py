#!/usr/bin/env python3
"""Generate a sample berth timing Parquet file for hs-trains.

Each row represents a berth step event: the moment a train's description
moved into a new berth on the signalling panel.

Columns produced
----------------
train_id   : str     – Train identifier / headcode, e.g. "1A23"
berth_id   : str     – Berth name (matches BerthDescription::name in the model)
elapsed_s  : float64 – Sectional running time from journey start, in seconds
length_m   : float64 – Length of this berth, in metres

Usage
-----
    uv run make_timing_parquet.py berth_timing.parquet
    uv run make_timing_parquet.py berth_timing.parquet --trains 1A23 2B45 --berths 15
"""

import argparse
import random

import polars as pl


def make_berth_timing(
    train_id: str,
    n_berths: int,
) -> list[dict]:
    """Return rows for one train moving through *n_berths* berths."""
    rng = random.Random(hash(train_id))  # reproducible per train ID
    rows = []
    for i in range(n_berths):
        elapsed_s = rng.uniform(45.0, 150.0)   # travel time for this berth
        length_m = rng.uniform(600.0, 1400.0)  # length of this berth
        rows.append(
            {
                "train_id": train_id,
                "berth_id": f"{train_id}_B{i + 1:02d}",
                "elapsed_s": round(elapsed_s, 3),
                "length_m": round(length_m, 1),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a sample berth timing Parquet file."
    )
    parser.add_argument("output", help="Output Parquet file path")
    parser.add_argument(
        "--trains",
        nargs="+",
        default=["1A23", "2B45", "3C67"],
        metavar="HEADCODE",
        help="Train identifiers to generate (default: 1A23 2B45 3C67)",
    )
    parser.add_argument(
        "--berths",
        type=int,
        default=12,
        metavar="N",
        help="Number of berths per train (default: 12)",
    )
    args = parser.parse_args()

    all_rows: list[dict] = []
    for train_id in args.trains:
        all_rows.extend(make_berth_timing(train_id, args.berths))

    df = pl.DataFrame(all_rows, schema={
        "train_id": pl.String,
        "berth_id": pl.String,
        "elapsed_s": pl.Float64,
        "length_m": pl.Float64,
    })
    df.write_parquet(args.output)

    print(f"Written {len(df)} rows ({len(args.trains)} trains × {args.berths} berths) to '{args.output}'")
    print(df)


if __name__ == "__main__":
    main()
