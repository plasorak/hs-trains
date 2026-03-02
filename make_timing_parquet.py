#!/usr/bin/env python3
"""Generate a sample berth timing Parquet file for rusty-trains.

Each row represents a berth step event: the moment a train's description
moved into a new berth on the signalling panel.

Columns produced
----------------
train_id     : str   – Train identifier / headcode, e.g. "1A23"
berth_id     : str   – Berth name (matches BerthDescription::name in the model)
timestamp_ms : int64 – Unix epoch timestamp in milliseconds
position_m   : float – Along-track distance from route origin, in metres

Usage
-----
    uv run make_timing_parquet.py berth_timing.parquet
    uv run make_timing_parquet.py berth_timing.parquet --trains 1A23 2B45 --berths 15
"""

import argparse
import random
from datetime import datetime, timezone

import pandas as pd


def make_berth_timing(
    train_id: str,
    n_berths: int,
    start_time_ms: int,
    start_pos_m: float = 0.0,
) -> list[dict]:
    """Return rows for one train moving through *n_berths* berths."""
    rng = random.Random(hash(train_id))  # reproducible per train ID
    rows = []
    t = start_time_ms
    x = start_pos_m
    for i in range(n_berths):
        rows.append(
            {
                "train_id": train_id,
                "berth_id": f"{train_id}_B{i + 1:02d}",
                "timestamp_ms": t,
                "position_m": round(x, 1),
            }
        )
        x += rng.uniform(600.0, 1400.0)   # 600–1400 m between berth boundaries
        t += int(rng.uniform(45.0, 150.0) * 1000)  # 45–150 s travel time
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

    # Trains depart 5 minutes apart from a fixed reference time.
    base_time_ms = int(
        datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc).timestamp() * 1000
    )

    all_rows: list[dict] = []
    for i, train_id in enumerate(args.trains):
        start_ms = base_time_ms + i * 5 * 60 * 1000  # 5-minute headway
        all_rows.extend(make_berth_timing(train_id, args.berths, start_ms))

    df = pd.DataFrame(all_rows).astype(
        {"train_id": "string", "berth_id": "string", "timestamp_ms": "int64", "position_m": "float64"}
    )
    df.to_parquet(args.output, index=False)

    print(f"Written {len(df)} rows ({len(args.trains)} trains × {args.berths} berths) to '{args.output}'")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
