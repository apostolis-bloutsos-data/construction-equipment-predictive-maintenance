"""Download and cache the AI4I 2020 Predictive Maintenance dataset.

The source is the official UCI Machine Learning Repository (dataset id 601).
"""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

UCI_DATA_URL = "https://archive.ics.uci.edu/static/public/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip"
EXPECTED_SHAPE = (10_000, 14)


def main(force: bool = False) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_path = repo_root / "data" / "raw" / "ai4i2020.csv"
    processed_path = repo_root / "data" / "processed" / "ai4i2020.parquet"

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists() and not force:
        print(f"Raw dataset already exists: {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        print("Downloading AI4I 2020 from UCI...")
        df = pd.read_csv(UCI_DATA_URL, compression="zip")
        df.to_csv(raw_path, index=False)
        print(f"Saved raw CSV: {raw_path}")

    if df.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Unexpected dataset shape {df.shape}; expected {EXPECTED_SHAPE}."
        )

    df.to_parquet(processed_path, index=False)
    print(f"Saved processed Parquet: {processed_path}")
    print(f"Validated dataset shape: {df.shape}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the raw dataset even if a local copy already exists.",
    )
    args = parser.parse_args()
    main(force=args.force)
