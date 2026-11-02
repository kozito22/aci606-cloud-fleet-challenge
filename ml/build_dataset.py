"""Build a multi-fleet training dataset with a fleet_id column.

The evaluator uses an unseen fleet (another seed, new robot IDs). To measure
that honestly we need several independent fleets and a train/test split by
fleet (GroupShuffleSplit). The official generator produces one fleet per run,
so this script runs simulator/generate_dataset.py unchanged once per seed and
concatenates the results adding `fleet_id`.

Usage:
    python ml/build_dataset.py                    # 16 fleets (default)
    python ml/build_dataset.py --fleets 24 --samples 4000
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "simulator" / "generate_dataset.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fleets", type=int, default=16)
    p.add_argument("--samples", type=int, default=3000, help="samples per fleet")
    p.add_argument("--base-seed", type=int, default=1000)
    p.add_argument("--output", default=str(ROOT / "data" / "training.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.fleets):
            seed = args.base_seed + i
            robots = 4 + i % 7  # 4..10 robots: fleets of different sizes
            out = Path(tmp) / f"fleet_{seed}.csv"
            subprocess.run(
                [sys.executable, str(GENERATOR), "--samples", str(args.samples),
                 "--robots", str(robots), "--seed", str(seed), "--output", str(out)],
                check=True, capture_output=True,
            )
            df = pd.read_csv(out)
            df.insert(0, "fleet_id", f"F{seed}")
            frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    print(f"Dataset written to {output}: {len(data)} rows, {data['fleet_id'].nunique()} fleets")
    print(data["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
