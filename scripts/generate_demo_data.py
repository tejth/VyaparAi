"""Regenerate the synthetic demo CSVs and scenario notes (fixed seed => reproducible).

Usage:  python scripts/generate_demo_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sample_generator import SEED, write_csvs  # noqa: E402


def main() -> None:
    data = write_csvs(ROOT / "data", SEED)
    print(f"demo_sales.csv       {len(data.sales):>7,} rows")
    print(f"demo_inventory.csv   {len(data.inventory):>7,} rows")
    print(f"demo_purchases.csv   {len(data.purchases):>7,} rows")


if __name__ == "__main__":
    main()
