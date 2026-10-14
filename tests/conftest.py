import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import loader  # noqa: E402


def make_sales(rows: list[dict] | None = None) -> pd.DataFrame:
    base = {"transaction_id": "T1", "transaction_date": "2026-01-05", "product_id": "P1", "product_name": "Rice", "quantity": 2,
            "unit_price": 100, "unit_cost": 80, "discount": 0, "total_amount": 200, "payment_method": "UPI", "category": "Staples"}
    rows = rows or [{}]
    return pd.DataFrame([{**base, **r} for r in rows]).astype(str)


def make_inventory(rows: list[dict] | None = None) -> pd.DataFrame:
    base = {"product_id": "P1", "product_name": "Rice", "category": "Staples", "current_stock": 50, "reorder_level": 20,
            "supplier_lead_time_days": 5, "unit_cost": 80, "selling_price": 100, "last_sale_date": ""}
    rows = rows or [{}]
    return pd.DataFrame([{**base, **r} for r in rows]).astype(str)


def daily_sales(days: int = 120, per_day: int = 5, pid: str = "P1", start: str = "2026-01-01", price: float = 100, cost: float = 80,
                name: str = "Rice") -> pd.DataFrame:
    """One sales line per day with a fixed quantity (clean, deterministic)."""
    dates = pd.date_range(start, periods=days)
    return pd.DataFrame({"transaction_id": [f"{pid}-{i}" for i in range(days)], "transaction_date": dates.strftime("%Y-%m-%d"), "product_id": pid,
                         "product_name": name, "quantity": per_day, "unit_price": price, "unit_cost": cost, "discount": 0,
                         "total_amount": per_day * price, "payment_method": "UPI", "category": "Staples"}).astype(str)


def dataset_from(sales_raw: pd.DataFrame, inv_raw: pd.DataFrame | None = None, pur_raw: pd.DataFrame | None = None) -> loader.Dataset:
    return loader.ingest(sales_raw, inv_raw, pur_raw)


@pytest.fixture(scope="session")
def demo_ds() -> loader.Dataset:
    return loader.load_demo_dataset()


@pytest.fixture(scope="session")
def demo_analysis(demo_ds):
    from src.analytics.pipeline import run_analysis
    return run_analysis(demo_ds)
