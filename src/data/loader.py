"""Reading CSVs, running validation and assembling an analysis-ready ``Dataset``."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

import config
from src.data import validator as V
from src.utils.logging_config import get_logger

log = get_logger(__name__)

DEMO_LABEL = "demo"
UPLOADED_LABEL = "uploaded"


@dataclass
class Dataset:
    sales: pd.DataFrame
    inventory: pd.DataFrame
    purchases: pd.DataFrame
    reports: dict[str, V.ValidationReport] = field(default_factory=dict)
    cross_issues: list[V.Issue] = field(default_factory=list)
    source: str = DEMO_LABEL

    @property
    def is_empty(self) -> bool:
        return len(self.sales) == 0

    @property
    def as_of(self) -> pd.Timestamp:
        if self.is_empty:
            return pd.Timestamp.today().normalize()
        return pd.Timestamp(self.sales["transaction_date"].max())

    @property
    def first_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.sales["transaction_date"].min()) if not self.is_empty else self.as_of

    def record_count(self) -> int:
        return len(self.sales) + len(self.inventory) + len(self.purchases)

    def quality_status(self) -> str:
        sts = [r.status for r in self.reports.values()]
        if "Errors" in sts:
            return "Errors (some rows excluded)"
        if "Warnings" in sts or any(i.severity == "warning" for i in self.cross_issues):
            return "Warnings"
        return "Clean"


# ----------------------------------------------------------------------------- file reading
def read_csv(source: str | Path | IO[bytes] | IO[str]) -> pd.DataFrame:
    """Read a CSV as text (validation parses types). Tolerates UTF-8 BOM / Latin-1 and empty files."""
    def _read(enc: str) -> pd.DataFrame:
        if hasattr(source, "seek"):
            source.seek(0)  # type: ignore[union-attr]
        return pd.read_csv(source, dtype=str, encoding=enc, skipinitialspace=True)

    try:
        try:
            return _read("utf-8-sig")
        except UnicodeDecodeError:
            return _read("latin-1")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_demo_raw() -> dict[str, pd.DataFrame]:
    out = {}
    for kind, fname in (("sales", "demo_sales.csv"), ("inventory", "demo_inventory.csv"), ("purchases", "demo_purchases.csv")):
        path = config.DATA_DIR / fname
        if not path.exists():  # regenerate deterministically if files were deleted
            from src.data.sample_generator import write_csvs

            write_csvs(config.DATA_DIR)
        out[kind] = read_csv(path)
    return out


# ----------------------------------------------------------------------------- ingestion
def ingest(raw_sales: pd.DataFrame | None, raw_inventory: pd.DataFrame | None, raw_purchases: pd.DataFrame | None,
           mappings: dict[str, dict[str, str | None]] | None = None, discount_mode: str = "percent",
           source: str = UPLOADED_LABEL) -> Dataset:
    """Map columns, validate every file, and assemble a Dataset (problem rows are reported, not hidden)."""
    mappings = mappings or {}

    def mapped(kind: str, raw: pd.DataFrame | None) -> pd.DataFrame | None:
        if raw is None:
            return None
        m = mappings.get(kind) or V.suggest_mapping(raw.columns, V.SPECS[kind])
        return V.apply_mapping(raw, m)

    s_map, i_map, p_map = mapped("sales", raw_sales), mapped("inventory", raw_inventory), mapped("purchases", raw_purchases)
    sales, rs = V.validate_sales(s_map if s_map is not None else pd.DataFrame(), discount_mode=discount_mode)
    if i_map is None:
        inventory, ri = V._empty_inventory(), V.ValidationReport("inventory")
        ri.add("info", "no_inventory_file", "No inventory file was provided.", impact="Inventory intelligence is unavailable.")
    else:
        inventory, ri = V.validate_inventory(i_map)
    purchases, rp = V.validate_purchases(p_map)
    for kind, raw, rep in (("sales", raw_sales, rs), ("inventory", raw_inventory, ri), ("purchases", raw_purchases, rp)):
        if raw is not None:
            rep.mapping = mappings.get(kind) or V.suggest_mapping(raw.columns, V.SPECS[kind])
    return finalize(sales, inventory, purchases, {"sales": rs, "inventory": ri, "purchases": rp}, source)


def finalize(sales: pd.DataFrame, inventory: pd.DataFrame, purchases: pd.DataFrame,
             reports: dict[str, V.ValidationReport], source: str) -> Dataset:
    """Cost fallbacks and derived fields. Estimates are flagged, never presented as recorded costs."""
    sales = sales.copy()
    inventory = inventory.copy()
    cost_ref = inventory.set_index("product_id")["unit_cost"] if len(inventory) else pd.Series(dtype="float64")

    sales["unit_cost_eff"] = sales["unit_cost"]
    sales["cost_source"] = np.where(sales["unit_cost"].notna(), "recorded", "unknown")
    if len(sales) and len(cost_ref):
        est = sales["product_id"].map(cost_ref)
        use = sales["unit_cost"].isna() & est.notna()
        sales.loc[use, "unit_cost_eff"] = est[use]
        sales.loc[use, "cost_source"] = "inventory_estimate"

    # Inventory gap-filling from sales history (flagged).
    inventory["unit_cost_estimated"] = False
    inventory["selling_price_estimated"] = False
    inventory["lead_time_assumed"] = False
    if len(inventory):
        if len(sales):
            rec = sales[(sales["cost_source"] == "recorded") & ~sales["is_duplicate"]].sort_values("transaction_date")
            last_cost = rec.groupby("product_id")["unit_cost"].last()
            last_price = sales[~sales["is_duplicate"] & ~sales["is_return"]].sort_values("transaction_date").groupby("product_id")["unit_price"].last()
            miss = inventory["unit_cost"].isna()
            inventory.loc[miss, "unit_cost"] = inventory.loc[miss, "product_id"].map(last_cost)
            inventory.loc[miss & inventory["unit_cost"].notna(), "unit_cost_estimated"] = True
            miss = inventory["selling_price"].isna()
            inventory.loc[miss, "selling_price"] = inventory.loc[miss, "product_id"].map(last_price)
            inventory.loc[miss & inventory["selling_price"].notna(), "selling_price_estimated"] = True
            last_sale = sales[~sales["is_duplicate"] & ~sales["is_return"]].groupby("product_id")["transaction_date"].max()
            miss = inventory["last_sale_date"].isna()
            inventory.loc[miss, "last_sale_date"] = inventory.loc[miss, "product_id"].map(last_sale)
            # Names / categories from sales when the inventory file lacks them.
            names = sales.drop_duplicates("product_id", keep="last").set_index("product_id")
            blank_cat = inventory["category"].eq("Uncategorised")
            inventory.loc[blank_cat, "category"] = inventory.loc[blank_cat, "product_id"].map(names["category"]).fillna("Uncategorised")
        no_lead = inventory["supplier_lead_time_days"].isna()
        inventory["lead_time_assumed"] = no_lead
        inventory.loc[no_lead, "supplier_lead_time_days"] = 7.0

    return Dataset(sales=sales.reset_index(drop=True), inventory=inventory.reset_index(drop=True),
                   purchases=purchases.reset_index(drop=True), reports=reports,
                   cross_issues=V.cross_validate(sales, inventory, purchases), source=source)


def load_demo_dataset() -> Dataset:
    raw = load_demo_raw()
    return ingest(raw["sales"], raw["inventory"], raw["purchases"], source=DEMO_LABEL)


def demo_csv_bytes() -> dict[str, bytes]:
    """Bytes of the sample CSVs for download buttons."""
    load_demo_raw()  # ensures the files exist
    return {f: (config.DATA_DIR / f).read_bytes() for f in ("demo_sales.csv", "demo_inventory.csv", "demo_purchases.csv")}
