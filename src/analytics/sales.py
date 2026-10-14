"""Sales analytics built on top of ``profitability`` (no duplicated metric logic)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analytics import profitability as P


@dataclass
class PeriodComparison:
    sufficient: bool
    window_days: int
    current_label: str
    previous_label: str
    metrics: pd.DataFrame            # columns: metric, current, previous, change, change_pct
    note: str


def period_comparison(sales: pd.DataFrame, as_of: pd.Timestamp, window_days: int = 30) -> PeriodComparison:
    """Compare the latest ``window_days`` with the immediately preceding window of equal length."""
    cur_start = as_of - pd.Timedelta(days=window_days - 1)
    prev_end = cur_start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=window_days - 1)
    cur_label = f"{cur_start:%d %b %Y} to {as_of:%d %b %Y}"
    prev_label = f"{prev_start:%d %b %Y} to {prev_end:%d %b %Y}"
    first = sales["transaction_date"].min() if len(sales) else None
    if first is None or pd.isna(first) or first > prev_start:
        return PeriodComparison(False, window_days, cur_label, prev_label,
                                pd.DataFrame(columns=["metric", "current", "previous", "change", "change_pct"]),
                                f"Not enough history: a {window_days}-day comparison needs data from {prev_start:%d %b %Y}.")
    cur = P.compute_summary(P.filter_sales(sales, cur_start, as_of))
    prev = P.compute_summary(P.filter_sales(sales, prev_start, prev_end))
    rows = []
    for label, key in [("Revenue (₹)", "revenue"), ("Gross profit (₹)", "gross_profit"), ("Gross margin (%)", "gross_margin_pct"),
                       ("Units sold", "units_sold"), ("Transactions", "transactions")]:
        a, b = getattr(cur, key), getattr(prev, key)
        change = a - b if not (pd.isna(a) or pd.isna(b)) else np.nan
        rows.append({"metric": label, "current": a, "previous": b, "change": change,
                     "change_pct": (P.safe_div(change, abs(b)) * 100) if key != "gross_margin_pct" else np.nan})
    note = ("Differences between periods are observed changes. They can reflect seasonality, calendar effects, promotions or stock "
            "availability, so they do not by themselves show what caused the change.")
    return PeriodComparison(True, window_days, cur_label, prev_label, pd.DataFrame(rows), note)


def declining_products(sales: pd.DataFrame, as_of: pd.Timestamp, window_days: int = 30, decline_pct: float = 30.0,
                       min_prior_units: float = 10.0) -> pd.DataFrame:
    """Products whose units sold in the latest window are at least ``decline_pct`` below the previous window."""
    cols = ["product_id", "product_name", "category", "units_prev", "units_cur", "change_pct", "revenue_cur", "revenue_prev"]
    cur_start = as_of - pd.Timedelta(days=window_days - 1)
    prev_end = cur_start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=window_days - 1)
    if sales.empty or sales["transaction_date"].min() > prev_start:
        return pd.DataFrame(columns=cols)
    cur = P.by_product(P.filter_sales(sales, cur_start, as_of)).set_index("product_id")
    prev = P.by_product(P.filter_sales(sales, prev_start, prev_end)).set_index("product_id")
    if prev.empty:
        return pd.DataFrame(columns=cols)
    m = prev[["product_name", "category", "units", "revenue"]].join(cur[["units", "revenue"]], how="left", lsuffix="_prev", rsuffix="_cur")
    m[["units_cur", "revenue_cur"]] = m[["units_cur", "revenue_cur"]].fillna(0.0)
    m = m[m["units_prev"] >= min_prior_units]
    m["change_pct"] = (m["units_cur"] - m["units_prev"]) / m["units_prev"] * 100
    out = m[m["change_pct"] <= -abs(decline_pct)].reset_index().sort_values("change_pct")
    return out[cols].reset_index(drop=True)


def ranking(sales: pd.DataFrame, by: str = "revenue", top_n: int = 10, ascending: bool = False) -> pd.DataFrame:
    prod = P.by_product(sales)
    if prod.empty:
        return prod
    return prod.sort_values(by, ascending=ascending, na_position="last").head(top_n).reset_index(drop=True)


def discount_analysis(sales: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Discount depth by product, category and month, plus the share of lines sold at a discount."""
    df = P.recognized(sales)
    df = df[df["quantity"] > 0]
    if df.empty:
        empty = pd.DataFrame()
        return {"by_product": empty, "by_category": empty, "by_month": empty, "distribution": empty}

    def agg(g: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
        r = g.agg(gross_sales=("gross_sales", "sum"), discount_value=("discount_value", "sum"), lines=("transaction_id", "count"),
                  discounted_lines=("discount", lambda s: int((s > 0).sum())))
        r["discount_pct"] = r["discount_value"] / r["gross_sales"].replace(0, np.nan) * 100
        r["discounted_share_pct"] = r["discounted_lines"] / r["lines"] * 100
        return r.reset_index()

    by_prod = agg(df.groupby(["product_id", "product_name"])).sort_values("discount_value", ascending=False)
    by_cat = agg(df.groupby("category")).sort_values("discount_value", ascending=False)
    df = df.assign(month=df["transaction_date"].dt.to_period("M").dt.to_timestamp())
    by_month = agg(df.groupby("month")).sort_values("month")
    bins = [-0.01, 0, 5, 10, 20, 30, 50, 100]
    labels = ["0%", "0-5%", "5-10%", "10-20%", "20-30%", "30-50%", "50-100%"]
    dist = pd.cut(df["discount"], bins=bins, labels=labels).value_counts().reindex(labels).rename_axis("discount_band").reset_index(name="lines")
    return {"by_product": by_prod, "by_category": by_cat, "by_month": by_month, "distribution": dist}


def correlation_note() -> str:
    return ("How to read these charts: a pattern between two measures (for example price and quantity) is a correlation, not proof "
            "that one causes the other. Trends describe what already happened in this data. Causal conclusions would need controlled "
            "comparisons (for example a planned price test), which this dataset does not contain.")
