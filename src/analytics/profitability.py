"""Single source of truth for revenue, cost of goods sold (COGS) and GROSS profit.

Definitions (all amounts in ₹)
------------------------------
* revenue            = recorded ``total_amount`` of each sales line (net of discount; returns are negative).
* COGS               = quantity x effective unit cost (returns reduce COGS). Effective cost is the recorded
                       cost, or the inventory cost as a flagged estimate, otherwise unknown (NaN).
* gross profit       = revenue - COGS, computed ONLY over lines whose cost is known.
* gross margin %     = gross profit / revenue of cost-known lines x 100 (NaN when that revenue is 0).
* Gross profit is NOT net profit: rent, salaries, utilities, taxes and other overheads are not in the data.
* Exact duplicate lines (flagged by validation) are excluded by default.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


def safe_div(a: float, b: float, default: float = float("nan")) -> float:
    try:
        if b is None or pd.isna(b) or float(b) == 0.0 or pd.isna(a):
            return default
        return float(a) / float(b)
    except Exception:
        return default


def filter_sales(sales: pd.DataFrame, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
                 categories: list[str] | None = None, product_ids: list[str] | None = None) -> pd.DataFrame:
    df = sales
    if start is not None:
        df = df[df["transaction_date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["transaction_date"] <= pd.Timestamp(end)]
    if categories:
        df = df[df["category"].isin(categories)]
    if product_ids:
        df = df[df["product_id"].isin(product_ids)]
    return df


def recognized(sales: pd.DataFrame, exclude_duplicates: bool = True) -> pd.DataFrame:
    """Add revenue / COGS / gross-profit columns. Unknown cost stays NaN (never zero)."""
    df = sales[~sales["is_duplicate"]] if exclude_duplicates and "is_duplicate" in sales.columns else sales
    df = df.copy()
    ucost = df["unit_cost_eff"] if "unit_cost_eff" in df.columns else df["unit_cost"]
    df["revenue"] = df["total_amount"]
    df["cost_known"] = ucost.notna()
    df["cogs"] = df["quantity"] * ucost
    df["gross_profit"] = df["revenue"] - df["cogs"]
    df["gross_sales"] = df["quantity"] * df["unit_price"]
    df["discount_value"] = df["gross_sales"] - df["revenue"]
    df["revenue_cost_known"] = df["revenue"].where(df["cost_known"], 0.0)
    return df


@dataclass
class FinancialSummary:
    revenue: float
    cogs: float
    gross_profit: float
    gross_margin_pct: float
    units_sold: float
    transactions: int
    active_products: int
    revenue_cost_known: float
    cost_coverage_pct: float
    discount_value: float
    discount_pct_of_gross: float
    refund_value: float
    return_lines: int
    duplicates_excluded: int
    avg_selling_price: float
    period_start: str
    period_end: str

    def as_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in asdict(self).items()}


def compute_summary(sales: pd.DataFrame, exclude_duplicates: bool = True) -> FinancialSummary:
    dups = int(sales["is_duplicate"].sum()) if (exclude_duplicates and len(sales) and "is_duplicate" in sales.columns) else 0
    df = recognized(sales, exclude_duplicates)
    if df.empty:
        nan = float("nan")
        return FinancialSummary(0.0, 0.0, 0.0, nan, 0.0, 0, 0, 0.0, nan, 0.0, nan, 0.0, 0, dups, nan, "", "")
    known = df[df["cost_known"]]
    revenue = float(df["revenue"].sum())
    rev_known = float(known["revenue"].sum())
    cogs = float(known["cogs"].sum())
    gp = rev_known - cogs
    pos = df[df["quantity"] > 0]
    gross_pos = float(pos["gross_sales"].sum())
    disc_val = float(pos["discount_value"].sum())
    return FinancialSummary(
        revenue=revenue, cogs=cogs, gross_profit=gp, gross_margin_pct=safe_div(gp, rev_known) * 100,
        units_sold=float(df["quantity"].sum()), transactions=int(df["transaction_id"].nunique()),
        active_products=int(pos["product_id"].nunique()), revenue_cost_known=rev_known,
        cost_coverage_pct=safe_div(rev_known, revenue) * 100 if revenue else float("nan"),
        discount_value=disc_val, discount_pct_of_gross=safe_div(disc_val, gross_pos) * 100,
        refund_value=float(-df.loc[df["quantity"] < 0, "revenue"].sum()), return_lines=int((df["quantity"] < 0).sum()),
        duplicates_excluded=dups, avg_selling_price=safe_div(float(pos["revenue"].sum()), float(pos["quantity"].sum())),
        period_start=f"{df['transaction_date'].min():%Y-%m-%d}", period_end=f"{df['transaction_date'].max():%Y-%m-%d}",
    )


def timeseries(sales: pd.DataFrame, freq: str = "D", exclude_duplicates: bool = True) -> pd.DataFrame:
    """Revenue / gross profit / units by day ('D'), week ('W') or month ('M'). Gaps are filled with zero sales."""
    df = recognized(sales, exclude_duplicates)
    cols = ["period", "revenue", "cogs", "gross_profit", "units", "transactions"]
    if df.empty:
        return pd.DataFrame(columns=cols + ["complete"])
    rule = {"D": "D", "W": "W-SUN", "M": "MS"}[freq]
    known = df[df["cost_known"]]
    g_all = df.set_index("transaction_date").resample(rule).agg(revenue=("revenue", "sum"), units=("quantity", "sum"),
                                                                transactions=("transaction_id", "nunique"))
    g_known = known.set_index("transaction_date").resample(rule).agg(rev_known=("revenue", "sum"), cogs=("cogs", "sum"))
    out = g_all.join(g_known, how="left").fillna({"rev_known": 0.0, "cogs": 0.0})
    out["gross_profit"] = out["rev_known"] - out["cogs"]
    out = out.reset_index().rename(columns={"transaction_date": "period"})
    out["complete"] = _complete_flags(out["period"], freq, df["transaction_date"].min(), df["transaction_date"].max())
    return out[cols + ["complete"]]


def _complete_flags(period: pd.Series, freq: str, first: pd.Timestamp, last: pd.Timestamp) -> pd.Series:
    """True when the whole day/week/month lies inside the data range (edge periods are usually partial).

    Weekly periods are labelled by their END date (Sunday); monthly periods by their first day.
    """
    if freq == "D":
        return pd.Series(True, index=period.index)
    if freq == "W":
        return (period - pd.Timedelta(days=6) >= first) & (period <= last)
    return (period >= first) & (period + pd.offsets.MonthEnd(0) <= last)


def complete_periods(ts: pd.DataFrame) -> pd.DataFrame:
    """Trend-chart view: drop partial first/last weeks or months, which would look like a false drop in sales."""
    if "complete" not in ts.columns:
        return ts
    full = ts[ts["complete"]]
    return full if len(full) >= 2 else ts          # tiny ranges: keep everything rather than show an empty chart


def by_product(sales: pd.DataFrame, exclude_duplicates: bool = True) -> pd.DataFrame:
    df = recognized(sales, exclude_duplicates)
    cols = ["product_id", "product_name", "category", "units", "revenue", "cogs", "gross_profit", "margin_pct",
            "avg_price", "avg_discount_pct", "cost_coverage_pct", "transactions"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    meta = df.sort_values("transaction_date").groupby("product_id").agg(product_name=("product_name", "last"), category=("category", "last"))
    pos = df[df["quantity"] > 0]
    g = df.groupby("product_id").agg(units=("quantity", "sum"), revenue=("revenue", "sum"),
                                     rev_known=("revenue_cost_known", "sum"), cogs=("cogs", "sum"),
                                     transactions=("transaction_id", "nunique"))
    g["gross_profit"] = g["rev_known"] - g["cogs"]
    g["margin_pct"] = np.where(g["rev_known"] != 0, g["gross_profit"] / g["rev_known"].replace(0, np.nan) * 100, np.nan)
    g["cost_coverage_pct"] = np.where(g["revenue"] != 0, g["rev_known"] / g["revenue"].replace(0, np.nan) * 100, np.nan)
    p = pos.groupby("product_id").agg(pos_rev=("revenue", "sum"), pos_units=("quantity", "sum"),
                                      gross=("gross_sales", "sum"), disc=("discount_value", "sum"))
    g = g.join(p)
    g["avg_price"] = g["pos_rev"] / g["pos_units"].replace(0, np.nan)
    g["avg_discount_pct"] = g["disc"] / g["gross"].replace(0, np.nan) * 100
    out = meta.join(g).reset_index()
    return out[cols].sort_values("revenue", ascending=False).reset_index(drop=True)


def by_category(sales: pd.DataFrame, exclude_duplicates: bool = True) -> pd.DataFrame:
    df = recognized(sales, exclude_duplicates)
    cols = ["category", "units", "revenue", "cogs", "gross_profit", "margin_pct", "revenue_share_pct", "products"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    g = df.groupby("category").agg(units=("quantity", "sum"), revenue=("revenue", "sum"), rev_known=("revenue_cost_known", "sum"),
                                   cogs=("cogs", "sum"), products=("product_id", "nunique"))
    g["gross_profit"] = g["rev_known"] - g["cogs"]
    g["margin_pct"] = np.where(g["rev_known"] != 0, g["gross_profit"] / g["rev_known"].replace(0, np.nan) * 100, np.nan)
    total = g["revenue"].sum()
    g["revenue_share_pct"] = g["revenue"] / total * 100 if total else np.nan
    return g.reset_index()[cols].sort_values("revenue", ascending=False).reset_index(drop=True)


def loss_making_products(prod: pd.DataFrame, min_revenue: float = 0.0) -> pd.DataFrame:
    """Products whose gross profit over the period is negative (cost-known revenue only)."""
    if prod.empty:
        return prod
    return prod[(prod["gross_profit"] < 0) & (prod["revenue"] >= min_revenue)].sort_values("gross_profit")


def recent_margin_drop(sales: pd.DataFrame, as_of: pd.Timestamp, window_days: int = 30) -> pd.DataFrame:
    """Per product: gross margin % in the latest window vs the prior window (cost-known lines only)."""
    cur_start = as_of - pd.Timedelta(days=window_days - 1)
    prev_start = cur_start - pd.Timedelta(days=window_days)
    cur = by_product(filter_sales(sales, cur_start, as_of))
    prev = by_product(filter_sales(sales, prev_start, cur_start - pd.Timedelta(days=1)))
    if cur.empty or prev.empty:
        return pd.DataFrame(columns=["product_id", "product_name", "margin_prev", "margin_cur", "margin_change_pts", "revenue_cur"])
    m = cur[["product_id", "product_name", "margin_pct", "revenue", "gross_profit"]].merge(
        prev[["product_id", "margin_pct", "gross_profit"]], on="product_id", suffixes=("_cur", "_prev"))
    m["margin_change_pts"] = m["margin_pct_cur"] - m["margin_pct_prev"]
    m = m.rename(columns={"margin_pct_cur": "margin_cur", "margin_pct_prev": "margin_prev", "revenue": "revenue_cur"})
    return m.sort_values("margin_change_pts")[["product_id", "product_name", "margin_prev", "margin_cur", "margin_change_pts", "revenue_cur",
                                               "gross_profit_cur", "gross_profit_prev"]].reset_index(drop=True)
