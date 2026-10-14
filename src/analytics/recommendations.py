"""Rule-based, ranked recommendations computed from analytics outputs (no LLM involved).

* Every recommendation lists evidence, an action, and assumptions. Exposure figures are ESTIMATES, shown only
  when they can be calculated, and always with their basis. They are never promises of savings.
* Overlap handling: a product's exposure is counted once in the headline total, under the highest-priority
  recommendation that covers it (order of generation below). Later overlapping items stay in the list but are
  marked ``counted=False`` so the total does not double count.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from src.analytics import profitability as P
from src.data.loader import Dataset
from src.utils.formatting import inr, num, pct

PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}


@dataclass
class Recommendation:
    kind: str
    title: str
    priority: str
    affected: str
    evidence: list[str]
    action: str
    exposure_inr: float = float("nan")
    exposure_basis: str = ""
    assumptions: str = ""
    product_ids: list[str] = field(default_factory=list)
    counted: bool = True

    def as_dict(self) -> dict:
        return {"kind": self.kind, "title": self.title, "priority": self.priority, "affected": self.affected,
                "evidence": self.evidence, "action": self.action,
                "exposure_inr": None if pd.isna(self.exposure_inr) else float(self.exposure_inr),
                "exposure_basis": self.exposure_basis, "assumptions": self.assumptions, "counted": self.counted}


class _Ledger:
    """Tracks which products already contributed to the headline exposure."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def claim(self, product_ids: list[str]) -> bool:
        ids = set(product_ids)
        if ids and ids & self.seen:
            return False
        self.seen |= ids
        return True


def build_recommendations(ds: Dataset, inv: pd.DataFrame, anomalies: pd.DataFrame, rules: config.BusinessRules | None = None,
                          as_of: pd.Timestamp | None = None) -> list[Recommendation]:
    rules = rules or config.DEFAULT_RULES
    as_of = as_of or ds.as_of
    recs: list[Recommendation] = []
    ledger = _Ledger()
    an = anomalies if anomalies is not None else pd.DataFrame(columns=["detection"])

    def add(rec: Recommendation) -> None:
        if not pd.isna(rec.exposure_inr):
            rec.counted = ledger.claim(rec.product_ids)
        recs.append(rec)

    # 1) Negative gross margins (money is being lost on every sale)
    for r in an[an["detection"] == "Negative gross margin"].itertuples():
        add(Recommendation(
            "negative_margin", f"Investigate negative margin: {r.product_name}", "High", r.product_name,
            [f"Gross margin over the last 90 days: {r.observed}.", r.explanation],
            "Review the selling price against the latest supplier cost and recent discounts; consider a price revision or a new supplier quote.",
            r.amount_inr, "Gross loss on this product over the last 90 days (observed, from recorded costs).",
            "Assumes recorded unit costs are correct. Gross loss only; other overheads are not in the data.", [r.product_id]))

    # 2) Stock-outs and reorders
    need = inv[inv["status"].astype(str).isin(["Stocked out", "Reorder now"])].copy() if len(inv) else inv
    if len(need):
        need = need.sort_values("daily_gp_at_risk", ascending=False, na_position="last")
        for r in need.itertuples():
            unit_margin = (r.selling_price - r.unit_cost) if pd.notna(r.selling_price) and pd.notna(r.unit_cost) else np.nan
            short_days = max(r.lead_time_days - (0 if r.avg_daily_demand == 0 else r.days_of_stock), 0)
            exp = r.avg_daily_demand * unit_margin * short_days if pd.notna(unit_margin) else np.nan
            out = str(r.status) == "Stocked out"
            add(Recommendation(
                "stockout" if out else "reorder", f"{'Restock now' if out else 'Reorder soon'}: {r.product_name}", "High" if out else "Medium",
                r.product_name,
                [f"Stock {num(r.current_stock)} units" + (f" (+{num(r.incoming_qty)} on order)" if r.incoming_qty else "") + f"; demand {r.avg_daily_demand:.2f}/day.",
                 f"Lead time {r.lead_time_days:.0f} days; reorder point {r.reorder_point:.1f}; confidence {r.confidence}."],
                f"Review and place an order of about {int(r.suggested_order_qty)} units (not placed automatically)." if r.suggested_order_qty
                else "Confirm the open order will arrive on time.",
                exp, "Estimated gross profit lost before new stock can arrive: daily demand x unit margin x days without stock during the lead time.",
                "Assumes recent demand continues and every missed sale is lost rather than delayed.", [r.product_id]))

    # 3) Frequent heavy discounting
    for r in an[an["detection"] == "Frequent heavy discounting"].itertuples():
        add(Recommendation(
            "discounting", f"Review heavy discounting: {r.product_name}", r.severity, r.product_name, [r.observed, r.explanation],
            "Decide whether the promotion is intended; if not, restrict who can give discounts above the threshold.",
            r.amount_inr, "Value of discounts given on these lines (observed: list price x quantity - amount charged).",
            "Some discounted sales might not have happened without the discount, so this is not the amount recoverable.", [r.product_id]))
    big = an[an["detection"] == "Unusually large discount"]
    if len(big):
        add(Recommendation(
            "large_discounts", f"Review {len(big)} unusually large discounts", "Medium", f"{big['product_name'].nunique()} product(s)",
            [f"Discounts of {pct(big['observed'].str.rstrip('%').astype(float).min(), 0)} to {pct(big['observed'].str.rstrip('%').astype(float).max(), 0)} versus a {rules.high_discount_pct:.0f}% threshold."],
            "Check each flagged bill for authorisation and reason.", float(big["amount_inr"].sum()),
            "Total discount value on these lines (observed).", "Some may be legitimate clearance or damaged-goods sales.", list(big["product_id"].unique())))

    # 4) Dead / slow / excess stock
    tied = inv[inv["status"].astype(str).isin(["Dead stock", "Slow-moving", "Excess stock"])] if len(inv) else inv
    for r in tied.sort_values("inventory_value", ascending=False, na_position="last").head(8).itertuples():
        st = str(r.status)
        add(Recommendation(
            "slow_stock", f"{'Clear' if st == 'Dead stock' else 'Slow down buying of'} {r.product_name}", "Medium" if st == "Dead stock" else "Low",
            r.product_name,
            [f"{st}: {num(r.current_stock)} units, demand {r.avg_daily_demand:.2f}/day, last sale {r.days_since_last_sale} days ago.",
             f"Cost value in stock: {inr(r.inventory_value)}."],
            r.action, r.excess_value if pd.notna(r.excess_value) and r.excess_value > 0 else r.inventory_value,
            "Cost value of stock above what current demand needs (working capital tied up). Cash is only released if the stock actually sells.",
            "Assumes the stock is still sellable; check expiry and condition first.", [r.product_id]))

    # 5) Data quality: duplicates, inconsistent totals, missing costs, suspicious entries
    dup = an[an["detection"] == "Duplicate transaction"]
    if len(dup):
        add(Recommendation("duplicates", f"Inspect {len(dup)} possible duplicate transactions", "Medium", f"{dup['product_id'].nunique()} product(s)",
                           [f"{len(dup)} repeated lines worth {inr(dup['amount_inr'].sum(), 2)}; they are excluded from revenue."],
                           "Compare with the original bills and fix the source of the double entry.", float(dup["amount_inr"].sum()),
                           "Revenue that would be overstated if the duplicates were counted (observed).", "A repeat could be a genuine second sale.", []))
    bad_total = an[an["detection"] == "Inconsistent total"]
    if len(bad_total):
        add(Recommendation("totals", f"Correct {len(bad_total)} lines whose totals do not add up", "Medium", f"{bad_total['product_id'].nunique()} product(s)",
                           [f"Recorded amounts differ from quantity x price x (1 - discount) by {inr(bad_total['amount_inr'].sum(), 2)} in total."],
                           "Verify each bill and correct the amount, price or discount.", float(bad_total["amount_inr"].sum()),
                           "Absolute difference between recorded and calculated amounts (observed).", "Recorded totals are used in all revenue figures.", []))
    entry = an[an["detection"].isin(["Abnormally large quantity", "Unexpected price change"]) & (an["severity"] == "High")]
    if len(entry):
        add(Recommendation("entry_errors", f"Check {len(entry)} likely data-entry errors", "Medium", f"{entry['product_id'].nunique()} product(s)",
                           [f"{len(entry)} lines have quantities or prices far outside the product's normal range (for example an extra zero)."],
                           "Open each bill and correct typos; confirm genuine bulk sales separately.", np.nan, "", "Some may be genuine wholesale orders.", []))
    miss = an[an["detection"] == "Missing unit cost"]
    if len(miss):
        add(Recommendation("missing_cost", f"Add costs for {len(miss)} product(s) with unknown cost", "Low", f"{len(miss)} product(s)",
                           ["Their margin is excluded from gross-profit figures rather than assumed to be zero."], "Add purchase costs to the sales or inventory file.",
                           np.nan, "", "", []))

    # 6) Declining sales (skip exposure double-counting with stock-outs via the ledger)
    for r in an[an["detection"] == "Sudden sales decline"].itertuples():
        add(Recommendation(
            "decline", f"Monitor falling sales: {r.product_name}", r.severity, r.product_name, [f"{r.observed} vs {r.expected}.", r.explanation],
            "Check whether the product was out of stock first; then review price and placement.",
            r.amount_inr, "Revenue difference between the last 30 days and the previous 30 days (observed; not all of it is recoverable).",
            "Seasonality can explain part of a decline. If the product was out of stock, the stock-out item above already covers this exposure.", [r.product_id]))

    # 7) Category-level changes
    cur_start = as_of - pd.Timedelta(days=29)
    prev_start, prev_end = cur_start - pd.Timedelta(days=30), cur_start - pd.Timedelta(days=1)
    if ds.first_date <= prev_start:
        cur = P.by_category(P.filter_sales(ds.sales, cur_start, as_of)).set_index("category")["revenue"]
        prev = P.by_category(P.filter_sales(ds.sales, prev_start, prev_end)).set_index("category")["revenue"]
        chg = ((cur - prev) / prev.replace(0, np.nan) * 100).dropna()
        for cat, v in chg[chg <= -15].sort_values().items():
            add(Recommendation("category", f"Evaluate falling category demand: {cat}", "Low", cat,
                               [f"Revenue {inr(cur.get(cat, 0))} in the last 30 days vs {inr(prev[cat])} in the 30 days before ({v:.0f}%)."],
                               "Compare with last year's seasonality and check stock availability within the category.", float(prev[cat] - cur.get(cat, 0)),
                               "Revenue difference between two 30-day windows (observed).", "Could be seasonal; one year of data cannot confirm.", []))

    recs.sort(key=lambda r: (PRIORITY_ORDER[r.priority], not r.counted, -(r.exposure_inr if not pd.isna(r.exposure_inr) else -1)))
    return recs


def total_counted_exposure(recs: list[Recommendation]) -> float:
    """Sum of exposure estimates with overlapping products counted once."""
    return float(sum(r.exposure_inr for r in recs if r.counted and not pd.isna(r.exposure_inr)))


def to_frame(recs: list[Recommendation]) -> pd.DataFrame:
    return pd.DataFrame([{**r.as_dict(), "evidence": " | ".join(r.evidence)} for r in recs])
