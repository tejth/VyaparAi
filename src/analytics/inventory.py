"""Inventory intelligence: demand velocity, reorder points, stock-out risk, slow / dead / excess stock.

Formulas (transparent, all configurable through ``config.BusinessRules``)
-----------------------------------------------------------------------
avg_daily_demand   = units sold (excluding returns and duplicate lines) in the demand window / days observed
demand_during_LT   = avg_daily_demand x supplier_lead_time_days
safety_stock       = z x std(daily demand) x sqrt(lead_time_days)        (z = 1.65 -> about 95% service level)
reorder_point      = demand_during_LT + safety_stock
order_up_to_level  = avg_daily_demand x (lead_time_days + target_coverage_days) + safety_stock
suggested_order    = max(0, ceil(order_up_to_level - current_stock - incoming_stock))   only when a reorder is due
days_of_stock      = current_stock / avg_daily_demand

Open purchase orders (purchase rows dated AFTER the latest sales date) are treated as incoming stock.
These figures are estimates from past sales; they are not guarantees. Nothing here changes inventory
or places orders.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd

import config
from src.analytics import profitability as P
from src.data.loader import Dataset

STATUS_ORDER = ["Stocked out", "Reorder now", "Stock-out risk", "Dead stock", "Slow-moving", "Excess stock", "Healthy", "No demand data"]


def demand_stats(sales: pd.DataFrame, as_of: pd.Timestamp, window_days: int, history_start: pd.Timestamp) -> pd.DataFrame:
    """Average and std of daily demand per product over the window (zero-sales days included)."""
    window_start = max(as_of - pd.Timedelta(days=window_days - 1), history_start)
    n_days = int((as_of - window_start).days) + 1
    df = sales[(~sales["is_duplicate"]) & (~sales["is_return"]) & (sales["transaction_date"] >= window_start)
               & (sales["transaction_date"] <= as_of)]
    if df.empty:
        return pd.DataFrame(columns=["product_id", "avg_daily_demand", "std_daily_demand", "sale_days", "window_days"])
    daily = df.groupby(["product_id", "transaction_date"])["quantity"].sum().unstack(fill_value=0)
    daily = daily.reindex(columns=pd.date_range(window_start, as_of), fill_value=0)
    out = pd.DataFrame({"avg_daily_demand": daily.mean(axis=1), "std_daily_demand": daily.std(axis=1, ddof=1).fillna(0.0),
                        "sale_days": (daily > 0).sum(axis=1)})
    out["window_days"] = n_days
    return out.reset_index()


def incoming_stock(purchases: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Open orders: purchase lines dated after the latest sales date."""
    if purchases.empty:
        return pd.DataFrame(columns=["product_id", "incoming_qty", "next_arrival"])
    open_ = purchases[purchases["purchase_date"] > as_of]
    if open_.empty:
        return pd.DataFrame(columns=["product_id", "incoming_qty", "next_arrival"])
    return open_.groupby("product_id").agg(incoming_qty=("quantity_purchased", "sum"), next_arrival=("purchase_date", "min")).reset_index()


def analyze_inventory(ds: Dataset, rules: config.BusinessRules | None = None, as_of: pd.Timestamp | None = None,
                      demand_override: dict[str, float] | None = None) -> pd.DataFrame:
    """One row per inventory product with demand, reorder maths, status, action and explanation.

    ``demand_override`` can supply model-based daily demand per product (used only where provided).
    """
    rules = rules or config.DEFAULT_RULES
    inv = ds.inventory
    cols = ["product_id", "product_name", "category", "current_stock", "incoming_qty", "avg_daily_demand", "days_of_stock",
            "lead_time_days", "reorder_point", "file_reorder_level", "safety_stock", "suggested_order_qty", "status", "flags",
            "inventory_value", "unit_cost", "selling_price", "margin_pct", "days_since_last_sale", "action", "explanation",
            "confidence", "data_warnings", "daily_gp_at_risk", "excess_value", "demand_source"]
    if inv.empty:
        return pd.DataFrame(columns=cols)

    as_of = as_of or ds.as_of
    sales = ds.sales
    history_start = ds.first_date
    history_days = int((as_of - history_start).days) + 1
    stats = demand_stats(sales, as_of, rules.demand_window_days, history_start).set_index("product_id")
    inc = incoming_stock(ds.purchases, as_of).set_index("product_id")
    pos = sales[(~sales["is_duplicate"]) & (~sales["is_return"])]
    last_sale = pos.groupby("product_id")["transaction_date"].max()

    rows = []
    for r in inv.itertuples(index=False):
        pid = r.product_id
        stock = float(r.current_stock)
        lead = float(r.supplier_lead_time_days) if pd.notna(r.supplier_lead_time_days) else 7.0
        warnings: list[str] = []
        st = stats.loc[pid] if pid in stats.index else None
        has_history = pid in last_sale.index
        d = float(st["avg_daily_demand"]) if st is not None else 0.0
        sd = float(st["std_daily_demand"]) if st is not None else 0.0
        sale_days = int(st["sale_days"]) if st is not None else 0
        demand_source = "recent sales average"
        if demand_override and pid in demand_override and pd.notna(demand_override[pid]):
            d, demand_source = float(max(demand_override[pid], 0.0)), "ML forecast"

        # Confidence in the demand estimate.
        if history_days < rules.min_history_days:
            confidence = "Low"
            warnings.append(f"Only {history_days} days of history (need {rules.min_history_days}+): demand is a rough estimate.")
        elif not has_history:
            confidence = "Low"
            warnings.append("No sales history for this product: demand cannot be estimated.")
        elif sale_days < 5:
            confidence = "Low"
            warnings.append(f"Sold on only {sale_days} day(s) in the window: demand is sparse and uncertain.")
        elif sale_days < 15:
            confidence = "Medium"
            warnings.append("Intermittent demand: estimate is approximate.")
        else:
            confidence = "High"
        if getattr(r, "lead_time_assumed", False):
            warnings.append("Supplier lead time not provided: assumed 7 days.")
        if getattr(r, "unit_cost_estimated", False):
            warnings.append("Unit cost estimated from sales history.")
        if pd.isna(r.unit_cost):
            warnings.append("Unit cost unknown: inventory value and margin cannot be calculated.")
        if r.negative_stock:
            warnings.append("Negative stock recorded: stock data is probably out of date.")
        if stock < 0:
            stock = 0.0

        safety = 0.0
        if confidence != "Low" or (has_history and history_days >= rules.min_history_days):
            safety = rules.low_stock_service_level_z * sd * math.sqrt(max(lead, 1.0))
        lead_demand = d * lead
        calc_rop = lead_demand + safety
        file_rop = float(r.reorder_level) if pd.notna(r.reorder_level) else float("nan")
        # Use the data-driven reorder point when we trust the demand estimate, otherwise fall back to the file's level.
        if confidence == "Low" and not math.isnan(file_rop):
            rop, rop_note = file_rop, "file reorder level (demand estimate unreliable)"
        else:
            rop, rop_note = calc_rop, "calculated reorder point"
        inc_qty = float(inc.loc[pid, "incoming_qty"]) if pid in inc.index else 0.0
        available = stock + inc_qty
        days_stock = stock / d if d > 0 else float("inf")
        order_up_to = d * (lead + rules.target_coverage_days) + safety
        reorder_due = d > 0 and available <= rop
        suggested = max(0, math.ceil(order_up_to - available)) if (reorder_due or (d > 0 and stock <= 0)) else 0
        last = last_sale.get(pid, pd.NaT)
        if pd.isna(last) and pd.notna(r.last_sale_date):
            last = r.last_sale_date
        dsl = int((as_of - last).days) if pd.notna(last) else None
        cost, price = r.unit_cost, r.selling_price
        margin = (price - cost) / price * 100 if (pd.notna(cost) and pd.notna(price) and price) else float("nan")
        value = stock * cost if pd.notna(cost) else float("nan")
        unit_margin = (price - cost) if (pd.notna(cost) and pd.notna(price)) else float("nan")

        # ---- status (single primary status, in priority order)
        flags: list[str] = []
        if d > 0 and stock <= 0:
            status = "Stocked out"
        elif reorder_due:
            status = "Reorder now"
        elif d > 0 and stock < lead_demand:
            status = "Stock-out risk"      # only possible when an open order is counted in `available`
        elif stock > 0 and ((dsl is not None and dsl >= rules.dead_stock_days_since_sale)
                            or (not has_history and history_days >= rules.dead_stock_days_since_sale)):
            status = "Dead stock"
        elif stock > 0 and has_history and (d < rules.slow_moving_daily_demand
                                            or (dsl is not None and dsl >= rules.slow_moving_days_since_sale)):
            status = "Slow-moving"
        elif d > 0 and days_stock > rules.excess_stock_days_cover:
            status = "Excess stock"
        elif not has_history:
            status = "No demand data"
        else:
            status = "Healthy"
        if status == "Dead stock" and history_days >= rules.dead_stock_days_since_sale:
            # Classified from days since last sale (a fact), not from the sparse demand estimate.
            confidence = "High"
            warnings = [w for w in warnings if "sparse" not in w]
        if not math.isnan(file_rop) and stock <= file_rop and status not in ("Stocked out", "Reorder now"):
            flags.append("Below file reorder level")
        if pd.notna(margin) and margin < rules.low_margin_pct:
            flags.append("Negative margin" if margin < 0 else "Low margin")
        if d > 0 and days_stock > rules.excess_stock_days_cover and status != "Excess stock":
            flags.append("Excess cover")

        excess_val = float("nan")
        if pd.notna(cost):
            keep_units = d * rules.excess_stock_days_cover
            excess_val = max(0.0, stock - keep_units) * cost if status in ("Excess stock", "Slow-moving", "Dead stock") else 0.0
        gp_risk = d * unit_margin if (d > 0 and pd.notna(unit_margin) and status in ("Stocked out", "Reorder now", "Stock-out risk")) else float("nan")

        action, why = _action_text(status, pid, r.product_name, stock, inc_qty, d, days_stock, lead, rop, rop_note, suggested, dsl, margin, price, cost, rules)
        rows.append({
            "product_id": pid, "product_name": r.product_name, "category": r.category, "current_stock": stock, "incoming_qty": inc_qty,
            "avg_daily_demand": d, "days_of_stock": days_stock, "lead_time_days": lead, "reorder_point": rop, "file_reorder_level": file_rop,
            "safety_stock": safety, "suggested_order_qty": suggested, "status": status, "flags": ", ".join(flags),
            "inventory_value": value, "unit_cost": cost, "selling_price": price, "margin_pct": margin, "days_since_last_sale": dsl,
            "action": action, "explanation": why, "confidence": confidence, "data_warnings": " ".join(warnings),
            "daily_gp_at_risk": gp_risk, "excess_value": excess_val, "demand_source": demand_source,
        })
    out = pd.DataFrame(rows)[cols]
    out["status"] = pd.Categorical(out["status"], STATUS_ORDER, ordered=True)
    return out.sort_values(["status", "inventory_value"], ascending=[True, False]).reset_index(drop=True)


def _action_text(status: str, pid: str, name: str, stock: float, incoming: float, d: float, days_stock: float, lead: float, rop: float,
                 rop_note: str, suggested: int, dsl: int | None, margin: float, price: float, cost: float,
                 rules: config.BusinessRules) -> tuple[str, str]:
    from src.utils.formatting import num, pct

    cover = "unlimited" if math.isinf(days_stock) else f"{days_stock:.1f} days"
    if status == "Stocked out":
        act = (f"Reorder about {suggested} units now (review before ordering)." if suggested > 0
               else f"Confirm the {num(incoming)} units on order arrive as scheduled.")
        return (act, f"Stock is 0 while the product normally sells {d:.2f} units/day, so sales are being lost. Lead time is {lead:.0f} days"
                + (f"; {num(incoming)} units are already on order." if incoming else "."))
    if status == "Reorder now":
        return (f"Reorder about {suggested} units.",
                f"Stock {num(stock)} (+{num(incoming)} on order) is at or below the {rop_note} of {rop:.1f} units. At {d:.2f} units/day it lasts "
                f"{cover} but the supplier needs {lead:.0f} days. The order-up-to level covers lead time plus {rules.target_coverage_days} days of demand.")
    if status == "Stock-out risk":
        return ("Confirm the open order will arrive before stock runs out.",
                f"Stock alone covers {cover}, shorter than the {lead:.0f}-day lead time. The {num(incoming)} units on order keep it above the "
                f"reorder point only if they arrive on time.")
    if status == "Dead stock":
        recover = (f" Selling down to cost needs a price above ₹{cost:,.0f}; maximum discount before selling below cost is {max((price - cost) / price * 100, 0):.0f}%."
                   if pd.notna(cost) and pd.notna(price) and price else "")
        return ("Do not reorder. Review for clearance, bundling or return-to-supplier.",
                f"No sale for {dsl} days (threshold {rules.dead_stock_days_since_sale}). Stock is slow, not necessarily unsellable: check expiry, damage "
                f"and seasonality before writing it off.{recover}")
    if status == "Slow-moving":
        return ("Pause reordering; consider a modest promotion.",
                f"Demand is {d:.2f} units/day (threshold {rules.slow_moving_daily_demand}) and the last sale was {dsl} days ago; stock covers {cover}. "
                "Slow-moving stock still sells, unlike dead stock.")
    if status == "Excess stock":
        return ("Hold off on reordering until stock falls.",
                f"Stock covers {cover}, more than the {rules.excess_stock_days_cover}-day threshold, tying up working capital.")
    if status == "No demand data":
        return ("Collect sales history before deciding.", "No sales records exist for this product, so demand cannot be estimated.")
    return ("No action needed.", f"Stock covers {cover} against a {lead:.0f}-day lead time.")


def summary_counts(analysis: pd.DataFrame) -> dict[str, float | int]:
    if analysis.empty:
        return {"low_stock": 0, "slow_moving": 0, "dead_stock": 0, "excess": 0, "stocked_out": 0, "inventory_value": 0.0,
                "inventory_value_known_pct": float("nan")}
    st = analysis["status"].astype(str)
    value = analysis["inventory_value"]
    return {
        "low_stock": int(st.isin(["Stocked out", "Reorder now", "Stock-out risk"]).sum()),
        "stocked_out": int((st == "Stocked out").sum()),
        "slow_moving": int(st.isin(["Slow-moving", "Dead stock"]).sum()),
        "dead_stock": int((st == "Dead stock").sum()),
        "excess": int((st == "Excess stock").sum()),
        "inventory_value": float(value.sum(skipna=True)),
        "inventory_value_known_pct": float(value.notna().mean() * 100),
    }


def reconcile_inventory(ds: Dataset, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Informational check: stock implied by (purchases - net sales) versus recorded stock.

    Only meaningful when the purchase file includes opening stock. Differences can come from
    duplicates, data-entry errors, shrinkage, returns to suppliers or missing purchase records.
    """
    as_of = as_of or ds.as_of
    if ds.purchases.empty or ds.inventory.empty:
        return pd.DataFrame(columns=["product_id", "product_name", "recorded_stock", "implied_stock", "difference"])
    pur = ds.purchases[ds.purchases["purchase_date"] <= as_of].groupby("product_id")["quantity_purchased"].sum()
    sold = ds.sales[~ds.sales["is_duplicate"]].groupby("product_id")["quantity"].sum()
    inv = ds.inventory.set_index("product_id")
    implied = pur.reindex(inv.index).fillna(0) - sold.reindex(inv.index).fillna(0)
    out = pd.DataFrame({"product_name": inv["product_name"], "recorded_stock": inv["current_stock"], "implied_stock": implied})
    out["difference"] = out["recorded_stock"] - out["implied_stock"]
    return out.reset_index().sort_values("difference", key=lambda s: s.abs(), ascending=False)


def with_rules(**overrides: float | int) -> config.BusinessRules:
    return replace(config.DEFAULT_RULES, **overrides)
