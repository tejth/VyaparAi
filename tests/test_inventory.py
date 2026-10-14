import math

import numpy as np
import pandas as pd
import pytest

import config
from src.analytics import inventory as I
from tests.conftest import dataset_from, daily_sales, make_inventory


def analyze(days=120, per_day=5, stock=100, lead=5, purchases=None, **rules):
    ds = dataset_from(daily_sales(days, per_day), make_inventory([{"current_stock": stock, "supplier_lead_time_days": lead, "reorder_level": 10}]), purchases)
    return I.analyze_inventory(ds, I.with_rules(**rules) if rules else None), ds


def test_reorder_point_and_safety_stock_formula():
    a, _ = analyze(stock=100, per_day=5, lead=5)
    r = a.iloc[0]
    assert r["avg_daily_demand"] == pytest.approx(5.0)
    assert r["safety_stock"] == pytest.approx(0.0)          # constant demand => std 0
    assert r["reorder_point"] == pytest.approx(25.0)         # 5/day x 5 days
    assert r["days_of_stock"] == pytest.approx(20.0)


def test_safety_stock_increases_with_variability():
    sales = daily_sales(120, 5)
    sales.loc[::2, "quantity"] = "9"
    sales.loc[1::2, "quantity"] = "1"
    sales["total_amount"] = (sales["quantity"].astype(int) * 100).astype(str)
    ds = dataset_from(sales, make_inventory([{"current_stock": 100, "supplier_lead_time_days": 4}]))
    r = I.analyze_inventory(ds).iloc[0]
    assert r["safety_stock"] > 5 and r["reorder_point"] == pytest.approx(r["avg_daily_demand"] * 4 + r["safety_stock"])


def test_reorder_needed_below_reorder_point_and_quantity_formula():
    a, _ = analyze(stock=20, per_day=5, lead=5)                # below ROP 25
    r = a.iloc[0]
    assert str(r["status"]) == "Reorder now"
    assert r["suggested_order_qty"] == math.ceil(5 * (5 + 30) + 0 - 20)       # order-up-to - stock
    assert r["suggested_order_qty"] > 0 and "Reorder" in r["action"]


def test_incoming_stock_reduces_suggested_order():
    pur = pd.DataFrame({"purchase_date": ["2026-06-01"], "product_id": ["P1"], "quantity_purchased": ["100"], "purchase_cost": ["8000"]})
    base, _ = analyze(stock=20)
    with_inc, ds = analyze(stock=20, purchases=pur)
    assert with_inc.iloc[0]["incoming_qty"] == 100 and with_inc.iloc[0]["suggested_order_qty"] < base.iloc[0]["suggested_order_qty"]
    assert with_inc.iloc[0]["suggested_order_qty"] >= 0


def test_never_negative_order_quantity(demo_analysis):
    assert (demo_analysis.inventory["suggested_order_qty"] >= 0).all()
    a, _ = analyze(stock=10_000)
    assert a.iloc[0]["suggested_order_qty"] == 0


def test_stocked_out_and_stockout_risk_statuses():
    a, _ = analyze(stock=0)
    assert str(a.iloc[0]["status"]) == "Stocked out" and a.iloc[0]["days_of_stock"] == 0
    a, _ = analyze(stock=8, per_day=5, lead=5)                 # 1.6 days of stock vs 5-day lead time
    assert str(a.iloc[0]["status"]) in ("Reorder now", "Stock-out risk")


def test_stock_covering_lead_time_is_healthy():
    a, _ = analyze(stock=100)
    assert str(a.iloc[0]["status"]) == "Healthy" and a.iloc[0]["suggested_order_qty"] == 0


def test_dead_and_slow_stock():
    sales = daily_sales(200, 5)
    sales = sales.iloc[:60]                                     # sales stop; the dataset ends later via another product
    other = daily_sales(200, 2, "P2", name="Other")
    ds = dataset_from(pd.concat([sales, other]), make_inventory([{"product_id": "P1", "current_stock": 30}, {"product_id": "P2", "product_name": "Other", "current_stock": 500}]))
    a = I.analyze_inventory(ds).set_index("product_id")
    assert str(a.loc["P1", "status"]) == "Dead stock" and a.loc["P1", "suggested_order_qty"] == 0 and "Do not reorder" in a.loc["P1", "action"]
    slow = dataset_from(daily_sales(120, 0.2 if False else 1).iloc[::10], make_inventory([{"current_stock": 40}]))
    r = I.analyze_inventory(slow).iloc[0]
    assert str(r["status"]) in ("Slow-moving", "Dead stock")


def test_excess_stock():
    a, _ = analyze(stock=2000, per_day=5)                       # 400 days of cover
    assert str(a.iloc[0]["status"]) == "Excess stock"


def test_product_without_sales_history_gets_no_demand_data_and_low_confidence():
    ds = dataset_from(daily_sales(60, 3), make_inventory([{"product_id": "P1"}, {"product_id": "NEW", "product_name": "New", "current_stock": 10}]))
    a = I.analyze_inventory(ds).set_index("product_id")
    assert str(a.loc["NEW", "status"]) == "No demand data" and a.loc["NEW", "confidence"] == "Low"
    assert a.loc["NEW", "avg_daily_demand"] == 0 and "No sales history" in a.loc["NEW", "data_warnings"]


def test_insufficient_history_flagged_low_confidence_and_uses_file_reorder_level():
    ds = dataset_from(daily_sales(10, 5), make_inventory([{"current_stock": 100, "reorder_level": 15}]))
    r = I.analyze_inventory(ds).iloc[0]
    assert r["confidence"] == "Low" and "days of history" in r["data_warnings"] and r["reorder_point"] == 15


def test_empty_inventory_is_handled():
    ds = dataset_from(daily_sales(30, 1))
    a = I.analyze_inventory(ds)
    assert a.empty and I.summary_counts(a)["low_stock"] == 0


def test_missing_cost_is_flagged_not_zero_valued():
    ds = dataset_from(daily_sales(60, 3), make_inventory([{"unit_cost": ""}]))
    r = I.analyze_inventory(ds).iloc[0]
    assert "cost" in r["data_warnings"].lower()
    assert not (r["inventory_value"] == 0)                       # either an estimate from sales or NaN, never a silent 0


def test_every_alert_has_required_fields(demo_analysis):
    inv = demo_analysis.inventory
    alerts = inv[inv["status"].astype(str).isin(["Stocked out", "Reorder now", "Stock-out risk"])]
    assert len(alerts) > 0
    for col in ["product_name", "current_stock", "avg_daily_demand", "days_of_stock", "lead_time_days", "action", "explanation", "confidence"]:
        assert alerts[col].notna().all(), col
    assert (alerts["explanation"].str.len() > 20).all()


def test_demo_scenarios_are_detected(demo_analysis):
    st = demo_analysis.inventory.set_index("product_name")["status"].astype(str)
    assert st["Aashirvaad Atta 5kg"] == "Stocked out"
    assert st["Quinoa 500g"] == "Dead stock" and st["Chia Seeds 250g"] == "Dead stock"
    assert st["Horlicks 500g"] == "Reorder now"
    assert demo_analysis.inventory_counts["low_stock"] >= 10


def test_demand_override_is_used_and_labelled():
    ds = dataset_from(daily_sales(120, 5), make_inventory([{"current_stock": 100}]))
    r = I.analyze_inventory(ds, demand_override={"P1": 20.0}).iloc[0]
    assert r["avg_daily_demand"] == 20 and r["demand_source"] == "ML forecast"


def test_reconciliation():
    ds = dataset_from(daily_sales(10, 5), make_inventory([{"current_stock": 50}]),
                      pd.DataFrame({"purchase_date": ["2026-01-01"], "product_id": ["P1"], "quantity_purchased": ["100"], "purchase_cost": ["8000"]}))
    rec = I.reconcile_inventory(ds)
    assert rec.iloc[0]["implied_stock"] == 50 and rec.iloc[0]["difference"] == 0


def test_stocked_out_with_large_open_order_never_suggests_negative_quantity():
    pur = pd.DataFrame({"purchase_date": ["2026-06-01"], "product_id": ["P1"], "quantity_purchased": ["5000"], "purchase_cost": ["400000"]})
    r = analyze(stock=0, purchases=pur)[0].iloc[0]
    assert str(r["status"]) == "Stocked out" and r["incoming_qty"] == 5000
    assert r["suggested_order_qty"] == 0                       # open order already covers demand: no further order
    assert "Confirm" in r["action"] and "on order" in r["action"]
