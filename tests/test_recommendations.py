import math

import pandas as pd
import pytest

from src.analytics import recommendations as R
from src.analytics.pipeline import run_analysis
from tests.conftest import dataset_from, daily_sales, make_inventory


def test_required_fields_on_every_recommendation(demo_analysis):
    recs = demo_analysis.recommendations
    assert len(recs) > 10
    for r in recs:
        assert r.title and r.priority in ("High", "Medium", "Low") and r.affected and r.evidence and r.action
        assert isinstance(r.assumptions, str)
        if not math.isnan(r.exposure_inr):
            assert r.exposure_basis                        # an estimate is never shown without its basis
            assert r.exposure_inr >= 0


def test_ranking_priority_then_exposure(demo_analysis):
    recs = demo_analysis.recommendations
    order = {"High": 0, "Medium": 1, "Low": 2}
    assert [order[r.priority] for r in recs] == sorted(order[r.priority] for r in recs)
    high_counted = [r.exposure_inr for r in recs if r.priority == "High" and r.counted and not math.isnan(r.exposure_inr)]
    assert high_counted == sorted(high_counted, reverse=True)


def test_overlapping_exposure_counted_once(demo_analysis):
    recs = demo_analysis.recommendations
    total = R.total_counted_exposure(recs)
    naive = sum(r.exposure_inr for r in recs if not math.isnan(r.exposure_inr))
    assert total < naive and any(not r.counted for r in recs)
    seen = set()
    for r in recs:                                         # products behind counted items must be disjoint
        if r.counted and not math.isnan(r.exposure_inr) and r.product_ids:
            assert not (set(r.product_ids) & seen)
            seen |= set(r.product_ids)


def test_expected_demo_recommendations_present(demo_analysis):
    titles = " | ".join(r.title for r in demo_analysis.recommendations)
    assert "Investigate negative margin: Fortune Sunflower Oil 1L" in titles
    assert "Restock now: Aashirvaad Atta 5kg" in titles
    assert "duplicate" in titles.lower() and "Clear Quinoa 500g" in titles


def test_reorder_recommendation_never_places_orders(demo_analysis):
    for r in demo_analysis.recommendations:
        if r.kind in ("stockout", "reorder") and "order" in r.action.lower():
            assert "not placed automatically" in r.action or "Review" in r.action or "Confirm" in r.action


def test_no_savings_promises(demo_analysis):
    text = " ".join(r.action + r.exposure_basis + r.assumptions for r in demo_analysis.recommendations).lower()
    assert "guarantee" not in text and "will save" not in text


def test_healthy_business_has_no_urgent_actions():
    ds = dataset_from(daily_sales(120, 3), make_inventory([{"current_stock": 200, "reorder_level": 10}]))
    an = run_analysis(ds, use_isolation_forest=False)
    assert [r for r in an.recommendations if r.priority == "High"] == []


def test_stockout_recommendation_with_calculable_exposure():
    ds = dataset_from(daily_sales(120, 5), make_inventory([{"current_stock": 0, "supplier_lead_time_days": 5}]))
    an = run_analysis(ds, use_isolation_forest=False)
    rec = [r for r in an.recommendations if r.kind == "stockout"][0]
    assert rec.priority == "High" and rec.exposure_inr == pytest.approx(5 * 20 * 5)       # demand 5/day x margin 20 x 5 days
    assert "unit margin" in rec.exposure_basis


def test_missing_cost_leaves_exposure_uncalculated_not_zero():
    s = daily_sales(120, 5)
    s["unit_cost"] = ""
    inv = make_inventory([{"current_stock": 0, "unit_cost": "", "selling_price": "100"}])
    an = run_analysis(dataset_from(s, inv), use_isolation_forest=False)
    rec = [r for r in an.recommendations if r.kind == "stockout"][0]
    assert math.isnan(rec.exposure_inr)


def test_empty_inputs():
    ds = dataset_from(daily_sales(40, 2))                  # no inventory file
    an = run_analysis(ds, use_isolation_forest=False)
    assert an.inventory.empty and isinstance(an.recommendations, list)
    assert R.to_frame([]).empty
