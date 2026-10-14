import numpy as np
import pandas as pd
import pytest

from src.analytics import profitability as P
from src.analytics import sales as S
from src.utils.formatting import indian_group, inr, inr_short
from tests.conftest import dataset_from, daily_sales, make_sales


def ds_of(rows):
    return dataset_from(make_sales(rows))


def test_revenue_gross_profit_and_margin():
    ds = ds_of([{"transaction_id": "A", "quantity": 2, "unit_price": 100, "unit_cost": 60, "total_amount": 200},
                {"transaction_id": "B", "quantity": 1, "unit_price": 50, "unit_cost": 30, "total_amount": 50}])
    s = P.compute_summary(ds.sales)
    assert s.revenue == 250 and s.cogs == 150 and s.gross_profit == 100
    assert s.gross_margin_pct == pytest.approx(40.0) and s.units_sold == 3


def test_discount_is_net_of_revenue_and_reported():
    ds = ds_of([{"quantity": 4, "unit_price": 100, "unit_cost": 60, "discount": 10, "total_amount": 360}])
    s = P.compute_summary(ds.sales)
    assert s.revenue == 360 and s.discount_value == pytest.approx(40) and s.discount_pct_of_gross == pytest.approx(10)
    assert s.gross_profit == pytest.approx(360 - 240)


def test_returns_reduce_revenue_and_cogs():
    ds = ds_of([{"transaction_id": "A", "quantity": 5, "total_amount": 500, "unit_cost": 60},
                {"transaction_id": "R", "quantity": -1, "total_amount": -100, "unit_cost": 60}])
    s = P.compute_summary(ds.sales)
    assert s.revenue == 400 and s.cogs == 240 and s.units_sold == 4 and s.refund_value == 100 and s.return_lines == 1


def test_zero_revenue_is_safe():
    s = P.compute_summary(ds_of([{"unit_price": 0, "total_amount": 0}]).sales)
    assert s.revenue == 0 and np.isnan(s.gross_margin_pct)
    empty = P.compute_summary(ds_of([{}]).sales.iloc[0:0])
    assert empty.revenue == 0 and np.isnan(empty.gross_margin_pct) and empty.transactions == 0


def test_missing_cost_is_excluded_from_margin_not_treated_as_zero():
    ds = ds_of([{"transaction_id": "A", "unit_cost": "60", "total_amount": 200}, {"transaction_id": "B", "unit_cost": "", "total_amount": 200}])
    s = P.compute_summary(ds.sales)
    assert s.revenue == 400                               # all revenue counts
    assert s.cogs == 120 and s.gross_profit == 80          # margin only over the cost-known line (200 - 120)
    assert s.gross_margin_pct == pytest.approx(40.0)       # NOT inflated by treating the unknown cost as 0
    assert s.cost_coverage_pct == pytest.approx(50.0)


def test_duplicates_excluded_by_default():
    ds = ds_of([{}, {}])
    assert P.compute_summary(ds.sales).revenue == 200
    assert P.compute_summary(ds.sales, exclude_duplicates=False).revenue == 400
    assert P.compute_summary(ds.sales).duplicates_excluded == 1


def test_timeseries_sums_match_summary_and_fill_gaps():
    ds = dataset_from(daily_sales(10, 3))
    d = P.timeseries(ds.sales, "D")
    assert len(d) == 10 and d["revenue"].sum() == P.compute_summary(ds.sales).revenue
    m = P.timeseries(ds.sales, "M")
    assert m["revenue"].sum() == d["revenue"].sum()


def test_by_product_and_category_consistency(demo_ds):
    s = P.compute_summary(demo_ds.sales)
    assert P.by_product(demo_ds.sales)["revenue"].sum() == pytest.approx(s.revenue)
    assert P.by_category(demo_ds.sales)["revenue"].sum() == pytest.approx(s.revenue)
    assert P.by_category(demo_ds.sales)["gross_profit"].sum() == pytest.approx(s.gross_profit)


def test_loss_making_detects_injected_scenario(demo_ds):
    loss = P.loss_making_products(P.by_product(P.filter_sales(demo_ds.sales, demo_ds.as_of - pd.Timedelta(days=89), demo_ds.as_of)))
    assert "Fortune Sunflower Oil 1L" in set(loss["product_name"])


def test_filters():
    ds = dataset_from(daily_sales(30, 1))
    f = P.filter_sales(ds.sales, pd.Timestamp("2026-01-10"), pd.Timestamp("2026-01-19"))
    assert len(f) == 10 and P.filter_sales(ds.sales, categories=["Nope"]).empty


def test_period_comparison_needs_history():
    ds = dataset_from(daily_sales(20, 2))
    assert not S.period_comparison(ds.sales, ds.as_of, 30).sufficient
    ds = dataset_from(daily_sales(70, 2))
    c = S.period_comparison(ds.sales, ds.as_of, 30)
    assert c.sufficient and c.metrics.loc[c.metrics["metric"] == "Revenue (₹)", "change"].iloc[0] == pytest.approx(0)


def test_declining_products():
    a = daily_sales(60, 5, "P1", "2026-01-01")
    a.loc[30:, "quantity"] = "1"                         # sharp drop in the last 30 days
    a.loc[30:, "total_amount"] = "100"
    ds = dataset_from(a)
    dec = S.declining_products(ds.sales, ds.as_of, 30, 30.0)
    assert list(dec["product_id"]) == ["P1"] and dec.iloc[0]["change_pct"] == pytest.approx(-80)


def test_indian_rupee_formatting():
    assert indian_group(1234567) == "12,34,567" and inr(1234567) == "₹12,34,567" and inr(-99999.5, 2) == "-₹99,999.50"
    assert inr_short(25_000_000) == "₹2.50 Cr" and inr_short(340_000) == "₹3.40 L" and inr(float("nan")) == "n/a"
    assert "$" not in inr(5) and "€" not in inr(5)


def test_partial_edge_periods_are_flagged_and_hidden_from_trends():
    ds = dataset_from(daily_sales(30, 2, start="2026-01-07"))               # starts Wednesday, ends Thursday 5 Feb
    w = P.timeseries(ds.sales, "W")
    assert list(w["period"].dt.strftime("%a")) == ["Sun"] * len(w)           # weeks are labelled by their end date
    assert not w["complete"].iloc[0] and not w["complete"].iloc[-1] and w["complete"].iloc[1:-1].all()
    shown = P.complete_periods(w)
    assert len(shown) == len(w) - 2 and (shown["units"] == 14).all()        # every shown week is a full 7 days x 2 units
    assert w["units"].iloc[0] < 14 and w["units"].iloc[-1] < 14             # the hidden ones really were partial
    m = P.timeseries(ds.sales, "M")
    assert not m["complete"].any() and len(P.complete_periods(m)) == len(m)  # nothing complete: keep rather than blank the chart
    assert P.timeseries(ds.sales, "D")["complete"].all()
    assert w["revenue"].sum() == P.compute_summary(ds.sales).revenue         # flagging never changes totals
