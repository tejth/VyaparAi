import numpy as np
import pandas as pd
import pytest

from src.ml import evaluation as E
from src.ml import forecasting as F
from tests.conftest import dataset_from, daily_sales


def weekly_pattern_series(days=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=days)
    base = np.array([3, 3, 4, 5, 8, 12, 14])[dates.dayofweek]
    return pd.Series(rng.poisson(base).astype(float), index=dates)


def sales_from_series(y: pd.Series, pid="P1") -> pd.DataFrame:
    df = pd.DataFrame({"transaction_id": [f"t{i}" for i in range(len(y))], "transaction_date": y.index.strftime("%Y-%m-%d"), "product_id": pid, "product_name": "Rice",
                       "quantity": y.values.astype(int).astype(str), "unit_price": "100", "unit_cost": "80", "discount": "0",
                       "total_amount": (y.values * 100).astype(int).astype(str), "payment_method": "UPI", "category": "Staples"})
    return dataset_from(df[df["quantity"].astype(int) > 0]).sales


# ---------------------------------------------------------------- metrics
def test_metrics_zero_safe():
    a, p = np.array([0, 0, 0]), np.array([1, 1, 1])
    assert E.mae(a, p) == 1 and np.isnan(E.wape(a, p))             # WAPE undefined (no demand), not inf/crash
    assert E.wape(np.array([10, 0, 10]), np.array([8, 1, 12])) == pytest.approx(5 / 20)
    assert E.bias(np.array([2, 2]), np.array([3, 3])) == 1


def test_time_split_is_chronological_and_requires_enough_history():
    assert E.time_split_index(30, 7) is None                       # too short
    cutoff = E.time_split_index(200, 7)
    assert cutoff is not None and cutoff < 200 and 200 - cutoff >= 7
    origins = E.backtest_origins(cutoff, 200, 7)
    assert origins[0] == cutoff - 1 and max(origins) + 7 <= 199      # every target in validation block, none beyond data
    assert all(o + 1 >= cutoff for o in origins)


# ---------------------------------------------------------------- features / leakage
def test_origin_features_use_only_past_data():
    y = np.arange(60, dtype=float)
    f1 = E_feats = F.origin_features(y, 40)
    y2 = y.copy()
    y2[41:] = 9999.0                                                  # change the future
    f2 = F.origin_features(y2, 40)
    assert f1 == f2                                                   # identical => no look-ahead
    assert f1["lag0"] == 40 and f1["lag1"] == 39 and f1["roll7"] == pytest.approx(np.mean(y[34:41]))


def test_same_weekday_mean_never_reads_beyond_origin():
    y = np.arange(100, dtype=float)
    o, h = 50, 10
    y2 = y.copy()
    y2[o + 1:] = -1e6
    assert F.same_weekday_mean(y, o, h) == F.same_weekday_mean(y2, o, h)


def test_build_rows_targets_and_shapes():
    y = np.arange(80, dtype=float)
    dates = pd.date_range("2026-01-01", periods=80)
    X, t, keys = F.build_rows(y, dates, [40, 41], 7, with_target=True, max_target=60)
    assert list(X.columns) == F.FEATURE_NAMES and len(X) == len(t) == len(keys) == 14
    assert all(o + h <= 60 for o, h in keys) and t[0] == y[41]
    Xf, tf, _ = F.build_rows(y, dates, [79], 7, with_target=False)
    assert len(Xf) == 7 and len(tf) == 0 and list(Xf["horizon"]) == list(range(1, 8))


def test_calendar_features_describe_target_day_not_origin():
    y = np.zeros(60)
    dates = pd.date_range("2026-01-01", periods=60)
    X, _, _ = F.build_rows(y, dates, [30], 3, with_target=False)
    assert list(X["dow"]) == [dates[31].dayofweek, dates[32].dayofweek, dates[33].dayofweek]


def test_outlier_capping():
    y = pd.Series([5.0] * 100)
    y.iloc[50] = 500
    capped, n = F.cap_outliers(y)
    assert n == 1 and capped.max() < 100


# ---------------------------------------------------------------- end to end
def test_forecast_on_weekly_pattern_selects_valid_model_and_is_sane():
    sales = sales_from_series(weekly_pattern_series(220))
    r = F.forecast_product(sales, "P1", 7)
    assert r.sufficient and len(r.forecast) == 7 and r.selected_model in F.MODEL_LABELS
    assert (r.forecast["forecast"] >= 0).all() and (r.forecast["lower"] <= r.forecast["upper"]).all()
    assert 40 < r.expected_units < 90                                 # true mean about 7 x 7 = 52
    ml, base = r.metric("random_forest", "mae"), min(r.metric("recent_average", "mae"), r.metric("seasonal_weekly", "mae"))
    chosen = r.metric(r.selected_model, "mae")
    assert chosen <= base + 1e-9 if r.selected_model != "random_forest" else chosen < base      # selection rule honoured
    assert r.forecast["date"].iloc[0] == sales["transaction_date"].max() + pd.Timedelta(days=1)


def test_ml_not_selected_when_it_does_not_beat_baseline():
    y = pd.Series(np.full(160, 5.0), index=pd.date_range("2026-01-01", periods=160))      # constant demand: baseline is perfect
    r = F.forecast_product(sales_from_series(y), "P1", 7)
    assert r.selected_model != "random_forest" and r.metric("recent_average", "mae") == pytest.approx(0)


def test_insufficient_history_warns_and_returns_no_forecast():
    r = F.forecast_product(dataset_from(daily_sales(10, 3)).sales, "P1", 7)
    assert not r.sufficient and r.forecast.empty and any("days of history" in w for w in r.warnings)


def test_unknown_product_and_no_history():
    r = F.forecast_product(dataset_from(daily_sales(60, 3)).sales, "NOPE", 7)
    assert not r.sufficient and any("No sales history" in w for w in r.warnings)


def test_zero_demand_period_forecasts_zero_without_crashing():
    y = weekly_pattern_series(200)
    y.iloc[-90:] = 0                                    # product stops selling; the shop's other products keep trading
    sales = pd.concat([sales_from_series(y, "P1"), sales_from_series(weekly_pattern_series(200, seed=1), "P2")], ignore_index=True)
    assert sales["transaction_date"].max() == pd.Timestamp("2026-07-19")
    r = F.forecast_product(sales, "P1", 7)
    assert r.sufficient and len(r.history) == 200 and r.history.iloc[-90:].sum() == 0
    assert r.avg_daily_forecast < 1.0 and not r.forecast["forecast"].isna().any()


def test_horizon_30_works(demo_ds):
    r = F.forecast_product(demo_ds.sales, "P020", 30)
    assert r.sufficient and len(r.forecast) == 30 and "WAPE" not in "".join(r.warnings)
    assert set(r.metrics["model"]) <= set(F.MODEL_LABELS)


def test_deterministic(demo_ds):
    a = F.forecast_product(demo_ds.sales, "P013", 7).forecast["forecast"].round(6).tolist()
    b = F.forecast_product(demo_ds.sales, "P013", 7).forecast["forecast"].round(6).tolist()
    assert a == b


def test_demand_override_only_for_ml_winners(demo_ds):
    res = {pid: F.forecast_product(demo_ds.sales, pid, 7) for pid in ["P013", "P020", "P003"]}
    ov = F.demand_override_from(res)
    assert set(ov) == {p for p, r in res.items() if r.selected_model == "random_forest"}
