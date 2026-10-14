"""Daily demand forecasting: baselines vs. a RandomForest, chosen on time-based validation.

Pipeline
--------
1. Build a daily demand series for one product (positive, non-duplicate units; zero-sales days included).
2. Cap extreme single-day volumes (probable data-entry errors / one-off bulk orders) so they do not distort the model.
3. Baselines: ``recent_average`` (mean of the last 28 days) and ``seasonal_weekly`` (mean of the same weekday over the last 4 weeks).
4. ML: ``RandomForestRegressor`` using a DIRECT multi-horizon design. Each training row is (origin, horizon h): features come only
   from data available at the origin (lags, rolling means/std, trend ratio, days since last sale, same-weekday mean) plus calendar
   features of the TARGET day (weekday, month, day of year) and h. The target is demand on day origin+h. There is therefore no
   leakage, and no recursive feeding of predictions back as lags.
5. Validation: the most recent block of days is held out (never random). The model is trained only on targets before the cutoff,
   then evaluated with rolling forecast origins inside the block. Baselines use the same origins and targets.
6. Selection: ML is used only if it beats the best baseline MAE by a margin; otherwise the best baseline is used.
7. The selected model is refit on all data and forecasts the next ``horizon`` days. An approximate range comes from validation errors.

Forecasts are estimates, not guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.ml import evaluation as E

RANDOM_STATE = 42
MIN_HISTORY_DAYS = 28           # below this no meaningful forecast is produced
MIN_ML_TRAIN_DAYS = 56          # minimum training days for ML validation
MIN_SALE_DAYS = 8               # products with fewer sale days are treated as too sparse
ML_MARGIN = 0.02                # ML must beat the best baseline MAE by >= 2% to be selected
LAGS = 7
FEATURE_NAMES = [*(f"lag{k}" for k in range(LAGS)), "roll7", "roll14", "roll28", "std7", "trend_ratio", "zero_share28",
                 "days_since_sale", "same_wd_mean4", "dow", "month", "doy", "horizon"]

MODEL_LABELS = {"recent_average": "Recent average (28 days)", "seasonal_weekly": "Weekly seasonal baseline",
                "random_forest": "Random Forest (ML)"}


# --------------------------------------------------------------------------- data prep
def daily_demand(sales: pd.DataFrame, product_id: str, end: pd.Timestamp | None = None) -> pd.Series:
    """Daily units sold for one product (returns and duplicate lines excluded), zero-filled."""
    df = sales[(sales["product_id"] == product_id) & (~sales["is_duplicate"]) & (~sales["is_return"])]
    if df.empty:
        return pd.Series(dtype="float64")
    s = df.groupby("transaction_date")["quantity"].sum()
    last = pd.Timestamp(end) if end is not None else sales["transaction_date"].max()
    first = sales["transaction_date"].min()
    return s.reindex(pd.date_range(first, last, freq="D"), fill_value=0.0).astype("float64")


def cap_outliers(y: pd.Series) -> tuple[pd.Series, int]:
    """Cap implausible single-day volumes at max(median + 8 x robust scale, 6 x median) of sale days."""
    nz = y[y > 0]
    if len(nz) < 20:
        return y, 0
    med = float(nz.median())
    scale = 1.4826 * float(np.median(np.abs(nz - med)))
    cap = max(med + 8 * max(scale, 0.25 * med), 6 * med)
    n = int((y > cap).sum())
    return y.clip(upper=cap), n


# --------------------------------------------------------------------------- features
def origin_features(y: np.ndarray, o: int) -> dict[str, float]:
    """Features known at origin index ``o`` (uses y[:o+1] only)."""
    w = y[: o + 1]
    last7, last14, last28 = w[-7:], w[-14:], w[-28:]
    nz = np.nonzero(w)[0]
    dss = float(min(o - nz[-1], 28)) if len(nz) else 28.0
    feats = {f"lag{k}": float(w[-1 - k]) if o - k >= 0 else 0.0 for k in range(LAGS)}
    feats.update(roll7=float(last7.mean()), roll14=float(last14.mean()), roll28=float(last28.mean()), std7=float(last7.std()),
                 trend_ratio=float(last7.mean() / last28.mean()) if last28.mean() > 0 else 1.0,
                 zero_share28=float((last28 == 0).mean()), days_since_sale=dss)
    return feats


def same_weekday_mean(y: np.ndarray, o: int, h: int, weeks: int = 4) -> float:
    """Mean demand on the target's weekday over the last ``weeks`` observed occurrences (all <= origin)."""
    idx = [o + h - 7 * k for k in range(1, weeks + 8) if o + h - 7 * k <= o]
    idx = [i for i in idx if i >= 0][:weeks]
    return float(np.mean(y[idx])) if idx else 0.0


def build_rows(y: np.ndarray, dates: pd.DatetimeIndex, origins: list[int], horizon: int, with_target: bool,
               max_target: int | None = None) -> tuple[pd.DataFrame, np.ndarray, list[tuple[int, int]]]:
    """Feature matrix for (origin, h) pairs. Targets (when requested) never exceed ``max_target``."""
    rows, targets, keys = [], [], []
    for o in origins:
        base = origin_features(y, o)
        for h in range(1, horizon + 1):
            t = o + h
            if with_target and (t >= len(y) or (max_target is not None and t > max_target)):
                break
            d = dates[t] if t < len(dates) else dates[-1] + pd.Timedelta(days=t - len(dates) + 1)
            rows.append({**base, "same_wd_mean4": same_weekday_mean(y, o, h), "dow": d.dayofweek, "month": d.month,
                         "doy": d.dayofyear, "horizon": h})
            keys.append((o, h))
            if with_target:
                targets.append(y[t])
    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    return X, np.asarray(targets, float), keys


def _fit_rf(y: np.ndarray, dates: pd.DatetimeIndex, horizon: int, max_target: int) -> RandomForestRegressor | None:
    origins = list(range(27, max_target))
    X, t, _ = build_rows(y, dates, origins, horizon, with_target=True, max_target=max_target)
    if len(X) < 100:
        return None
    rf = RandomForestRegressor(n_estimators=80, max_depth=8, min_samples_leaf=8, max_features=0.6, n_jobs=-1,
                               random_state=RANDOM_STATE)
    rf.fit(X, t)
    return rf


# --------------------------------------------------------------------------- baselines
def baseline_recent_average(y: np.ndarray, o: int, horizon: int) -> np.ndarray:
    return np.full(horizon, float(y[max(0, o - 27): o + 1].mean()))


def baseline_seasonal_weekly(y: np.ndarray, o: int, horizon: int) -> np.ndarray:
    return np.array([same_weekday_mean(y, o, h) for h in range(1, horizon + 1)])


# --------------------------------------------------------------------------- result container
@dataclass
class ForecastResult:
    product_id: str
    product_name: str
    horizon: int
    sufficient: bool
    history: pd.Series
    forecast: pd.DataFrame = field(default_factory=pd.DataFrame)   # date, forecast, lower, upper, + each model column
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)    # model, mae, wape, rmse, bias, n
    selected_model: str = ""
    ml_available: bool = False
    warnings: list[str] = field(default_factory=list)
    outliers_capped: int = 0
    backtest_origins: int = 0

    @property
    def expected_units(self) -> float:
        return float(self.forecast["forecast"].sum()) if len(self.forecast) else float("nan")

    @property
    def avg_daily_forecast(self) -> float:
        return float(self.forecast["forecast"].mean()) if len(self.forecast) else float("nan")

    def metric(self, model: str, key: str) -> float | None:
        m = self.metrics[self.metrics["model"] == model]
        return float(m.iloc[0][key]) if len(m) and pd.notna(m.iloc[0][key]) else None


LIMITATIONS = [
    "Forecasts are statistical estimates from past sales; they are not guarantees.",
    "Days when an item was out of stock show zero sales and can make demand look lower than it really was.",
    "With about one year of data the model cannot learn year-on-year seasonality reliably; festivals, promotions and price changes are not modelled explicitly.",
    "Validation is a backtest on the most recent weeks, so accuracy on future (different) conditions may be worse.",
    "Extreme one-day volumes are capped before training, so genuine bulk orders are not forecast.",
]


# --------------------------------------------------------------------------- main entry point
def forecast_product(sales: pd.DataFrame, product_id: str, horizon: int = 7, product_name: str | None = None,
                     as_of: pd.Timestamp | None = None) -> ForecastResult:
    """Forecast daily demand for the next ``horizon`` days for a single product."""
    name = product_name or _name_of(sales, product_id)
    raw = daily_demand(sales, product_id, end=as_of)
    res = ForecastResult(product_id, name, horizon, sufficient=False, history=raw)
    if raw.empty:
        res.warnings.append("No sales history for this product, so no forecast can be made.")
        return res
    sale_days = int((raw > 0).sum())
    if len(raw) < MIN_HISTORY_DAYS:
        res.warnings.append(f"Only {len(raw)} days of history (need at least {MIN_HISTORY_DAYS}); no reliable forecast.")
        return res
    if sale_days < MIN_SALE_DAYS:
        res.warnings.append(f"Sold on only {sale_days} day(s); demand is too sparse for a reliable forecast. A recent-average estimate is shown.")

    ys, capped = cap_outliers(raw)
    res.outliers_capped = capped
    if capped:
        res.warnings.append(f"{capped} extreme day(s) were capped before modelling (likely data-entry errors or one-off bulk orders).")
    y = ys.to_numpy(float)
    dates = ys.index
    n = len(y)
    last = n - 1
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")

    # --- validation
    cutoff = E.time_split_index(n, horizon, min_train=MIN_ML_TRAIN_DAYS)
    models_scored: dict[str, dict[str, float]] = {}
    residuals: dict[str, np.ndarray] = {}
    ml_ok = False
    if cutoff is not None and sale_days >= MIN_SALE_DAYS:
        origins = E.backtest_origins(cutoff, n, horizon)
        res.backtest_origins = len(origins)
        actual = np.concatenate([y[o + 1: o + 1 + horizon] for o in origins])
        for key, fn in (("recent_average", baseline_recent_average), ("seasonal_weekly", baseline_seasonal_weekly)):
            pred = np.concatenate([fn(y, o, horizon) for o in origins])
            models_scored[key] = E.score(actual, pred)
            residuals[key] = pred - actual
        rf_val = _fit_rf(y, dates, horizon, max_target=cutoff - 1)
        if rf_val is not None:
            Xv, _, _ = build_rows(y, dates, origins, horizon, with_target=False)
            pred = np.clip(rf_val.predict(Xv), 0, None)
            models_scored["random_forest"] = E.score(actual, pred)
            residuals["random_forest"] = pred - actual
            ml_ok = True
    else:
        res.warnings.append("Not enough history for a time-based validation; only the recent-average baseline is available "
                            f"(need roughly {MIN_ML_TRAIN_DAYS + horizon} days with regular sales).")

    # --- selection
    if models_scored:
        base_keys = [k for k in ("recent_average", "seasonal_weekly") if k in models_scored]
        best_base = min(base_keys, key=lambda k: models_scored[k]["mae"])
        selected = best_base
        if ml_ok and models_scored["random_forest"]["mae"] < models_scored[best_base]["mae"] * (1 - ML_MARGIN):
            selected = "random_forest"
        res.metrics = pd.DataFrame([{"model": k, **v} for k, v in models_scored.items()])
        res.metrics["selected"] = res.metrics["model"] == selected
    else:
        selected = "recent_average"

    # --- final forecasts on all data
    preds: dict[str, np.ndarray] = {"recent_average": baseline_recent_average(y, last, horizon),
                                    "seasonal_weekly": baseline_seasonal_weekly(y, last, horizon)}
    if ml_ok:
        rf_full = _fit_rf(y, dates, horizon, max_target=last)
        if rf_full is not None:
            Xf, _, _ = build_rows(y, dates, [last], horizon, with_target=False)
            Xf["doy"] = [d.dayofyear for d in future_dates]
            Xf["month"] = [d.month for d in future_dates]
            Xf["dow"] = [d.dayofweek for d in future_dates]
            preds["random_forest"] = np.clip(rf_full.predict(Xf), 0, None)
    res.ml_available = "random_forest" in preds
    if selected == "random_forest" and not res.ml_available:
        selected = "recent_average"
    if not models_scored:
        res.warnings.append("Showing a recent-average estimate only; accuracy could not be validated.")

    fc = pd.DataFrame({"date": future_dates, **{k: v for k, v in preds.items()}})
    fc["forecast"] = fc[selected]
    if selected in residuals and len(residuals[selected]) >= 10:
        q_lo, q_hi = np.quantile(residuals[selected], [0.9, 0.1])   # error = pred - actual
        fc["lower"] = np.clip(fc["forecast"] - q_lo, 0, None)
        fc["upper"] = np.clip(fc["forecast"] - q_hi, 0, None)
    else:
        fc["lower"], fc["upper"] = np.nan, np.nan
    res.forecast, res.selected_model, res.sufficient = fc, selected, True
    if selected != "random_forest" and ml_ok:
        res.warnings.append("The ML model did not beat the baseline on validation, so the baseline forecast is used.")
    if models_scored and selected in models_scored and (models_scored[selected]["wape"] > 0.8 if pd.notna(models_scored[selected]["wape"]) else False):
        res.warnings.append("Validation error is high (WAPE above 80%): demand is very irregular, treat this forecast as a rough guide.")
    return res


def _name_of(sales: pd.DataFrame, product_id: str) -> str:
    m = sales.loc[sales["product_id"] == product_id, "product_name"]
    return str(m.iloc[-1]) if len(m) else product_id


def forecast_many(sales: pd.DataFrame, product_ids: list[str], horizon: int = 7, progress=None) -> dict[str, ForecastResult]:
    out: dict[str, ForecastResult] = {}
    for i, pid in enumerate(product_ids):
        out[pid] = forecast_product(sales, pid, horizon)
        if progress:
            progress((i + 1) / max(len(product_ids), 1), out[pid].product_name)
    return out


def demand_override_from(results: dict[str, ForecastResult]) -> dict[str, float]:
    """Average daily forecast per product, only where a validated forecast exists and used the ML model."""
    return {pid: r.avg_daily_forecast for pid, r in results.items() if r.sufficient and r.selected_model == "random_forest"}
