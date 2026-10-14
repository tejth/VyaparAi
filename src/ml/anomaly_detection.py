"""Anomaly and profit-leakage detection: transparent rules + optional Isolation Forest.

Every finding is placed in ONE of three groups so confirmed problems are never mixed up with guesses:

* "Data quality"        - verifiable problems in the records (duplicates, totals that do not add up, missing costs).
* "Statistical anomaly" - values that are unusual versus the product's own history (quantity, price, daily spikes,
                          Isolation Forest outliers). They may be legitimate (bulk order, new price) and REQUIRE REVIEW.
* "Business risk"       - patterns that could be eroding profit (heavy discounts, negative margins, falling sales).

Nothing here is evidence of fraud; findings are potential risks that need a person to check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config
from src.analytics import profitability as P
from src.analytics import sales as S
from src.data.loader import Dataset
from src.utils.formatting import inr, num, pct

COLUMNS = ["record_id", "date", "product_id", "product_name", "group", "detection", "observed", "expected", "severity",
           "explanation", "next_step", "amount_inr"]
SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def _row(record_id, date, pid, name, group, detection, observed, expected, severity, explanation, next_step, amount=np.nan) -> dict:
    return {"record_id": record_id, "date": date, "product_id": pid, "product_name": name, "group": group, "detection": detection,
            "observed": observed, "expected": expected, "severity": severity, "explanation": explanation,
            "next_step": next_step, "amount_inr": amount}


def _robust_scale(s: pd.Series) -> tuple[float, float]:
    med = float(s.median())
    mad = 1.4826 * float((s - med).abs().median())
    return med, mad


def detect_anomalies(ds: Dataset, rules: config.BusinessRules | None = None, use_isolation_forest: bool = True,
                     as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    rules = rules or config.DEFAULT_RULES
    if ds.is_empty:
        return pd.DataFrame(columns=COLUMNS)
    as_of = as_of or ds.as_of
    sales = ds.sales
    rows: list[dict] = []
    live = sales[~sales["is_duplicate"]]

    # ------------------------------------------------------------------ data quality
    for r in sales[sales["is_duplicate"]].itertuples():
        rows.append(_row(r.transaction_id, r.transaction_date, r.product_id, r.product_name, "Data quality", "Duplicate transaction",
                         f"Line repeated ({num(r.quantity)} x {inr(r.unit_price, 2)})", "Each transaction_id appears once", "Medium",
                         "An identical line (same id, date, product, quantity, price and amount) already exists. The repeat is excluded from revenue.",
                         "Check the billing system or CSV export for a double entry.", abs(r.total_amount)))
    for r in sales[sales["total_mismatch"] & ~sales["is_duplicate"]].itertuples():
        diff = r.total_amount - r.expected_total
        rows.append(_row(r.transaction_id, r.transaction_date, r.product_id, r.product_name, "Data quality", "Inconsistent total",
                         f"Recorded {inr(r.total_amount, 2)}", f"{inr(r.expected_total, 2)} (qty x price x (1 - discount))",
                         "High" if abs(diff) >= 500 else "Medium",
                         f"The recorded total differs from quantity x price after discount by {inr(abs(diff), 2)}. The recorded amount is used in revenue.",
                         "Verify the bill; correct the amount or the price/discount.", abs(diff)))
    miss = live[live["cost_source"] == "unknown"].groupby(["product_id", "product_name"]).size().reset_index(name="n")
    for r in miss.itertuples():
        rows.append(_row(f"cost-{r.product_id}", as_of, r.product_id, r.product_name, "Data quality", "Missing unit cost",
                         f"{r.n} line(s) without cost", "Cost recorded for every line", "Medium",
                         "Cost is unknown for these lines and no inventory cost exists, so their margin is excluded rather than assumed to be zero.",
                         "Add the product's purchase cost to the sales or inventory file."))

    # ------------------------------------------------------------------ statistical: discount, quantity, price
    pos = live[(~live["is_return"]) & (live["unit_price"] > 0)].copy()
    big_disc = pos[pos["discount"] >= rules.high_discount_pct].copy()
    per_prod = big_disc.groupby("product_id")["transaction_id"].transform("count")
    extreme = big_disc[(big_disc["discount"] >= 50) | (per_prod < 5)]
    repeated = big_disc[(big_disc["discount"] < 50) & (per_prod >= 5)]
    for r in extreme.itertuples():
        given = r.quantity * r.unit_price - r.total_amount
        sev = "High" if r.discount >= 60 else ("Medium" if r.discount >= 40 else "Low")
        below = pd.notna(r.unit_cost_eff) and r.total_amount < r.quantity * r.unit_cost_eff
        rows.append(_row(r.transaction_id, r.transaction_date, r.product_id, r.product_name, "Business risk", "Unusually large discount",
                         pct(r.discount, 0), f"Below {rules.high_discount_pct:.0f}%", sev,
                         f"{pct(r.discount, 0)} discount on {num(r.quantity)} unit(s) gave away {inr(given, 2)}"
                         + (" and the line sold below cost." if below else "."),
                         "Confirm the discount was authorised and the reason (clearance, damaged goods, negotiated bulk price).", max(given, 0)))
    for (pid, name), g in repeated.groupby(["product_id", "product_name"]):
        given = float((g["quantity"] * g["unit_price"] - g["total_amount"]).sum())
        below = int((g["total_amount"] < g["quantity"] * g["unit_cost_eff"]).sum())
        rows.append(_row(f"disc-{pid}", g["transaction_date"].max(), pid, name, "Business risk", "Frequent heavy discounting",
                         f"{len(g)} lines at {g['discount'].min():.0f}-{g['discount'].max():.0f}% off",
                         f"Discounts below {rules.high_discount_pct:.0f}%", "High" if below / len(g) > 0.5 else "Medium",
                         f"{len(g)} lines (latest {g['transaction_date'].max():%d %b %Y}) carried heavy discounts worth {inr(given)} in total; {below} sold below cost.",
                         "Check whether a promotion was planned and whether margin can support it.", given))

    stats = pos.groupby("product_id")["quantity"].agg(["median", "count"])
    scale = pos.groupby("product_id")["quantity"].apply(lambda s: 1.4826 * float((s - s.median()).abs().median()))
    pos = pos.join(stats, on="product_id").join(scale.rename("qty_scale"), on="product_id")
    med1 = pos["median"].clip(lower=1)
    thr = np.maximum(8 * med1, med1 + rules.large_qty_zscore * np.maximum(pos["qty_scale"], med1))
    big_qty = pos[(pos["quantity"] > thr) & (pos["quantity"] >= 6) & (pos["count"] >= 20)]
    for r in big_qty.itertuples():
        ratio = r.quantity / max(r.median, 1)
        sev = "High" if ratio >= 20 else "Medium"
        rows.append(_row(r.transaction_id, r.transaction_date, r.product_id, r.product_name, "Statistical anomaly", "Abnormally large quantity",
                         f"{num(r.quantity)} units", f"Typical line: {num(r.median)} unit(s)", sev,
                         f"This line is {ratio:.0f}x the product's typical quantity. It could be a bulk/wholesale order or a typing error (an extra digit).",
                         "Check the bill. If it is a typo, correct it; if a genuine bulk sale, confirm stock and price.", r.total_amount))

    ref_price = pos.groupby("product_id")["unit_price"].median().rename("ref_price")
    pos = pos.join(ref_price, on="product_id")
    dev = (pos["unit_price"] - pos["ref_price"]) / pos["ref_price"] * 100
    price_out = pos[dev.abs() >= rules.price_change_pct]
    for r in price_out.itertuples():
        d = (r.unit_price - r.ref_price) / r.ref_price * 100
        rows.append(_row(r.transaction_id, r.transaction_date, r.product_id, r.product_name, "Statistical anomaly", "Unexpected price change",
                         f"{inr(r.unit_price, 2)} per unit", f"Usual price ~ {inr(r.ref_price, 2)} (+/-{rules.price_change_pct:.0f}%)",
                         "High" if abs(d) >= 100 else "Medium",
                         f"Unit price is {d:+.0f}% versus the product's usual price. Either a one-off price change or a data-entry error.",
                         "Verify the price on the bill and in the price list.", abs(r.quantity * (r.unit_price - r.ref_price))))

    # ------------------------------------------------------------------ statistical: daily spikes
    daily = P.timeseries(sales, "D")
    if len(daily) >= 28:
        d = daily.assign(wd=daily["period"].dt.dayofweek)
        wd_med = d.groupby("wd")["revenue"].transform("median").replace(0, np.nan)
        ratio = (d["revenue"] / wd_med).dropna()
        med, scl = _robust_scale(ratio)
        if scl > 0:
            z = (ratio - med) / scl
            for i in z[z >= rules.spike_zscore].index:
                rev = float(d.loc[i, "revenue"])
                rows.append(_row(f"day-{d.loc[i, 'period']:%Y%m%d}", d.loc[i, "period"], "", "All products", "Statistical anomaly", "Revenue spike",
                                 inr(rev), f"Typical {d.loc[i, 'period']:%A}: {inr(wd_med[i])}", "Medium" if z[i] < 2 * rules.spike_zscore else "High",
                                 f"Revenue was {rev / wd_med[i]:.1f}x the typical {d.loc[i, 'period']:%A}. Often a bulk order, festival day or an entry error.",
                                 "Review the largest bills of that day.", rev - float(wd_med[i])))

    # ------------------------------------------------------------------ business risk: margins and sales declines
    prod = P.by_product(P.filter_sales(sales, as_of - pd.Timedelta(days=89), as_of))
    for r in prod[(prod["margin_pct"] < rules.low_margin_pct) & (prod["revenue"] > 0)].itertuples():
        neg = r.gross_profit < 0
        rows.append(_row(f"margin-{r.product_id}", as_of, r.product_id, r.product_name, "Business risk",
                         "Negative gross margin" if neg else "Low gross margin", pct(r.margin_pct), f"At least {rules.low_margin_pct:.0f}%",
                         "High" if neg else "Low",
                         (f"Over the last 90 days gross profit was {inr(r.gross_profit)} on revenue of {inr(r.revenue)}: the product is selling below its cost."
                          if neg else f"Gross margin of {pct(r.margin_pct)} over the last 90 days leaves little room to cover overheads."),
                         "Review the selling price, the supplier cost and recent discounts." if neg else "Check whether the price can be revised or the cost reduced.",
                         max(-r.gross_profit, 0) if neg else np.nan))
    margin_drop = P.recent_margin_drop(sales, as_of)
    for r in margin_drop[margin_drop["margin_change_pts"] <= -3].itertuples():
        rows.append(_row(f"mdrop-{r.product_id}", as_of, r.product_id, r.product_name, "Business risk", "Margin decline",
                         pct(r.margin_cur), f"Previous 30 days: {pct(r.margin_prev)}", "Medium",
                         f"Gross margin fell {abs(r.margin_change_pts):.1f} percentage points versus the previous 30 days.",
                         "Look for new discounts or a supplier cost increase."))

    for r in S.declining_products(sales, as_of, 30, rules.decline_pct).itertuples():
        stockout = r.units_cur == 0
        rows.append(_row(f"decline-{r.product_id}", as_of, r.product_id, r.product_name, "Business risk", "Sudden sales decline",
                         f"{num(r.units_cur)} units (last 30 days)", f"{num(r.units_prev)} units (previous 30 days)",
                         "High" if r.change_pct <= -60 else "Medium",
                         f"Units sold fell {abs(r.change_pct):.0f}%. Possible causes include stock-outs, seasonality, a competitor or a price change; the data alone cannot tell which.",
                         "Check stock availability first, then pricing and shelf placement.", r.revenue_prev - r.revenue_cur))

    # ------------------------------------------------------------------ optional Isolation Forest (extra, not a replacement)
    if use_isolation_forest and len(pos) >= 500:
        flagged = {(x[0], x[1]) for x in [(row["record_id"], row["detection"]) for row in rows]}
        feats = pd.DataFrame({
            "qty_ratio": np.log1p(pos["quantity"] / pos["median"].clip(lower=1)),
            "price_ratio": pos["unit_price"] / pos["ref_price"],
            "discount": pos["discount"],
            "margin": ((pos["unit_price"] * (1 - pos["discount"] / 100) - pos["unit_cost_eff"]) / pos["unit_price"]).clip(-1, 1).fillna(0.0),
        })
        iso = IsolationForest(n_estimators=100, contamination=0.001, random_state=42, n_jobs=-1).fit(feats)
        out = pos[iso.predict(feats) == -1]
        already = {row["record_id"] for row in rows if row["group"] != "Business risk" or row["detection"] == "Unusually large discount"}
        for r in out.itertuples():
            if r.transaction_id in already:
                continue
            rows.append(_row(r.transaction_id, r.transaction_date, r.product_id, r.product_name, "Statistical anomaly", "Isolation Forest outlier",
                             f"{num(r.quantity)} units @ {inr(r.unit_price, 2)}, {pct(r.discount, 0)} off",
                             "Typical combination for this product", "Low",
                             "The combination of quantity, price, discount and margin is unusual compared with all other sales lines (multivariate outlier).",
                             "Review the bill; most outliers are legitimate.", np.nan))

    out_df = pd.DataFrame(rows, columns=COLUMNS)
    if out_df.empty:
        return out_df
    out_df["_sev"] = out_df["severity"].map(SEVERITY_ORDER)
    out_df = out_df.sort_values(["_sev", "date"], ascending=[True, False]).drop(columns="_sev").reset_index(drop=True)
    return out_df


def summarize(anoms: pd.DataFrame) -> dict[str, int]:
    if anoms.empty:
        return {"total": 0, "high": 0, "data_quality": 0, "statistical": 0, "business_risk": 0}
    return {"total": len(anoms), "high": int((anoms["severity"] == "High").sum()),
            "data_quality": int((anoms["group"] == "Data quality").sum()),
            "statistical": int((anoms["group"] == "Statistical anomaly").sum()),
            "business_risk": int((anoms["group"] == "Business risk").sum())}
