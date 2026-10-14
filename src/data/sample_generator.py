"""Synthetic demo data for a fictional Indian general-store / kirana retailer.

Everything here is SYNTHETIC. It is generated with a fixed seed so the demo is
reproducible. Stock is simulated day by day (opening stock + purchases - sales), so
current_stock reconciles with the sales and purchase files, except where anomalies are
deliberately injected (documented in the returned scenario list).

All monetary values are in Indian Rupees (INR, ₹). `discount` is a PERCENTAGE (0-100)
applied to unit_price x quantity. `purchase_cost` is the TOTAL cost of the purchase line.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START = date(2025, 10, 1)
N_DAYS = 365  # 1 Oct 2025 .. 30 Sep 2026
END = START + timedelta(days=N_DAYS - 1)

# name, category, base_cost (₹), margin %, base units/day, lead time (days), seasonal profile, tags
PRODUCTS: list[tuple] = [
    # Staples
    ("Aashirvaad Atta 5kg", "Staples", 215, 9, 6.0, 3, "flat", ["no_restock"]),
    ("Fortune Chakki Atta 10kg", "Staples", 395, 8, 3.5, 3, "flat", []),
    ("India Gate Basmati Rice 5kg", "Staples", 560, 10, 2.5, 4, "festive", []),
    ("Sona Masoori Rice 25kg", "Staples", 1180, 7, 1.2, 4, "flat", []),
    ("Tata Salt 1kg", "Staples", 21, 12, 9.0, 2, "flat", ["no_restock"]),
    ("Madhur Sugar 1kg", "Staples", 41, 6, 8.0, 2, "festive", []),
    # Pulses
    ("Toor Dal 1kg", "Pulses", 128, 10, 3.0, 3, "flat", ["no_restock"]),
    ("Moong Dal 1kg", "Pulses", 112, 10, 2.2, 3, "flat", []),
    ("Chana Dal 1kg", "Pulses", 82, 11, 2.5, 3, "flat", []),
    ("Masoor Dal 1kg", "Pulses", 86, 10, 2.0, 3, "flat", []),
    ("Rajma 1kg", "Pulses", 118, 12, 1.0, 3, "winter", []),
    ("Kabuli Chana 1kg", "Pulses", 132, 11, 0.9, 3, "flat", []),
    # Edible oils
    ("Fortune Sunflower Oil 1L", "Edible Oils", 128, 6, 7.0, 3, "flat", ["neg_margin"]),
    ("Saffola Gold Oil 1L", "Edible Oils", 175, 8, 2.5, 3, "flat", []),
    ("Kachi Ghani Mustard Oil 1L", "Edible Oils", 148, 9, 3.2, 3, "winter", []),
    ("Dhara Groundnut Oil 1L", "Edible Oils", 210, 8, 1.4, 4, "flat", []),
    ("Fortune Soya Oil 1L", "Edible Oils", 118, 7, 3.5, 3, "flat", ["no_restock"]),
    ("Patanjali Cow Ghee 1kg", "Edible Oils", 560, 10, 1.6, 4, "festive", []),
    # Dairy & bakery
    ("Amul Butter 500g", "Dairy & Bakery", 262, 7, 3.0, 2, "flat", ["no_restock"]),
    ("Amul Gold Milk 1L", "Dairy & Bakery", 62, 5, 14.0, 1, "flat", []),
    ("Mother Dairy Paneer 200g", "Dairy & Bakery", 78, 12, 3.5, 1, "flat", []),
    ("Amul Cheese Slices 200g", "Dairy & Bakery", 118, 10, 2.0, 2, "flat", []),
    ("Britannia Bread 400g", "Dairy & Bakery", 36, 9, 8.0, 1, "flat", []),
    ("Amul Masti Dahi 400g", "Dairy & Bakery", 34, 10, 5.0, 1, "summer", []),
    ("Eggs Tray 12 pcs", "Dairy & Bakery", 78, 8, 5.0, 1, "winter", []),
    # Snacks & confectionery
    ("Maggi Masala Noodles 4-pack", "Snacks & Confectionery", 62, 11, 8.0, 3, "flat", ["no_restock"]),
    ("Parle-G Biscuits 250g", "Snacks & Confectionery", 20, 12, 14.0, 3, "flat", []),
    ("Britannia Good Day 200g", "Snacks & Confectionery", 38, 12, 7.0, 3, "flat", []),
    ("Haldiram Bhujia 400g", "Snacks & Confectionery", 105, 15, 3.5, 4, "festive", ["discount_erosion"]),
    ("Lays Classic Salted 52g", "Snacks & Confectionery", 18, 15, 11.0, 3, "summer", []),
    ("Kurkure Masala Munch 90g", "Snacks & Confectionery", 18, 15, 9.0, 3, "flat", []),
    ("Sunfeast Dark Fantasy 300g", "Snacks & Confectionery", 98, 13, 2.4, 3, "flat", []),
    ("Haldiram Soan Papdi 500g", "Snacks & Confectionery", 190, 18, 1.8, 4, "festive_strong", []),
    ("Cadbury Dairy Milk 50g", "Snacks & Confectionery", 36, 12, 6.0, 3, "festive", []),
    # Beverages
    ("Tata Tea Gold 1kg", "Beverages", 480, 9, 2.5, 3, "winter", []),
    ("Red Label Tea 500g", "Beverages", 245, 9, 3.0, 3, "winter", []),
    ("Nescafe Classic 100g", "Beverages", 295, 12, 1.8, 4, "winter", []),
    ("Bru Instant Coffee 100g", "Beverages", 220, 11, 1.5, 4, "winter", []),
    ("Coca-Cola 2L", "Beverages", 85, 10, 3.0, 2, "summer", []),
    ("Thums Up 750ml", "Beverages", 40, 10, 5.0, 2, "summer", ["no_restock"]),
    ("Frooti Mango 1L", "Beverages", 62, 11, 4.0, 2, "summer", []),
    ("Real Fruit Juice 1L", "Beverages", 98, 12, 2.2, 3, "summer", []),
    ("Bisleri Water 1L", "Beverages", 15, 18, 18.0, 2, "summer", []),
    ("Horlicks 500g", "Beverages", 255, 10, 1.5, 4, "flat", []),
    # Personal care
    ("Dettol Soap 3-pack", "Personal Care", 112, 14, 3.0, 3, "flat", ["discount_erosion"]),
    ("Lux Soap 4-pack", "Personal Care", 118, 14, 2.8, 3, "flat", []),
    ("Colgate MaxFresh 150g", "Personal Care", 92, 13, 4.5, 3, "flat", []),
    ("Clinic Plus Shampoo 340ml", "Personal Care", 188, 15, 2.0, 4, "flat", []),
    ("Head & Shoulders 340ml", "Personal Care", 292, 14, 1.2, 4, "flat", []),
    ("Parachute Coconut Oil 500ml", "Personal Care", 168, 12, 2.8, 3, "winter", ["decline"]),
    ("Vaseline Body Lotion 400ml", "Personal Care", 245, 15, 0.9, 4, "winter", []),
    ("Whisper Ultra Pads 20s", "Personal Care", 168, 13, 1.6, 4, "flat", []),
    # Household
    ("Surf Excel Easy Wash 1kg", "Household", 198, 11, 3.2, 3, "flat", ["no_restock"]),
    ("Vim Dishwash Liquid 500ml", "Household", 98, 14, 3.0, 3, "flat", []),
    ("Harpic Toilet Cleaner 1L", "Household", 168, 14, 1.8, 4, "flat", []),
    ("Lizol Floor Cleaner 1L", "Household", 195, 15, 1.5, 4, "flat", ["decline"]),
    ("Good Knight Refill 45ml", "Household", 78, 15, 2.5, 3, "monsoon", []),
    ("Comfort Fabric Conditioner 860ml", "Household", 195, 15, 1.0, 4, "flat", []),
    ("Garbage Bags 30 pcs", "Household", 95, 20, 1.1, 4, "flat", []),
    # Spices
    ("Everest Turmeric Powder 200g", "Spices", 62, 18, 2.5, 3, "flat", []),
    ("Everest Chilli Powder 200g", "Spices", 78, 18, 2.2, 3, "flat", []),
    ("MDH Garam Masala 100g", "Spices", 85, 18, 2.0, 3, "winter", []),
    ("Catch Cumin Seeds 100g", "Spices", 72, 18, 1.4, 3, "flat", []),
    ("Everest Chaat Masala 100g", "Spices", 68, 18, 1.2, 3, "flat", []),
    ("Kashmiri Saffron 1g", "Spices", 190, 25, 0.2, 5, "festive", ["slow"]),
    # Dry fruits
    ("California Almonds 250g", "Dry Fruits", 245, 16, 1.0, 4, "festive_strong", []),
    ("Cashew Nuts 250g", "Dry Fruits", 285, 16, 0.9, 4, "festive_strong", []),
    ("Raisins 250g", "Dry Fruits", 82, 18, 1.1, 4, "festive", []),
    ("Dates 500g", "Dry Fruits", 165, 15, 0.9, 4, "festive", []),
    ("Walnut Kernels 250g", "Dry Fruits", 310, 15, 0.3, 5, "festive", ["slow"]),
    # Gourmet & health (slow / dead stock scenarios)
    ("Quinoa 500g", "Gourmet & Health", 240, 22, 0.5, 6, "flat", ["dead:150"]),
    ("Chia Seeds 250g", "Gourmet & Health", 210, 22, 0.4, 6, "flat", ["dead:190"]),
    ("Extra Virgin Olive Oil 500ml", "Gourmet & Health", 520, 20, 0.4, 6, "flat", ["dead:215"]),
    ("Imported Penne Pasta 500g", "Gourmet & Health", 135, 20, 0.15, 6, "flat", ["slow", "overstock"]),
    ("Muesli Granola 500g", "Gourmet & Health", 215, 20, 0.12, 6, "flat", ["slow", "overstock"]),
]

_MONTH_PROFILES: dict[str, dict[int, float]] = {
    "flat": {m: 1.0 for m in range(1, 13)},
    "summer": {1: 0.8, 2: 0.9, 3: 1.2, 4: 1.6, 5: 1.9, 6: 1.7, 7: 1.2, 8: 1.0, 9: 0.9, 10: 0.85, 11: 0.8, 12: 0.8},
    "winter": {1: 1.4, 2: 1.3, 3: 1.0, 4: 0.8, 5: 0.7, 6: 0.7, 7: 0.8, 8: 0.85, 9: 0.9, 10: 1.0, 11: 1.35, 12: 1.45},
    "festive": {1: 1.0, 2: 0.95, 3: 1.2, 4: 0.95, 5: 0.9, 6: 0.9, 7: 0.9, 8: 1.0, 9: 1.1, 10: 1.6, 11: 1.3, 12: 1.1},
    "festive_strong": {1: 1.0, 2: 0.9, 3: 1.3, 4: 0.8, 5: 0.7, 6: 0.7, 7: 0.7, 8: 0.9, 9: 1.2, 10: 2.3, 11: 1.7, 12: 1.2},
    "monsoon": {1: 0.8, 2: 0.8, 3: 0.9, 4: 0.9, 5: 1.0, 6: 1.5, 7: 1.8, 8: 1.8, 9: 1.5, 10: 1.0, 11: 0.8, 12: 0.8},
}
_WEEKDAY = {0: 0.9, 1: 0.9, 2: 0.95, 3: 1.0, 4: 1.1, 5: 1.25, 6: 1.35}
_PAYMENTS = ["UPI", "Cash", "Card", "Credit"]
_PAYMENT_P = [0.50, 0.30, 0.12, 0.08]


@dataclass
class DemoData:
    sales: pd.DataFrame
    inventory: pd.DataFrame
    purchases: pd.DataFrame
    scenarios: list[dict] = field(default_factory=list)


def _split_quantity(rng: np.random.Generator, total: int) -> list[int]:
    if total <= 2:
        return [total]
    k = int(rng.integers(1, min(total, 4) + 1))
    if k == 1:
        return [total]
    cuts = sorted(rng.choice(np.arange(1, total), size=k - 1, replace=False).tolist())
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def generate(seed: int = SEED) -> DemoData:
    rng = np.random.default_rng(seed)
    sales_rows: list[dict] = []
    purchase_rows: list[dict] = []
    inv_rows: list[dict] = []
    forced: dict[tuple[str, int], int] = {}  # legitimate bulk orders (stock is arranged for them)

    # Legitimate bulk wholesale orders are placed through the simulation so stock reconciles.
    name_to_idx = {p[0]: i for i, p in enumerate(PRODUCTS)}
    bulk_specs = [("Madhur Sugar 1kg", date(2026, 3, 12), 400, 5.0), ("Bisleri Water 1L", date(2026, 5, 8), 350, 8.0)]
    bulk_discount: dict[tuple[str, int], float] = {}
    for pname, d, q, disc in bulk_specs:
        pid = f"P{name_to_idx[pname] + 1:03d}"
        forced[(pid, (d - START).days)] = q
        bulk_discount[(pid, (d - START).days)] = disc

    for idx, (name, cat, cost0, margin, lam, lead, profile, tags) in enumerate(PRODUCTS):
        pid = f"P{idx + 1:03d}"
        base_price = max(round(cost0 * (1 + margin / 100) * 1.02), math.ceil(cost0 * 1.02))
        safety = math.ceil(1.65 * math.sqrt(lam * lead)) + 1
        reorder_level = max(3, math.ceil(lam * lead) + safety)
        order_qty = max(5, int(round(lam * 35)))
        dead_day = next((int(t.split(":")[1]) for t in tags if t.startswith("dead:")), None)
        stop_restock_day = 325 if "no_restock" in tags else (60 if (dead_day is not None or "overstock" in tags) else N_DAYS)
        opening = int(round(lam * (600 if "overstock" in tags else 45))) + int(rng.integers(0, 6))
        if dead_day is not None:
            # Over-bought relative to demand: stock is still on the shelf after sales stop.
            opening = int(math.ceil(lam * dead_day * 1.15)) + 45
        opening = max(opening, 8)

        stock = opening
        pending: list[tuple[int, int, float]] = []  # (arrival_day, qty, unit_cost)
        purchase_rows.append(
            {"purchase_date": START.isoformat(), "product_id": pid, "quantity_purchased": opening,
             "purchase_cost": round(opening * cost0, 2)}
        )
        last_sale_day: int | None = None
        cost_now = cost0
        price_now = base_price

        for t in range(N_DAYS):
            day = START + timedelta(days=t)
            # Cost inflation ~ +5%/yr; scenario cost shock for the loss-making oil.
            cost_now = cost0 * (1 + 0.00015 * t)
            if "neg_margin" in tags and t >= 270:
                cost_now *= 1.22
            price_now = base_price
            if t >= 180 and "neg_margin" not in tags:
                price_now = round(base_price * 1.03)

            # Receive stock.
            arrived = [o for o in pending if o[0] <= t]
            pending = [o for o in pending if o[0] > t]
            for _, qty, ucost in arrived:
                stock += qty
                purchase_rows.append(
                    {"purchase_date": day.isoformat(), "product_id": pid, "quantity_purchased": qty,
                     "purchase_cost": round(qty * ucost, 2)}
                )

            # Demand.
            mult = _MONTH_PROFILES[profile][day.month] * _WEEKDAY[day.weekday()] * (1 + 0.0004 * t)
            if "decline" in tags and t >= 300:
                mult *= 0.28
            if "decline" in tags and 240 <= t < 300:
                mult *= 0.7
            if dead_day is not None and t >= dead_day:
                mult = 0.0
            demand = int(rng.poisson(max(lam * mult, 0.0)))

            forced_q = forced.get((pid, t), 0)
            demand = min(demand, stock)
            stock -= demand
            if forced_q:
                # Stock is purchased specifically for the wholesale order, so it nets to zero.
                purchase_rows.append(
                    {"purchase_date": day.isoformat(), "product_id": pid, "quantity_purchased": forced_q,
                     "purchase_cost": round(forced_q * cost_now, 2)}
                )
            quantities = _split_quantity(rng, demand) if demand > 0 else []
            if forced_q:
                quantities.append(forced_q)
            for q in quantities:
                disc = 0.0
                r = rng.random()
                if r < 0.04:
                    disc = 5.0
                elif r < 0.07:
                    disc = 10.0
                if "discount_erosion" in tags and t >= 275 and rng.random() < 0.45:
                    disc = float(rng.choice([30.0, 35.0, 40.0]))
                if forced_q and q == forced_q:
                    disc = bulk_discount[(pid, t)]
                total = round(q * price_now * (1 - disc / 100), 2)
                sales_rows.append(
                    {"transaction_date": day.isoformat(), "product_id": pid, "product_name": name,
                     "quantity": int(q), "unit_price": float(price_now), "unit_cost": round(cost_now, 2),
                     "discount": disc, "total_amount": total,
                     "payment_method": str(rng.choice(_PAYMENTS, p=_PAYMENT_P)), "category": cat}
                )
            if demand > 0 or forced_q:
                last_sale_day = t

            # Reorder policy.
            on_order = sum(o[1] for o in pending)
            if t < stop_restock_day and (stock + on_order) <= reorder_level:
                arrival = t + lead + int(rng.integers(0, 2))
                if arrival < N_DAYS:
                    pending.append((arrival, order_qty, cost_now))

        inv_rows.append(
            {"product_id": pid, "product_name": name, "category": cat, "current_stock": int(stock),
             "reorder_level": int(reorder_level), "supplier_lead_time_days": int(lead),
             "unit_cost": round(cost_now, 2), "selling_price": float(price_now),
             "last_sale_date": (START + timedelta(days=last_sale_day)).isoformat() if last_sale_day is not None else ""}
        )

    sales = pd.DataFrame(sales_rows)
    inventory = pd.DataFrame(inv_rows)
    purchases = pd.DataFrame(purchase_rows)

    # Open (incoming) purchase orders after the snapshot date for a few at-risk products.
    incoming = []
    for pname, qty, d in [("Amul Butter 500g", 60, date(2026, 10, 3)), ("Surf Excel Easy Wash 1kg", 50, date(2026, 10, 5))]:
        row = inventory[inventory["product_name"] == pname].iloc[0]
        incoming.append({"purchase_date": d.isoformat(), "product_id": row["product_id"],
                         "quantity_purchased": qty, "purchase_cost": round(qty * float(row["unit_cost"]), 2)})
    purchases = pd.concat([purchases, pd.DataFrame(incoming)], ignore_index=True)

    sales = sales.sort_values(["transaction_date", "product_id"], kind="mergesort").reset_index(drop=True)
    sales.insert(0, "transaction_id", [f"TXN{i + 1:07d}" for i in range(len(sales))])
    sales, scen_list = _inject_anomalies(sales, rng, bulk_specs)
    sales = sales.sort_values(["transaction_date", "transaction_id"], kind="mergesort").reset_index(drop=True)
    return DemoData(
        sales=_finalise(sales), inventory=inventory,
        purchases=purchases.sort_values(["purchase_date", "product_id"]).reset_index(drop=True),
        scenarios=scen_list,
    )


def _finalise(sales: pd.DataFrame) -> pd.DataFrame:
    cols = ["transaction_id", "transaction_date", "product_id", "product_name", "quantity", "unit_price",
            "unit_cost", "discount", "total_amount", "payment_method", "category"]
    return sales[cols]


def _inject_anomalies(sales: pd.DataFrame, rng: np.random.Generator, bulk_specs) -> tuple[pd.DataFrame, list[dict]]:
    """Mutate a copy of `sales` with documented anomalies and return it with the scenario list."""
    sales = sales.copy().reset_index(drop=True)
    scenarios: list[dict] = []
    bulk_idx = set()
    for pname, d, q, _ in bulk_specs:
        m = sales[(sales["product_name"] == pname) & (sales["transaction_date"] == d.isoformat()) & (sales["quantity"] == q)]
        bulk_idx.update(m.index.tolist())
    pool = [i for i in sales.index
            if i not in bulk_idx and sales.at[i, "quantity"] >= 1 and sales.at[i, "category"] != "Gourmet & Health"]

    def pick(n: int) -> list[int]:
        chosen = rng.choice(pool, size=n, replace=False).tolist()
        for c in chosen:
            pool.remove(c)
        return chosen

    # 1) Excessive one-off discounts (70-90%).
    ids = pick(6)
    for i in ids:
        sales.at[i, "discount"] = float(rng.choice([70.0, 80.0, 90.0]))
        sales.at[i, "total_amount"] = round(sales.at[i, "quantity"] * sales.at[i, "unit_price"] * (1 - sales.at[i, "discount"] / 100), 2)
    scenarios.append({"title": "Excessive one-off discounts", "kind": "Business risk",
                      "detail": "6 transactions carry 70-90% discounts, far above the normal 0-10% range; totals are internally consistent.",
                      "rows": [(sales.at[i, "transaction_date"], sales.at[i, "product_name"]) for i in ids]})

    # 2) Quantity data-entry errors (x100).
    ids = pick(4)
    for i in ids:
        sales.at[i, "quantity"] = int(sales.at[i, "quantity"]) * 100
        sales.at[i, "total_amount"] = round(sales.at[i, "quantity"] * sales.at[i, "unit_price"] * (1 - sales.at[i, "discount"] / 100), 2)
    scenarios.append({"title": "Likely quantity data-entry errors", "kind": "Data quality / statistical outlier",
                      "detail": "4 transactions have quantities 100x the normal size (an extra '00' typed).",
                      "rows": [(sales.at[i, "transaction_date"], sales.at[i, "product_name"]) for i in ids]})

    # 3) Unexpected price changes (x10).
    ids = pick(3)
    for i in ids:
        sales.at[i, "unit_price"] = float(sales.at[i, "unit_price"]) * 10
        sales.at[i, "total_amount"] = round(sales.at[i, "quantity"] * sales.at[i, "unit_price"] * (1 - sales.at[i, "discount"] / 100), 2)
    scenarios.append({"title": "Unexpected price spikes", "kind": "Data quality / statistical outlier",
                      "detail": "3 transactions have a unit price 10x the usual selling price.",
                      "rows": [(sales.at[i, "transaction_date"], sales.at[i, "product_name"]) for i in ids]})

    # 4) Inconsistent totals.
    ids = pick(4)
    for i in ids:
        sales.at[i, "total_amount"] = round(sales.at[i, "total_amount"] * 0.6, 2)
    scenarios.append({"title": "Inconsistent totals", "kind": "Confirmed data-quality problem",
                      "detail": "4 transactions where total_amount does not equal quantity x unit_price x (1 - discount).",
                      "rows": [(sales.at[i, "transaction_date"], sales.at[i, "product_name"]) for i in ids]})

    # 5) Missing costs.
    ids = pick(15)
    for i in ids:
        sales.at[i, "unit_cost"] = np.nan
    scenarios.append({"title": "Missing unit costs", "kind": "Confirmed data-quality problem",
                      "detail": "15 transactions have a blank unit_cost; their margin is flagged as unknown, never treated as zero cost.",
                      "rows": [(sales.at[i, "transaction_date"], sales.at[i, "product_name"]) for i in ids]})

    # 6) Duplicates (same transaction content, same id).
    ids = rng.choice(pool, size=8, replace=False).tolist()
    dups = sales.loc[ids].copy()
    sales = pd.concat([sales, dups], ignore_index=True)
    scenarios.append({"title": "Duplicate transactions", "kind": "Confirmed data-quality problem",
                      "detail": "8 transactions appear twice with identical content and the same transaction_id.",
                      "rows": [(r["transaction_date"], r["product_name"]) for _, r in dups.iterrows()]})

    # 7) Refunds (negative quantity).
    base = sales.loc[rng.choice(pool, size=10, replace=False)].copy()
    base["quantity"] = -np.minimum(base["quantity"].astype(int), 2)
    base["total_amount"] = (base["quantity"] * base["unit_price"] * (1 - base["discount"] / 100)).round(2)
    base["transaction_id"] = [f"RET{i + 1:05d}" for i in range(len(base))]
    sales = pd.concat([sales, base], ignore_index=True)
    scenarios.append({"title": "Customer refunds / returns", "kind": "Convention demo",
                      "detail": "10 rows with negative quantity and negative total_amount; they reduce recognised revenue and COGS.",
                      "rows": [(r["transaction_date"], r["product_name"]) for _, r in base.iterrows()]})

    return sales, scenarios


def write_csvs(out_dir: Path, seed: int = SEED) -> DemoData:
    data = generate(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    data.sales.to_csv(out_dir / "demo_sales.csv", index=False)
    data.inventory.to_csv(out_dir / "demo_inventory.csv", index=False)
    data.purchases.to_csv(out_dir / "demo_purchases.csv", index=False)
    return data
