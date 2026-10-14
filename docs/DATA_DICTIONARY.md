# Data dictionary

All money is in **Indian rupees (₹)**. Dates are `YYYY-MM-DD` (day-first formats such as `05/01/2026` are read as 5 January 2026).
Column names are matched automatically (for example `Qty` → `quantity`, `Rate` → `unit_price`) and can be remapped in **Data Management**.

## Sales transactions (`demo_sales.csv`) – required

| Column | Required | Type | Meaning |
|---|---|---|---|
| `transaction_id` | no | text | Bill / invoice line id. Used for duplicate detection. Generated if absent. |
| `transaction_date` | **yes** | date | Date of sale. Rows with an unreadable date are set aside and reported. |
| `product_id` | **yes** | text | SKU / item code. |
| `product_name` | no | text | Display name (falls back to `product_id`). |
| `quantity` | **yes** | number | Units sold. **Negative = customer return/refund.** Zero or non-numeric rows are set aside. |
| `unit_price` | **yes** | ₹ | List selling price per unit before discount. |
| `unit_cost` | no | ₹ | Cost per unit. Blank = unknown (never treated as zero). |
| `discount` | no | % | Discount as a percentage (0–100) of quantity × unit_price. Use the "rupee amounts" option on upload if your file stores amounts. |
| `total_amount` | no | ₹ | Recorded net amount of the line (after discount). If blank it is computed and flagged. |
| `payment_method` | no | text | UPI, Cash, Card, Credit... |
| `category` | no | text | Product category (`Uncategorised` if absent). |

## Inventory (`demo_inventory.csv`)

| Column | Required | Meaning |
|---|---|---|
| `product_id` | **yes** | SKU, same codes as the sales file. |
| `current_stock` | **yes** | Units on hand at the latest sales date. Negative values are kept and flagged. |
| `product_name`, `category` | no | Fall back to the sales file. |
| `reorder_level` | no | Your own reorder point (units). VyaparAI also calculates a data-driven one. |
| `supplier_lead_time_days` | no | Days from ordering to receiving. 7 days is assumed (and flagged) if absent. |
| `unit_cost`, `selling_price` | no | Current cost and price (₹). Estimated from sales if absent (flagged). |
| `last_sale_date` | no | Derived from sales if absent. |

## Purchases / restocking (`demo_purchases.csv`) – optional

| Column | Required | Meaning |
|---|---|---|
| `purchase_date` | **yes** | Date goods were (or will be) received. Lines dated **after** the latest sales date are treated as **stock on order**. |
| `product_id` | **yes** | SKU. |
| `quantity_purchased` | **yes** | Units received, must be positive. |
| `purchase_cost` | no | **Total** cost of the line (₹), not per unit. |

## Financial conventions

* **Revenue** = recorded `total_amount` of each non-duplicate line (net of discount, returns negative).
* **Cost of goods sold** = quantity × unit cost, for lines whose cost is known (recorded, or estimated from the inventory cost and labelled as such).
* **Gross profit** = revenue − COGS over cost-known lines. **It is not net profit**: rent, salaries, electricity, taxes and other overheads are not in the data.
* **Gross margin %** = gross profit ÷ revenue of cost-known lines × 100 (blank when revenue is zero).
* **Exact duplicate lines** (same id, date, product, quantity, price, discount and amount) are flagged and excluded from metrics.
* **Unknown costs** are never converted to zero. They are estimated from inventory cost (flagged) or excluded from margin figures, and the share of revenue with a known cost is shown.
