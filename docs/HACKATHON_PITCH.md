# VyaparAI: hackathon pitch

## Elevator pitch
VyaparAI is a free, local-first business copilot for small Indian retailers. Upload your sales and stock CSVs and it shows, in rupees, what to reorder, which products lose money, where discounts leak margin and which records look wrong. Every number comes from transparent Python calculations and ML forecasts. A free open-weight language model only explains them, and it works even when no model is installed.

## Problem
A kirana owner or small wholesaler has sales and stock data but no analyst. The result: best-sellers run out unnoticed, a supplier price rise turns a product loss-making, heavy discounting erodes thin margins, cash sits in slow stock, and billing typos go unseen. Existing BI tools are expensive, cloud-based or built for analysts, and "chat with your data" tools often make up numbers.

## Target customers
Small retailers, grocery and general stores, small wholesalers and distributors who keep sales/stock in a billing app or spreadsheet that can export CSV, and who cannot pay for SaaS or send data to a cloud.

## Solution
1. **Ingest and validate** CSVs with column mapping, a validation report and no silent row loss.
2. **Calculate** revenue, COGS, *gross* profit, margins, stock cover, reorder points (one source of truth).
3. **Predict** 7/30-day demand with baselines versus a Random Forest, picked on time-based validation.
4. **Detect** data-quality problems, statistical anomalies and business risks, kept separate.
5. **Recommend** ranked actions with evidence, assumptions and an exposure estimate (no double counting).
6. **Explain** through an AI analyst that is grounded in those results.

## Architecture
CSV → validator/loader → SQLite → analytics (profitability, inventory) + ML (forecasting, anomalies) → recommendations → read-only tool layer → analyst (local LLM or template fallback) → Streamlit UI. Layers are separate; all pages share the same functions.

## AI / ML innovation
* **Honest forecasting:** direct multi-horizon Random Forest with features known only at the forecast origin (no leakage), rolling-origin backtest on the most recent weeks, WAPE instead of MAPE (zero-demand days), and the ML model is used **only if it beats the baseline** by a margin. On the demo data it wins for some products and loses for others, and the app shows which.
* **Layered anomaly detection:** transparent rules first (duplicates, inconsistent totals, discount, price and quantity outliers, margin and sales shifts), Isolation Forest as a low-severity extra. Findings are grouped as *data quality* / *statistical anomaly* / *business risk* and worded as "requires review".
* **Inventory maths a shop owner can follow:** reorder point = lead-time demand + safety stock; order-up-to level; stock on order counted; never a negative order; low-data fallbacks labelled with a confidence level.
* **Profit-change decomposition:** a gross-profit change is split into a volume effect and a per-unit margin effect, with the biggest contributors.

## GenAI implementation
* **Free and local:** Ollama (e.g. `llama3.2:3b`) or a Hugging Face open model run on the user's machine. No API key, account or payment.
* **Tool-grounded:** the question is routed to one of 10 fixed read-only tools with validated arguments. No model-written SQL, Python or shell is ever executed.
* **Facts in, words out:** the model receives calculated JSON and must copy numbers; a grounding check flags any number it cannot trace, and the data used is shown with every answer.
* **Graceful degradation:** if the model is offline, slow or missing, a labelled template answer built from the same facts is shown.

## Why this is more than a chatbot
The chatbot is the last 10%. The value is the deterministic engine underneath: validated data, one definition of gross profit, tested inventory maths, leakage-free forecasts and ranked recommendations. Remove the language model and the product still works; that is the design.

## Business impact and how to measure it (not yet measured)
This is a prototype on synthetic data. **No customer savings, adoption or accuracy in real shops has been measured.** To measure impact in a pilot: (1) stock-out days and lost-sales estimate before/after, (2) share of revenue sold below target margin, (3) value of stock with no sale for 120+ days, (4) number of billing errors caught and corrected, (5) forecast WAPE on live data versus the baseline, (6) time spent on weekly purchasing decisions.

## Technical differentiation
Fully free and offline-capable; rupee-native formatting (lakh/crore); explicit conventions for returns, duplicates and unknown costs; time-based validation with baselines; grounded and auditable AI; 139 automated tests including mutation checks of the leakage and cost-handling rules.

## Limitations and roadmap
**Limitations:** one year of synthetic data cannot show year-on-year seasonality; stock-outs look like zero demand; no GST/tax, supplier minimums, expiry or overheads; no authentication (single user, local); small local models word answers imperfectly; the Hugging Face path and a real Ollama model were not exercised in the build environment (Ollama/HF calls are tested with mocks).
**Roadmap:** stockout-aware demand estimation, GST-aware margins, overheads for net profit, billing-app connectors (Tally, Vyapar, Zoho Books exports), Hindi and regional-language questions, WhatsApp-style daily summary, a pilot with real shops, and proper multi-user auth before any hosted deployment.

## Three-minute live demo script
**0:00 – Frame (20 s).** "A small shop's data, a free laptop, no cloud." Open the app: sidebar badge says *Demo data (synthetic)*; the Fallback/AI badge shows the model status.
**0:20 – Overview (35 s).** Revenue ₹1.30 Cr, gross profit ₹12.7 L (9.7% margin). Point at the caption: *gross, not net*. Show the weekly trend and the top alerts.
**0:55 – Inventory (40 s).** 8 products stocked out. Open **Aashirvaad Atta 5kg**: demand 5/day, 3-day lead time, suggested order about 177 units, with the formula under *How it works*. "It suggests; it never orders." Show **Quinoa**: dead stock, "do not reorder".
**1:35 – Profit leakage (40 s).** Fortune Sunflower Oil is loss-making (−₹19.7 K) after a supplier cost rise; Haldiram Bhujia and Dettol show discount erosion. Filter to *Data quality*: 8 duplicates and 4 totals that do not add up. Stress "requires review, not fraud".
**2:15 – Forecast (25 s).** Pick Amul Gold Milk, 30 days. Show baselines vs Random Forest and the *Selected* verdict: "ML is only used when it earns it."
**2:40 – AI analyst (20 s).** Ask "Why did gross profit decrease?" Show the answer, then *Calculated data used*. If Ollama is running, say the model only phrased those numbers; if not, point at the *Fallback mode* label.

## Likely judges' questions
**Is the forecast accurate?** On a rolling backtest of the last weeks of synthetic data, typical WAPE is about 20–30% for steady products and above 100% for very sparse ones, and the app says so. ML beats the baseline for some products only. We make no accuracy claim for real shops yet.
**How do you stop the LLM hallucinating numbers?** It never calculates. It sees only tool output, must copy figures, and a checker flags any number not in the facts. The facts are displayed beside every answer. Small models can still word things badly, which is why the fallback and the facts panel exist.
**Can the model run SQL or code?** No. Only 10 named read-only tools with schema-validated arguments; unknown tools, extra arguments, wrong types and out-of-range values are rejected (tested). The planner's proposed calls are validated the same way.
**Why not just call GPT?** Cost, privacy and reliability for a shop with thin margins and patchy internet. The core product needs no LLM, and the optional one is free and local.
**How do you handle bad data?** Mapping, validation report, rejected rows listed with reasons, duplicates and mismatched totals flagged, unknown costs never treated as zero, returns netted. A messy Indian-style export is covered by an end-to-end test.
**Are the savings real?** Exposure figures are labelled estimates with stated formulas and overlapping items counted once. We do not claim savings; section "Business impact" lists how a pilot would measure them.
**Is the data real?** No: synthetic, reproducible (fixed seed), with planted scenarios documented in `docs/DEMO_SCENARIOS.md`. Planted problems are found, which shows detection works; it is not evidence about real shops.
**Security?** Local single-user app, parameterised SQL, no secrets, no network calls except to a local model. It has no login, so it must not be exposed publicly without authentication.
**What would you do next?** A pilot with 3–5 shops, stock-out-aware demand estimation, GST/overheads, billing-app connectors, and Hindi support.
