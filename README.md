# VyaparAI – The AI Business Copilot for Small Businesses

> **Turn Your Business Data Into Smarter Decisions.**

VyaparAI turns the sales and stock records of a small retailer into answers: what to reorder, which products lose money, where discounts leak profit, what demand will look like next week, and which records need a second look. It runs **entirely on your own computer, in Indian rupees (₹), with no paid service of any kind**: no OpenAI/Anthropic/Gemini key, no cloud database, no hosting fee.

* Charts, forecasts, anomaly detection and recommendations are plain Python and scikit-learn. They work with **no language model at all**.
* The conversational AI analyst uses a **free open-weight model running locally** (Ollama, or a Hugging Face model). If none is installed it falls back to clearly labelled template answers built from the same calculated numbers.

> **All demo data is synthetic** (a fictional Indian kirana store). No result shown with the demo data is a real business result.

## The business problem

Small shops record sales and stock but rarely see the whole picture: a best-seller quietly stocks out, a product sells below cost after a supplier price rise, heavy discounts erase margin, cash sits in goods that do not sell, and a typo in a bill goes unnoticed. VyaparAI calculates these things deterministically and explains them in plain language.

## Features

| Section | What it does |
|---|---|
| **Overview** | Revenue, cost of goods sold, **gross** profit and margin, units, low-stock and slow-moving counts, stock value, anomaly count; trends, top products, categories, alerts, ranked recommendations; date and category filters |
| **Sales Analytics** | Daily/weekly/monthly trends, product and category rankings, quantity-vs-revenue, discount analysis, period-over-period comparison, declining products, CSV export |
| **Inventory Intelligence** | Reorder point and safety stock, suggested order quantity (accounts for stock on order), stock-out risk, slow / dead / excess stock, promotion candidates, stock reconciliation, editable thresholds |
| **Demand Forecasting** | 7 or 30-day daily forecasts: recent-average and weekly-seasonal baselines vs a Random Forest, time-based validation, MAE/WAPE, honest model selection |
| **Profit Leakage Detection** | Data-quality problems, statistical anomalies and business risks kept separate, with severity, expected value, explanation and next step; Isolation Forest as an extra signal |
| **AI Business Analyst** | Ask questions in plain language. Numbers come from validated read-only Python tools; the model only words the result |
| **Data Management** | Upload CSVs, map columns, validation report, rejected-row download, demo reset, sample files |
| **Settings and Model Status** | Data source, data quality, Ollama/Hugging Face status, forecast models used, degraded modes, thresholds, backup |

## Architecture

```
CSV upload / demo data
        |
 src/data   validator -> loader -> repository (SQLite, parameterised queries)
        |
 src/analytics  profitability (single source of truth for revenue/COGS/gross profit)
                sales, inventory, recommendations, pipeline
 src/ml         forecasting, evaluation, anomaly_detection
        |
 src/genai      business_tools (10 validated read-only tools)
                analyst (route -> run tools -> LLM words FACTS, or template fallback)
                ollama_client (Ollama + optional Hugging Face)
        |
 src/ui + pages/   Streamlit pages (cached; models are not retrained on every rerun)
```

UI, data access, business logic, ML and GenAI are separate layers. Every page uses the same functions, so a number is calculated in exactly one place.

## Technology stack (all free and open source)

Python 3.11+ · Streamlit · Pandas / NumPy · Plotly · scikit-learn (Random Forest, Isolation Forest) · SQLite · pytest · Ollama *or* Hugging Face `transformers` (optional).

## Prerequisites

* Python 3.11 or newer
* About 1 GB free disk for the app and dependencies, plus **2–5 GB if you add a local language model** (optional)
* 8 GB RAM recommended for the 3B model; smaller models work with less

## Installation

**Windows (PowerShell)**
```powershell
cd vyaparai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS**
```bash
cd vyaparai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optionally copy `.env.example` to `.env` to change settings. No keys are needed.

## Run the app

```bash
streamlit run app.py
```
Open http://localhost:8501. The synthetic demo data loads automatically on first start (once; restarts do not duplicate it). The sidebar always shows whether you are looking at **Demo data** or **Your uploaded data**.

## Free local language model (optional)

The app works without this. Add it for conversational answers. The model is **not bundled**: you download it separately (a few GB).

### Option A: Ollama (recommended)
1. Install Ollama (free) from https://ollama.com.
2. Download a model and make sure the server is running:
   ```bash
   ollama pull llama3.2:3b      # about 2 GB; try llama3.2:1b or qwen2.5:1.5b on low-RAM machines
   ollama serve                 # skip if the Ollama desktop app is already running
   ```
3. Reload the **AI Business Analyst** page. The badge turns green and shows the model name. To use another model, set `OLLAMA_MODEL` in `.env`.

### Option B: Hugging Face model, run locally (no Ollama)
```bash
pip install -r requirements-hf.txt
```
Set `VYAPARAI_LLM_BACKEND=huggingface` (optional) and `HF_MODEL` in `.env`. The default `Qwen/Qwen2.5-0.5B-Instruct` is a small free model downloaded once from the Hugging Face Hub (public models need no account or token); it is fast on CPU but less fluent than a 3B model.

### How the analyst stays honest
1. The question is routed to a small, fixed set of **read-only tools** (the model cannot run SQL, Python or shell commands, change stock or place orders). Tool arguments are validated; unknown tools or arguments are rejected.
2. The tools return calculated facts with the period covered and a label: observed, estimate or recommendation.
3. The model only phrases those facts. A **grounding check** flags any number in its answer that is not in the facts, and the exact data used is shown under every answer.
4. If the model is offline, missing or times out, a **Fallback mode** answer is generated from templates and labelled as such.

## Load demo data / upload your own

* **Demo data:** loaded automatically. **Data Management → Reset to demo data** reloads it. The generator is `python scripts/generate_demo_data.py` (fixed seed). See `docs/DEMO_SCENARIOS.md` for the planted problems and `docs/DATA_DICTIONARY.md` for columns.
* **Your data:** **Data Management → Upload CSV files**. Provide a sales file; inventory and purchases files are optional but unlock inventory features. Headers are matched automatically (`Qty`, `Rate`, `SKU`, `Bill Date`...), and you can fix the mapping. After validation your data becomes the active dataset; the demo data is kept separately and untouched.
* Problem rows are **never silently discarded**: unreadable rows are listed with a reason (and downloadable), suspicious rows are kept and flagged, and the report explains how each issue affects the analysis. Discounts are percentages by default (a rupee-amount option exists). Dates like `05/01/2026` are read **day-first**.

## Run the tests

```bash
pip install -r requirements.txt
python -m pytest
```
The suite (139 tests) covers validation, revenue/gross-profit/discount maths, reorder and stock-out logic, forecast features and time-based validation, leakage guards, anomaly detection, recommendation ranking and de-duplication, tool-argument validation, and behaviour when Ollama is unavailable (mocked HTTP). Tests do not need Ollama or internet access.

## Key definitions

* **Gross profit** = revenue − cost of goods sold. It is **not net profit**: rent, salaries and other overheads are not in the data.
* **Reorder point** = demand during supplier lead time + safety stock; **safety stock** = z × std(daily demand) × √lead time; **suggested order** = order-up-to level − stock − stock on order, never below zero.
* **Unknown costs** are never treated as zero: they are estimated from inventory cost (flagged) or excluded from margin figures.
* **Exposure figures** in recommendations are labelled estimates with their calculation basis, with overlapping items counted once. They are not promised savings.

## Database

SQLite file `data/vyaparai.db` (created on first run). Tables: `sales`, `inventory`, `purchases`, `rejected_rows`, `dataset_meta`, `app_config`, `analysis_runs`, `forecast_meta`. Demo and uploaded rows live in the same tables, separated by a `dataset` column.
* **Initialise:** automatic. **Reset:** Settings → *Erase ALL stored data* (asks for confirmation), or delete the file. **Back up:** Settings → *Create backup* (writes to `data/backups/`) or copy the file while the app is stopped.

## Troubleshooting

| Symptom | Fix |
|---|---|
| AI Analyst says *Fallback mode* | Ollama is not running or the model is not pulled. Run `ollama serve` and `ollama pull llama3.2:3b`; check Settings → Model status |
| Model is slow or times out | Use a smaller model (`llama3.2:1b`), or raise `VYAPARAI_LLM_TIMEOUT` |
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements.txt` |
| Everything looks stale after replacing the DB file | Restart the app (analyses are cached) |
| Forecast says "not enough history" | Needs 28+ days of sales (about 12 weeks for ML validation) and regular sales |
| Upload rows rejected | Open the validation report; download the rejected rows, fix them and upload again |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |

## Model limitations

* Forecasts are **estimates, not guarantees**. With about a year of data the model cannot learn year-on-year seasonality; festivals, promotions and price changes are not modelled. Days when an item was out of stock look like zero demand. Accuracy is measured on a backtest of recent weeks and can be worse in new conditions. ML is used only where it beats a simple baseline.
* Anomalies are **statistical flags that need review**, not evidence of fraud. Many are legitimate (bulk orders, price changes).
* Inventory recommendations rest on recent sales; they do not know about supplier minimums, shelf life, cash limits or upcoming events.
* Small local language models can phrase things imperfectly. That is why numbers come from code, the facts are shown, and ungrounded numbers are flagged.
* Reorder suggestions are suggestions only. VyaparAI never changes inventory or buys anything.

## Privacy

All data stays on your computer in a local SQLite file. Nothing is sent to any cloud service; Ollama and Hugging Face models run locally (the Hugging Face model *download* contacts huggingface.co once). The app has **no login**: it is built for a single user on a trusted machine. Do not expose it on a public network without adding proper authentication and authorisation. Logs never contain data rows or prompts.

## Hackathon demonstration

See `docs/HACKATHON_PITCH.md` for the pitch, a 3-minute demo script and judge Q&A.
