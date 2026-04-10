# 📊 Automated Excel Data Analysis Agent

An end-to-end intelligent agent that automatically ingests Excel files, infers schemas, performs data quality checks, builds a semantic model, runs statistical analysis, generates smart visualizations, and produces a stakeholder-friendly HTML report — all with minimal manual intervention.

---

## 🚀 Quick Start

```bash
# 1. Clone / open the project
cd "data analysis agent"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate sample data (optional)
python sample_data/generate_sample.py

# 4. Launch the Streamlit app
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser, upload any `.xlsx` / `.xls` file, and click **Run Analysis**.

---

## 📁 Project Structure

```
data analysis agent/
├── agent/
│   ├── __init__.py
│   ├── ingestion.py          # Excel loading, sheet detection, header inference
│   ├── schema_inference.py   # Column type detection, data dictionary
│   ├── quality_checks.py     # Data quality, outlier/anomaly detection
│   ├── semantic_layer.py     # Dimensions, measures, KPIs, YAML export
│   ├── analysis_engine.py    # Stats, correlations, time-series, insights
│   ├── visualization.py      # Smart Plotly chart generation
│   ├── report_generator.py   # Self-contained HTML report
│   ├── orchestrator.py       # Pipeline orchestration
│   ├── query_engine.py       # NL→SQL + quota-safe LLM routing
│   └── hc_attrition/         # Quota-Safe Query Orchestrator package
│       ├── __init__.py
│       ├── fy_helper.py      # April–March FY date utilities
│       ├── catalogue.py      # 25-question catalogue + intent classification
│       ├── sql_templates.py  # Deterministic SQL templates (all 25 questions)
│       ├── query_router.py   # Tier 1/2/3 routing with confidence scoring
│       ├── llm_orchestrator.py # Circuit breaker, retry/backoff, cache, rate limiter
│       └── observability.py  # Per-query traces + aggregate metrics
├── tests/
│   ├── test_ingestion.py
│   ├── test_schema_inference.py
│   ├── test_quality_checks.py
│   ├── test_semantic_layer.py
│   └── test_hc_attrition.py  # 83 tests for the quota-safe orchestrator
├── sample_data/
│   └── generate_sample.py    # Generates sample_sales.xlsx (1,000 rows)
├── catalogue.csv             # 25 HC/Attrition business questions
├── app.py                    # Streamlit UI
├── config.yaml               # Configurable thresholds + orchestrator settings
├── requirements.txt
└── README.md
```

---

## 🧩 Pipeline Steps

| # | Module | What it does |
|---|--------|-------------|
| 1 | `ingestion.py` | Loads `.xlsx`/`.xls`, detects best sheet, handles merged cells & title rows |
| 2 | `schema_inference.py` | Infers column types (numeric, categorical, datetime, boolean, identifier, text) |
| 3 | `quality_checks.py` | Checks nulls, duplicates, outliers (IQR), mixed types |
| 4 | `semantic_layer.py` | Classifies dimensions/measures/KPIs, exports YAML |
| 5 | `analysis_engine.py` | Summaries, correlations, time-series trends, segment analysis, insights |
| 6 | `visualization.py` | Auto-selects and generates Plotly charts |
| 7 | `report_generator.py` | Builds self-contained HTML report with all charts embedded |

---

## ⚙️ Configuration

Edit `config.yaml` to tune the pipeline:

```yaml
schema:
  cardinality_threshold: 50        # max unique values for categorical
  identifier_min_uniqueness: 0.9   # uniqueness ratio to treat as ID

quality:
  outlier_method: iqr
  outlier_iqr_factor: 1.5

analysis:
  top_n_categories: 15
  correlation_min_columns: 2
  time_series_min_points: 10

visualization:
  max_charts: 30
  max_scatter_points: 5000

reporting:
  max_insights: 20
```

---

## 🧪 Running Tests

```bash
cd "data analysis agent"
python -m pytest tests/ -v --tb=short
```

---

## 📋 Streamlit UI Tabs

| Tab | Contents |
|-----|----------|
| 📋 Data Quality | Per-column null %, outlier count, duplicate rows, quality score |
| 🗂️ Schema | Data dictionary with inferred types, samples, notes |
| 🧠 Semantic Layer | Dimensions, measures, time fields, KPIs, YAML export |
| 📈 Analysis & Insights | Prioritized business insights with confidence scores |
| 📊 Charts | All interactive Plotly charts |
| 📄 Download Report | Self-contained HTML report download + preview |

---

## 📦 Requirements

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `openpyxl` / `xlrd` | Excel file reading |
| `scipy` | Statistical calculations |
| `plotly` | Interactive visualizations |
| `streamlit` | Web UI |
| `PyYAML` | Config and semantic layer export |
| `jinja2` | HTML templating |
| `python-dateutil` | Robust date parsing |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🔒 Quota-Safe Architecture and Model Routing

### Overview

The Quota-Safe Query Orchestrator ensures HC/Attrition analytics queries work
reliably under Gemini rate limits (429 RESOURCE_EXHAUSTED, 503 transient errors)
without user-facing failures.

### Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   QueryRouter       │  classify_intent() + extract_filters()
│   (Tier 1/2/3)      │
└─────────┬───────────┘
          │
   ┌──────┴──────────────────────────┐
   │                                 │
   ▼ confidence ≥ 0.75               ▼ confidence < 0.75
┌──────────────────┐        ┌────────────────────┐
│ Tier 1           │        │ Tier 2/3 (LLM)     │
│ Deterministic    │        │ LLMOrchestrator    │
│ SQL Template     │        └────────┬───────────┘
│ (no LLM call)    │                 │
└──────────────────┘    ┌────────────┼────────────┐
          │             │            │             │
          │          Rate         Retry/       Circuit
          │         Limiter      Backoff       Breaker
          │             │            │             │
          │         ┌───┴────────────┴─────────────┴───┐
          │         │     Fallback Chain                │
          │         │  gemini-2.0-flash                 │
          │         │  → gemini-2.0-flash-lite          │
          │         │  → template-only                  │
          │         └───────────────────────────────────┘
          │
          ▼
    Response Cache  (keyed by normalized_query + table + filters)
          │
          ▼
    ObservabilityTracker  (QueryTrace + aggregate metrics)
```

### Tier Routing

| Tier | Trigger | Model | Behaviour |
|------|---------|-------|-----------|
| **1 — Deterministic** | confidence ≥ 0.75 for known catalogue intent | None | SQL template rendered; zero LLM calls |
| **2 — Lightweight LLM** | 0.40 ≤ confidence < 0.75 | `gemini-2.0-flash-lite` | Lightweight model with schema-trimmed prompt |
| **3 — Full LLM** | confidence < 0.40 or unknown intent | configured pro model | Full prompt; novel/complex queries |

### Quota-Safety Features

| Feature | Description |
|---------|-------------|
| **Deterministic templates** | 25 SQL templates for HC/Attrition questions; no LLM call for common patterns |
| **Response cache** | TTL-based cache (default 5 min) keyed by query+table+filters; eliminates duplicate LLM calls |
| **Rate limiter** | Sliding-window RPM guard per session and globally |
| **Retry + backoff** | 503 → exponential backoff with jitter; 429 → honors `retryDelay` from API error |
| **Circuit breaker** | Per-model (CLOSED→OPEN→HALF_OPEN); trips after N failures, auto-recovers after cooldown |
| **Fallback chain** | `gemini-2.0-flash → gemini-2.0-flash-lite → template-only` (configurable) |

### Configuration (`config.yaml`)

```yaml
orchestrator:
  deterministic_first: true
  max_llm_calls_per_query: 1
  default_model: "gemini-2.0-flash"
  fallback_chain:
    - "gemini-2.0-flash"
    - "gemini-2.0-flash-lite"
    - "template-only"
  confidence_threshold_deterministic: 0.75
  confidence_threshold_tier2: 0.40
  rpm_per_session: 10
  rpm_global: 60
  max_retries: 3
  base_backoff_seconds: 1.0
  circuit_breaker_failure_threshold: 3
  circuit_breaker_cooldown_seconds: 60
  cache_ttl_seconds: 300
```

### Example Traces

**Normal attrition query (Tier 1 — no LLM):**
```
QUERY: "show me attrition for June 2025"
→ intent=attrition_overall, confidence=0.92
→ TIER 1 DETERMINISTIC — SQL template rendered
→ cache_miss → result cached for 300s
→ latency=2ms, model=deterministic, retries=0
```

**429 failover path:**
```
QUERY: "what is the voluntary attrition trend?"
→ intent=attrition_vol_invol, confidence=0.78
→ TIER 1 DETERMINISTIC (confidence ≥ 0.75) — template used
→ No LLM call needed — 429 never reached
```

**Complex novel query (Tier 3 — LLM):**
```
QUERY: "correlation between satisfaction scores and attrition"
→ intent=None, confidence=0.0
→ TIER 3 FULL LLM — gemini-2.0-flash
→ success on attempt 1, latency=1240ms
```

### Financial Year Logic

All attrition YTD calculations use April–March FY:
- `fy_start(date(2025, 6, 15))` → `2025-04-01`
- `fy_end(date(2025, 6, 15))` → `2026-03-31`
- `fy_label(date(2025, 6, 15))` → `"FY2025-26"`

### Acceptance Criteria Status

| Criterion | Status |
|-----------|--------|
| "show me attrition for June 2025" works without Pro model | ✅ Tier 1 deterministic template |
| Repeated attrition queries don't cascade 429 | ✅ Cache + deterministic templates |
| 429/503 → fallback path, output still returned | ✅ Circuit breaker + fallback chain |
| Complex novel query still supported via LLM | ✅ Tier 3 routes to full model |
| Logs show routing/fallback decisions | ✅ ObservabilityTracker + QueryTrace |
| Existing headcount queries continue to work | ✅ Deterministic templates + backward-compatible |
