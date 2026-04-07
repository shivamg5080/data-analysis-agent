"""
Analysis Engine Module
======================
Performs automated statistical analysis using the semantic layer.
Generates summaries, correlations, time-series trends, segment analysis,
anomaly detection, and stakeholder-oriented insights.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_analysis(
    df: pd.DataFrame,
    column_types: dict[str, str],
    semantic_layer: dict[str, Any],
    analysis_columns: list[str],
    config: dict | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Run the full automated analysis pipeline.

    Returns
    -------
    dict with keys:
        - ``dataset_summary``     : dict
        - ``column_analyses``     : list[dict]
        - ``correlations``        : dict | None
        - ``time_series``         : list[dict] | None
        - ``segment_analyses``    : list[dict]
        - ``insights``            : list[dict]
        - ``confidence_score``    : float
        - ``narrative_summary``   : str | None
    """
    cfg = (config or {}).get("analysis", {})
    top_n: int = cfg.get("top_n_categories", 15)
    min_corr_cols: int = cfg.get("correlation_min_columns", 2)
    ts_min_pts: int = cfg.get("time_series_min_points", 10)

    measures = semantic_layer.get("measures", [])
    dimensions = semantic_layer.get("dimensions", [])
    time_fields = semantic_layer.get("time_fields", [])

    measure_cols = [m["raw_column"] for m in measures if m["raw_column"] in df.columns]
    dimension_cols = [d["raw_column"] for d in dimensions if d["raw_column"] in df.columns]
    time_cols = [t["raw_column"] for t in time_fields if t["raw_column"] in df.columns]

    logger.info("Running analysis on %d analysis columns", len(analysis_columns))

    # ---- 1. Dataset summary ------------------------------------------------
    dataset_summary = _dataset_summary(df, column_types, analysis_columns)

    # ---- 2. Column-by-column analysis --------------------------------------
    column_analyses = _analyse_all_columns(df, analysis_columns, column_types, top_n)

    # ---- 3. Correlations ---------------------------------------------------
    correlations = None
    numeric_cols = [c for c in measure_cols if c in df.columns]
    if len(numeric_cols) >= min_corr_cols:
        correlations = _compute_correlations(df, numeric_cols)

    # ---- 4. Time-series analysis ------------------------------------------
    time_series: list[dict] = []
    for tc in time_cols:
        ts_result = _analyse_time_series(df, tc, measure_cols, ts_min_pts)
        if ts_result:
            time_series.append(ts_result)

    # ---- 5. Segment analysis ----------------------------------------------
    segment_analyses: list[dict] = []
    for dim_col in dimension_cols[:4]:  # limit to 4 dimensions
        for meas_col in measure_cols[:3]:  # top 3 measures
            seg = _segment_analysis(df, dim_col, meas_col, top_n)
            if seg:
                segment_analyses.append(seg)

    # ---- 6. Insights -------------------------------------------------------
    insights = _generate_insights(
        df, dataset_summary, column_analyses, correlations, time_series, segment_analyses,
        column_types, measure_cols, dimension_cols, time_cols, config
    )

    # ---- 7. Narrative Summary (AI) -----------------------------------------
    narrative_summary = None
    if api_key:
        narrative_summary = _generate_narrative_summary(api_key, insights, dataset_summary)

    # ---- 8. Confidence score -----------------------------------------------
    confidence = _compute_confidence(df, column_analyses, insights)

    logger.info("Analysis complete. %d insights generated.", len(insights))

    return {
        "dataset_summary": dataset_summary,
        "column_analyses": column_analyses,
        "correlations": correlations,
        "time_series": time_series,
        "segment_analyses": segment_analyses,
        "insights": insights,
        "confidence_score": confidence,
        "narrative_summary": narrative_summary,
    }


def _generate_narrative_summary(api_key: str, insights: list[dict], summary: dict) -> str | None:
    """Uses Gemini to create a cohesive narrative for non-tech stakeholders."""
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        insights_text = "\n".join([f"- {i['title']}: {i['detail']}" for i in insights[:12]])
        
        prompt = f"""You are a senior data consultant. Write a concise, professional Executive Summary 
for a non-technical stakeholder based on the following data insights.

Context: Dataset has {summary['rows']:,} rows and {summary['columns']} columns.
Key Findings:
{insights_text}

Rules:
1. Write 2-3 short, impactful paragraphs.
2. Focus on the 'So What?' — what do these numbers mean for the business?
3. Use a friendly, professional tone.
4. Do NOT use markdown headings (#). Use bold text for emphasis.
5. End with a one-sentence 'Next Steps' recommendation.
"""
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return response.text.strip()
    except Exception as e:
        logger.warning("Failed to generate narrative summary: %s", e)
        return None


# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------

def _dataset_summary(
    df: pd.DataFrame, column_types: dict[str, str], analysis_columns: list[str]
) -> dict:
    """Build a top-level dataset summary."""
    type_counts: dict[str, int] = {}
    for col in analysis_columns:
        t = column_types.get(col, "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "analysis_columns": len(analysis_columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "column_type_counts": type_counts,
        "complete_rows": int((~df.isnull().any(axis=1)).sum()),
        "complete_row_pct": round((~df.isnull().any(axis=1)).mean() * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
    }


# ---------------------------------------------------------------------------
# Column analysis
# ---------------------------------------------------------------------------

def _analyse_all_columns(
    df: pd.DataFrame,
    analysis_columns: list[str],
    column_types: dict[str, str],
    top_n: int,
) -> list[dict]:
    results = []
    for col in analysis_columns:
        col_type = column_types.get(col, "unknown")
        series = df[col]
        non_null = series.dropna()

        result: dict[str, Any] = {
            "column": col,
            "type": col_type,
            "null_count": int(series.isna().sum()),
            "null_pct": round(series.isna().mean() * 100, 2),
        }

        if col_type == "numeric":
            result["stats"] = _safe_numeric_stats(non_null)

        elif col_type in ("categorical", "boolean"):
            vc = non_null.astype(str).value_counts().head(top_n)
            result["value_counts"] = {k: int(v) for k, v in vc.items()}
            result["top_value"] = vc.index[0] if len(vc) else None
            result["top_value_pct"] = round(vc.iloc[0] / len(non_null) * 100, 2) if len(vc) else 0

        elif col_type == "datetime":
            parsed = pd.to_datetime(non_null, errors="coerce").dropna()
            if len(parsed):
                result["date_min"] = str(parsed.min().date())
                result["date_max"] = str(parsed.max().date())
                result["date_range_days"] = (parsed.max() - parsed.min()).days

        results.append(result)
    return results


def _safe_numeric_stats(series: pd.Series) -> dict:
    """Compute numeric stats safely."""
    try:
        numeric = pd.to_numeric(
            series.astype(str).str.replace(",", "").str.strip(), errors="coerce"
        ).dropna()
        if len(numeric) == 0:
            return {}
        return {
            "count": len(numeric),
            "mean": round(float(numeric.mean()), 4),
            "median": round(float(numeric.median()), 4),
            "std": round(float(numeric.std()), 4),
            "min": round(float(numeric.min()), 4),
            "max": round(float(numeric.max()), 4),
            "q1": round(float(numeric.quantile(0.25)), 4),
            "q3": round(float(numeric.quantile(0.75)), 4),
            "sum": round(float(numeric.sum()), 4),
            "skewness": round(float(scipy_stats.skew(numeric)), 4),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

def _compute_correlations(df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    """Compute Pearson correlation matrix for numeric columns."""
    try:
        subset = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        corr = subset.corr(method="pearson")
        # Find strong correlations
        strong: list[dict] = []
        for i, c1 in enumerate(corr.columns):
            for j, c2 in enumerate(corr.columns):
                if i >= j:
                    continue
                val = corr.loc[c1, c2]
                if abs(val) >= 0.7:
                    strong.append({
                        "col1": c1,
                        "col2": c2,
                        "correlation": round(float(val), 4),
                        "strength": "strong positive" if val >= 0.7 else "strong negative",
                    })
        return {
            "matrix": corr.round(4).to_dict(),
            "columns": numeric_cols,
            "strong_correlations": strong,
        }
    except Exception as e:
        logger.warning("Correlation computation failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Time-series analysis
# ---------------------------------------------------------------------------

def _analyse_time_series(
    df: pd.DataFrame,
    time_col: str,
    measure_cols: list[str],
    min_points: int,
) -> dict | None:
    """Analyse trends for each measure over time."""
    try:
        temp = df[[time_col] + [c for c in measure_cols if c in df.columns]].copy()
        temp[time_col] = pd.to_datetime(temp[time_col], errors="coerce")
        temp = temp.dropna(subset=[time_col])
        if len(temp) < min_points:
            return None

        temp["_month"] = temp[time_col].dt.to_period("M")
        result: dict[str, Any] = {"time_column": time_col, "trends": []}

        for m_col in measure_cols:
            if m_col not in df.columns:
                continue
            try:
                numeric_col = pd.to_numeric(temp[m_col], errors="coerce")
                monthly = numeric_col.groupby(temp["_month"]).sum().reset_index()
                monthly.columns = ["period", "value"]
                monthly["period"] = monthly["period"].astype(str)

                # Linear trend
                if len(monthly) >= 3:
                    x = np.arange(len(monthly))
                    slope, _, r, _, _ = scipy_stats.linregress(x, monthly["value"])
                    trend_dir = "upward" if slope > 0 else "downward"
                else:
                    slope, r, trend_dir = 0.0, 0.0, "stable"

                result["trends"].append({
                    "measure": m_col,
                    "monthly_data": monthly.to_dict(orient="records"),
                    "trend_direction": trend_dir,
                    "slope": round(float(slope), 4),
                    "r_squared": round(float(r ** 2), 4),
                })
            except Exception:
                continue

        return result if result["trends"] else None
    except Exception as e:
        logger.warning("Time-series analysis failed for '%s': %s", time_col, e)
        return None


# ---------------------------------------------------------------------------
# Segment analysis
# ---------------------------------------------------------------------------

def _segment_analysis(
    df: pd.DataFrame, dim_col: str, meas_col: str, top_n: int
) -> dict | None:
    """Aggregate a measure by a dimension and return top-N segments."""
    try:
        numeric = pd.to_numeric(df[meas_col], errors="coerce")
        grp = numeric.groupby(df[dim_col].astype(str))
        agg = grp.agg(["sum", "mean", "count"]).round(4)
        agg = agg.sort_values("sum", ascending=False).head(top_n)
        return {
            "dimension": dim_col,
            "measure": meas_col,
            "segments": agg.reset_index().rename(
                columns={dim_col: "segment", "sum": "total", "mean": "avg", "count": "count"}
            ).to_dict(orient="records"),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Insight generation
# ---------------------------------------------------------------------------

def _generate_insights(
    df: pd.DataFrame,
    dataset_summary: dict,
    column_analyses: list[dict],
    correlations: dict | None,
    time_series: list[dict],
    segment_analyses: list[dict],
    column_types: dict,
    measure_cols: list[str],
    dimension_cols: list[str],
    time_cols: list[str],
    config: dict | None,
) -> list[dict]:
    """Generate prioritised business insights."""
    cfg = (config or {}).get("reporting", {})
    max_insights: int = cfg.get("max_insights", 20)
    insights: list[dict] = []

    def add(priority: int, category: str, title: str, detail: str, confidence: float):
        insights.append({
            "priority": priority,
            "category": category,
            "title": title,
            "detail": detail,
            "confidence": round(confidence, 2),
        })

    # Dataset size insight
    rows = dataset_summary["rows"]
    add(1, "Dataset Overview",
        f"Dataset contains {rows:,} records across {dataset_summary['columns']} columns",
        f"{dataset_summary['complete_row_pct']}% of rows are complete (no missing values). "
        f"{dataset_summary.get('duplicate_rows', 0):,} duplicate rows detected.",
        0.99)

    # Null insights
    high_null_cols = [
        a for a in column_analyses if a["null_pct"] > 30
    ]
    if high_null_cols:
        cols_str = ", ".join(f"'{a['column']}' ({a['null_pct']:.0f}%)" for a in high_null_cols[:5])
        add(2, "Data Quality",
            f"{len(high_null_cols)} column(s) have high missing value rates",
            f"Columns with >30% nulls: {cols_str}. These columns may need imputation or exclusion.",
            0.95)

    # Numeric range insights
    for ca in column_analyses:
        if ca["type"] == "numeric" and ca.get("stats"):
            s = ca["stats"]
            mean_val = s.get("mean", 0)
            max_val = s.get("max", 0)
            if mean_val and max_val and max_val > mean_val * 5:
                add(4, "Distribution",
                    f"'{ca['column']}' has extreme outliers (max={max_val:,.2f}, mean={mean_val:,.2f})",
                    f"The maximum value is {max_val / mean_val:.1f}× the average — investigate potential erroneous entries.",
                    0.85)

    # Correlation insights
    if correlations and correlations.get("strong_correlations"):
        for sc in correlations["strong_correlations"][:3]:
            add(3, "Correlation",
                f"Strong {sc['strength']} between '{sc['col1']}' and '{sc['col2']}' (r={sc['correlation']})",
                f"These two measures tend to move together. Consider whether this is causal or coincidental.",
                0.90)

    # Time-series insights
    for ts in time_series:
        for trend in ts.get("trends", [])[:2]:
            if trend["r_squared"] > 0.5:
                add(3, "Trend",
                    f"'{trend['measure']}' shows a {trend['trend_direction']} trend over time (R²={trend['r_squared']:.2f})",
                    f"The linear trend explains {trend['r_squared'] * 100:.0f}% of variance in {trend['measure']}.",
                    0.85)

    # Segment insights
    for seg in segment_analyses[:3]:
        segs = seg.get("segments", [])
        if segs:
            top = segs[0]
            total_sum = sum(s.get("total", 0) for s in segs)
            if total_sum:
                top_pct = top.get("total", 0) / total_sum * 100
                add(3, "Segmentation",
                    f"'{top.get('segment')}' dominates '{seg['measure']}' by {seg['dimension']} ({top_pct:.1f}%)",
                    f"The top segment accounts for {top_pct:.1f}% of total {seg['measure']}. "
                    f"Consider focusing strategy on this segment.",
                    0.88)

    # Dominant category insight
    for ca in column_analyses:
        if ca["type"] in ("categorical", "boolean") and ca.get("top_value_pct", 0) > 60:
            add(4, "Distribution",
                f"'{ca['column']}' is dominated by '{ca['top_value']}' ({ca['top_value_pct']:.0f}% of values)",
                f"This imbalance could bias aggregate statistics. Consider stratified analysis.",
                0.80)

    # Sort by priority and cap
    insights.sort(key=lambda x: x["priority"])
    return insights[:max_insights]


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _compute_confidence(
    df: pd.DataFrame, column_analyses: list[dict], insights: list[dict]
) -> float:
    """Return an overall analysis confidence score (0.0–1.0)."""
    score = 1.0
    # Penalise for nulls
    avg_null = np.mean([a["null_pct"] for a in column_analyses]) if column_analyses else 0
    score -= avg_null / 100 * 0.3
    # Reward for having many rows
    if len(df) < 50:
        score -= 0.2
    elif len(df) < 200:
        score -= 0.1
    return round(max(0.5, min(1.0, score)), 2)
