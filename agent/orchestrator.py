"""
Orchestrator Module
===================
Main pipeline that wires all agent modules together:
  file → ingest → schema → quality → semantic → analyse → visualise → report

Returns a structured results dict consumed by the Streamlit UI.
"""

from __future__ import annotations

import io
import logging
import os
import time
from datetime import datetime
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default config (merged with user config.yaml if provided)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict[str, Any] = {
    "ingestion": {
        "max_rows": 200_000,
        "best_sheet_strategy": "largest",
        "header_detection_rows": 20,
    },
    "schema": {
        "cardinality_threshold": 50,
        "identifier_min_uniqueness": 0.9,
    },
    "quality": {
        "outlier_method": "iqr",
        "outlier_iqr_factor": 1.5,
    },
    "analysis": {
        "top_n_categories": 15,
        "correlation_min_columns": 2,
        "time_series_min_points": 10,
    },
    "visualization": {
        "max_charts": 30,
        "max_scatter_points": 5000,
    },
    "reporting": {
        "max_insights": 20,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load and merge YAML config with defaults."""
    cfg = dict(_DEFAULT_CONFIG)
    if config_path and os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
            _deep_merge(cfg, user_cfg)
            logger.info("Loaded config from %s", config_path)
        except Exception as e:
            logger.warning("Failed to load config '%s': %s — using defaults", config_path, e)
    return cfg


def run_pipeline(
    file: str | bytes | io.BytesIO,
    filename: str = "upload.xlsx",
    config_path: str | None = None,
    output_report_path: str | None = None,
    progress_callback=None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Execute the full analysis pipeline.

    Parameters
    ----------
    file:
        Path string, raw bytes, or BytesIO of the Excel file.
    filename:
        Display name for the file (used in the report).
    config_path:
        Optional path to a config.yaml file.
    output_report_path:
        If provided, the HTML report is written to this path.
    progress_callback:
        Optional callable(step: int, total: int, message: str) for progress.

    Returns
    -------
    dict with keys:
        metadata, schema, quality, semantic, analysis, charts, report_html,
        logs, elapsed_seconds
    """
    # Lazy imports to keep module importable even if deps missing
    from agent.ingestion import load_excel
    from agent.schema_inference import infer_schema
    from agent.quality_checks import run_quality_checks
    from agent.semantic_layer import build_semantic_layer
    from agent.analysis_engine import run_analysis
    from agent.visualization import generate_visualizations
    from agent.report_generator import generate_report

    config = load_config(config_path)
    logs: list[str] = []
    t0 = time.time()

    def step(n: int, total: int, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] Step {n}/{total}: {msg}"
        logs.append(entry)
        logger.info(entry)
        if progress_callback:
            try:
                progress_callback(n, total, msg)
            except Exception:
                pass

    TOTAL_STEPS = 7

    # ---- 1. Ingestion -------------------------------------------------------
    step(1, TOTAL_STEPS, "Loading and parsing Excel file…")
    ingest_result = load_excel(file, filename=filename, config=config)
    df = ingest_result["dataframe"]
    sheet_name = ingest_result.get("sheet_name", "Sheet1")
    logs += ingest_result.get("warnings", [])

    # ---- 2. Schema inference ------------------------------------------------
    step(2, TOTAL_STEPS, "Inferring column schema and data dictionary…")
    schema_result = infer_schema(df, config=config)
    column_types = schema_result["column_types"]
    analysis_columns = schema_result["analysis_columns"]
    data_dict = schema_result["data_dictionary"]

    # ---- 3. Quality checks --------------------------------------------------
    step(3, TOTAL_STEPS, "Running data quality checks…")
    quality_result = run_quality_checks(df, column_types, analysis_columns, config=config)

    # ---- 4. Semantic layer --------------------------------------------------
    step(4, TOTAL_STEPS, "Building semantic layer (dimensions, measures, KPIs)…")
    semantic_result = build_semantic_layer(df, column_types, analysis_columns, config=config)

    # ---- 5. Analysis --------------------------------------------------------
    step(5, TOTAL_STEPS, "Running automated statistical analysis…")
    analysis_result = run_analysis(
        df, column_types, semantic_result, analysis_columns, config=config, api_key=api_key
    )

    # ---- 6. Visualizations --------------------------------------------------
    step(6, TOTAL_STEPS, "Generating smart visualizations…")
    charts = generate_visualizations(
        df, column_types, analysis_result, semantic_result, analysis_columns, config=config
    )

    # ---- 7. Report ----------------------------------------------------------
    step(7, TOTAL_STEPS, "Generating HTML report…")
    pipeline_result = {
        "metadata": {
            "filename": filename,
            "sheet": sheet_name,
            "rows_raw": ingest_result.get("rows_raw", len(df)),
            "pipeline_run_at": datetime.now().isoformat(),
        },
        "schema": {
            "column_types": column_types,
            "analysis_columns": analysis_columns,
            "data_dictionary": data_dict,
        },
        "quality": quality_result,
        "semantic": semantic_result,
        "analysis": analysis_result,
        "charts": charts,
        "dataframe": df,
    }
    report_html = generate_report(pipeline_result, output_path=output_report_path)
    pipeline_result["report_html"] = report_html
    pipeline_result["logs"] = logs
    pipeline_result["elapsed_seconds"] = round(time.time() - t0, 2)

    logger.info(
        "Pipeline complete in %.1fs — %d insights, %d charts",
        pipeline_result["elapsed_seconds"],
        len(analysis_result.get("insights", [])),
        len(charts),
    )
    return pipeline_result


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (in-place)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
