"""
Semantic Layer Module
=====================
Builds a dbt-inspired semantic layer on top of raw column data.
Classifies columns into dimensions, measures, time fields, entities,
and auto-generates KPIs. Exports a YAML metadata file for reproducibility.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Semantic types
SEM_DIMENSION = "dimension"
SEM_MEASURE = "measure"
SEM_TIME = "time"
SEM_ENTITY = "entity"
SEM_KPI = "kpi"

# Aggregation choices per numeric context
AGG_AVG_KEYWORDS = ["rate", "ratio", "average", "avg", "score", "percent",
                    "pct", "satisfaction", "rating", "age"]
AGG_COUNT_KEYWORDS = ["count", "cnt", "num", "number", "id"]

# ---------------------------------------------------------------------------
# Domain overrides (HR / Attrition)
# ---------------------------------------------------------------------------

HR_SEMANTIC_OVERRIDES: dict[str, dict[str, Any]] = {
 "attrition": {
    "type": SEM_KPI,
    "description": "Employee attrition rate calculated as exits divided by headcount, aligned to financial year (April–March).",
    "synonyms": ["attrition rate", "employee churn", "exit rate", "employee turnover"],
    "allowed_operations": ["calculate", "trend", "filter"],
    "related_columns": ["date_of_exit", "lwd", "empid", "empstatus", "endofmonth"],
    "formula": "exits / headcount",
    "time_logic": {
        "financial_year_start": "April",
        "financial_year_end": "March",
        "monthly_behavior": "cumulative exits from April to selected month / headcount",
        "yearly_behavior": "calculate from April of that year",
        "range_behavior": "cumulative exits over range / headcount"
    },
    "required_columns": ["Exit_Date OR Attrition flag", "Employee_ID", "Headcount"],
    "sql_template": {
        "monthly": "SUM(exits from April to selected month) / headcount",
        "yearly": "SUM(exits from April of that year) / headcount"
    },
    "edge_cases": [
        "Handle missing exit dates",
        "Avoid division by zero",
        "Ensure correct financial year mapping",
        "Handle partial data months"
    ]
},
    "empid": {
        "type": SEM_ENTITY,
        "description": "Primary employee identifier used for headcount and attrition",
        "synonyms": ["employee id", "emp id", "employee number", "associate id", "staff id"],
        "allowed_operations": ["count", "distinct", "filter"],
        "related_columns": ["employee_name", "empstatus", "manager_id"],
    },
    "employee_name": {
        "type": SEM_DIMENSION,
        "description": "Employee full name for drill-down",
        "synonyms": ["employee", "staff name", "associate name"],
        "allowed_operations": ["filter"],
        "related_columns": ["empid"],
    },
    "empstatus": {
        "type": SEM_DIMENSION,
        "description": "Employment status at snapshot date (Active/Inactive/Exited)",
        "synonyms": ["status", "employment status", "active flag", "current status"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["endofmonth", "lwd", "date_of_exit"],
    },
    "date_of_joining": {
        "type": SEM_TIME,
        "description": "Hire/start date used for tenure and cohorts",
        "synonyms": ["hire date", "doj", "start date", "join date"],
        "allowed_operations": ["filter", "group_by_month", "group_by_year", "date_diff"],
        "related_columns": ["tenure_months", "tenure_days", "newhireflag"],
    },
    "lwd": {
        "type": SEM_TIME,
        "description": "Last working day; identifies leavers",
        "synonyms": ["last working day", "last day", "lwd"],
        "allowed_operations": ["filter", "group_by_month", "group_by_quarter", "group_by_year"],
        "related_columns": ["date_of_exit", "final_exit_type", "final_reason_of_exit"],
    },
    "date_of_exit": {
        "type": SEM_TIME,
        "description": "Official exit/termination date",
        "synonyms": ["exit date", "separation date", "termination date"],
        "allowed_operations": ["filter", "group_by_month", "group_by_year"],
        "related_columns": ["lwd", "final_exit_type"],
    },
    "endofmonth": {
        "type": SEM_TIME,
        "description": "Month-end snapshot date for headcount",
        "synonyms": ["month end", "snapshot date", "as of date"],
        "allowed_operations": ["filter", "group_by_month", "group_by_year"],
        "related_columns": ["empstatus", "newhireflag"],
    },
    "newhireflag": {
        "type": SEM_DIMENSION,
        "description": "Flag for new hires in snapshot period",
        "synonyms": ["new hire", "new joiner", "recent hire", "joined this month"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["date_of_joining"],
    },
    "grade": {
        "type": SEM_DIMENSION,
        "description": "Employee grade / band",
        "synonyms": ["grade", "band", "level band"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["joblevel"],
    },
    "joblevel": {
        "type": SEM_DIMENSION,
        "description": "Job level / title band",
        "synonyms": ["job level", "level", "designation level"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["grade"],
    },
    "employeetype": {
        "type": SEM_DIMENSION,
        "description": "Employment type (Full Time/Contract)",
        "synonyms": ["employee type", "worker type", "fte/contract"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["empstatus"],
    },
    "ic_pm": {
        "type": SEM_DIMENSION,
        "description": "Role type: Individual Contributor vs People Manager",
        "synonyms": ["ic/pm", "individual contributor", "people manager"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["manager_id"],
    },
    "emprating": {
        "type": SEM_DIMENSION,
        "description": "Performance rating",
        "synonyms": ["rating", "performance rating", "appraisal"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["empstatus"],
    },
    "gender": {
        "type": SEM_DIMENSION,
        "description": "Gender of employee",
        "synonyms": ["sex", "gender identity"],
        "allowed_operations": ["filter", "group"],
        "related_columns": [],
    },
    "manager": {
        "type": SEM_DIMENSION,
        "description": "Manager display name with id",
        "synonyms": ["manager", "supervisor", "reporting manager"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["manager_id", "manager_name"],
    },
    "manager_id": {
        "type": SEM_ENTITY,
        "description": "Manager employee ID for team roll-up",
        "synonyms": ["manager id", "supervisor id", "lead id", "team lead id"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["manager_name", "manageremail", "empid"],
    },
    "manager_name": {
        "type": SEM_DIMENSION,
        "description": "Manager name",
        "synonyms": ["manager", "supervisor name", "team lead"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["manager_id"],
    },
    "manageremail": {
        "type": SEM_DIMENSION,
        "description": "Manager email address",
        "synonyms": ["manager email", "supervisor email"],
        "allowed_operations": ["filter"],
        "related_columns": ["manager_id", "manager_name"],
    },
    "functional role": {
        "type": SEM_DIMENSION,
        "description": "Functional role / job title",
        "synonyms": ["role", "job role", "designation", "position"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["department", "sub department"],
    },
    "department": {
        "type": SEM_DIMENSION,
        "description": "Department / function",
        "synonyms": ["dept", "function", "org unit"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["sub department", "lob", "businessgroup"],
    },
    "sub department": {
        "type": SEM_DIMENSION,
        "description": "Sub-department / team",
        "synonyms": ["sub dept", "sub-function", "team"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["department"],
    },
    "lob": {
        "type": SEM_DIMENSION,
        "description": "Line of business",
        "synonyms": ["lob", "business line", "line of business"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["businessgroup"],
    },
    "businessgroup": {
        "type": SEM_DIMENSION,
        "description": "Business group / division",
        "synonyms": ["business group", "division", "org group"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["lob", "cxo"],
    },
    "cxo": {
        "type": SEM_DIMENSION,
        "description": "CXO / executive owner of group",
        "synonyms": ["cxo", "executive sponsor", "business head"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["cxo_emp_id"],
    },
    "cxo_emp_id": {
        "type": SEM_ENTITY,
        "description": "CXO employee ID",
        "synonyms": ["cxo id", "executive id"],
        "allowed_operations": ["filter"],
        "related_columns": ["cxo"],
    },
    "hrbpname": {
        "type": SEM_DIMENSION,
        "description": "HRBP name",
        "synonyms": ["hrbp", "hr business partner", "people partner"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["hrbpempid", "hrbpemail"],
    },
    "hrbpemail": {
        "type": SEM_DIMENSION,
        "description": "HRBP email",
        "synonyms": ["hrbp email", "hr partner email"],
        "allowed_operations": ["filter"],
        "related_columns": ["hrbpname"],
    },
    "hrbpempid": {
        "type": SEM_ENTITY,
        "description": "HRBP employee ID",
        "synonyms": ["hrbp id", "hr partner id"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["hrbpname"],
    },
    "final_reason_of_exit": {
        "type": SEM_DIMENSION,
        "description": "Final standardized exit reason",
        "synonyms": ["final exit reason", "termination reason", "reason of exit"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["final_exit_type"],
    },
    "final_exit_type": {
        "type": SEM_DIMENSION,
        "description": "Final exit category (Voluntary/Involuntary)",
        "synonyms": ["attrition type", "exit category", "voluntary/involuntary"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["final_reason_of_exit"],
    },
    "employee_reason_of_exit": {
        "type": SEM_DIMENSION,
        "description": "Exit reason reported by employee",
        "synonyms": ["employee exit reason", "self-reported reason"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["final_reason_of_exit"],
    },
    "emp_type_of_exit": {
        "type": SEM_DIMENSION,
        "description": "Exit type reported by employee",
        "synonyms": ["employee exit type", "self exit type"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["final_exit_type"],
    },
    "manager_reason_of_exit": {
        "type": SEM_DIMENSION,
        "description": "Exit reason reported by manager",
        "synonyms": ["manager exit reason"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["final_reason_of_exit"],
    },
    "manager_type_of_exit": {
        "type": SEM_DIMENSION,
        "description": "Exit type reported by manager",
        "synonyms": ["manager exit type"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["final_exit_type"],
    },
    "hrbp_reason_of_exit": {
        "type": SEM_DIMENSION,
        "description": "Exit reason validated by HRBP",
        "synonyms": ["hrbp exit reason"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["final_reason_of_exit"],
    },
    "hrbp_type_of_exit": {
        "type": SEM_DIMENSION,
        "description": "Exit type validated by HRBP",
        "synonyms": ["hrbp exit type"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["final_exit_type"],
    },
    "er_reason_of_exit": {
        "type": SEM_DIMENSION,
        "description": "Employee relations exit reason",
        "synonyms": ["er exit reason", "employee relations reason"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["final_reason_of_exit"],
    },
    "er_type_of_exit": {
        "type": SEM_DIMENSION,
        "description": "Employee relations exit type",
        "synonyms": ["er exit type"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["final_exit_type"],
    },
    "tenure_bucket": {
        "type": SEM_DIMENSION,
        "description": "Tenure band (e.g., 1-2 Years)",
        "synonyms": ["tenure band", "length of service bucket"],
        "allowed_operations": ["filter", "group"],
        "related_columns": ["tenure_months", "tenure_days"],
    },
    "tenure_months": {
        "type": SEM_MEASURE,
        "description": "Tenure in months at snapshot/exit",
        "synonyms": ["months of service", "tenure months"],
        "allowed_operations": ["avg", "min", "max"],
        "related_columns": ["tenure_bucket", "date_of_joining"],
    },
    "tenure_days": {
        "type": SEM_MEASURE,
        "description": "Tenure in days at snapshot/exit",
        "synonyms": ["days of service", "tenure days"],
        "allowed_operations": ["avg", "min", "max"],
        "related_columns": ["tenure_months", "date_of_joining"],
    },
    "resignation_date": {
        "type": SEM_TIME,
        "description": "Resignation date (notice period starts)",
        "synonyms": ["resigned on", "notice date", "resignation date"],
        "allowed_operations": ["filter", "date_diff"],
        "related_columns": ["lwd"],
    },
}

HR_QUERY_PATTERNS = [
    {
        "pattern": "headcount by <dimension>",
        "maps_to": "Active Headcount grouped by <dimension> at endofmonth",
    },
    {
        "pattern": "attrition rate by <dimension>",
        "maps_to": "Leavers in period / Avg Active HC in period * 100 grouped by <dimension>",
    },
    {
        "pattern": "attrition rate last month",
        "maps_to": "Attrition rate using lwd/date_of_exit in prior month and avg active HC",
    },
    {
        "pattern": "attrition trend last 12 months",
        "maps_to": "Rolling 12M attrition rate using lwd/date_of_exit and avg HC",
    },
    {
        "pattern": "notice period employees",
        "maps_to": "resignation_date IS NOT NULL AND lwd > today()",
    },
    {
        "pattern": "new hires this month",
        "maps_to": "COUNT(empid) WHERE newhireflag='Yes' for endofmonth",
    },
    {
        "pattern": "exit reasons by department",
        "maps_to": "Leaver count grouped by final_reason_of_exit and department",
    },
]

HR_AMBIGUITY_RULES = [
    {
        "term": "attrition",
        "default": "use lwd/date_of_exit as leaver event and avg active headcount in period",
    },
    {
        "term": "team",
        "maps_to": "manager_id",
    },
    {
        "term": "function",
        "maps_to": "department",
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_semantic_layer(
    df: pd.DataFrame,
    column_types: dict[str, str],
    analysis_columns: list[str],
    config: dict | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """
    Build and return the semantic layer.

    Returns
    -------
    dict with keys:
        - ``dimensions``  : list[dict]
        - ``measures``    : list[dict]
        - ``time_fields`` : list[dict]
        - ``entities``    : list[dict]
        - ``kpis``        : list[dict]
        - ``yaml``        : str — YAML representation
        - ``lineage``     : dict — raw col → semantic field mapping
        - ``summary``     : str — human-readable summary
    """
    cfg = (config or {}).get("semantic_layer", {})
    auto_kpi: bool = cfg.get("auto_kpi_generation", True)
    standardize: bool = cfg.get("standardize_names", True)
    synonym_map: dict = cfg.get("synonym_mapping", {})

    # Build synonym lookup: raw_keyword → canonical_name
    synonym_lookup = _build_synonym_lookup(synonym_map)

    dimensions: list[dict] = []
    measures: list[dict] = []
    time_fields: list[dict] = []
    entities: list[dict] = []
    lineage: dict[str, str] = {}

    hr_context = _has_hr_schema(analysis_columns)

    for col in analysis_columns:
        col_type = column_types.get(col, "unknown")
        canonical = _canonical_name(col, synonym_lookup, standardize)
        lineage[col] = canonical

        override = _get_override(col)
        sample_values = _get_sample_values(df, col, col_type)

        entry_base = {
            "raw_column": col,
            "semantic_name": canonical,
            "description": override.get("description") or _infer_description(col, col_type),
            "nullable": bool(df[col].isna().any()),
            "synonyms": override.get("synonyms", []),
            "allowed_operations": override.get("allowed_operations", _default_operations(col_type)),
            "related_columns": override.get("related_columns", []),
            "sample_values": sample_values,
        }

        override_type = override.get("type")
        if override_type:
            col_type = {"time": "datetime", "measure": "numeric", "dimension": "categorical", "entity": "identifier"}.get(override_type, col_type)

        if col_type == "datetime":
            time_fields.append({
                **entry_base,
                "semantic_type": SEM_TIME,
                "granularities": ["day", "month", "quarter", "year"],
                "time_role": "snapshot" if _normalize_key(col) == "endofmonth" else "event",
            })
        elif col_type == "identifier":
            entities.append({
                **entry_base,
                "semantic_type": SEM_ENTITY,
            })
        elif col_type == "numeric":
            agg = _infer_aggregation(col)
            measures.append({
                **entry_base,
                "semantic_type": SEM_MEASURE,
                "aggregation": agg,
                "format": _infer_format(col),
            })
        elif col_type in ("categorical", "boolean"):
            dimensions.append({
                **entry_base,
                "semantic_type": SEM_DIMENSION,
                "values": _get_dimension_values(df, col),
            })
        else:
            # text → treat as dimension
            dimensions.append({
                **entry_base,
                "semantic_type": SEM_DIMENSION,
            })

    # ---- Auto-generate KPIs ------------------------------------------------
    kpis: list[dict] = []
    if auto_kpi:
        if hr_context:
            kpis = _generate_hr_kpis(analysis_columns)
        else:
            kpis = _generate_kpis(measures, time_fields, dimensions, df, column_types)

    relationships = _build_relationships(analysis_columns) if hr_context else []
    query_patterns = HR_QUERY_PATTERNS if hr_context else []
    ambiguity_rules = HR_AMBIGUITY_RULES if hr_context else []

    # ---- YAML export -------------------------------------------------------
    layer_dict = {
        "semantic_layer": {
            "version": "1.2",
            "dimensions": dimensions,
            "measures": measures,
            "time_fields": time_fields,
            "entities": entities,
            "kpis": kpis,
            "relationships": relationships,
            "query_patterns": query_patterns,
            "ambiguity_rules": ambiguity_rules,
        }
    }
    yaml_str = yaml.dump(layer_dict, default_flow_style=False, allow_unicode=True, sort_keys=False)

    if output_path:
        Path(output_path).write_text(yaml_str, encoding="utf-8")
        logger.info("Semantic layer saved to: %s", output_path)

    summary = _build_summary(dimensions, measures, time_fields, entities, kpis, relationships, query_patterns)
    logger.info(
        "Semantic layer built: %d dimensions, %d measures, %d time, %d entities, %d KPIs",
        len(dimensions), len(measures), len(time_fields), len(entities), len(kpis)
    )

    return {
        "dimensions": dimensions,
        "measures": measures,
        "time_fields": time_fields,
        "entities": entities,
        "kpis": kpis,
        "relationships": relationships,
        "query_patterns": query_patterns,
        "ambiguity_rules": ambiguity_rules,
        "yaml": yaml_str,
        "lineage": lineage,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _has_hr_schema(columns: list[str]) -> bool:
    keys = {_normalize_key(c) for c in columns}
    return {"empid", "empstatus", "endofmonth"}.issubset(keys)


def _get_override(col: str) -> dict[str, Any]:
    key = _normalize_key(col)
    for k, v in HR_SEMANTIC_OVERRIDES.items():
        if _normalize_key(k) == key:
            return v
    return {}


def _default_operations(col_type: str) -> list[str]:
    if col_type == "numeric":
        return ["sum", "avg", "min", "max"]
    if col_type == "datetime":
        return ["filter", "group_by_month", "group_by_quarter", "group_by_year"]
    if col_type == "identifier":
        return ["count", "distinct", "filter"]
    return ["filter", "group"]


def _get_sample_values(df: pd.DataFrame, col: str, col_type: str) -> list[str]:
    try:
        vals = df[col].dropna().unique()[:5]
        return [str(v) for v in vals]
    except Exception:
        return []


def _build_synonym_lookup(synonym_map: dict) -> dict[str, str]:
    """Build keyword → canonical mapping from config synonym_mapping."""
    lookup: dict[str, str] = {}
    for canonical, synonyms in synonym_map.items():
        for syn in synonyms:
            lookup[syn.lower()] = canonical
    return lookup


def _canonical_name(col: str, synonym_lookup: dict[str, str], standardize: bool) -> str:
    """Convert raw column name to canonical semantic name."""
    name = col.lower().strip()
    # Check full match first
    if name in synonym_lookup:
        return synonym_lookup[name]
    # Check partial keyword match
    for kw, canonical in synonym_lookup.items():
        if kw in name:
            return re.sub(kw, canonical, name)
    if standardize:
        return re.sub(r"[^a-z0-9]", "_", name).strip("_")
    return name


def _infer_description(col: str, col_type: str) -> str:
    """Generate a human-readable description for a column."""
    readable = col.replace("_", " ").title()
    type_descriptions = {
        "numeric": f"Numeric measure: {readable}",
        "categorical": f"Categorical dimension: {readable}",
        "datetime": f"Time field: {readable}",
        "boolean": f"Boolean flag: {readable}",
        "identifier": f"Unique identifier: {readable}",
        "text": f"Free-text field: {readable}",
    }
    return type_descriptions.get(col_type, f"Field: {readable}")


def _infer_aggregation(col: str) -> str:
    """Infer the best aggregation for a measure column based on its name."""
    name = col.lower()
    if any(kw in name for kw in AGG_AVG_KEYWORDS):
        return "avg"
    if any(kw in name for kw in AGG_COUNT_KEYWORDS):
        return "count"
    return "sum"


def _infer_format(col: str) -> str:
    """Infer display format (currency, percentage, number)."""
    name = col.lower()
    if any(kw in name for kw in ["price", "revenue", "cost", "amount", "sales", "profit", "income"]):
        return "currency"
    if any(kw in name for kw in ["pct", "percent", "rate", "ratio"]):
        return "percentage"
    return "number"


def _get_dimension_values(df: pd.DataFrame, col: str) -> list[str]:
    """Return sorted unique values for a dimension (capped at 50)."""
    try:
        vals = df[col].dropna().unique()[:50]
        return sorted([str(v) for v in vals])
    except Exception:
        return []


def _build_relationships(columns: list[str]) -> list[dict[str, str]]:
    cols = {_normalize_key(c) for c in columns}
    rels = []
    if {"manager_id", "empid"}.issubset(cols):
        rels.append({"from": "manager_id", "to": "empid", "type": "many_to_one", "description": "Employees roll up to a manager"})
    if {"department", "subdepartment"}.issubset(cols):
        rels.append({"from": "department", "to": "sub department", "type": "one_to_many", "description": "Department to sub-department hierarchy"})
    if {"lob", "businessgroup"}.issubset(cols):
        rels.append({"from": "lob", "to": "businessgroup", "type": "many_to_one", "description": "LOB maps to business group"})
    if {"businessgroup", "cxo"}.issubset(cols):
        rels.append({"from": "businessgroup", "to": "cxo", "type": "many_to_one", "description": "Business group maps to CXO"})
    if {"hrbpempid", "empid"}.issubset(cols):
        rels.append({"from": "hrbpempid", "to": "empid", "type": "many_to_one", "description": "Employees mapped to HRBP"})
    return rels


def _generate_hr_kpis(columns: list[str]) -> list[dict]:
    cols = {_normalize_key(c) for c in columns}
    kpis: list[dict] = []
    if {"empid", "empstatus", "endofmonth"}.issubset(cols):
        kpis.append({
            "name": "active_headcount",
            "semantic_type": SEM_KPI,
            "formula": "COUNT(DISTINCT empid) WHERE empstatus='Active' AND endofmonth=<period>",
            "description": "Total active headcount at month-end snapshot",
            "format": "number",
            "grain": "monthly snapshot",
        })
    if {"empid", "lwd", "endofmonth"}.issubset(cols):
        kpis.append({
            "name": "attrition_rate",
            "semantic_type": SEM_KPI,
            "formula": "Leavers in period / Avg Active HC in period * 100",
            "description": "Attrition rate using leavers (lwd/date_of_exit) and avg active HC",
            "format": "percentage",
            "grain": "monthly / ytd",
        })
    if {"empid", "final_exit_type"}.issubset(cols):
        kpis.append({
            "name": "voluntary_attrition_pct",
            "semantic_type": SEM_KPI,
            "formula": "Leavers WHERE final_exit_type='Voluntary' / Total Leavers * 100",
            "description": "Share of voluntary attrition",
            "format": "percentage",
        })
        kpis.append({
            "name": "involuntary_attrition_pct",
            "semantic_type": SEM_KPI,
            "formula": "Leavers WHERE final_exit_type='Involuntary' / Total Leavers * 100",
            "description": "Share of involuntary attrition",
            "format": "percentage",
        })
    if {"empid", "newhireflag", "endofmonth"}.issubset(cols):
        kpis.append({
            "name": "new_hire_count",
            "semantic_type": SEM_KPI,
            "formula": "COUNT(DISTINCT empid) WHERE newhireflag='Yes' AND endofmonth=<period>",
            "description": "Number of new hires in the period",
            "format": "number",
        })
    if {"empid", "resignation_date", "lwd"}.issubset(cols):
        kpis.append({
            "name": "notice_period_count",
            "semantic_type": SEM_KPI,
            "formula": "COUNT(empid) WHERE resignation_date IS NOT NULL AND lwd > today()",
            "description": "Employees resigned but not yet left",
            "format": "number",
        })
    if {"empid", "tenure_months", "lwd"}.issubset(cols):
        kpis.append({
            "name": "first_year_attrition_pct",
            "semantic_type": SEM_KPI,
            "formula": "Leavers WHERE tenure_months <= 12 / Total Leavers * 100",
            "description": "Share of leavers with tenure <= 12 months",
            "format": "percentage",
        })
    return kpis


def _generate_kpis(
    measures: list[dict],
    time_fields: list[dict],
    dimensions: list[dict],
    df: pd.DataFrame,
    column_types: dict[str, str],
) -> list[dict]:
    """Auto-generate business KPIs from available measures and dimensions."""
    kpis: list[dict] = []
    measure_names = [m["semantic_name"] for m in measures]
    raw_measure_cols = {m["semantic_name"]: m["raw_column"] for m in measures}

    # KPI 1: Total of each sum measure
    for m in measures:
        if m.get("aggregation") == "sum":
            kpis.append({
                "name": f"total_{m['semantic_name']}",
                "semantic_type": SEM_KPI,
                "formula": f"SUM({m['raw_column']})",
                "description": f"Total {m['raw_column'].replace('_', ' ')} across all records",
                "format": m.get("format", "number"),
                "source_measures": [m["semantic_name"]],
            })

    # KPI 2: Average of avg measures
    for m in measures:
        if m.get("aggregation") == "avg" and m.get("format") != "currency":
            kpis.append({
                "name": f"avg_{m['semantic_name']}",
                "semantic_type": SEM_KPI,
                "formula": f"AVG({m['raw_column']})",
                "description": f"Average {m['raw_column'].replace('_', ' ')}",
                "format": m.get("format", "number"),
                "source_measures": [m["semantic_name"]],
            })

    # KPI 3: Revenue per quantity (if both exist)
    revenue_col = next((m for m in measure_names if "revenue" in m or "sales" in m or "amount" in m), None)
    qty_col = next((m for m in measure_names if "quantity" in m or "qty" in m or "units" in m), None)
    if revenue_col and qty_col:
        kpis.append({
            "name": "revenue_per_unit",
            "semantic_type": SEM_KPI,
            "formula": f"SUM({raw_measure_cols.get(revenue_col,'revenue')}) / SUM({raw_measure_cols.get(qty_col,'quantity')})",
            "description": "Average revenue generated per unit sold",
            "format": "currency",
            "source_measures": [revenue_col, qty_col],
        })

    # KPI 4: Count of records per dimension (if dimensions exist)
    if dimensions:
        for dim in dimensions[:2]:  # limit to first 2
            kpis.append({
                "name": f"record_count_by_{dim['semantic_name']}",
                "semantic_type": SEM_KPI,
                "formula": f"COUNT(*) GROUP BY {dim['raw_column']}",
                "description": f"Number of records per {dim['raw_column'].replace('_', ' ')}",
                "format": "number",
                "source_measures": [],
            })

    return kpis


def _build_summary(
    dimensions: list[dict],
    measures: list[dict],
    time_fields: list[dict],
    entities: list[dict],
    kpis: list[dict],
    relationships: list[dict],
    query_patterns: list[dict],
) -> str:
    lines = [
        f"**Semantic Layer Summary**",
        f"- **{len(dimensions)} Dimension(s)**: {', '.join(d['semantic_name'] for d in dimensions[:8])}{'...' if len(dimensions) > 8 else ''}",
        f"- **{len(measures)} Measure(s)**: {', '.join(m['semantic_name'] for m in measures[:8])}{'...' if len(measures) > 8 else ''}",
        f"- **{len(time_fields)} Time Field(s)**: {', '.join(t['semantic_name'] for t in time_fields)}",
        f"- **{len(entities)} Entity/Identifier(s)**: {', '.join(e['semantic_name'] for e in entities[:5])}",
        f"- **{len(kpis)} KPI(s)**",
        f"- **{len(relationships)} Relationship(s)**",
        f"- **{len(query_patterns)} Query Pattern(s)**",
    ]
    return "\n".join(lines)
