"""
SQL Template Engine
====================
Provides deterministic SQL templates for each of the 25 catalogue questions.

Templates are keyed by intent_key (catalogue ID → intent string).
Parameters are applied via safe string interpolation — only known
literal values (dates, enum strings) are substituted; table/column
names are never taken from user input to prevent SQL injection.

All templates target DuckDB-compatible SQL.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _col(name: str) -> str:
    """Double-quote a column name for DuckDB safety."""
    # Strip any existing quotes first, then re-quote
    clean = name.strip('"').strip("'")
    return f'"{clean}"'


def _tbl(name: str) -> str:
    """Double-quote a table name for DuckDB safety."""
    clean = name.strip('"').strip("'")
    return f'"{clean}"'


def _maybe_and_filter(col: str, value: Optional[str]) -> str:
    """Return ``AND "col" = 'value'`` if *value* is truthy, else empty string."""
    if not value:
        return ""
    # Sanitize value: allow only word chars, spaces, hyphens
    safe_val = re.sub(r"[^\w\s\-]", "", value)
    return f"  AND {_col(col)} = '{safe_val}'\n"


def _maybe_date_filter(
    col: str,
    start: Optional[str],
    end: Optional[str],
) -> str:
    """Return date range filter clause if dates are provided."""
    if not (_is_valid_date(start) and _is_valid_date(end)):
        return ""
    return f"  AND {_col(col)} BETWEEN '{start}' AND '{end}'\n"


def _is_valid_date(value: Optional[str]) -> bool:
    if not value:
        return False
    _iso = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    return bool(_iso.match(value))


def _latest_month_filter(col: str, table: str) -> str:
    """Filter to the latest month available in the dataset."""
    return (
        f"  AND DATE_TRUNC('month', {_col(col)}) = (\n"
        f"    SELECT DATE_TRUNC('month', MAX({_col(col)})) FROM {table}\n"
        f"  )\n"
    )


def _fy_start_expr(date_expr: str) -> str:
    return (
        "CASE WHEN EXTRACT('month' FROM {expr}) >= 4 "
        "THEN DATE_TRUNC('year', {expr}) + INTERVAL '3 months' "
        "ELSE DATE_TRUNC('year', {expr}) - INTERVAL '9 months' END"
    ).format(expr=date_expr)


def _attrition_params_ctes(
    table: str,
    month_start: Optional[str],
    month_end: Optional[str],
    latest_col: str = "endofmonth",
) -> tuple:
    """Return CTE list, month_start expr, month_end expr for attrition queries."""
    t = _tbl(table)
    if _is_valid_date(month_start) and _is_valid_date(month_end):
        start_expr = f"DATE '{month_start}'"
        end_expr = f"DATE '{month_end}'"
        params_cte = (
            "params AS (\n"
            f"  SELECT {start_expr} AS month_start,\n"
            f"         {end_expr} AS month_end,\n"
            f"         {_fy_start_expr(end_expr)} AS fy_start\n"
            ")"
        )
        return [params_cte], start_expr, end_expr

    latest_cte = (
        "latest_month AS (\n"
        f"  SELECT\n"
        f"    DATE_TRUNC('month', MAX({_col(latest_col)})) AS month_start,\n"
        f"    DATE_TRUNC('month', MAX({_col(latest_col)})) + INTERVAL '1 month' - INTERVAL '1 day' AS month_end\n"
        f"  FROM {t}\n"
        ")"
    )
    params_cte = (
        "params AS (\n"
        "  SELECT month_start,\n"
        "         month_end,\n"
        f"         {_fy_start_expr('month_end')} AS fy_start\n"
        "  FROM latest_month\n"
        ")"
    )
    return [latest_cte, params_cte], "month_start", "month_end"


def _with_ctes(ctes: list) -> str:
    if not ctes:
        return ""
    return "WITH " + ",\n".join(ctes) + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_template(
    intent_key: str,
    table_name: str,
    filters: Optional[Dict[str, Optional[str]]] = None,
) -> Optional[str]:
    """Render a SQL template for the given intent.

    Parameters
    ----------
    intent_key:
        One of the 25 canonical intent keys from the catalogue.
    table_name:
        The registered DuckDB table name (from the uploaded file).
    filters:
        Optional dict of extracted user filters (department, grade, etc.).

    Returns
    -------
    A SQL string, or ``None`` if no template exists for *intent_key*.
    """
    filters = filters or {}
    fn = _TEMPLATE_MAP.get(intent_key)
    if fn is None:
        return None
    try:
        sql = fn(table_name, filters)
        logger.debug("Template rendered for intent=%s table=%s", intent_key, table_name)
        return sql
    except Exception as exc:
        logger.warning("Template render failed for intent=%s: %s", intent_key, exc)
        return None


def list_supported_intents() -> list[str]:
    """Return all intent keys that have a SQL template defined."""
    return list(_TEMPLATE_MAP.keys())


# ---------------------------------------------------------------------------
# Template functions (one per catalogue question / intent)
# ---------------------------------------------------------------------------

def _q1_hc_total_snapshot(table: str, f: dict) -> str:
    """Q1: Total active headcount (snapshot)."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    grade_f = _maybe_and_filter("grade", f.get("grade"))
    gender_f = _maybe_and_filter("gender", f.get("gender"))
    lob_f = _maybe_and_filter("lob", f.get("lob"))
    date_f = _maybe_date_filter("endofmonth", f.get("month_start"), f.get("month_end"))
    if not date_f:
        date_f = _latest_month_filter("endofmonth", t)
    return (
        f"SELECT COUNT(DISTINCT \"empid\") AS active_headcount\n"
        f"FROM {t}\n"
        f"WHERE \"empstatus\" = 'Active'\n"
        f"{dept_f}{grade_f}{gender_f}{lob_f}{date_f}"
        f";"
    )


def _q2_hc_trend_mom(table: str, f: dict) -> str:
    """Q2: Headcount month-on-month trend (last 12 months)."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    lob_f = _maybe_and_filter("lob", f.get("lob"))
    grade_f = _maybe_and_filter("grade", f.get("grade"))
    return (
        f"SELECT\n"
        f"  \"endofmonth\" AS month,\n"
        f"  COUNT(DISTINCT \"empid\") AS active_headcount\n"
        f"FROM {t}\n"
        f"WHERE \"empstatus\" = 'Active'\n"
        f"  AND \"endofmonth\" >= (CURRENT_DATE - INTERVAL '12 months')\n"
        f"{dept_f}{lob_f}{grade_f}"
        f"GROUP BY \"endofmonth\"\n"
        f"ORDER BY \"endofmonth\"\n"
        f";"
    )


def _q3_hc_by_gender(table: str, f: dict) -> str:
    """Q3: Headcount split by gender across departments."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    lob_f = _maybe_and_filter("lob", f.get("lob"))
    date_f = _maybe_date_filter("endofmonth", f.get("month_start"), f.get("month_end"))
    if not date_f:
        date_f = _latest_month_filter("endofmonth", t)
    return (
        f"SELECT\n"
        f"  \"department\",\n"
        f"  \"gender\",\n"
        f"  COUNT(DISTINCT \"empid\") AS headcount,\n"
        f"  ROUND(\n"
        f"    COUNT(DISTINCT \"empid\") * 100.0 /\n"
        f"    SUM(COUNT(DISTINCT \"empid\")) OVER (PARTITION BY \"department\"),\n"
        f"    2\n"
        f"  ) AS gender_pct\n"
        f"FROM {t}\n"
        f"WHERE \"empstatus\" = 'Active'\n"
        f"{dept_f}{lob_f}{date_f}"
        f"GROUP BY \"department\", \"gender\"\n"
        f"ORDER BY \"department\", \"gender\"\n"
        f";"
    )


def _q4_hc_team(table: str, f: dict) -> str:
    """Q4: Headcount for a specific manager's team."""
    t = _tbl(table)
    mgr_f = _maybe_and_filter("manager_id", f.get("manager_id"))
    dept_f = _maybe_and_filter("department", f.get("department"))
    date_f = _maybe_date_filter("endofmonth", f.get("month_start"), f.get("month_end"))
    if not date_f:
        date_f = _latest_month_filter("endofmonth", t)
    return (
        f"SELECT\n"
        f"  \"manager_name\",\n"
        f"  COUNT(DISTINCT \"empid\") AS team_headcount\n"
        f"FROM {t}\n"
        f"WHERE \"empstatus\" = 'Active'\n"
        f"{mgr_f}{dept_f}{date_f}"
        f"GROUP BY \"manager_name\"\n"
        f"ORDER BY team_headcount DESC\n"
        f";"
    )


def _q5_hc_by_grade(table: str, f: dict) -> str:
    """Q5: Headcount by grade/job level."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    date_f = _maybe_date_filter("endofmonth", f.get("month_start"), f.get("month_end"))
    if not date_f:
        date_f = _latest_month_filter("endofmonth", t)
    return (
        f"SELECT\n"
        f"  \"grade\",\n"
        f"  \"joblevel\",\n"
        f"  COUNT(DISTINCT \"empid\") AS headcount\n"
        f"FROM {t}\n"
        f"WHERE \"empstatus\" = 'Active'\n"
        f"{dept_f}{date_f}"
        f"GROUP BY \"grade\", \"joblevel\"\n"
        f"ORDER BY \"grade\", \"joblevel\"\n"
        f";"
    )


def _q6_hc_new_hires(table: str, f: dict) -> str:
    """Q6: New hires this month vs last month."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    grade_f = _maybe_and_filter("grade", f.get("grade"))
    gender_f = _maybe_and_filter("gender", f.get("gender"))
    lob_f = _maybe_and_filter("lob", f.get("lob"))
    date_f = _maybe_date_filter("endofmonth", f.get("month_start"), f.get("month_end"))
    return (
        f"SELECT\n"
        f"  \"endofmonth\" AS month,\n"
        f"  COUNT(DISTINCT \"empid\") AS new_hire_count\n"
        f"FROM {t}\n"
        f"WHERE \"newhireflag\" = 'Yes'\n"
        f"  AND \"endofmonth\" >= (CURRENT_DATE - INTERVAL '2 months')\n"
        f"{dept_f}{grade_f}{gender_f}{lob_f}{date_f}"
        f"GROUP BY \"endofmonth\"\n"
        f"ORDER BY \"endofmonth\"\n"
        f";"
    )


def _q7_hc_by_tenure(table: str, f: dict) -> str:
    """Q7: Headcount by tenure bucket."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    grade_f = _maybe_and_filter("grade", f.get("grade"))
    date_f = _maybe_date_filter("endofmonth", f.get("month_start"), f.get("month_end"))
    if not date_f:
        date_f = _latest_month_filter("endofmonth", t)
    return (
        f"SELECT\n"
        f"  \"tenure_bucket\",\n"
        f"  COUNT(DISTINCT \"empid\") AS headcount\n"
        f"FROM {t}\n"
        f"WHERE \"empstatus\" = 'Active'\n"
        f"{dept_f}{grade_f}{date_f}"
        f"GROUP BY \"tenure_bucket\"\n"
        f"ORDER BY MIN(\"tenure_months\")\n"
        f";"
    )


def _q8_hc_by_emp_type(table: str, f: dict) -> str:
    """Q8: Headcount by employee type (Full Time / Contract)."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    lob_f = _maybe_and_filter("lob", f.get("lob"))
    date_f = _maybe_date_filter("endofmonth", f.get("month_start"), f.get("month_end"))
    if not date_f:
        date_f = _latest_month_filter("endofmonth", t)
    return (
        f"SELECT\n"
        f"  \"employeetype\",\n"
        f"  COUNT(DISTINCT \"empid\") AS headcount,\n"
        f"  ROUND(\n"
        f"    COUNT(DISTINCT \"empid\") * 100.0 /\n"
        f"    SUM(COUNT(DISTINCT \"empid\")) OVER (),\n"
        f"    2\n"
        f"  ) AS pct_of_total\n"
        f"FROM {t}\n"
        f"WHERE \"empstatus\" = 'Active'\n"
        f"{dept_f}{lob_f}{date_f}"
        f"GROUP BY \"employeetype\"\n"
        f"ORDER BY headcount DESC\n"
        f";"
    )


def _q9_hc_by_business_group(table: str, f: dict) -> str:
    """Q9: Headcount comparison across business groups."""
    t = _tbl(table)
    lob_f = _maybe_and_filter("lob", f.get("lob"))
    bg_f = _maybe_and_filter("businessgroup", f.get("businessgroup"))
    date_f = _maybe_date_filter("endofmonth", f.get("month_start"), f.get("month_end"))
    if not date_f:
        date_f = _latest_month_filter("endofmonth", t)
    return (
        f"SELECT\n"
        f"  \"lob\",\n"
        f"  \"businessgroup\",\n"
        f"  COUNT(DISTINCT \"empid\") AS headcount\n"
        f"FROM {t}\n"
        f"WHERE \"empstatus\" = 'Active'\n"
        f"{lob_f}{bg_f}{date_f}"
        f"GROUP BY \"lob\", \"businessgroup\"\n"
        f"ORDER BY headcount DESC\n"
        f";"
    )


def _q10_hc_ic_pm_split(table: str, f: dict) -> str:
    """Q10: IC vs PM split."""
    t = _tbl(table)
    mgr_f = _maybe_and_filter("manager_id", f.get("manager_id"))
    dept_f = _maybe_and_filter("department", f.get("department"))
    date_f = _maybe_date_filter("endofmonth", f.get("month_start"), f.get("month_end"))
    if not date_f:
        date_f = _latest_month_filter("endofmonth", t)
    return (
        f"SELECT\n"
        f"  \"ic_pm\",\n"
        f"  COUNT(DISTINCT \"empid\") AS headcount,\n"
        f"  ROUND(\n"
        f"    COUNT(DISTINCT \"empid\") * 100.0 /\n"
        f"    SUM(COUNT(DISTINCT \"empid\")) OVER (),\n"
        f"    2\n"
        f"  ) AS pct\n"
        f"FROM {t}\n"
        f"WHERE \"empstatus\" = 'Active'\n"
        f"{mgr_f}{dept_f}{date_f}"
        f"GROUP BY \"ic_pm\"\n"
        f"ORDER BY headcount DESC\n"
        f";"
    )


def _q11_attrition_overall(table: str, f: dict) -> str:
    """Q11: Overall attrition rate (month or YTD)."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    lob_f = _maybe_and_filter("lob", f.get("lob"))
    grade_f = _maybe_and_filter("grade", f.get("grade"))
    ctes, _, _ = _attrition_params_ctes(
        table, f.get("month_start"), f.get("month_end")
    )
    ctes.extend([
        (
            "leavers AS (\n"
            f"  SELECT COUNT(DISTINCT \"empid\") AS leaver_count\n"
            f"  FROM {t}\n"
            "  CROSS JOIN params p\n"
            "  WHERE \"empstatus\" = 'Inactive'\n"
            "    AND CAST(\"lwd\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
            f"{dept_f}{lob_f}{grade_f}"
            ")"
        ),
        (
            "monthly_hc AS (\n"
            "  SELECT\n"
            "    DATE_TRUNC('month', \"endofmonth\") AS month,\n"
            "    COUNT(DISTINCT \"empid\") AS active_hc\n"
            f"  FROM {t}\n"
            "  CROSS JOIN params p\n"
            "  WHERE \"empstatus\" = 'Active'\n"
            "    AND CAST(\"endofmonth\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
            f"{dept_f}{lob_f}{grade_f}"
            "  GROUP BY DATE_TRUNC('month', \"endofmonth\")\n"
            ")"
        ),
        (
            "avg_hc AS (\n"
            "  SELECT ROUND(AVG(active_hc)::NUMERIC, 2) AS avg_active_hc\n"
            "  FROM monthly_hc\n"
            ")"
        ),
    ])
    return (
        _with_ctes(ctes)
        + "SELECT\n"
        "  l.leaver_count,\n"
        "  a.avg_active_hc,\n"
        "  ROUND(\n"
        "    l.leaver_count * 100.0 / NULLIF(a.avg_active_hc, 0),\n"
        "    2\n"
        "  ) AS attrition_rate_pct,\n"
        "  p.month_end AS month_end\n"
        "FROM leavers l, avg_hc a, params p\n"
        ";"
    )


def _q12_attrition_vol_invol(table: str, f: dict) -> str:
    """Q12: Voluntary vs involuntary attrition split."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    grade_f = _maybe_and_filter("grade", f.get("grade"))
    ctes, _, _ = _attrition_params_ctes(
        table, f.get("month_start"), f.get("month_end")
    )
    return (
        _with_ctes(ctes)
        + f"SELECT\n"
        f"  \"final_exit_type\",\n"
        f"  COUNT(DISTINCT \"empid\") AS leaver_count,\n"
        f"  ROUND(\n"
        f"    COUNT(DISTINCT \"empid\") * 100.0 /\n"
        f"    SUM(COUNT(DISTINCT \"empid\")) OVER (),\n"
        f"    2\n"
        f"  ) AS pct_of_total\n"
        f"FROM {t}\n"
        f"CROSS JOIN params p\n"
        f"WHERE \"empstatus\" = 'Inactive'\n"
        f"  AND CAST(\"lwd\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
        f"{dept_f}{grade_f}"
        f"GROUP BY \"final_exit_type\"\n"
        f"ORDER BY leaver_count DESC\n"
        f";"
    )


def _q13_attrition_exit_reasons(table: str, f: dict) -> str:
    """Q13: Top reasons employees are leaving."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    grade_f = _maybe_and_filter("grade", f.get("grade"))
    exit_f = _maybe_and_filter("final_exit_type", f.get("exit_type"))
    ctes, _, _ = _attrition_params_ctes(
        table, f.get("month_start"), f.get("month_end")
    )
    return (
        _with_ctes(ctes)
        + f"SELECT\n"
        f"  \"final_reason_of_exit\",\n"
        f"  \"final_exit_type\",\n"
        f"  COUNT(DISTINCT \"empid\") AS leaver_count\n"
        f"FROM {t}\n"
        f"CROSS JOIN params p\n"
        f"WHERE \"empstatus\" = 'Inactive'\n"
        f"  AND CAST(\"lwd\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
        f"{dept_f}{grade_f}{exit_f}"
        f"GROUP BY \"final_reason_of_exit\", \"final_exit_type\"\n"
        f"ORDER BY leaver_count DESC\n"
        f"LIMIT 20\n"
        f";"
    )


def _q14_attrition_team(table: str, f: dict) -> str:
    """Q14: Attrition for a specific manager's team."""
    t = _tbl(table)
    mgr_f = _maybe_and_filter("manager_id", f.get("manager_id"))
    ctes, _, _ = _attrition_params_ctes(
        table, f.get("month_start"), f.get("month_end")
    )
    return (
        _with_ctes(ctes)
        + f"SELECT\n"
        f"  \"manager_name\",\n"
        f"  COUNT(DISTINCT \"empid\") AS leavers\n"
        f"FROM {t}\n"
        f"CROSS JOIN params p\n"
        f"WHERE \"empstatus\" = 'Inactive'\n"
        f"  AND CAST(\"lwd\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
        f"{mgr_f}"
        f"GROUP BY \"manager_name\"\n"
        f"ORDER BY leavers DESC\n"
        f";"
    )


def _q15_attrition_by_tenure(table: str, f: dict) -> str:
    """Q15: Attrition rate by tenure bucket."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    ctes, _, _ = _attrition_params_ctes(
        table, f.get("month_start"), f.get("month_end")
    )
    ctes.extend([
        (
            "leavers AS (\n"
            "  SELECT \"tenure_bucket\", COUNT(DISTINCT \"empid\") AS leaver_count\n"
            f"  FROM {t}\n"
            "  CROSS JOIN params p\n"
            "  WHERE \"empstatus\" = 'Inactive'\n"
            "    AND CAST(\"lwd\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
            f"{dept_f}"
            "  GROUP BY \"tenure_bucket\"\n"
            ")"
        ),
        (
            "monthly_hc AS (\n"
            "  SELECT\n"
            "    DATE_TRUNC('month', \"endofmonth\") AS month,\n"
            "    \"tenure_bucket\",\n"
            "    COUNT(DISTINCT \"empid\") AS active_hc\n"
            f"  FROM {t}\n"
            "  CROSS JOIN params p\n"
            "  WHERE \"empstatus\" = 'Active'\n"
            "    AND CAST(\"endofmonth\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
            f"{dept_f}"
            "  GROUP BY DATE_TRUNC('month', \"endofmonth\"), \"tenure_bucket\"\n"
            ")"
        ),
        (
            "avg_hc AS (\n"
            "  SELECT \"tenure_bucket\", AVG(active_hc) AS avg_active_hc\n"
            "  FROM monthly_hc\n"
            "  GROUP BY \"tenure_bucket\"\n"
            ")"
        ),
    ])
    return (
        _with_ctes(ctes)
        + "SELECT\n"
        "  COALESCE(l.\"tenure_bucket\", a.\"tenure_bucket\") AS tenure_bucket,\n"
        "  COALESCE(l.leaver_count, 0) AS leavers,\n"
        "  COALESCE(a.avg_active_hc, 0) AS avg_active_hc,\n"
        "  ROUND(\n"
        "    COALESCE(l.leaver_count, 0) * 100.0 /\n"
        "    NULLIF(COALESCE(a.avg_active_hc, 0), 0),\n"
        "    2\n"
        "  ) AS attrition_rate_pct\n"
        "FROM leavers l\n"
        "FULL OUTER JOIN avg_hc a USING (\"tenure_bucket\")\n"
        "ORDER BY attrition_rate_pct DESC\n"
        ";"
    )


def _q16_attrition_by_grade(table: str, f: dict) -> str:
    """Q16: Attrition rate by grade/job level."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    ctes, _, _ = _attrition_params_ctes(
        table, f.get("month_start"), f.get("month_end")
    )
    ctes.extend([
        (
            "leavers AS (\n"
            "  SELECT \"grade\", COUNT(DISTINCT \"empid\") AS leaver_count\n"
            f"  FROM {t}\n"
            "  CROSS JOIN params p\n"
            "  WHERE \"empstatus\" = 'Inactive'\n"
            "    AND CAST(\"lwd\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
            f"{dept_f}"
            "  GROUP BY \"grade\"\n"
            ")"
        ),
        (
            "monthly_hc AS (\n"
            "  SELECT\n"
            "    DATE_TRUNC('month', \"endofmonth\") AS month,\n"
            "    \"grade\",\n"
            "    COUNT(DISTINCT \"empid\") AS active_hc\n"
            f"  FROM {t}\n"
            "  CROSS JOIN params p\n"
            "  WHERE \"empstatus\" = 'Active'\n"
            "    AND CAST(\"endofmonth\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
            f"{dept_f}"
            "  GROUP BY DATE_TRUNC('month', \"endofmonth\"), \"grade\"\n"
            ")"
        ),
        (
            "avg_hc AS (\n"
            "  SELECT \"grade\", AVG(active_hc) AS avg_active_hc\n"
            "  FROM monthly_hc\n"
            "  GROUP BY \"grade\"\n"
            ")"
        ),
    ])
    return (
        _with_ctes(ctes)
        + "SELECT\n"
        "  COALESCE(l.\"grade\", a.\"grade\") AS grade,\n"
        "  COALESCE(l.leaver_count, 0) AS leavers,\n"
        "  COALESCE(a.avg_active_hc, 0) AS avg_active_hc,\n"
        "  ROUND(\n"
        "    COALESCE(l.leaver_count, 0) * 100.0 /\n"
        "    NULLIF(COALESCE(a.avg_active_hc, 0), 0),\n"
        "    2\n"
        "  ) AS attrition_rate_pct\n"
        "FROM leavers l\n"
        "FULL OUTER JOIN avg_hc a USING (\"grade\")\n"
        "ORDER BY attrition_rate_pct DESC\n"
        ";"
    )


def _q17_attrition_by_dept(table: str, f: dict) -> str:
    """Q17: Attrition comparison across departments."""
    t = _tbl(table)
    lob_f = _maybe_and_filter("lob", f.get("lob"))
    dept_f = _maybe_and_filter("department", f.get("department"))
    ctes, _, _ = _attrition_params_ctes(
        table, f.get("month_start"), f.get("month_end")
    )
    ctes.extend([
        (
            "leavers AS (\n"
            "  SELECT \"department\", COUNT(DISTINCT \"empid\") AS leaver_count\n"
            f"  FROM {t}\n"
            "  CROSS JOIN params p\n"
            "  WHERE \"empstatus\" = 'Inactive'\n"
            "    AND CAST(\"lwd\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
            f"{lob_f}{dept_f}"
            "  GROUP BY \"department\"\n"
            ")"
        ),
        (
            "monthly_hc AS (\n"
            "  SELECT\n"
            "    DATE_TRUNC('month', \"endofmonth\") AS month,\n"
            "    \"department\",\n"
            "    COUNT(DISTINCT \"empid\") AS active_hc\n"
            f"  FROM {t}\n"
            "  CROSS JOIN params p\n"
            "  WHERE \"empstatus\" = 'Active'\n"
            "    AND CAST(\"endofmonth\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
            f"{lob_f}{dept_f}"
            "  GROUP BY DATE_TRUNC('month', \"endofmonth\"), \"department\"\n"
            ")"
        ),
        (
            "avg_hc AS (\n"
            "  SELECT \"department\", ROUND(AVG(active_hc)::NUMERIC, 2) AS avg_active_hc\n"
            "  FROM monthly_hc\n"
            "  GROUP BY \"department\"\n"
            ")"
        ),
        (
            "org_leavers AS (\n"
            "  SELECT COUNT(DISTINCT \"empid\") AS leaver_count\n"
            f"  FROM {t}\n"
            "  CROSS JOIN params p\n"
            "  WHERE \"empstatus\" = 'Inactive'\n"
            "    AND CAST(\"lwd\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
            f"{lob_f}"
            ")"
        ),
        (
            "org_monthly_hc AS (\n"
            "  SELECT\n"
            "    DATE_TRUNC('month', \"endofmonth\") AS month,\n"
            "    COUNT(DISTINCT \"empid\") AS active_hc\n"
            f"  FROM {t}\n"
            "  CROSS JOIN params p\n"
            "  WHERE \"empstatus\" = 'Active'\n"
            "    AND CAST(\"endofmonth\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
            f"{lob_f}"
            "  GROUP BY DATE_TRUNC('month', \"endofmonth\")\n"
            ")"
        ),
        (
            "org_avg_hc AS (\n"
            "  SELECT AVG(active_hc) AS avg_active_hc\n"
            "  FROM org_monthly_hc\n"
            ")"
        ),
        (
            "org_avg AS (\n"
            "  SELECT\n"
            "    l.leaver_count * 100.0 / NULLIF(a.avg_active_hc, 0) AS org_attrition_pct\n"
            "  FROM org_leavers l\n"
            "  CROSS JOIN org_avg_hc a\n"
            ")"
        ),
    ])
    return (
        _with_ctes(ctes)
        + "SELECT\n"
        "  COALESCE(l.\"department\", a.\"department\") AS department,\n"
        "  COALESCE(l.leaver_count, 0) AS leavers,\n"
        "  COALESCE(a.avg_active_hc, 0) AS avg_active_hc,\n"
        "  ROUND(\n"
        "    COALESCE(l.leaver_count, 0) * 100.0 /\n"
        "    NULLIF(COALESCE(a.avg_active_hc, 0), 0),\n"
        "    2\n"
        "  ) AS dept_attrition_pct,\n"
        "  ROUND(o.org_attrition_pct, 2) AS org_avg_attrition_pct\n"
        "FROM leavers l\n"
        "FULL OUTER JOIN avg_hc a USING (\"department\")\n"
        "CROSS JOIN org_avg o\n"
        "ORDER BY dept_attrition_pct DESC\n"
        ";"
    )


def _q18_attrition_trend(table: str, f: dict) -> str:
    """Q18: Attrition trend month-on-month (last 12 months)."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    lob_f = _maybe_and_filter("lob", f.get("lob"))
    return (
        f"WITH monthly_leavers AS (\n"
        f"  SELECT\n"
        f"    DATE_TRUNC('month', \"lwd\") AS month,\n"
        f"    COUNT(DISTINCT \"empid\") AS leavers\n"
        f"  FROM {t}\n"
        f"  WHERE \"empstatus\" = 'Inactive'\n"
        f"    AND \"lwd\" >= (CURRENT_DATE - INTERVAL '12 months')\n"
        f"  {dept_f.strip()}\n"
        f"  {lob_f.strip()}\n"
        f"  GROUP BY DATE_TRUNC('month', \"lwd\")\n"
        f"),\n"
        f"monthly_hc AS (\n"
        f"  SELECT\n"
        f"    DATE_TRUNC('month', \"endofmonth\") AS month,\n"
        f"    COUNT(DISTINCT \"empid\") AS active_hc\n"
        f"  FROM {t}\n"
        f"  WHERE \"empstatus\" = 'Active'\n"
        f"    AND \"endofmonth\" >= (CURRENT_DATE - INTERVAL '12 months')\n"
        f"  {dept_f.strip()}\n"
        f"  {lob_f.strip()}\n"
        f"  GROUP BY DATE_TRUNC('month', \"endofmonth\")\n"
        f")\n"
        f"SELECT\n"
        f"  COALESCE(l.month, h.month) AS month,\n"
        f"  COALESCE(l.leavers, 0) AS leavers,\n"
        f"  COALESCE(h.active_hc, 0) AS active_hc,\n"
        f"  ROUND(\n"
        f"    COALESCE(l.leavers, 0) * 100.0 /\n"
        f"    NULLIF(COALESCE(h.active_hc, 0), 0),\n"
        f"    2\n"
        f"  ) AS monthly_attrition_pct\n"
        f"FROM monthly_leavers l\n"
        f"FULL OUTER JOIN monthly_hc h USING (month)\n"
        f"ORDER BY month\n"
        f";"
    )


def _q19_attrition_by_gender(table: str, f: dict) -> str:
    """Q19: Gender-wise attrition rate."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    ctes, _, _ = _attrition_params_ctes(
        table, f.get("month_start"), f.get("month_end")
    )
    ctes.extend([
        (
            "leavers AS (\n"
            "  SELECT \"gender\", COUNT(DISTINCT \"empid\") AS leaver_count\n"
            f"  FROM {t}\n"
            "  CROSS JOIN params p\n"
            "  WHERE \"empstatus\" = 'Inactive'\n"
            "    AND CAST(\"lwd\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
            f"{dept_f}"
            "  GROUP BY \"gender\"\n"
            ")"
        ),
        (
            "monthly_hc AS (\n"
            "  SELECT\n"
            "    DATE_TRUNC('month', \"endofmonth\") AS month,\n"
            "    \"gender\",\n"
            "    COUNT(DISTINCT \"empid\") AS active_hc\n"
            f"  FROM {t}\n"
            "  CROSS JOIN params p\n"
            "  WHERE \"empstatus\" = 'Active'\n"
            "    AND CAST(\"endofmonth\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
            f"{dept_f}"
            "  GROUP BY DATE_TRUNC('month', \"endofmonth\"), \"gender\"\n"
            ")"
        ),
        (
            "avg_hc AS (\n"
            "  SELECT \"gender\", ROUND(AVG(active_hc)::NUMERIC, 2) AS avg_active_hc\n"
            "  FROM monthly_hc\n"
            "  GROUP BY \"gender\"\n"
            ")"
        ),
    ])
    return (
        _with_ctes(ctes)
        + "SELECT\n"
        "  COALESCE(l.\"gender\", a.\"gender\") AS gender,\n"
        "  COALESCE(l.leaver_count, 0) AS leavers,\n"
        "  COALESCE(a.avg_active_hc, 0) AS avg_active_hc,\n"
        "  ROUND(\n"
        "    COALESCE(l.leaver_count, 0) * 100.0 /\n"
        "    NULLIF(COALESCE(a.avg_active_hc, 0), 0),\n"
        "    2\n"
        "  ) AS attrition_rate_pct\n"
        "FROM leavers l\n"
        "FULL OUTER JOIN avg_hc a USING (\"gender\")\n"
        "ORDER BY attrition_rate_pct DESC\n"
        ";"
    )


def _q20_attrition_notice_period(table: str, f: dict) -> str:
    """Q20: Employees in notice period (resigned, not yet left)."""
    t = _tbl(table)
    mgr_f = _maybe_and_filter("manager_id", f.get("manager_id"))
    return (
        f"SELECT\n"
        f"  \"empid\",\n"
        f"  \"resignation_date\",\n"
        f"  \"lwd\",\n"
        f"  (CAST(\"lwd\" AS DATE) - CURRENT_DATE) AS days_remaining,\n"
        f"  \"empstatus\"\n"
        f"FROM {t}\n"
        f"WHERE \"resignation_date\" IS NOT NULL\n"
        f"  AND CAST(\"lwd\" AS DATE) > CURRENT_DATE\n"
        f"{mgr_f}"
        f"ORDER BY \"lwd\"\n"
        f";"
    )


def _q21_attrition_first_year(table: str, f: dict) -> str:
    """Q21: First-year attrition rate (leaving within 12 months)."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    grade_f = _maybe_and_filter("grade", f.get("grade"))
    ctes, _, _ = _attrition_params_ctes(
        table, f.get("month_start"), f.get("month_end")
    )
    ctes.extend([
        (
            "total_leavers AS (\n"
            "  SELECT COUNT(DISTINCT \"empid\") AS total_count\n"
            f"  FROM {t}\n"
            "  CROSS JOIN params p\n"
            "  WHERE \"empstatus\" = 'Inactive'\n"
            "    AND CAST(\"lwd\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
            f"{dept_f}{grade_f}"
            ")"
        ),
        (
            "first_year_leavers AS (\n"
            "  SELECT COUNT(DISTINCT \"empid\") AS first_year_count\n"
            f"  FROM {t}\n"
            "  CROSS JOIN params p\n"
            "  WHERE \"empstatus\" = 'Inactive'\n"
            "    AND \"tenure_months\" <= 12\n"
            "    AND CAST(\"lwd\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
            f"{dept_f}{grade_f}"
            ")"
        ),
    ])
    return (
        _with_ctes(ctes)
        + "SELECT\n"
        "  f.first_year_count,\n"
        "  t.total_count AS total_leavers,\n"
        "  ROUND(\n"
        "    f.first_year_count * 100.0 / NULLIF(t.total_count, 0),\n"
        "    2\n"
        "  ) AS first_year_attrition_pct\n"
        "FROM first_year_leavers f, total_leavers t\n"
        ";"
    )


def _q22_attrition_exit_by_dept_grade(table: str, f: dict) -> str:
    """Q22: Exit reasons cross-tabbed by department or grade."""
    t = _tbl(table)
    ctes, _, _ = _attrition_params_ctes(
        table, f.get("month_start"), f.get("month_end")
    )
    return (
        _with_ctes(ctes)
        + f"SELECT\n"
        f"  \"department\",\n"
        f"  \"grade\",\n"
        f"  \"final_reason_of_exit\",\n"
        f"  \"final_exit_type\",\n"
        f"  COUNT(DISTINCT \"empid\") AS leaver_count\n"
        f"FROM {t}\n"
        f"CROSS JOIN params p\n"
        f"WHERE \"empstatus\" = 'Inactive'\n"
        f"  AND CAST(\"lwd\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
        f"GROUP BY \"department\", \"grade\", \"final_reason_of_exit\", \"final_exit_type\"\n"
        f"ORDER BY leaver_count DESC\n"
        f";"
    )


def _q23_attrition_new_hire(table: str, f: dict) -> str:
    """Q23: Attrition rate for new hires (joined in last 6 months)."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    grade_f = _maybe_and_filter("grade", f.get("grade"))
    ctes, _, _ = _attrition_params_ctes(
        table, f.get("month_start"), f.get("month_end")
    )
    ctes.extend([
        (
            "new_hire_leavers AS (\n"
            "  SELECT COUNT(DISTINCT \"empid\") AS leaver_count\n"
            f"  FROM {t}\n"
            "  CROSS JOIN params p\n"
            "  WHERE \"empstatus\" = 'Inactive'\n"
            "    AND \"tenure_months\" <= 6\n"
            "    AND CAST(\"lwd\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
            f"{dept_f}{grade_f}"
            ")"
        ),
        (
            "new_hires AS (\n"
            "  SELECT COUNT(DISTINCT \"empid\") AS hire_count\n"
            f"  FROM {t}\n"
            "  CROSS JOIN params p\n"
            "  WHERE \"newhireflag\" = 'Yes'\n"
            "    AND CAST(\"endofmonth\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
            f"{dept_f}{grade_f}"
            ")"
        ),
    ])
    return (
        _with_ctes(ctes)
        + "SELECT\n"
        "  n.hire_count AS new_hires,\n"
        "  l.leaver_count AS new_hire_leavers,\n"
        "  ROUND(\n"
        "    l.leaver_count * 100.0 / NULLIF(n.hire_count, 0),\n"
        "    2\n"
        "  ) AS new_hire_attrition_pct\n"
        "FROM new_hire_leavers l, new_hires n\n"
        ";"
    )


def _q24_attrition_perf_rating(table: str, f: dict) -> str:
    """Q24: Performance rating distribution of employees who left."""
    t = _tbl(table)
    mgr_f = _maybe_and_filter("manager_id", f.get("manager_id"))
    dept_f = _maybe_and_filter("department", f.get("department"))
    grade_f = _maybe_and_filter("grade", f.get("grade"))
    ctes, _, _ = _attrition_params_ctes(
        table, f.get("month_start"), f.get("month_end")
    )
    return (
        _with_ctes(ctes)
        + f"SELECT\n"
        f"  \"emprating\",\n"
        f"  COUNT(DISTINCT \"empid\") AS leaver_count\n"
        f"FROM {t}\n"
        f"CROSS JOIN params p\n"
        f"WHERE \"empstatus\" = 'Inactive'\n"
        f"  AND CAST(\"lwd\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
        f"{mgr_f}{dept_f}{grade_f}"
        f"GROUP BY \"emprating\"\n"
        f"ORDER BY leaver_count DESC\n"
        f";"
    )


def _q25_attrition_avg_tenure(table: str, f: dict) -> str:
    """Q25: Average tenure of employees who left."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    grade_f = _maybe_and_filter("grade", f.get("grade"))
    exit_f = _maybe_and_filter("final_exit_type", f.get("exit_type"))
    ctes, _, _ = _attrition_params_ctes(
        table, f.get("month_start"), f.get("month_end")
    )
    return (
        _with_ctes(ctes)
        + f"SELECT\n"
        f"  ROUND(AVG(\"tenure_months\"), 1) AS avg_tenure_months,\n"
        f"  ROUND(AVG(\"tenure_days\") / 365.0, 1) AS avg_tenure_years,\n"
        f"  COUNT(DISTINCT \"empid\") AS total_leavers\n"
        f"FROM {t}\n"
        f"CROSS JOIN params p\n"
        f"WHERE \"empstatus\" = 'Inactive'\n"
        f"  AND CAST(\"lwd\" AS DATE) BETWEEN p.fy_start AND p.month_end\n"
        f"{dept_f}{grade_f}{exit_f}"
        f";"
    )


def _q1a_hc_total_snapshot_current_month(table: str, f: dict) -> str:
    """Q1a: Total active headcount snapshot as of end of current month (auto-detect FY)."""
    t = _tbl(table)
    return (
        "WITH fy_params AS (\n"
        "  SELECT\n"
        "    CASE\n"
        "      WHEN EXTRACT('month' FROM CURRENT_DATE) >= 4\n"
        "      THEN (DATE_TRUNC('year', CURRENT_DATE) + INTERVAL '3 months')::DATE\n"
        "      ELSE (DATE_TRUNC('year', CURRENT_DATE) - INTERVAL '9 months')::DATE\n"
        "    END AS fy_start,\n"
        "    (DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day')::DATE AS month_end,\n"
        "    CASE\n"
        "      WHEN EXTRACT('month' FROM CURRENT_DATE) >= 4\n"
        "      THEN CONCAT('FY', EXTRACT('year' FROM CURRENT_DATE)::INT, '-',\n"
        "                  (EXTRACT('year' FROM CURRENT_DATE)::INT + 1) % 100)\n"
        "      ELSE CONCAT('FY', (EXTRACT('year' FROM CURRENT_DATE)::INT - 1), '-',\n"
        "                  EXTRACT('year' FROM CURRENT_DATE)::INT % 100)\n"
        "    END AS fy_label,\n"
        "    TO_CHAR(CURRENT_DATE, 'MMMM YYYY') AS current_month_label\n"
        ")\n"
        f"SELECT\n"
        f"  COUNT(DISTINCT e.\"empid\") AS active_headcount,\n"
        f"  fp.fy_label AS fiscal_year,\n"
        f"  fp.current_month_label AS current_month,\n"
        f"  fp.month_end AS as_of_date\n"
        f"FROM {t} e\n"
        f"CROSS JOIN fy_params fp\n"
        f"WHERE e.\"empstatus\" = 'Active'\n"
        f"  AND CAST(e.\"endofmonth\" AS DATE) = fp.month_end\n"
        f"GROUP BY fp.fy_label, fp.current_month_label, fp.month_end\n"
        f";"
    )


def _q1b_hc_total_snapshot_december_2025(table: str, f: dict) -> str:
    """Q1b: Total active headcount snapshot as of December 31, 2025."""
    t = _tbl(table)
    return (
        f"SELECT\n"
        f"  COUNT(DISTINCT \"empid\") AS active_headcount,\n"
        f"  'FY2025-26' AS fiscal_year,\n"
        f"  'December 2025' AS current_month,\n"
        f"  CAST('2025-12-31' AS DATE) AS as_of_date\n"
        f"FROM {t}\n"
        f"WHERE \"empstatus\" = 'Active'\n"
        f"  AND CAST(\"endofmonth\" AS DATE) = CAST('2025-12-31' AS DATE)\n"
        f";"
    )


def _q3a_hc_by_gender_current_month(table: str, f: dict) -> str:
    """Q3a: Headcount by gender as of end of current month (auto-detect FY)."""
    t = _tbl(table)
    return (
        "WITH fy_params AS (\n"
        "  SELECT\n"
        "    CASE\n"
        "      WHEN EXTRACT('month' FROM CURRENT_DATE) >= 4\n"
        "      THEN (DATE_TRUNC('year', CURRENT_DATE) + INTERVAL '3 months')::DATE\n"
        "      ELSE (DATE_TRUNC('year', CURRENT_DATE) - INTERVAL '9 months')::DATE\n"
        "    END AS fy_start,\n"
        "    (DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day')::DATE AS month_end,\n"
        "    CASE\n"
        "      WHEN EXTRACT('month' FROM CURRENT_DATE) >= 4\n"
        "      THEN CONCAT('FY', EXTRACT('year' FROM CURRENT_DATE)::INT, '-',\n"
        "                  (EXTRACT('year' FROM CURRENT_DATE)::INT + 1) % 100)\n"
        "      ELSE CONCAT('FY', (EXTRACT('year' FROM CURRENT_DATE)::INT - 1), '-',\n"
        "                  EXTRACT('year' FROM CURRENT_DATE)::INT % 100)\n"
        "    END AS fy_label,\n"
        "    TO_CHAR(CURRENT_DATE, 'MMMM YYYY') AS current_month_label\n"
        ")\n"
        f"SELECT\n"
        f"  COALESCE(e.\"gender\", 'Unknown') AS gender,\n"
        f"  COUNT(DISTINCT e.\"empid\") AS active_headcount,\n"
        f"  ROUND(\n"
        f"    COUNT(DISTINCT e.\"empid\") * 100.0 /\n"
        f"    SUM(COUNT(DISTINCT e.\"empid\")) OVER (),\n"
        f"    2\n"
        f"  ) AS percentage_of_total,\n"
        f"  fp.fy_label AS fiscal_year,\n"
        f"  fp.current_month_label AS current_month,\n"
        f"  fp.month_end AS as_of_date\n"
        f"FROM {t} e\n"
        f"CROSS JOIN fy_params fp\n"
        f"WHERE e.\"empstatus\" = 'Active'\n"
        f"  AND CAST(e.\"endofmonth\" AS DATE) = fp.month_end\n"
        f"GROUP BY e.\"gender\", fp.fy_label, fp.current_month_label, fp.month_end\n"
        f"ORDER BY active_headcount DESC, e.\"gender\"\n"
        f";"
    )


def _q3b_hc_by_gender_december_2025(table: str, f: dict) -> str:
    """Q3b: Headcount by gender as of December 31, 2025."""
    t = _tbl(table)
    return (
        f"SELECT\n"
        f"  COALESCE(\"gender\", 'Unknown') AS gender,\n"
        f"  COUNT(DISTINCT \"empid\") AS active_headcount,\n"
        f"  ROUND(\n"
        f"    COUNT(DISTINCT \"empid\") * 100.0 /\n"
        f"    SUM(COUNT(DISTINCT \"empid\")) OVER (),\n"
        f"    2\n"
        f"  ) AS percentage_of_total,\n"
        f"  'FY2025-26' AS fiscal_year,\n"
        f"  'December 2025' AS current_month,\n"
        f"  CAST('2025-12-31' AS DATE) AS as_of_date\n"
        f"FROM {t}\n"
        f"WHERE \"empstatus\" = 'Active'\n"
        f"  AND CAST(\"endofmonth\" AS DATE) = CAST('2025-12-31' AS DATE)\n"
        f"GROUP BY \"gender\"\n"
        f"ORDER BY active_headcount DESC, \"gender\"\n"
        f";"
    )


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

_TEMPLATE_MAP: Dict[str, object] = {
    "hc_total_snapshot": _q1_hc_total_snapshot,
    "hc_trend_mom": _q2_hc_trend_mom,
    "hc_by_gender": _q3_hc_by_gender,
    "hc_team": _q4_hc_team,
    "hc_by_grade": _q5_hc_by_grade,
    "hc_new_hires": _q6_hc_new_hires,
    "hc_by_tenure": _q7_hc_by_tenure,
    "hc_by_emp_type": _q8_hc_by_emp_type,
    "hc_by_business_group": _q9_hc_by_business_group,
    "hc_ic_pm_split": _q10_hc_ic_pm_split,
    "attrition_overall": _q11_attrition_overall,
    "attrition_vol_invol": _q12_attrition_vol_invol,
    "attrition_exit_reasons": _q13_attrition_exit_reasons,
    "attrition_team": _q14_attrition_team,
    "attrition_by_tenure": _q15_attrition_by_tenure,
    "attrition_by_grade": _q16_attrition_by_grade,
    "attrition_by_dept": _q17_attrition_by_dept,
    "attrition_trend": _q18_attrition_trend,
    "attrition_by_gender": _q19_attrition_by_gender,
    "attrition_notice_period": _q20_attrition_notice_period,
    "attrition_first_year": _q21_attrition_first_year,
    "attrition_exit_by_dept_grade": _q22_attrition_exit_by_dept_grade,
    "attrition_new_hire": _q23_attrition_new_hire,
    "attrition_perf_rating": _q24_attrition_perf_rating,
    "attrition_avg_tenure": _q25_attrition_avg_tenure,
    "hc_total_snapshot_current_month": _q1a_hc_total_snapshot_current_month,
    "hc_total_snapshot_december_2025": _q1b_hc_total_snapshot_december_2025,
    "hc_by_gender_current_month": _q3a_hc_by_gender_current_month,
    "hc_by_gender_december_2025": _q3b_hc_by_gender_december_2025,
}
