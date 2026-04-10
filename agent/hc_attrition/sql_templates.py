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
    if not start or not end:
        return ""
    # Validate ISO-8601 date format to prevent injection
    _iso = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    if not (_iso.match(start) and _iso.match(end)):
        return ""
    return f"  AND {_col(col)} BETWEEN '{start}' AND '{end}'\n"


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
    date_f = _maybe_date_filter("endofmonth", f.get("month_start"), f.get("month_end"))
    return (
        f"SELECT\n"
        f"  \"lob\",\n"
        f"  \"businessgroup\",\n"
        f"  COUNT(DISTINCT \"empid\") AS headcount\n"
        f"FROM {t}\n"
        f"WHERE \"empstatus\" = 'Active'\n"
        f"{lob_f}{date_f}"
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
    date_f = _maybe_date_filter("lwd", f.get("month_start"), f.get("month_end"))

    if not date_f:
        # Default: current calendar month
        date_f = "  AND DATE_TRUNC('month', \"lwd\") = DATE_TRUNC('month', CURRENT_DATE)\n"

    return (
        f"WITH leavers AS (\n"
        f"  SELECT COUNT(DISTINCT \"empid\") AS leaver_count\n"
        f"  FROM {t}\n"
        f"  WHERE \"empstatus\" = 'Inactive'\n"
        f"  {dept_f.strip()}\n"
        f"  {lob_f.strip()}\n"
        f"  {grade_f.strip()}\n"
        f"  {date_f.strip()}\n"
        f"),\n"
        f"avg_hc AS (\n"
        f"  SELECT COUNT(DISTINCT \"empid\") AS total_active\n"
        f"  FROM {t}\n"
        f"  WHERE \"empstatus\" = 'Active'\n"
        f"  {dept_f.strip()}\n"
        f"  {lob_f.strip()}\n"
        f"  {grade_f.strip()}\n"
        f")\n"
        f"SELECT\n"
        f"  l.leaver_count,\n"
        f"  a.total_active,\n"
        f"  ROUND(\n"
        f"    l.leaver_count * 100.0 / NULLIF(a.total_active + l.leaver_count / 2.0, 0),\n"
        f"    2\n"
        f"  ) AS attrition_rate_pct\n"
        f"FROM leavers l, avg_hc a\n"
        f";"
    )


def _q12_attrition_vol_invol(table: str, f: dict) -> str:
    """Q12: Voluntary vs involuntary attrition split."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    grade_f = _maybe_and_filter("grade", f.get("grade"))
    date_f = _maybe_date_filter("lwd", f.get("month_start"), f.get("month_end"))
    return (
        f"SELECT\n"
        f"  \"final_exit_type\",\n"
        f"  COUNT(DISTINCT \"empid\") AS leaver_count,\n"
        f"  ROUND(\n"
        f"    COUNT(DISTINCT \"empid\") * 100.0 /\n"
        f"    SUM(COUNT(DISTINCT \"empid\")) OVER (),\n"
        f"    2\n"
        f"  ) AS pct_of_total\n"
        f"FROM {t}\n"
        f"WHERE \"empstatus\" = 'Inactive'\n"
        f"{dept_f}{grade_f}{date_f}"
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
    date_f = _maybe_date_filter("lwd", f.get("month_start"), f.get("month_end"))
    return (
        f"SELECT\n"
        f"  \"final_reason_of_exit\",\n"
        f"  \"final_exit_type\",\n"
        f"  COUNT(DISTINCT \"empid\") AS leaver_count\n"
        f"FROM {t}\n"
        f"WHERE \"empstatus\" = 'Inactive'\n"
        f"{dept_f}{grade_f}{exit_f}{date_f}"
        f"GROUP BY \"final_reason_of_exit\", \"final_exit_type\"\n"
        f"ORDER BY leaver_count DESC\n"
        f"LIMIT 20\n"
        f";"
    )


def _q14_attrition_team(table: str, f: dict) -> str:
    """Q14: Attrition for a specific manager's team."""
    t = _tbl(table)
    mgr_f = _maybe_and_filter("manager_id", f.get("manager_id"))
    date_f = _maybe_date_filter("lwd", f.get("month_start"), f.get("month_end"))

    if not date_f:
        date_f = "  AND DATE_TRUNC('month', \"lwd\") = DATE_TRUNC('month', CURRENT_DATE)\n"

    return (
        f"SELECT\n"
        f"  \"manager_name\",\n"
        f"  COUNT(DISTINCT \"empid\") AS leavers\n"
        f"FROM {t}\n"
        f"WHERE \"empstatus\" = 'Inactive'\n"
        f"{mgr_f}{date_f}"
        f"GROUP BY \"manager_name\"\n"
        f"ORDER BY leavers DESC\n"
        f";"
    )


def _q15_attrition_by_tenure(table: str, f: dict) -> str:
    """Q15: Attrition rate by tenure bucket."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    date_f = _maybe_date_filter("lwd", f.get("month_start"), f.get("month_end"))
    return (
        f"WITH leavers AS (\n"
        f"  SELECT \"tenure_bucket\", COUNT(DISTINCT \"empid\") AS leaver_count\n"
        f"  FROM {t}\n"
        f"  WHERE \"empstatus\" = 'Inactive'\n"
        f"  {dept_f.strip()}\n"
        f"  {date_f.strip()}\n"
        f"  GROUP BY \"tenure_bucket\"\n"
        f"),\n"
        f"total_hc AS (\n"
        f"  SELECT \"tenure_bucket\", COUNT(DISTINCT \"empid\") AS active_count\n"
        f"  FROM {t}\n"
        f"  WHERE \"empstatus\" = 'Active'\n"
        f"  {dept_f.strip()}\n"
        f"  GROUP BY \"tenure_bucket\"\n"
        f")\n"
        f"SELECT\n"
        f"  COALESCE(l.\"tenure_bucket\", h.\"tenure_bucket\") AS tenure_bucket,\n"
        f"  COALESCE(l.leaver_count, 0) AS leavers,\n"
        f"  COALESCE(h.active_count, 0) AS active_hc,\n"
        f"  ROUND(\n"
        f"    COALESCE(l.leaver_count, 0) * 100.0 /\n"
        f"    NULLIF(COALESCE(h.active_count, 0) + COALESCE(l.leaver_count, 0), 0),\n"
        f"    2\n"
        f"  ) AS attrition_rate_pct\n"
        f"FROM leavers l\n"
        f"FULL OUTER JOIN total_hc h USING (\"tenure_bucket\")\n"
        f"ORDER BY attrition_rate_pct DESC\n"
        f";"
    )


def _q16_attrition_by_grade(table: str, f: dict) -> str:
    """Q16: Attrition rate by grade/job level."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    date_f = _maybe_date_filter("lwd", f.get("month_start"), f.get("month_end"))
    return (
        f"WITH leavers AS (\n"
        f"  SELECT \"grade\", COUNT(DISTINCT \"empid\") AS leaver_count\n"
        f"  FROM {t}\n"
        f"  WHERE \"empstatus\" = 'Inactive'\n"
        f"  {dept_f.strip()}\n"
        f"  {date_f.strip()}\n"
        f"  GROUP BY \"grade\"\n"
        f"),\n"
        f"total_hc AS (\n"
        f"  SELECT \"grade\", COUNT(DISTINCT \"empid\") AS active_count\n"
        f"  FROM {t}\n"
        f"  WHERE \"empstatus\" = 'Active'\n"
        f"  {dept_f.strip()}\n"
        f"  GROUP BY \"grade\"\n"
        f")\n"
        f"SELECT\n"
        f"  COALESCE(l.\"grade\", h.\"grade\") AS grade,\n"
        f"  COALESCE(l.leaver_count, 0) AS leavers,\n"
        f"  COALESCE(h.active_count, 0) AS active_hc,\n"
        f"  ROUND(\n"
        f"    COALESCE(l.leaver_count, 0) * 100.0 /\n"
        f"    NULLIF(COALESCE(h.active_count, 0) + COALESCE(l.leaver_count, 0), 0),\n"
        f"    2\n"
        f"  ) AS attrition_rate_pct\n"
        f"FROM leavers l\n"
        f"FULL OUTER JOIN total_hc h USING (\"grade\")\n"
        f"ORDER BY attrition_rate_pct DESC\n"
        f";"
    )


def _q17_attrition_by_dept(table: str, f: dict) -> str:
    """Q17: Attrition comparison across departments."""
    t = _tbl(table)
    lob_f = _maybe_and_filter("lob", f.get("lob"))
    date_f = _maybe_date_filter("lwd", f.get("month_start"), f.get("month_end"))
    return (
        f"WITH leavers AS (\n"
        f"  SELECT \"department\", COUNT(DISTINCT \"empid\") AS leaver_count\n"
        f"  FROM {t}\n"
        f"  WHERE \"empstatus\" = 'Inactive'\n"
        f"  {lob_f.strip()}\n"
        f"  {date_f.strip()}\n"
        f"  GROUP BY \"department\"\n"
        f"),\n"
        f"total_hc AS (\n"
        f"  SELECT \"department\", COUNT(DISTINCT \"empid\") AS active_count\n"
        f"  FROM {t}\n"
        f"  WHERE \"empstatus\" = 'Active'\n"
        f"  {lob_f.strip()}\n"
        f"  GROUP BY \"department\"\n"
        f"),\n"
        f"org_avg AS (\n"
        f"  SELECT\n"
        f"    SUM(l.leaver_count) * 100.0 /\n"
        f"    NULLIF(SUM(h.active_count) + SUM(l.leaver_count), 0) AS org_attrition_pct\n"
        f"  FROM leavers l\n"
        f"  JOIN total_hc h USING (\"department\")\n"
        f")\n"
        f"SELECT\n"
        f"  COALESCE(l.\"department\", h.\"department\") AS department,\n"
        f"  COALESCE(l.leaver_count, 0) AS leavers,\n"
        f"  COALESCE(h.active_count, 0) AS active_hc,\n"
        f"  ROUND(\n"
        f"    COALESCE(l.leaver_count, 0) * 100.0 /\n"
        f"    NULLIF(COALESCE(h.active_count, 0) + COALESCE(l.leaver_count, 0), 0),\n"
        f"    2\n"
        f"  ) AS dept_attrition_pct,\n"
        f"  ROUND(o.org_attrition_pct, 2) AS org_avg_attrition_pct\n"
        f"FROM leavers l\n"
        f"FULL OUTER JOIN total_hc h USING (\"department\")\n"
        f"CROSS JOIN org_avg o\n"
        f"ORDER BY dept_attrition_pct DESC\n"
        f";"
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
    date_f = _maybe_date_filter("lwd", f.get("month_start"), f.get("month_end"))
    return (
        f"WITH leavers AS (\n"
        f"  SELECT \"gender\", COUNT(DISTINCT \"empid\") AS leaver_count\n"
        f"  FROM {t}\n"
        f"  WHERE \"empstatus\" = 'Inactive'\n"
        f"  {dept_f.strip()}\n"
        f"  {date_f.strip()}\n"
        f"  GROUP BY \"gender\"\n"
        f"),\n"
        f"active_hc AS (\n"
        f"  SELECT \"gender\", COUNT(DISTINCT \"empid\") AS active_count\n"
        f"  FROM {t}\n"
        f"  WHERE \"empstatus\" = 'Active'\n"
        f"  {dept_f.strip()}\n"
        f"  GROUP BY \"gender\"\n"
        f")\n"
        f"SELECT\n"
        f"  COALESCE(l.\"gender\", h.\"gender\") AS gender,\n"
        f"  COALESCE(l.leaver_count, 0) AS leavers,\n"
        f"  COALESCE(h.active_count, 0) AS active_hc,\n"
        f"  ROUND(\n"
        f"    COALESCE(l.leaver_count, 0) * 100.0 /\n"
        f"    NULLIF(COALESCE(h.active_count, 0) + COALESCE(l.leaver_count, 0), 0),\n"
        f"    2\n"
        f"  ) AS attrition_rate_pct\n"
        f"FROM leavers l\n"
        f"FULL OUTER JOIN active_hc h USING (\"gender\")\n"
        f"ORDER BY attrition_rate_pct DESC\n"
        f";"
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
    date_f = _maybe_date_filter("lwd", f.get("month_start"), f.get("month_end"))
    return (
        f"WITH total_leavers AS (\n"
        f"  SELECT COUNT(DISTINCT \"empid\") AS total_count\n"
        f"  FROM {t}\n"
        f"  WHERE \"empstatus\" = 'Inactive'\n"
        f"  {dept_f.strip()}\n"
        f"  {grade_f.strip()}\n"
        f"  {date_f.strip()}\n"
        f"),\n"
        f"first_year_leavers AS (\n"
        f"  SELECT COUNT(DISTINCT \"empid\") AS first_year_count\n"
        f"  FROM {t}\n"
        f"  WHERE \"empstatus\" = 'Inactive'\n"
        f"    AND \"tenure_months\" <= 12\n"
        f"  {dept_f.strip()}\n"
        f"  {grade_f.strip()}\n"
        f"  {date_f.strip()}\n"
        f")\n"
        f"SELECT\n"
        f"  f.first_year_count,\n"
        f"  t.total_count AS total_leavers,\n"
        f"  ROUND(\n"
        f"    f.first_year_count * 100.0 / NULLIF(t.total_count, 0),\n"
        f"    2\n"
        f"  ) AS first_year_attrition_pct\n"
        f"FROM first_year_leavers f, total_leavers t\n"
        f";"
    )


def _q22_attrition_exit_by_dept_grade(table: str, f: dict) -> str:
    """Q22: Exit reasons cross-tabbed by department or grade."""
    t = _tbl(table)
    date_f = _maybe_date_filter("lwd", f.get("month_start"), f.get("month_end"))
    return (
        f"SELECT\n"
        f"  \"department\",\n"
        f"  \"grade\",\n"
        f"  \"final_reason_of_exit\",\n"
        f"  \"final_exit_type\",\n"
        f"  COUNT(DISTINCT \"empid\") AS leaver_count\n"
        f"FROM {t}\n"
        f"WHERE \"empstatus\" = 'Inactive'\n"
        f"{date_f}"
        f"GROUP BY \"department\", \"grade\", \"final_reason_of_exit\", \"final_exit_type\"\n"
        f"ORDER BY leaver_count DESC\n"
        f";"
    )


def _q23_attrition_new_hire(table: str, f: dict) -> str:
    """Q23: Attrition rate for new hires (joined in last 6 months)."""
    t = _tbl(table)
    dept_f = _maybe_and_filter("department", f.get("department"))
    grade_f = _maybe_and_filter("grade", f.get("grade"))
    date_f = _maybe_date_filter("endofmonth", f.get("month_start"), f.get("month_end"))
    return (
        f"WITH new_hire_leavers AS (\n"
        f"  SELECT COUNT(DISTINCT \"empid\") AS leaver_count\n"
        f"  FROM {t}\n"
        f"  WHERE \"empstatus\" = 'Inactive'\n"
        f"    AND \"tenure_months\" <= 6\n"
        f"  {dept_f.strip()}\n"
        f"  {grade_f.strip()}\n"
        f"  {date_f.strip()}\n"
        f"),\n"
        f"new_hires AS (\n"
        f"  SELECT COUNT(DISTINCT \"empid\") AS hire_count\n"
        f"  FROM {t}\n"
        f"  WHERE \"newhireflag\" = 'Yes'\n"
        f"  {dept_f.strip()}\n"
        f"  {grade_f.strip()}\n"
        f"  {date_f.strip()}\n"
        f")\n"
        f"SELECT\n"
        f"  n.hire_count AS new_hires,\n"
        f"  l.leaver_count AS new_hire_leavers,\n"
        f"  ROUND(\n"
        f"    l.leaver_count * 100.0 / NULLIF(n.hire_count, 0),\n"
        f"    2\n"
        f"  ) AS new_hire_attrition_pct\n"
        f"FROM new_hire_leavers l, new_hires n\n"
        f";"
    )


def _q24_attrition_perf_rating(table: str, f: dict) -> str:
    """Q24: Performance rating distribution of employees who left."""
    t = _tbl(table)
    mgr_f = _maybe_and_filter("manager_id", f.get("manager_id"))
    dept_f = _maybe_and_filter("department", f.get("department"))
    grade_f = _maybe_and_filter("grade", f.get("grade"))
    date_f = _maybe_date_filter("lwd", f.get("month_start"), f.get("month_end"))
    return (
        f"SELECT\n"
        f"  \"emprating\",\n"
        f"  COUNT(DISTINCT \"empid\") AS leaver_count\n"
        f"FROM {t}\n"
        f"WHERE \"empstatus\" = 'Inactive'\n"
        f"{mgr_f}{dept_f}{grade_f}{date_f}"
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
    date_f = _maybe_date_filter("lwd", f.get("month_start"), f.get("month_end"))
    return (
        f"SELECT\n"
        f"  ROUND(AVG(\"tenure_months\"), 1) AS avg_tenure_months,\n"
        f"  ROUND(AVG(\"tenure_days\") / 365.0, 1) AS avg_tenure_years,\n"
        f"  COUNT(DISTINCT \"empid\") AS total_leavers\n"
        f"FROM {t}\n"
        f"WHERE \"empstatus\" = 'Inactive'\n"
        f"{dept_f}{grade_f}{exit_f}{date_f}"
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
}
