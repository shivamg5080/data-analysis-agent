"""
HC/Attrition Question Catalogue
================================
Loads the 25-question business catalogue from CSV and provides
intent classification via regex pattern matching.

Intent keys map 1-to-1 with catalogue question IDs (see INTENT_TO_ID).
"""

from __future__ import annotations

import csv
import logging
import os
import re
from datetime import date
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default catalogue path: repo root / catalogue.csv
_HERE = os.path.dirname(__file__)
DEFAULT_CATALOGUE_PATH = os.path.normpath(
    os.path.join(_HERE, "..", "..", "catalogue.csv")
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CatalogueEntry:
    """A single row from the HC/Attrition business catalogue."""

    id: int
    audience: str
    theme: str
    business_question: str
    question_type: str
    time_grain: str
    key_filters_slicers: List[str]
    metric_kpi: str
    source_columns_raw: List[str]
    fact_table: str
    dimension_tables: List[str]
    priority: str
    intent_key: str = field(default="")


# ---------------------------------------------------------------------------
# Intent patterns  (ordered: more specific patterns first)
# ---------------------------------------------------------------------------

INTENT_PATTERNS: Dict[str, List[re.Pattern]] = {
    # --- Headcount ---
    "hc_total_snapshot": [
        re.compile(r"total\s+(active\s+)?headcount", re.I),
        re.compile(r"(overall|current|active)\s+headcount", re.I),
        re.compile(r"headcount\s+(as\s+of|this\s+month|end\s+of|snapshot)", re.I),
        re.compile(r"how\s+many\s+(employees?|people|staff)\s+(are\s+there|do\s+we\s+have|currently)", re.I),
    ],
    "hc_trend_mom": [
        re.compile(r"headcount\s+(trend|trended|trending)", re.I),
        re.compile(r"headcount\s+(month.?on.?month|mom)", re.I),
        re.compile(r"(headcount|hc)\s+(over|for)\s+(last|past)\s+\d+\s+months?", re.I),
        re.compile(r"month.?on.?month\s+(headcount|hc)", re.I),
    ],
    "hc_by_gender": [
        re.compile(r"headcount\s+(by|split\s+by|breakdown\s+by)\s+gender", re.I),
        re.compile(r"gender\s+(ratio|split|breakdown|distribution|wise)\s+(headcount|hc)", re.I),
        re.compile(r"(male|female)\s+headcount", re.I),
    ],
    "hc_team": [
        re.compile(r"(my\s+team|team\s+headcount|headcount.{0,20}my\s+team)", re.I),
        re.compile(r"(how\s+many|count)\s+(people|employees?|staff)\s+(in|on)\s+my\s+(team|department)", re.I),
        re.compile(r"employees?\s+in\s+my\s+team", re.I),
    ],
    "hc_by_grade": [
        re.compile(r"headcount\s+(by|per)\s+grade", re.I),
        re.compile(r"headcount\s+(by|per)\s+job\s+level", re.I),
        re.compile(r"grade.{0,10}(headcount|hc)", re.I),
        re.compile(r"(headcount|hc)\s+by\s+level", re.I),
    ],
    "hc_new_hires": [
        re.compile(r"(new\s+hires?|new\s+joiners?)\s+(this\s+month|last\s+month|vs|comparison)", re.I),
        re.compile(r"how\s+many\s+(new\s+hires?|people|employees?)\s+joined", re.I),
        re.compile(r"joined\s+(this|last)\s+month", re.I),
    ],
    "hc_by_tenure": [
        re.compile(r"headcount\s+(by|per)\s+tenure", re.I),
        re.compile(r"tenure\s+(bucket|group|band).{0,10}(headcount|hc)", re.I),
    ],
    "hc_by_emp_type": [
        re.compile(r"headcount\s+(by|per)\s+(employee\s+type|emp\s+type|employment\s+type)", re.I),
        re.compile(r"(full.?time|contract|permanent).{0,10}(headcount|hc|split)", re.I),
        re.compile(r"(headcount|hc|split)\s+(by|per)\s+employee\s+type", re.I),
        re.compile(r"employee\s+type.{0,15}(headcount|hc|split)", re.I),
    ],
    "hc_by_business_group": [
        re.compile(r"headcount\s+(by|per|across)\s+(business\s+group|lob|line\s+of\s+business)", re.I),
        re.compile(r"(compare|how\s+does)\s+headcount.{0,20}(business\s+groups?|lob)", re.I),
        re.compile(r"headcount.{0,20}across\s+(business\s+groups?|lobs?)", re.I),
        re.compile(r"(business\s+groups?|lob).{0,15}headcount\s+(comparison|compare)", re.I),
    ],
    "hc_ic_pm_split": [
        re.compile(r"\b(ic|individual\s+contributor)\s+(vs|versus|and)\s+(pm|people\s+manager)", re.I),
        re.compile(r"\b(ic|pm)\s+split\b", re.I),
    ],
    # --- Attrition (specific dimension breakdowns checked BEFORE generic overall) ---
    "attrition_vol_invol": [
        re.compile(r"(voluntary|involuntary)\s+(vs\.?\s+)?(involuntary|voluntary)?\s+attrition", re.I),
        re.compile(r"attrition\s+(split|breakdown)\s+(by\s+)?(voluntary|involuntary|exit\s+type)", re.I),
        re.compile(r"vol\s+(vs|versus|and)\s+invol(untary)?\s+attrition", re.I),
        re.compile(r"(vol|invol)\s+attrition\s+split", re.I),
    ],
    "attrition_exit_reasons": [
        re.compile(r"(top|main|key|primary)\s+(exit|leaving|attrition)\s+reasons?", re.I),
        re.compile(r"why\s+(are\s+)?(employees?|people|staff)\s+(leaving|exiting|quitting)", re.I),
        re.compile(r"(exit|leaving|resignation)\s+reasons?\s+(breakdown|distribution|analysis)", re.I),
        re.compile(r"reasons?\s+(for|of|behind)\s+(attrition|leaving|exit|resignations?)", re.I),
        re.compile(r"top\s+reasons?\s+(employees?|people)\s+are\s+leaving", re.I),
    ],
    "attrition_team": [
        re.compile(r"(how\s+many|count).{0,20}(left|resigned|exited).{0,20}(my\s+team|my\s+department)", re.I),
        re.compile(r"(my\s+team|team)\s+attrition", re.I),
        re.compile(r"(left|leaving|resigned)\s+(from|in)\s+(my\s+team|my\s+department)", re.I),
        re.compile(r"attrition.{0,10}my\s+team", re.I),
    ],
    "attrition_by_tenure": [
        re.compile(r"attrition\s+rate?\s*(by|per|across)\s+tenure", re.I),
        re.compile(r"attrition.{0,15}by\s+tenure\s+bucket", re.I),
        re.compile(r"tenure\s+(bucket|group|band)\s+attrition", re.I),
        re.compile(r"attrition\s+(rate|%)\s+by\s+tenure", re.I),
    ],
    "attrition_by_grade": [
        re.compile(r"attrition\s+rate?\s*(by|per|across)\s+(grade|job\s+level)", re.I),
        re.compile(r"attrition.{0,15}by\s+grade", re.I),
        re.compile(r"(grade|job\s+level).{0,15}attrition\s+(rate|%)", re.I),
    ],
    "attrition_by_dept": [
        re.compile(r"attrition\s+rate?\s*(by|per|across|compare)\s+(department|dept|lob)", re.I),
        re.compile(r"attrition.{0,15}across\s+(department|dept)", re.I),
        re.compile(r"(department|dept).{0,15}attrition\s+(comparison|compare|rate|%)", re.I),
        re.compile(r"(compare|how\s+does)\s+attrition.{0,20}(department|dept|lob)", re.I),
        re.compile(r"attrition.{0,10}compare.{0,20}(department|dept)", re.I),
    ],
    "attrition_trend": [
        re.compile(r"attrition\s+(trend|trended|trending|month.?on.?month)", re.I),
        re.compile(r"(rolling|last)\s+(12\s+)?months?\s+attrition\s+(trend|rate)", re.I),
        re.compile(r"attrition\s+(over|for)\s+the\s+last\s+\d+\s+months?", re.I),
    ],
    "attrition_by_gender": [
        re.compile(r"(gender.?wise|gender.?based)\s+attrition", re.I),
        re.compile(r"attrition\s+(by|per)\s+gender", re.I),
        re.compile(r"(male|female)\s+attrition\s+(rate|%)", re.I),
    ],
    "attrition_notice_period": [
        re.compile(r"(in.?notice|notice.?period|serving\s+notice)", re.I),
        re.compile(r"(resigned|resignation)\s+(but\s+)?(not\s+yet\s+left|still\s+working|in\s+notice)", re.I),
        re.compile(r"employees?\s+(who\s+have\s+)?resigned\s+but\s+(have\s+)?not\s+(yet\s+)?left", re.I),
        re.compile(r"who\s+(have\s+)?resigned\s+and\s+(are\s+)?still\s+(working|here)", re.I),
    ],
    "attrition_first_year": [
        re.compile(r"(first.?year|1.?year|within\s+12\s+months?)\s+attrition", re.I),
        re.compile(r"attrition\s+(within|in)\s+(first\s+year|12\s+months?|one\s+year)", re.I),
        re.compile(r"employees?\s+(leaving|who\s+left)\s+(within|in)\s+(first\s+year|12\s+months?)", re.I),
    ],
    "attrition_exit_by_dept_grade": [
        re.compile(r"exit\s+reasons?\s+(differ|by|per|across)\s+(department|dept|grade)", re.I),
        re.compile(r"(department|grade)\s+(vs|and)\s+exit\s+reasons?", re.I),
        re.compile(r"exit\s+reasons?\s+(cross.?tab|matrix)", re.I),
        re.compile(r"exit\s+reasons?\s+differ\b", re.I),
        re.compile(r"do\s+exit\s+reasons?\s+differ", re.I),
    ],
    "attrition_new_hire": [
        re.compile(r"(new.?hire|recent\s+hire)\s+attrition", re.I),
        re.compile(r"attrition\s+(for|of)\s+new\s+hires?", re.I),
        re.compile(r"attrition\s+rate?\s*(for|of)\s+new\s+hires?", re.I),
        re.compile(r"employees?\s+joined\s+in\s+(last\s+)?6\s+months?\s+attrition", re.I),
        re.compile(r"attrition.{0,15}joined\s+in\s+(last\s+)?\d+\s+months?", re.I),
    ],
    "attrition_perf_rating": [
        re.compile(r"performance\s+ratings?\s+(of|for)\s+(employees?\s+who\s+left|leavers?)", re.I),
        re.compile(r"(leavers?|who\s+left).{0,20}performance\s+ratings?", re.I),
        re.compile(r"rating\s+distribution.{0,20}(left|resigned|exited)", re.I),
    ],
    "attrition_avg_tenure": [
        re.compile(r"(average|avg)\s+tenure\s+(of\s+)?(leavers?|employees?\s+who\s+left)", re.I),
        re.compile(r"(how\s+long).{0,20}(employees?\s+stay|stay\s+before\s+leaving)", re.I),
        re.compile(r"tenure\s+(of\s+)?(leavers?|who\s+left)", re.I),
    ],
    # --- Generic attrition overall (checked last so specific ones win) ---
    "attrition_overall": [
        re.compile(r"(overall|total|general)\s+attrition\s+(rate|%|percent|percentage)", re.I),
        # Rate this month / YTD (not followed by 'by' a dimension)
        re.compile(r"attrition\s+(rate|%|percent|percentage)\s+(this\s+month|ytd|year.to.date)", re.I),
        re.compile(r"(what\s+is|show\s+me|give\s+me)\s+(the\s+)?attrition\s+(rate|for)(?!\s+(?:new\s+hire|by|per|across))", re.I),
        re.compile(r"attrition\s+for\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\w+)\s+\d{4}", re.I),
        re.compile(r"attrition\s+(this\s+month|ytd|year.to.date)", re.I),
    ],
}

# Ordered list matching catalogue IDs 1–25
_INTENT_KEYS_ORDERED: List[str] = [
    "hc_total_snapshot",       # 1
    "hc_trend_mom",            # 2
    "hc_by_gender",            # 3
    "hc_team",                 # 4
    "hc_by_grade",             # 5
    "hc_new_hires",            # 6
    "hc_by_tenure",            # 7
    "hc_by_emp_type",          # 8
    "hc_by_business_group",    # 9
    "hc_ic_pm_split",          # 10
    "attrition_overall",       # 11
    "attrition_vol_invol",     # 12
    "attrition_exit_reasons",  # 13
    "attrition_team",          # 14
    "attrition_by_tenure",     # 15
    "attrition_by_grade",      # 16
    "attrition_by_dept",       # 17
    "attrition_trend",         # 18
    "attrition_by_gender",     # 19
    "attrition_notice_period", # 20
    "attrition_first_year",    # 21
    "attrition_exit_by_dept_grade",  # 22
    "attrition_new_hire",      # 23
    "attrition_perf_rating",   # 24
    "attrition_avg_tenure",    # 25
]

INTENT_TO_ID: Dict[str, int] = {
    key: idx + 1 for idx, key in enumerate(_INTENT_KEYS_ORDERED)
}


# ---------------------------------------------------------------------------
# Catalogue loader
# ---------------------------------------------------------------------------

def load_catalogue(path: str = DEFAULT_CATALOGUE_PATH) -> Dict[int, CatalogueEntry]:
    """Load HC/Attrition question catalogue from a CSV file.

    Parameters
    ----------
    path:
        Path to the catalogue CSV.  Falls back gracefully if the file is missing.

    Returns
    -------
    ``dict[question_id, CatalogueEntry]``
    """
    entries: Dict[int, CatalogueEntry] = {}

    if not os.path.isfile(path):
        logger.warning("Catalogue CSV not found at %s — using empty catalogue", path)
        return entries

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                q_id = int(row["id"])
                intent_key = (
                    _INTENT_KEYS_ORDERED[q_id - 1]
                    if 1 <= q_id <= len(_INTENT_KEYS_ORDERED)
                    else f"q{q_id}"
                )
                entry = CatalogueEntry(
                    id=q_id,
                    audience=row.get("audience", ""),
                    theme=row.get("theme", ""),
                    business_question=row.get("business_question", ""),
                    question_type=row.get("question_type", ""),
                    time_grain=row.get("time_grain", ""),
                    key_filters_slicers=[
                        f.strip()
                        for f in row.get("key_filters_slicers", "").split("|")
                        if f.strip()
                    ],
                    metric_kpi=row.get("metric_kpi", ""),
                    source_columns_raw=[
                        c.strip()
                        for c in row.get("source_columns_raw", "").split("|")
                        if c.strip()
                    ],
                    fact_table=row.get("fact_table", ""),
                    dimension_tables=[
                        t.strip()
                        for t in row.get("dimension_tables", "").split("|")
                        if t.strip()
                    ],
                    priority=row.get("priority", "P2"),
                    intent_key=intent_key,
                )
                entries[q_id] = entry
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping catalogue row %s — %s", row, exc)

    logger.info("Loaded %d catalogue entries from %s", len(entries), path)
    return entries


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

def classify_intent(query: str) -> Tuple[Optional[str], float]:
    """Match *query* against known intent patterns.

    Returns
    -------
    ``(intent_key, confidence)``
        *confidence* is in ``[0.0, 1.0]``.
        Returns ``(None, 0.0)`` when no pattern matches.

    Notes
    -----
    Confidence is boosted by:
    - the proportion of the query that the match covers, and
    - whether the match is longer (more specific).

    The ``INTENT_PATTERNS`` dict is ordered so that more specific dimension
    breakdowns (by_grade, by_tenure, etc.) appear before the generic
    ``attrition_overall`` bucket.  When two intents score equally the first
    one encountered wins, so the ordering matters.
    """
    query_lower = query.lower().strip()
    if not query_lower:
        return None, 0.0

    best_intent: Optional[str] = None
    best_score: float = 0.0

    for intent_key, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            m = pattern.search(query_lower)
            if m:
                match_len = len(m.group(0))
                query_len = max(len(query_lower), 1)
                # Base confidence of 0.75, boosted by match length ratio up to 0.25
                score = min(0.75 + (match_len / query_len) * 0.25, 1.0)
                # Use > (strictly greater) so first match at equal score wins.
                # This ensures more specific intents (ordered first) are preferred
                # over the generic attrition_overall bucket.
                if score > best_score:
                    best_score = score
                    best_intent = intent_key

    return best_intent, best_score


# ---------------------------------------------------------------------------
# Filter extraction
# ---------------------------------------------------------------------------

def extract_filters(query: str, intent_key: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Extract dimension filters from a natural-language *query*.

    Returns a dict with keys matching common HC/Attrition dimensions.
    Values are ``None`` when the dimension is not found in the query.
    """
    from agent.hc_attrition.fy_helper import (
        parse_month_year,
        parse_month_only,
        parse_specific_date,
        parse_fy_range,
        parse_full_year,
        month_range,
    )

    filters: Dict[str, Optional[str]] = {
        "department": None,
        "grade": None,
        "lob": None,
        "businessgroup": None,
        "manager_id": None,
        "gender": None,
        "month_start": None,   # ISO format: YYYY-MM-DD
        "month_end": None,     # ISO format: YYYY-MM-DD
        "year": None,
        "tenure_bucket": None,
        "exit_type": None,
    }

    # Department
    dept_m = re.search(
        r"(?:department|dept)\s*(?:is\s+|=\s*|:\s*)?[\"']?(\w[\w\s\-]{0,29}?)[\"']?"
        r"(?=\s+(?:and|or|,|\.|$)|\s+\w+\s+(?:department|dept|grade|lob)|$)",
        query, re.I
    )
    if dept_m:
        filters["department"] = dept_m.group(1).strip()

    # Grade
    grade_m = re.search(
        r"(?:grade|level)\s*(?:is\s+|=\s*|:\s*)?[\"']?(\w[\w\s\-]{0,19}?)[\"']?"
        r"(?=\s+(?:and|or|,|\.|$)|$)",
        query, re.I
    )
    if grade_m:
        filters["grade"] = grade_m.group(1).strip()

    # Gender
    gender_m = re.search(r"\b(male|female)\b", query, re.I)
    if gender_m:
        filters["gender"] = gender_m.group(1).lower()

    is_attrition = bool(
        (intent_key and intent_key.startswith("attrition_"))
        or re.search(r"\battrition\b", query, re.I)
    )

    # Explicit date (YYYY-MM-DD)
    explicit_date = parse_specific_date(query)
    if explicit_date:
        first, last = month_range(explicit_date.year, explicit_date.month)
        filters["month_start"] = first.isoformat()
        filters["month_end"] = last.isoformat()
        filters["year"] = str(explicit_date.year)

    # Month + year
    if not filters["month_start"]:
        parsed = parse_month_year(query)
        if parsed:
            filters["month_start"] = parsed[0].isoformat()
            filters["month_end"] = parsed[1].isoformat()
            filters["year"] = str(parsed[0].year)

    # FY / full year for attrition queries
    if not filters["month_start"] and is_attrition:
        fy_range = parse_fy_range(query)
        if fy_range:
            filters["month_start"] = fy_range[0].isoformat()
            filters["month_end"] = fy_range[1].isoformat()
            filters["year"] = str(fy_range[0].year)

    if not filters["month_start"] and is_attrition:
        full_year = parse_full_year(query)
        if full_year:
            filters["month_start"] = full_year[0].isoformat()
            filters["month_end"] = full_year[1].isoformat()
            filters["year"] = str(full_year[1].year)

    # Month only (use latest available year)
    if not filters["month_start"]:
        parsed = parse_month_only(query)
        if parsed:
            filters["month_start"] = parsed[0].isoformat()
            filters["month_end"] = parsed[1].isoformat()
            filters["year"] = str(parsed[0].year)

    # Year only (use current month of that year)
    if not filters["month_start"]:
        year_m = re.search(r"\b(20\d{2})\b", query)
        if year_m:
            ref_year = int(year_m.group(1))
            ref_month = date.today().month
            first, last = month_range(ref_year, ref_month)
            filters["month_start"] = first.isoformat()
            filters["month_end"] = last.isoformat()
            filters["year"] = str(ref_year)

    # Exit type
    if re.search(r"\bvoluntary\b", query, re.I):
        filters["exit_type"] = "Voluntary"
    elif re.search(r"\binvoluntary\b", query, re.I):
        filters["exit_type"] = "Involuntary"

    # LOB
    lob_m = re.search(
        r"(?:lob|line\s+of\s+business)\s*(?:is\s+|=\s*|:\s*)?[\"']?(\w[\w\s\-]{0,29}?)[\"']?"
        r"(?=\s+(?:and|or|,|\.|$)|\s+\w+\s+(?:department|dept|grade|lob)|$)",
        query, re.I
    )
    if lob_m:
        filters["lob"] = lob_m.group(1).strip()

    # Business group
    bg_m = re.search(
        r"business\s+group\s*(?:is\s+|=\s*|:\s*)?[\"']?(\w[\w\s\-]{0,29}?)[\"']?"
        r"(?=\s+(?:and|or|,|\.|$)|$)",
        query, re.I
    )
    if bg_m:
        filters["businessgroup"] = bg_m.group(1).strip()

    # Manager ID (simple digit or alphanumeric code)
    mgr_m = re.search(r"manager(?:\s+id)?\s*[=:]\s*[\"']?(\w+)[\"']?", query, re.I)
    if mgr_m:
        filters["manager_id"] = mgr_m.group(1)

    return filters
