"""
Financial Year Helper Utilities
================================
Provides April–March financial year computations for HC/Attrition analytics.

All helpers are pure functions (no side effects) for easy unit testing.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Optional, Tuple

# Indian FY starts in April
FY_START_MONTH = 4  # April
# If base_year - parsed_year exceeds this, assume the next century (e.g., 1999 → 00).
YEAR_ROLLOVER_THRESHOLD = 50


# ---------------------------------------------------------------------------
# Core FY helpers
# ---------------------------------------------------------------------------

def fy_start(reference_date: date) -> date:
    """Return April 1 of the FY that contains *reference_date*.

    Examples
    --------
    >>> fy_start(date(2025, 6, 15))
    date(2025, 4, 1)
    >>> fy_start(date(2025, 2, 10))
    date(2024, 4, 1)
    """
    if reference_date.month >= FY_START_MONTH:
        return date(reference_date.year, FY_START_MONTH, 1)
    return date(reference_date.year - 1, FY_START_MONTH, 1)


def fy_end(reference_date: date) -> date:
    """Return March 31 of the FY that contains *reference_date*.

    Examples
    --------
    >>> fy_end(date(2025, 6, 15))
    date(2026, 3, 31)
    >>> fy_end(date(2025, 2, 10))
    date(2025, 3, 31)
    """
    start = fy_start(reference_date)
    return date(start.year + 1, 3, 31)


def fy_for_date(reference_date: date) -> Tuple[date, date]:
    """Return ``(fy_start, fy_end)`` for the FY containing *reference_date*."""
    return fy_start(reference_date), fy_end(reference_date)


def fy_label(reference_date: date) -> str:
    """Return a human-readable FY label such as ``'FY2025-26'``.

    Examples
    --------
    >>> fy_label(date(2025, 6, 15))
    'FY2025-26'
    """
    start = fy_start(reference_date)
    return f"FY{start.year}-{str(start.year + 1)[-2:]}"


# ---------------------------------------------------------------------------
# Month name lookup
# ---------------------------------------------------------------------------

_MONTHS: dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_MONTH_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
    r"|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?"
    r"|nov(?:ember)?|dec(?:ember)?)\b[\s\-/]*(\d{4})",
    re.IGNORECASE,
)
_MONTH_ONLY_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
    r"|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?"
    r"|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
_FULL_DATE_RE = re.compile(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b")
_FY_RE = re.compile(r"\bFY\s*([0-9]{2,4})\s*[-/–]\s*([0-9]{2,4})\b", re.IGNORECASE)
_FULL_YEAR_RE = re.compile(r"\bfull\s+year\s+(20\d{2})\b", re.IGNORECASE)


def parse_month_year(text: str) -> Optional[Tuple[date, date]]:
    """Parse a natural-language month/year reference from *text*.

    Supports formats like "June 2025", "Jun 2025", "June-2025", "06/2025".

    Returns
    -------
    ``(first_day, last_day)`` for the matched month, or ``None`` if no match.

    Examples
    --------
    >>> parse_month_year("show me attrition for June 2025")
    (date(2025, 6, 1), date(2025, 6, 30))
    """
    # Try named-month pattern first
    match = _MONTH_RE.search(text)
    if match:
        month_str = match.group(1).lower()
        year = int(match.group(2))
        month = _MONTHS.get(month_str)
        if month and 2000 <= year <= 2100:
            first = date(year, month, 1)
            last = _last_day_of_month(year, month)
            return first, last

    # Try numeric MM/YYYY or YYYY-MM
    num_match = re.search(
        r"(?:(\d{1,2})[/\-](\d{4})|(\d{4})[/\-](\d{1,2}))", text
    )
    if num_match:
        if num_match.group(1):
            month, year = int(num_match.group(1)), int(num_match.group(2))
        else:
            year, month = int(num_match.group(3)), int(num_match.group(4))
        if 1 <= month <= 12 and 2000 <= year <= 2100:
            first = date(year, month, 1)
            last = _last_day_of_month(year, month)
            return first, last

    return None


def parse_month_only(text: str, reference_date: Optional[date] = None) -> Optional[Tuple[date, date]]:
    """Parse a month-only reference (e.g., "June") and infer the year."""
    match = _MONTH_ONLY_RE.search(text)
    if not match:
        return None
    month_str = match.group(1).lower()
    month = _MONTHS.get(month_str)
    if not month:
        return None
    if reference_date is None:
        reference_date = date.today()
    year = reference_date.year
    if month > reference_date.month:
        # Month is in the future relative to the reference date; use latest available year.
        year -= 1
    first = date(year, month, 1)
    last = _last_day_of_month(year, month)
    return first, last


def parse_specific_date(text: str) -> Optional[date]:
    """Parse a YYYY-MM-DD date from the text."""
    match = _FULL_DATE_RE.search(text)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def parse_fy_range(text: str) -> Optional[Tuple[date, date]]:
    """Parse FY strings like 'FY2025-26' and return the FY start/end dates."""
    match = _FY_RE.search(text)
    if not match:
        return None
    start_year = _normalize_year(match.group(1))
    end_year = _normalize_year(match.group(2), start_year)
    return date(start_year, 4, 1), date(end_year, 3, 31)


def parse_full_year(text: str) -> Optional[Tuple[date, date]]:
    """Parse 'full year 2025' to FY 2024-25 (Apr 1 2024 - Mar 31 2025)."""
    match = _FULL_YEAR_RE.search(text)
    if not match:
        return None
    year = int(match.group(1))
    return date(year - 1, 4, 1), date(year, 3, 31)


def _last_day_of_month(year: int, month: int) -> date:
    """Return the last calendar day of the given month."""
    if month == 12:
        return date(year + 1, 1, 1) - timedelta(days=1)
    return date(year, month + 1, 1) - timedelta(days=1)


def month_range(year: int, month: int) -> Tuple[date, date]:
    """Return the first and last dates for the given year/month."""
    first = date(year, month, 1)
    last = _last_day_of_month(year, month)
    return first, last


def _normalize_year(token: str, base_year: Optional[int] = None) -> int:
    """Normalize 2-digit or 4-digit year tokens to a 4-digit year."""
    token = token.strip()
    year = int(token)
    if len(token) == 2:
        if base_year:
            century = base_year // 100
        else:
            century = date.today().year // 100
        year = century * 100 + year
        if base_year and year < base_year and (base_year - year) > YEAR_ROLLOVER_THRESHOLD:
            year += 100
    return year


# ---------------------------------------------------------------------------
# Range helpers
# ---------------------------------------------------------------------------

def ytd_range(reference_date: Optional[date] = None) -> Tuple[date, date]:
    """Return ``(fy_start, reference_date)`` for YTD calculation.

    Uses the current FY (April–March).
    """
    if reference_date is None:
        reference_date = date.today()
    return fy_start(reference_date), reference_date


def last_n_months_range(n: int, reference_date: Optional[date] = None) -> Tuple[date, date]:
    """Return ``(start, end)`` covering the last *n* calendar months.

    The *end* date is *reference_date* (or today).
    """
    if reference_date is None:
        reference_date = date.today()
    month = reference_date.month - n
    year = reference_date.year
    while month <= 0:
        month += 12
        year -= 1
    start = date(year, month, 1)
    return start, reference_date
