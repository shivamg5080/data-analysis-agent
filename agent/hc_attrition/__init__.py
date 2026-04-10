"""
HC/Attrition Analytics Package
================================
Provides quota-safe query orchestration, deterministic SQL templates,
and catalogue-driven routing for headcount and attrition analytics.
"""

from agent.hc_attrition.fy_helper import (
    fy_start,
    fy_end,
    fy_for_date,
    fy_label,
    parse_month_year,
    ytd_range,
    last_n_months_range,
)
from agent.hc_attrition.catalogue import (
    load_catalogue,
    classify_intent,
    extract_filters,
    INTENT_TO_ID,
    INTENT_PATTERNS,
    CatalogueEntry,
)
from agent.hc_attrition.query_router import QueryRouter, RouteResult, RouteTier
from agent.hc_attrition.llm_orchestrator import LLMOrchestrator, CircuitBreaker, RateLimiter
from agent.hc_attrition.observability import ObservabilityTracker, QueryTrace

__all__ = [
    # FY helpers
    "fy_start", "fy_end", "fy_for_date", "fy_label",
    "parse_month_year", "ytd_range", "last_n_months_range",
    # Catalogue
    "load_catalogue", "classify_intent", "extract_filters",
    "INTENT_TO_ID", "INTENT_PATTERNS", "CatalogueEntry",
    # Router
    "QueryRouter", "RouteResult", "RouteTier",
    # LLM Orchestrator
    "LLMOrchestrator", "CircuitBreaker", "RateLimiter",
    # Observability
    "ObservabilityTracker", "QueryTrace",
]
