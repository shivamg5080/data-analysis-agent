"""
Query Router — Deterministic-First Tier Routing
================================================
Routes incoming user questions through three tiers:

  Tier 1 (DETERMINISTIC)
    High-confidence match against the 25-question catalogue.
    Returns a SQL template — **no LLM call needed**.

  Tier 2 (LLM_TIER2)
    Moderate confidence.  Use a lightweight/flash model.

  Tier 3 (LLM_TIER3)
    Low/no confidence.  Complex or novel query; use the full pro model.

Usage::

    router = QueryRouter(config=cfg.get("orchestrator", {}))
    result = router.route("show me attrition for June 2025", table_name="fact_attrition")
    if result.tier == RouteTier.DETERMINISTIC:
        sql = result.sql
    else:
        # call LLM with result.preferred_model
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class RouteTier(str, Enum):
    DETERMINISTIC = "deterministic"
    LLM_TIER2 = "llm_tier2"
    LLM_TIER3 = "llm_tier3"


@dataclass
class RouteResult:
    """Result of a routing decision."""

    tier: RouteTier
    confidence: float
    intent_key: Optional[str] = None
    catalogue_id: Optional[int] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    sql: Optional[str] = None                    # Populated for Tier 1
    preferred_model: str = ""                    # Populated for Tier 2/3
    why: str = ""                                # Human-readable rationale

    def is_deterministic(self) -> bool:
        return self.tier == RouteTier.DETERMINISTIC

    def model_tier_label(self) -> str:
        labels = {
            RouteTier.DETERMINISTIC: "Tier 1 — Deterministic Template",
            RouteTier.LLM_TIER2: "Tier 2 — Lightweight LLM",
            RouteTier.LLM_TIER3: "Tier 3 — Full LLM (Pro)",
        }
        return labels.get(self.tier, str(self.tier))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

# HC/Attrition column signatures used to detect table type
_HC_COLUMNS: Set[str] = {
    "empid", "empstatus", "endofmonth", "department",
    "grade", "gender", "lob", "businessgroup",
}
_ATTRITION_COLUMNS: Set[str] = {
    "empid", "empstatus", "lwd", "final_exit_type",
    "final_reason_of_exit", "tenure_months",
}
_MIN_OVERLAP = 3  # at least 3 matching columns to identify a table


class QueryRouter:
    """Deterministic-first query router for HC/Attrition questions.

    Parameters
    ----------
    config:
        The ``orchestrator`` section of ``config.yaml``.
        Relevant keys:

        - ``deterministic_first`` (bool, default True)
        - ``confidence_threshold_deterministic`` (float, default 0.75)
        - ``confidence_threshold_tier2`` (float, default 0.40)
        - ``fallback_chain`` (list of model names)
        - ``catalogue_path`` (str, path to catalogue CSV)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}

        self._deterministic_first: bool = bool(
            cfg.get("deterministic_first", True)
        )
        self._threshold_det: float = float(
            cfg.get("confidence_threshold_deterministic", 0.75)
        )
        self._threshold_t2: float = float(
            cfg.get("confidence_threshold_tier2", 0.40)
        )
        fallback_chain: List[str] = cfg.get("fallback_chain") or [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "template-only",
        ]
        # Tier 2 model = first non-pro model in chain
        self._tier2_model = self._pick_tier2(fallback_chain)
        # Tier 3 model = first (possibly pro) model in chain
        self._tier3_model = fallback_chain[0] if fallback_chain else "gemini-2.0-flash"

        catalogue_path: str = cfg.get("catalogue_path", "catalogue.csv")

        # Lazy-load catalogue and import helpers at call time to avoid
        # circular imports at module load.
        self._catalogue_path = catalogue_path
        self._catalogue: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def route(
        self,
        query: str,
        table_name: str = "",
        table_columns: Optional[List[str]] = None,
    ) -> RouteResult:
        """Classify *query* and return a ``RouteResult``.

        Parameters
        ----------
        query:
            The raw user question.
        table_name:
            The DuckDB-registered table name (used in SQL templates).
        table_columns:
            List of column names present in the table.  Used to validate
            whether a deterministic template can actually execute.
        """
        from agent.hc_attrition.catalogue import classify_intent, extract_filters, INTENT_TO_ID
        from agent.hc_attrition.sql_templates import render_template

        intent_key, confidence = classify_intent(query)
        filters = extract_filters(query)

        logger.info(
            "Router: query=%r intent=%s confidence=%.2f",
            query[:80], intent_key, confidence,
        )

        # ---- Tier 1: Deterministic -------------------------------------------
        if (
            self._deterministic_first
            and intent_key
            and confidence >= self._threshold_det
        ):
            # Try to render the SQL template
            sql = render_template(intent_key, table_name, filters) if table_name else None

            # Validate columns if we have a column list
            if sql and table_columns:
                if not self._table_looks_like_hc_or_attrition(table_columns):
                    sql = None  # Table doesn't match; don't use template

            catalogue_id = INTENT_TO_ID.get(intent_key)

            if sql:
                logger.info(
                    "Route → DETERMINISTIC | intent=%s catalogue_id=%s",
                    intent_key, catalogue_id,
                )
                return RouteResult(
                    tier=RouteTier.DETERMINISTIC,
                    confidence=confidence,
                    intent_key=intent_key,
                    catalogue_id=catalogue_id,
                    filters=filters,
                    sql=sql,
                    why=(
                        f"High-confidence match ({confidence:.0%}) for catalogue question "
                        f"#{catalogue_id} ({intent_key}). "
                        "Using deterministic SQL template — no LLM call required."
                    ),
                )

        # ---- Tier 2: Lightweight LLM -----------------------------------------
        if intent_key and confidence >= self._threshold_t2:
            logger.info(
                "Route → TIER2 | intent=%s confidence=%.2f model=%s",
                intent_key, confidence, self._tier2_model,
            )
            catalogue_id = INTENT_TO_ID.get(intent_key)
            return RouteResult(
                tier=RouteTier.LLM_TIER2,
                confidence=confidence,
                intent_key=intent_key,
                catalogue_id=catalogue_id,
                filters=filters,
                preferred_model=self._tier2_model,
                why=(
                    f"Moderate confidence ({confidence:.0%}) — routing to lightweight model "
                    f"`{self._tier2_model}` (intent: {intent_key})."
                ),
            )

        # ---- Tier 3: Full LLM ------------------------------------------------
        logger.info(
            "Route → TIER3 | intent=%s confidence=%.2f model=%s",
            intent_key, confidence, self._tier3_model,
        )
        return RouteResult(
            tier=RouteTier.LLM_TIER3,
            confidence=confidence,
            intent_key=intent_key,
            catalogue_id=INTENT_TO_ID.get(intent_key) if intent_key else None,
            filters=filters,
            preferred_model=self._tier3_model,
            why=(
                f"Low/no confidence ({confidence:.0%}) — routing to full LLM "
                f"`{self._tier3_model}` for complex/novel query."
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_tier2(chain: List[str]) -> str:
        """Return the first non-pro, non-sentinel model from the chain."""
        for m in chain:
            if m == "template-only":
                continue
            # Prefer flash/lite models for tier 2
            if "flash" in m.lower() or "lite" in m.lower():
                return m
        # Fall back to whatever is available
        return next((m for m in chain if m != "template-only"), "gemini-2.0-flash")

    @staticmethod
    def _table_looks_like_hc_or_attrition(columns: List[str]) -> bool:
        """Return True if *columns* overlap enough with HC or Attrition signatures."""
        col_set = {c.lower().strip() for c in columns}
        hc_overlap = len(col_set & _HC_COLUMNS)
        attr_overlap = len(col_set & _ATTRITION_COLUMNS)
        return hc_overlap >= _MIN_OVERLAP or attr_overlap >= _MIN_OVERLAP
