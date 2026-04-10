"""
Observability & Diagnostics
============================
Structured per-query traces and aggregate metrics for the quota-safe
query orchestrator.

Usage::

    tracker = ObservabilityTracker()
    trace = tracker.start_trace("show me attrition for June 2025")
    trace.route_tier = RouteTier.DETERMINISTIC
    trace.intent_key = "attrition_overall"
    trace.confidence = 0.92
    tracker.finish_trace(trace)

    print(tracker.summary())
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query trace
# ---------------------------------------------------------------------------

@dataclass
class QueryTrace:
    """Captures the complete lifecycle of a single query."""

    query: str
    trace_id: str = ""

    # Routing
    route_tier: str = "unknown"         # "deterministic" | "llm_tier2" | "llm_tier3"
    intent_key: Optional[str] = None
    confidence: float = 0.0
    catalogue_id: Optional[int] = None

    # Model selection
    model_used: str = ""
    preferred_model: str = ""
    fallback_events: List[str] = field(default_factory=list)

    # Token / cost
    token_estimate: int = 0

    # Retries
    retry_count: int = 0
    error_codes: List[str] = field(default_factory=list)

    # Cache
    cache_hit: bool = False

    # Latency
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float = 0.0

    # Result
    status: str = "pending"    # "success" | "fallback" | "error" | "pending"
    sql: Optional[str] = None

    @property
    def latency_ms(self) -> float:
        """Elapsed time in milliseconds (0 if not yet finished)."""
        if self.finished_at:
            return (self.finished_at - self.started_at) * 1000
        return 0.0

    def finish(self, status: str = "success") -> None:
        """Mark the trace as complete."""
        self.finished_at = time.monotonic()
        self.status = status

    def as_dict(self) -> Dict[str, Any]:
        """Serialize the trace to a loggable / displayable dict."""
        return {
            "trace_id": self.trace_id,
            "query": self.query[:120],
            "route_tier": self.route_tier,
            "intent_key": self.intent_key,
            "confidence": round(self.confidence, 3),
            "catalogue_id": self.catalogue_id,
            "model_used": self.model_used,
            "preferred_model": self.preferred_model,
            "fallback_events": self.fallback_events,
            "token_estimate": self.token_estimate,
            "retry_count": self.retry_count,
            "error_codes": self.error_codes,
            "cache_hit": self.cache_hit,
            "latency_ms": round(self.latency_ms, 1),
            "status": self.status,
        }

    def why_routed_summary(self) -> str:
        """Human-readable routing explanation for the UI "Show SQL & Reasoning" panel."""
        parts: List[str] = []

        tier_labels = {
            "deterministic": "🟢 Tier 1 — Deterministic template",
            "llm_tier2": "🟡 Tier 2 — Lightweight LLM (flash)",
            "llm_tier3": "🔴 Tier 3 — Full LLM (pro)",
            "cache": "⚡ Cached result",
            "unknown": "❓ Unknown",
        }
        parts.append(tier_labels.get(self.route_tier, self.route_tier))

        if self.intent_key:
            parts.append(f"Intent matched: **{self.intent_key}** (confidence {self.confidence:.0%})")

        if self.cache_hit:
            parts.append("Result served from cache — no LLM call made.")

        if self.model_used and self.model_used not in ("cache", "template-only", ""):
            parts.append(f"Model used: `{self.model_used}`")

        if self.fallback_events:
            parts.append(f"Fallback events: {', '.join(self.fallback_events)}")

        if self.retry_count:
            parts.append(f"Retries: {self.retry_count}")

        parts.append(f"Latency: {self.latency_ms:.0f} ms")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Aggregate tracker
# ---------------------------------------------------------------------------

class ObservabilityTracker:
    """Collects query traces and maintains aggregate metrics.

    Thread-safe.
    """

    def __init__(self, max_history: int = 200):
        self._max_history = max_history
        self._history: List[QueryTrace] = []
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {
            "total_queries": 0,
            "deterministic_hits": 0,
            "llm_calls": 0,
            "llm_calls_429": 0,
            "llm_calls_503": 0,
            "cache_hits": 0,
            "fallbacks": 0,
            "errors": 0,
        }

    # ------------------------------------------------------------------
    # Trace lifecycle
    # ------------------------------------------------------------------

    def start_trace(self, query: str, preferred_model: str = "") -> QueryTrace:
        """Create and register a new in-flight trace."""
        import uuid
        trace = QueryTrace(
            query=query,
            trace_id=str(uuid.uuid4())[:8],
            preferred_model=preferred_model,
        )
        return trace

    def finish_trace(self, trace: QueryTrace) -> None:
        """Finalise *trace* and update aggregate counters."""
        if not trace.finished_at:
            trace.finish()

        with self._lock:
            self._counters["total_queries"] += 1

            if trace.route_tier == "deterministic":
                self._counters["deterministic_hits"] += 1
            elif trace.route_tier in ("llm_tier2", "llm_tier3"):
                self._counters["llm_calls"] += 1

            if trace.cache_hit:
                self._counters["cache_hits"] += 1

            if trace.fallback_events:
                self._counters["fallbacks"] += len(trace.fallback_events)

            for code in trace.error_codes:
                if "429" in code:
                    self._counters["llm_calls_429"] += 1
                elif "503" in code:
                    self._counters["llm_calls_503"] += 1

            if trace.status == "error":
                self._counters["errors"] += 1

            self._history.append(trace)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        self._log_trace(trace)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a snapshot of aggregate metrics."""
        with self._lock:
            total = max(self._counters["total_queries"], 1)
            return {
                **self._counters,
                "deterministic_hit_rate_pct": round(
                    self._counters["deterministic_hits"] * 100 / total, 1
                ),
                "cache_hit_rate_pct": round(
                    self._counters["cache_hits"] * 100 / total, 1
                ),
                "fallback_rate_pct": round(
                    min(self._counters["fallbacks"], total) * 100 / total, 1
                ),
                "history_count": len(self._history),
            }

    def recent_traces(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the *n* most recent completed traces as dicts."""
        with self._lock:
            return [t.as_dict() for t in self._history[-n:]]

    def counters(self) -> Dict[str, int]:
        """Return raw counters (copy)."""
        with self._lock:
            return dict(self._counters)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_trace(self, trace: QueryTrace) -> None:
        d = trace.as_dict()
        logger.info(
            "QUERY_TRACE | tier=%s intent=%s conf=%.2f model=%s "
            "cache=%s retries=%d latency=%.0fms status=%s events=%s",
            d["route_tier"],
            d["intent_key"] or "-",
            d["confidence"],
            d["model_used"] or "-",
            d["cache_hit"],
            d["retry_count"],
            d["latency_ms"],
            d["status"],
            d["fallback_events"],
        )
