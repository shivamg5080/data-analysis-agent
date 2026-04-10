"""
Quota-Safe LLM Orchestrator
=============================
Implements:
  - Per-session + global rate limiter (RPM + token budget)
  - Retry with exponential backoff + jitter
  - 429 handling: honors ``retryDelay`` from API error details when available
  - Circuit breaker per model (CLOSED → OPEN → HALF_OPEN → CLOSED)
  - Configurable fallback chain (pro → flash → template-only)
  - Output cache keyed by (normalized_query, table_name, filters_hash)

Usage::

    orchestrator = LLMOrchestrator(client, config=cfg["orchestrator"])
    result = orchestrator.generate(model, prompt)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults (overridden by config.yaml orchestrator section)
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "max_retries": 3,
    "base_backoff_seconds": 1.0,
    "max_backoff_seconds": 60.0,
    "jitter": True,
    "rpm_per_session": 10,
    "rpm_global": 60,
    "token_budget_per_query": 8000,
    "circuit_breaker_failure_threshold": 3,
    "circuit_breaker_cooldown_seconds": 60,
    "circuit_breaker_half_open_max_calls": 1,
    "cache_ttl_seconds": 300,
    "cache_max_entries": 500,
}


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Sliding-window rate limiter (requests per minute).

    Thread-safe via a ``threading.Lock``.
    """

    def __init__(self, rpm: int = 60):
        self._rpm = rpm
        self._window_seconds = 60.0
        self._timestamps: List[float] = []
        self._lock = threading.Lock()

    @property
    def rpm(self) -> int:
        return self._rpm

    def acquire(self, timeout: float = 120.0) -> bool:
        """Block until a request slot is available.

        Returns ``True`` if acquired, ``False`` if timed out.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                now = time.monotonic()
                # Evict timestamps older than the sliding window
                self._timestamps = [
                    t for t in self._timestamps if now - t < self._window_seconds
                ]
                if len(self._timestamps) < self._rpm:
                    self._timestamps.append(now)
                    return True
                # Calculate how long to wait for the oldest slot to expire
                wait = self._window_seconds - (now - self._timestamps[0]) + 0.05
            time.sleep(min(wait, deadline - time.monotonic()))
        return False

    def record(self) -> None:
        """Manually record a request (used after external calls)."""
        with self._lock:
            self._timestamps.append(time.monotonic())


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class CircuitState(Enum):
    CLOSED = "CLOSED"         # Normal operation
    OPEN = "OPEN"             # Failing; fast-fail
    HALF_OPEN = "HALF_OPEN"   # Probing recovery


@dataclass
class CircuitBreaker:
    """Per-model circuit breaker.

    States
    ------
    CLOSED
        Normal; all calls go through.
    OPEN
        Too many failures; calls are rejected immediately.
        After ``cooldown_seconds`` transitions to HALF_OPEN.
    HALF_OPEN
        A single probe call is allowed.  Success → CLOSED; failure → OPEN.
    """

    model_name: str
    failure_threshold: int = 3
    cooldown_seconds: float = 60.0
    half_open_max_calls: int = 1

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False, repr=False)
    _failure_count: int = field(default=0, init=False, repr=False)
    _opened_at: float = field(default=0.0, init=False, repr=False)
    _half_open_calls: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._evaluate_state()

    def _evaluate_state(self) -> CircuitState:
        """Evaluate state transitions (must be called under lock)."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.cooldown_seconds:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info(
                    "CircuitBreaker[%s]: OPEN → HALF_OPEN after %.0fs cooldown",
                    self.model_name, elapsed,
                )
        return self._state

    def is_available(self) -> bool:
        """Return whether a call is allowed through the breaker."""
        with self._lock:
            state = self._evaluate_state()
            if state == CircuitState.CLOSED:
                return True
            if state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            # OPEN
            return False

    def record_success(self) -> None:
        """Record a successful call (may reset the breaker)."""
        with self._lock:
            if self._state in (CircuitState.HALF_OPEN, CircuitState.CLOSED):
                if self._state == CircuitState.HALF_OPEN:
                    logger.info(
                        "CircuitBreaker[%s]: HALF_OPEN → CLOSED (probe succeeded)",
                        self.model_name,
                    )
                self._state = CircuitState.CLOSED
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failure (may trip the breaker)."""
        with self._lock:
            self._failure_count += 1
            if self._state == CircuitState.HALF_OPEN:
                # Probe failed → back to OPEN
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "CircuitBreaker[%s]: HALF_OPEN → OPEN (probe failed)",
                    self.model_name,
                )
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "CircuitBreaker[%s]: CLOSED → OPEN (%d failures ≥ threshold %d)",
                    self.model_name, self._failure_count, self.failure_threshold,
                )

    def reset(self) -> None:
        """Manually reset the breaker to CLOSED."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class _ResponseCache:
    """Simple TTL-based in-memory cache (thread-safe)."""

    def __init__(self, ttl_seconds: float = 300, max_entries: int = 500):
        self._ttl = ttl_seconds
        self._max = max_entries
        self._store: Dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    @staticmethod
    def make_key(normalized_query: str, table_name: str, filters: dict) -> str:
        """Build a stable cache key from query + context."""
        raw = json.dumps(
            {"q": normalized_query, "t": table_name, "f": filters},
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Tuple[bool, Any]:
        """Return ``(hit, value)``."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False, None
            if time.monotonic() > entry.expires_at:
                del self._store[key]
                return False, None
            return True, entry.value

    def put(self, key: str, value: Any) -> None:
        """Store *value* under *key*."""
        with self._lock:
            # Evict oldest entries if at capacity
            if len(self._store) >= self._max:
                oldest_key = min(self._store, key=lambda k: self._store[k].expires_at)
                del self._store[oldest_key]
            self._store[key] = _CacheEntry(
                value=value,
                expires_at=time.monotonic() + self._ttl,
            )

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._store)


# ---------------------------------------------------------------------------
# LLM Orchestrator
# ---------------------------------------------------------------------------

class LLMOrchestrator:
    """Quota-safe LLM call manager.

    Responsibilities
    ----------------
    1. Rate limiting (per-session + global sliding window).
    2. Retry with exponential backoff + jitter for 503 errors.
    3. 429 handling: honors ``retryDelay`` from API error details.
    4. Circuit breaker per model.
    5. Configurable fallback chain.
    6. Response cache.

    Parameters
    ----------
    client:
        A ``google.genai.Client`` instance.
    fallback_chain:
        Ordered list of model names.  The last entry may be the sentinel
        ``"template-only"`` to indicate deterministic-only fallback.
    config:
        Dict from the ``orchestrator`` section of ``config.yaml``.
    """

    def __init__(
        self,
        client: Any,
        fallback_chain: List[str],
        config: Optional[Dict[str, Any]] = None,
    ):
        self._client = client
        self._fallback_chain = fallback_chain or ["gemini-2.0-flash"]
        cfg = dict(_DEFAULTS)
        if config:
            cfg.update(config)

        self._max_retries: int = int(cfg["max_retries"])
        self._base_backoff: float = float(cfg["base_backoff_seconds"])
        self._max_backoff: float = float(cfg["max_backoff_seconds"])
        self._jitter: bool = bool(cfg["jitter"])

        self._session_limiter = RateLimiter(rpm=int(cfg["rpm_per_session"]))
        self._global_limiter = RateLimiter(rpm=int(cfg["rpm_global"]))

        cb_threshold = int(cfg["circuit_breaker_failure_threshold"])
        cb_cooldown = float(cfg["circuit_breaker_cooldown_seconds"])
        cb_half_open = int(cfg["circuit_breaker_half_open_max_calls"])

        self._breakers: Dict[str, CircuitBreaker] = {
            m: CircuitBreaker(
                model_name=m,
                failure_threshold=cb_threshold,
                cooldown_seconds=cb_cooldown,
                half_open_max_calls=cb_half_open,
            )
            for m in self._fallback_chain
            if m != "template-only"
        }

        self._cache = _ResponseCache(
            ttl_seconds=float(cfg["cache_ttl_seconds"]),
            max_entries=int(cfg["cache_max_entries"]),
        )

        # Global counters for observability
        self.stats: Dict[str, int] = {
            "total_calls": 0,
            "cache_hits": 0,
            "retry_429": 0,
            "retry_503": 0,
            "fallbacks": 0,
            "circuit_opens": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_content(
        self,
        prompt: str,
        *,
        cache_key: Optional[str] = None,
        preferred_model: Optional[str] = None,
    ) -> Tuple[Optional[Any], str, List[str]]:
        """Call the LLM with retry/fallback/circuit-breaker/cache.

        Parameters
        ----------
        prompt:
            The full prompt string to send.
        cache_key:
            If provided, the cache is checked/populated with this key.
        preferred_model:
            Override the first model to try (must be in fallback_chain).

        Returns
        -------
        ``(response, model_used, events)``
            *response* is the raw API response or ``None`` if all models failed.
            *model_used* is the model that succeeded (or ``"template-only"``).
            *events* is a list of human-readable routing/retry events.
        """
        events: List[str] = []
        self.stats["total_calls"] += 1

        # Cache lookup
        if cache_key:
            hit, cached = self._cache.get(cache_key)
            if hit:
                self.stats["cache_hits"] += 1
                events.append("cache_hit")
                logger.debug("LLM cache hit for key=%s...", cache_key[:16])
                return cached, "cache", events

        # Build model order
        chain = self._build_chain(preferred_model)

        for model_name in chain:
            if model_name == "template-only":
                events.append("fallback:template-only")
                return None, "template-only", events

            breaker = self._breakers.get(model_name)
            if breaker and not breaker.is_available():
                events.append(f"circuit_open:{model_name}")
                logger.info(
                    "CircuitBreaker OPEN for %s — skipping to next model", model_name
                )
                self.stats["circuit_opens"] += 1
                continue

            response, model_events = self._try_model(model_name, prompt)
            events.extend(model_events)

            if response is not None:
                if breaker:
                    breaker.record_success()
                if cache_key:
                    self._cache.put(cache_key, response)
                return response, model_name, events

            # Model failed
            if breaker:
                breaker.record_failure()
            self.stats["fallbacks"] += 1
            events.append(f"fallback_from:{model_name}")

        # All models exhausted
        logger.warning("All models in fallback chain failed")
        return None, "all_failed", events

    def cache_key_for(
        self,
        normalized_query: str,
        table_name: str,
        filters: dict,
    ) -> str:
        """Convenience wrapper for ``_ResponseCache.make_key``."""
        return _ResponseCache.make_key(normalized_query, table_name, filters)

    def put_cache(self, key: str, value: Any) -> None:
        """Manually populate the cache (e.g. for deterministic results)."""
        self._cache.put(key, value)

    def get_cache(self, key: str) -> Tuple[bool, Any]:
        """Manually look up the cache."""
        return self._cache.get(key)

    def clear_cache(self) -> None:
        """Evict all cached entries."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return self._cache.size()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_chain(self, preferred_model: Optional[str]) -> List[str]:
        """Return the model order to try, starting with *preferred_model*."""
        chain = list(self._fallback_chain)
        if preferred_model and preferred_model in chain:
            chain.remove(preferred_model)
            chain.insert(0, preferred_model)
        return chain

    def _try_model(
        self, model_name: str, prompt: str
    ) -> Tuple[Optional[Any], List[str]]:
        """Attempt to call *model_name* with retry logic.

        Returns ``(response_or_None, events)``.
        """
        events: List[str] = []

        # Acquire rate limit slots
        if not self._session_limiter.acquire(timeout=10):
            events.append(f"rate_limited_session:{model_name}")
            logger.warning("Session rate limit hit for model %s", model_name)
            return None, events
        if not self._global_limiter.acquire(timeout=10):
            events.append(f"rate_limited_global:{model_name}")
            logger.warning("Global rate limit hit for model %s", model_name)
            return None, events

        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=model_name, contents=prompt
                )
                logger.info(
                    "LLM success: model=%s attempt=%d", model_name, attempt
                )
                events.append(f"success:{model_name}:attempt={attempt}")
                return response, events

            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                is_429 = self._is_quota_error(exc)
                is_503 = self._is_transient_error(exc)

                if is_429:
                    self.stats["retry_429"] += 1
                    delay = self._extract_retry_delay(exc) or self._backoff(attempt)
                    events.append(f"429:{model_name}:wait={delay:.1f}s")
                    logger.warning(
                        "429 on %s (attempt %d/%d) — waiting %.1fs",
                        model_name, attempt, self._max_retries, delay,
                    )
                    time.sleep(delay)
                elif is_503:
                    self.stats["retry_503"] += 1
                    delay = self._backoff(attempt)
                    events.append(f"503:{model_name}:wait={delay:.1f}s")
                    logger.warning(
                        "503 on %s (attempt %d/%d) — waiting %.1fs",
                        model_name, attempt, self._max_retries, delay,
                    )
                    time.sleep(delay)
                else:
                    # Non-retryable error
                    events.append(f"error:{model_name}:{err_str[:60]}")
                    logger.warning("Non-retryable error on %s: %s", model_name, err_str)
                    return None, events

                if attempt == self._max_retries:
                    events.append(f"max_retries_exhausted:{model_name}")
                    logger.warning(
                        "Max retries (%d) exhausted for %s", self._max_retries, model_name
                    )

        return None, events

    def _backoff(self, attempt: int) -> float:
        """Compute exponential backoff with optional jitter."""
        delay = min(self._base_backoff * (2 ** (attempt - 1)), self._max_backoff)
        if self._jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        return delay

    @staticmethod
    def _extract_retry_delay(exc: Exception) -> Optional[float]:
        """Extract ``retryDelay`` from a 429 API error response when available."""
        err_str = str(exc)
        # Google API errors often embed JSON with retryDelay
        try:
            match = re.search(r'"retryDelay":\s*"?([\d.]+)', err_str)
            if match:
                return float(match.group(1))
        except (ValueError, AttributeError):
            pass
        return None

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        """Detect 429 / RESOURCE_EXHAUSTED errors."""
        msg = str(exc).lower()
        return any(m in msg for m in [
            "429", "resource_exhausted", "quota", "rate limit", "rate_limit",
            "too many requests",
        ])

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        """Detect 503 / transient errors."""
        msg = str(exc).lower()
        return any(m in msg for m in [
            "503", "unavailable", "timeout", "temporar",
            "deadline exceeded", "connection reset", "service unavailable",
            "internal server error", "500",
        ])
