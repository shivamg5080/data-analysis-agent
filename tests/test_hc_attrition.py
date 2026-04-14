"""
Tests for the Quota-Safe Query Orchestrator
============================================

Covers:
  A) Intent classification for all 25 HC/Attrition catalogue questions
  B) April–March FY date logic
  C) SQL template correctness (structure and parameterisation)
  D) Retry/backoff + circuit breaker behaviour
  E) Integration: simulated 429/503 → graceful fallback
  F) Attrition query under forced quota conditions (deterministic path)
"""

from __future__ import annotations

import sys
import os
import re
import time
from datetime import date
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_exception(msg: str) -> Exception:
    return RuntimeError(msg)


# ===========================================================================
# B) April–March FY date logic
# ===========================================================================

class TestFYHelper:
    """Unit tests for agent/hc_attrition/fy_helper.py"""

    def test_fy_start_before_april(self):
        from agent.hc_attrition.fy_helper import fy_start
        assert fy_start(date(2025, 3, 31)) == date(2024, 4, 1)

    def test_fy_start_on_april_1(self):
        from agent.hc_attrition.fy_helper import fy_start
        assert fy_start(date(2025, 4, 1)) == date(2025, 4, 1)

    def test_fy_start_after_april(self):
        from agent.hc_attrition.fy_helper import fy_start
        assert fy_start(date(2025, 6, 15)) == date(2025, 4, 1)

    def test_fy_end_before_april(self):
        from agent.hc_attrition.fy_helper import fy_end
        # Jan 2025 is in FY 2024-25 → ends Mar 31 2025
        assert fy_end(date(2025, 1, 15)) == date(2025, 3, 31)

    def test_fy_end_after_april(self):
        from agent.hc_attrition.fy_helper import fy_end
        # Jun 2025 is in FY 2025-26 → ends Mar 31 2026
        assert fy_end(date(2025, 6, 15)) == date(2026, 3, 31)

    def test_fy_for_date(self):
        from agent.hc_attrition.fy_helper import fy_for_date
        start, end = fy_for_date(date(2025, 6, 15))
        assert start == date(2025, 4, 1)
        assert end == date(2026, 3, 31)

    def test_fy_label(self):
        from agent.hc_attrition.fy_helper import fy_label
        assert fy_label(date(2025, 6, 15)) == "FY2025-26"
        assert fy_label(date(2025, 1, 1)) == "FY2024-25"

    def test_parse_month_year_june_2025(self):
        from agent.hc_attrition.fy_helper import parse_month_year
        result = parse_month_year("show me attrition for June 2025")
        assert result is not None
        first, last = result
        assert first == date(2025, 6, 1)
        assert last == date(2025, 6, 30)

    def test_parse_month_year_abbreviation(self):
        from agent.hc_attrition.fy_helper import parse_month_year
        result = parse_month_year("headcount for Jan 2024")
        assert result is not None
        assert result[0] == date(2024, 1, 1)
        assert result[1] == date(2024, 1, 31)

    def test_parse_month_year_december(self):
        from agent.hc_attrition.fy_helper import parse_month_year
        result = parse_month_year("December 2024")
        assert result is not None
        first, last = result
        assert first == date(2024, 12, 1)
        assert last == date(2024, 12, 31)

    def test_parse_month_year_no_match(self):
        from agent.hc_attrition.fy_helper import parse_month_year
        assert parse_month_year("show me total headcount") is None

    def test_ytd_range(self):
        from agent.hc_attrition.fy_helper import ytd_range, fy_start
        ref = date(2025, 7, 15)
        start, end = ytd_range(ref)
        assert start == fy_start(ref)
        assert end == ref

    def test_last_n_months_range(self):
        from agent.hc_attrition.fy_helper import last_n_months_range
        ref = date(2025, 6, 30)
        start, end = last_n_months_range(3, ref)
        assert end == ref
        assert start.month == 3
        assert start.year == 2025

    def test_fy_boundary_march_31(self):
        """March 31 2025 is in FY 2024-25, not FY 2025-26."""
        from agent.hc_attrition.fy_helper import fy_start, fy_end
        d = date(2025, 3, 31)
        assert fy_start(d) == date(2024, 4, 1)
        assert fy_end(d) == date(2025, 3, 31)


# ===========================================================================
# A) Intent classification for all 25 catalogue questions
# ===========================================================================

class TestIntentClassification:
    """Verify that each of the 25 catalogue questions maps to the correct intent."""

    # Representative natural-language paraphrases for each catalogue question
    _QUESTION_SAMPLES = [
        # (expected_intent, sample_query)
        ("hc_total_snapshot",       "What is the total active headcount this month?"),
        ("hc_trend_mom",            "How has headcount trended month-on-month over the last 12 months?"),
        ("hc_by_gender",            "What is the headcount split by gender across departments?"),
        ("hc_team",                 "How many employees are in my team right now?"),
        ("hc_by_grade",             "What is the headcount by grade?"),
        ("hc_new_hires",            "How many new hires joined this month vs last month?"),
        ("hc_by_tenure",            "What is the headcount by tenure bucket?"),
        ("hc_by_emp_type",          "Headcount split by employee type Full Time vs Contract?"),
        ("hc_by_business_group",    "How does headcount compare across business groups?"),
        ("hc_ic_pm_split",          "What is the IC vs PM split in my team?"),
        ("attrition_overall",       "What is the overall attrition rate this month?"),
        ("attrition_vol_invol",     "What is the voluntary vs involuntary attrition split?"),
        ("attrition_exit_reasons",  "What are the top reasons employees are leaving?"),
        ("attrition_team",          "How many people left my team this month?"),
        ("attrition_by_tenure",     "What is the attrition rate by tenure bucket?"),
        ("attrition_by_grade",      "What is the attrition rate by grade?"),
        ("attrition_by_dept",       "How does attrition compare across departments?"),
        ("attrition_trend",         "What is the attrition trend month-on-month for the last 12 months?"),
        ("attrition_by_gender",     "What is the gender-wise attrition rate?"),
        ("attrition_notice_period", "Which employees have resigned but not yet left notice period?"),
        ("attrition_first_year",    "What is the first-year attrition rate?"),
        ("attrition_exit_by_dept_grade", "Do exit reasons differ by department or grade?"),
        ("attrition_new_hire",      "What is the attrition rate for new hires joined in last 6 months?"),
        ("attrition_perf_rating",   "What is the performance rating distribution of employees who left?"),
        ("attrition_avg_tenure",    "What is the average tenure of employees who left?"),
    ]

    @pytest.mark.parametrize("expected_intent,query", _QUESTION_SAMPLES)
    def test_intent_classification(self, expected_intent: str, query: str):
        from agent.hc_attrition.catalogue import classify_intent
        intent, confidence = classify_intent(query)
        assert intent == expected_intent, (
            f"Query: {query!r}\n"
            f"Expected: {expected_intent!r}, Got: {intent!r} (confidence={confidence:.2f})"
        )
        assert confidence >= 0.75, (
            f"Confidence {confidence:.2f} below threshold 0.75 for: {query!r}"
        )

    def test_no_match_returns_none(self):
        from agent.hc_attrition.catalogue import classify_intent
        intent, confidence = classify_intent("what is the weather in London?")
        # Should either return None or very low confidence
        assert intent is None or confidence < 0.75

    def test_empty_query(self):
        from agent.hc_attrition.catalogue import classify_intent
        intent, confidence = classify_intent("")
        assert intent is None
        assert confidence == 0.0

    def test_attrition_june_2025_common_pattern(self):
        """The most common failing query must map to attrition_overall."""
        from agent.hc_attrition.catalogue import classify_intent
        intent, confidence = classify_intent("show me attrition for June 2025")
        assert intent == "attrition_overall"
        assert confidence >= 0.75


# ===========================================================================
# Filter extraction
# ===========================================================================

class TestFilterExtraction:
    def test_extract_month_year(self):
        from agent.hc_attrition.catalogue import extract_filters
        f = extract_filters("show me attrition for June 2025")
        assert f["month_start"] == "2025-06-01"
        assert f["month_end"] == "2025-06-30"
        assert f["year"] == "2025"

    def test_extract_exit_type_voluntary(self):
        from agent.hc_attrition.catalogue import extract_filters
        f = extract_filters("show me voluntary attrition for Q1")
        assert f["exit_type"] == "Voluntary"

    def test_extract_gender(self):
        from agent.hc_attrition.catalogue import extract_filters
        f = extract_filters("male headcount in Engineering department")
        assert f["gender"] == "male"

    def test_no_filters(self):
        from agent.hc_attrition.catalogue import extract_filters
        f = extract_filters("what is the total headcount?")
        assert all(v is None for v in f.values())


# ===========================================================================
# C) SQL Template correctness
# ===========================================================================

class TestSQLTemplates:
    """Verify that each template renders valid SQL."""

    def test_render_hc_total_snapshot_basic(self):
        from agent.hc_attrition.sql_templates import render_template
        sql = render_template("hc_total_snapshot", "fact_headcount")
        assert sql is not None
        assert "COUNT" in sql.upper()
        assert '"fact_headcount"' in sql
        assert "empstatus" in sql.lower()

    def test_render_attrition_overall_basic(self):
        from agent.hc_attrition.sql_templates import render_template
        sql = render_template("attrition_overall", "fact_attrition")
        assert sql is not None
        assert "Inactive" in sql
        assert "attrition_rate_pct" in sql

    def test_render_attrition_overall_with_date_filter(self):
        from agent.hc_attrition.sql_templates import render_template
        sql = render_template(
            "attrition_overall",
            "fact_attrition",
            {"month_start": "2025-06-01", "month_end": "2025-06-30"},
        )
        assert sql is not None
        assert "2025-06-01" in sql
        assert "2025-06-30" in sql

    def test_render_vol_invol_with_dept(self):
        from agent.hc_attrition.sql_templates import render_template
        sql = render_template(
            "attrition_vol_invol",
            "fact_attrition",
            {"department": "Engineering"},
        )
        assert sql is not None
        assert "Engineering" in sql
        assert "final_exit_type" in sql.lower()

    def test_render_hc_trend_mom(self):
        from agent.hc_attrition.sql_templates import render_template
        sql = render_template("hc_trend_mom", "my_table")
        assert sql is not None
        assert "GROUP BY" in sql.upper()
        assert "endofmonth" in sql.lower()

    def test_render_notice_period(self):
        from agent.hc_attrition.sql_templates import render_template
        sql = render_template("attrition_notice_period", "fact_attrition")
        assert sql is not None
        assert "resignation_date" in sql.lower()
        assert "CURRENT_DATE" in sql.upper()

    def test_render_first_year_attrition(self):
        from agent.hc_attrition.sql_templates import render_template
        sql = render_template("attrition_first_year", "fact_attrition")
        assert sql is not None
        assert "tenure_months" in sql.lower()
        assert "12" in sql

    def test_render_returns_none_for_unknown_intent(self):
        from agent.hc_attrition.sql_templates import render_template
        result = render_template("unknown_intent_xyz", "some_table")
        assert result is None

    def test_render_all_25_intents(self):
        """Every intent in the catalogue must produce non-None SQL."""
        from agent.hc_attrition.sql_templates import render_template, list_supported_intents
        intents = list_supported_intents()
        assert len(intents) == 29, f"Expected 29 intents, got {len(intents)}"
        for intent_key in intents:
            sql = render_template(intent_key, "test_table")
            assert sql is not None, f"No template rendered for intent: {intent_key}"
            # Basic SQL structure checks
            assert re.search(r"\bSELECT\b", sql, re.I), f"No SELECT in: {intent_key}"
            assert re.search(r"\bFROM\b", sql, re.I), f"No FROM in: {intent_key}"

    def test_sql_injection_prevention(self):
        """Malicious SQL metacharacters must be stripped from user-supplied filter values."""
        from agent.hc_attrition.sql_templates import render_template
        # Attempt SQL injection via department filter
        sql = render_template(
            "hc_total_snapshot",
            "fact_headcount",
            {"department": "Eng'; DROP TABLE employees; --"},
        )
        assert sql is not None
        # The sanitizer strips metacharacters (', ;) from the user-supplied value.
        # The value should appear inside a SQL string literal (between single quotes).
        # Extract the value portion inside single quotes following "= '"
        assert "= '" in sql, "Department filter should appear as string literal"
        # Find the user value between the quotes
        value_start = sql.index("= '") + 3
        value_end = sql.index("'", value_start)
        user_value_in_sql = sql[value_start:value_end]
        # Semicolons from user input must be stripped (can't terminate the statement)
        assert ";" not in user_value_in_sql, (
            f"Semicolons must be stripped from user input, got: {user_value_in_sql!r}"
        )
        # Single quotes from user input must be stripped (can't break string literals)
        assert "'" not in user_value_in_sql, (
            f"Single quotes must be stripped from user input, got: {user_value_in_sql!r}"
        )

    def test_date_filter_validation_rejects_invalid(self):
        """Invalid date formats must not appear in the rendered SQL."""
        from agent.hc_attrition.sql_templates import render_template
        sql = render_template(
            "attrition_overall",
            "fact_attrition",
            {"month_start": "not-a-date", "month_end": "also-bad"},
        )
        # Template should still render but without the bad date filter
        assert sql is not None
        assert "not-a-date" not in sql


# ===========================================================================
# D) Circuit breaker behaviour
# ===========================================================================

class TestCircuitBreaker:
    """Unit tests for CircuitBreaker state machine."""

    def _make_breaker(self, threshold=3, cooldown=60):
        from agent.hc_attrition.llm_orchestrator import CircuitBreaker
        return CircuitBreaker(
            model_name="test-model",
            failure_threshold=threshold,
            cooldown_seconds=cooldown,
        )

    def test_initial_state_closed(self):
        from agent.hc_attrition.llm_orchestrator import CircuitState
        cb = self._make_breaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available()

    def test_trips_after_threshold(self):
        from agent.hc_attrition.llm_orchestrator import CircuitState
        cb = self._make_breaker(threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.is_available()

    def test_does_not_trip_below_threshold(self):
        from agent.hc_attrition.llm_orchestrator import CircuitState
        cb = self._make_breaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available()

    def test_success_resets_failure_count(self):
        from agent.hc_attrition.llm_orchestrator import CircuitState
        cb = self._make_breaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # Count should reset; now need 3 more failures to trip
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_cooldown(self):
        from agent.hc_attrition.llm_orchestrator import CircuitBreaker, CircuitState
        cb = CircuitBreaker(
            model_name="test-model",
            failure_threshold=1,
            cooldown_seconds=0.01,  # Very short cooldown for testing
        )
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.05)
        # Checking state should trigger OPEN → HALF_OPEN transition
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.is_available()  # Probe call allowed

    def test_half_open_success_closes(self):
        from agent.hc_attrition.llm_orchestrator import CircuitBreaker, CircuitState
        cb = CircuitBreaker(
            model_name="test-model",
            failure_threshold=1,
            cooldown_seconds=0.01,
        )
        cb.record_failure()
        time.sleep(0.05)
        _ = cb.is_available()  # Trigger HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_manual_reset(self):
        from agent.hc_attrition.llm_orchestrator import CircuitState
        cb = self._make_breaker(threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available()


# ===========================================================================
# D continued) Rate limiter
# ===========================================================================

class TestRateLimiter:
    def test_allows_up_to_rpm(self):
        from agent.hc_attrition.llm_orchestrator import RateLimiter
        rl = RateLimiter(rpm=5)
        for _ in range(5):
            acquired = rl.acquire(timeout=1)
            assert acquired

    def test_blocks_over_rpm(self):
        """6th request within 60s should block (or timeout quickly)."""
        from agent.hc_attrition.llm_orchestrator import RateLimiter
        rl = RateLimiter(rpm=3)
        for _ in range(3):
            rl.acquire(timeout=1)
        # 4th request should time out immediately
        result = rl.acquire(timeout=0.1)
        assert not result


# ===========================================================================
# D continued) Retry / backoff
# ===========================================================================

class TestLLMOrchestratorRetry:
    """Verify retry and backoff behaviour of LLMOrchestrator."""

    def _make_mock_client(self, side_effects):
        """Return a mock genai.Client whose generate_content raises/returns the side_effects."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = side_effects
        return mock_client

    def test_retries_on_503_then_succeeds(self):
        from agent.hc_attrition.llm_orchestrator import LLMOrchestrator

        good_response = MagicMock()
        good_response.text = "SELECT 1"
        mock_client = self._make_mock_client([
            RuntimeError("503 service unavailable"),
            good_response,
        ])

        orch = LLMOrchestrator(
            client=mock_client,
            fallback_chain=["gemini-test"],
            config={
                "max_retries": 3,
                "base_backoff_seconds": 0.001,
                "max_backoff_seconds": 0.01,
                "jitter": False,
                "rpm_per_session": 100,
                "rpm_global": 100,
                "circuit_breaker_failure_threshold": 5,
                "circuit_breaker_cooldown_seconds": 60,
                "circuit_breaker_half_open_max_calls": 1,
                "cache_ttl_seconds": 1,
                "cache_max_entries": 10,
            },
        )
        response, model_used, events = orch.generate_content("test prompt")
        assert response is not None
        assert model_used == "gemini-test"
        assert any("503" in e for e in events)

    def test_retries_on_429_honors_retry_delay(self):
        from agent.hc_attrition.llm_orchestrator import LLMOrchestrator

        good_response = MagicMock()
        good_response.text = "SELECT 1"
        err = RuntimeError('429 resource_exhausted "retryDelay": "0.01"')
        mock_client = self._make_mock_client([err, good_response])

        orch = LLMOrchestrator(
            client=mock_client,
            fallback_chain=["gemini-test"],
            config={
                "max_retries": 3,
                "base_backoff_seconds": 0.001,
                "max_backoff_seconds": 0.1,
                "jitter": False,
                "rpm_per_session": 100,
                "rpm_global": 100,
                "circuit_breaker_failure_threshold": 5,
                "circuit_breaker_cooldown_seconds": 60,
                "circuit_breaker_half_open_max_calls": 1,
                "cache_ttl_seconds": 1,
                "cache_max_entries": 10,
            },
        )
        response, model_used, events = orch.generate_content("test prompt")
        assert response is not None
        assert any("429" in e for e in events)

    def test_fallback_to_second_model_on_repeated_429(self):
        from agent.hc_attrition.llm_orchestrator import LLMOrchestrator

        good_response = MagicMock()
        good_response.text = "SELECT 1"

        # First model always fails with 429, second succeeds
        call_count = {"n": 0}
        def side_effect_fn(*args, **kwargs):
            call_count["n"] += 1
            if kwargs.get("model") == "model-primary" or (
                args and args[0] in ("model-primary",)
            ):
                raise RuntimeError("429 quota exceeded")
            return good_response

        mock_client = MagicMock()
        # Track which model is called via keyword argument
        def gen_content(model, contents):
            call_count["n"] += 1
            if model == "model-primary":
                raise RuntimeError("429 quota exceeded")
            return good_response

        mock_client.models.generate_content.side_effect = gen_content

        orch = LLMOrchestrator(
            client=mock_client,
            fallback_chain=["model-primary", "model-fallback"],
            config={
                "max_retries": 2,
                "base_backoff_seconds": 0.001,
                "max_backoff_seconds": 0.01,
                "jitter": False,
                "rpm_per_session": 100,
                "rpm_global": 100,
                "circuit_breaker_failure_threshold": 2,
                "circuit_breaker_cooldown_seconds": 60,
                "circuit_breaker_half_open_max_calls": 1,
                "cache_ttl_seconds": 1,
                "cache_max_entries": 10,
            },
        )
        response, model_used, events = orch.generate_content("test prompt")
        assert response is not None
        assert model_used == "model-fallback"
        assert any("fallback" in e.lower() for e in events)

    def test_all_models_fail_returns_none(self):
        from agent.hc_attrition.llm_orchestrator import LLMOrchestrator

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("503 unavailable")

        orch = LLMOrchestrator(
            client=mock_client,
            fallback_chain=["model-a", "model-b"],
            config={
                "max_retries": 2,
                "base_backoff_seconds": 0.001,
                "max_backoff_seconds": 0.01,
                "jitter": False,
                "rpm_per_session": 100,
                "rpm_global": 100,
                "circuit_breaker_failure_threshold": 5,
                "circuit_breaker_cooldown_seconds": 60,
                "circuit_breaker_half_open_max_calls": 1,
                "cache_ttl_seconds": 1,
                "cache_max_entries": 10,
            },
        )
        response, model_used, events = orch.generate_content("test prompt")
        assert response is None
        assert "all_failed" in model_used

    def test_cache_prevents_duplicate_llm_call(self):
        from agent.hc_attrition.llm_orchestrator import LLMOrchestrator, _ResponseCache

        good_response = MagicMock()
        good_response.text = "SELECT 1"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = good_response

        orch = LLMOrchestrator(
            client=mock_client,
            fallback_chain=["model-a"],
            config={
                "max_retries": 1,
                "base_backoff_seconds": 0.001,
                "max_backoff_seconds": 0.01,
                "jitter": False,
                "rpm_per_session": 100,
                "rpm_global": 100,
                "circuit_breaker_failure_threshold": 5,
                "circuit_breaker_cooldown_seconds": 60,
                "circuit_breaker_half_open_max_calls": 1,
                "cache_ttl_seconds": 60,
                "cache_max_entries": 10,
            },
        )
        cache_key = orch.cache_key_for("same query", "table", {})

        # First call: LLM hit
        r1, m1, _ = orch.generate_content("prompt1", cache_key=cache_key)
        assert mock_client.models.generate_content.call_count == 1

        # Second call with same key: cache hit
        r2, m2, events2 = orch.generate_content("prompt1", cache_key=cache_key)
        assert mock_client.models.generate_content.call_count == 1  # No new calls
        assert m2 == "cache"
        assert "cache_hit" in events2


# ===========================================================================
# E) Integration: simulated 429/503 → graceful fallback
# ===========================================================================

class TestIntegrationFallback:
    """End-to-end tests simulating quota exhaustion through the QueryRouter."""

    def _make_router_and_orch(self, client):
        from agent.hc_attrition.query_router import QueryRouter
        from agent.hc_attrition.llm_orchestrator import LLMOrchestrator

        cfg = {
            "deterministic_first": True,
            "confidence_threshold_deterministic": 0.75,
            "confidence_threshold_tier2": 0.40,
            "fallback_chain": ["model-primary", "model-fallback", "template-only"],
            "max_retries": 2,
            "base_backoff_seconds": 0.001,
            "max_backoff_seconds": 0.01,
            "jitter": False,
            "rpm_per_session": 100,
            "rpm_global": 100,
            "circuit_breaker_failure_threshold": 5,
            "circuit_breaker_cooldown_seconds": 60,
            "circuit_breaker_half_open_max_calls": 1,
            "cache_ttl_seconds": 60,
            "cache_max_entries": 10,
        }
        router = QueryRouter(config=cfg)
        orch = LLMOrchestrator(client=client, fallback_chain=cfg["fallback_chain"], config=cfg)
        return router, orch

    def test_attrition_query_routes_deterministic_no_llm(self):
        """'show me attrition for June 2025' must resolve via Tier 1 template."""
        from agent.hc_attrition.query_router import RouteTier

        mock_client = MagicMock()
        router, _ = self._make_router_and_orch(mock_client)

        route = router.route(
            "show me attrition for June 2025",
            table_name="fact_attrition",
            table_columns=[
                "empid", "empstatus", "lwd", "final_exit_type",
                "tenure_months", "department", "endofmonth",
            ],
        )
        assert route.tier == RouteTier.DETERMINISTIC, (
            f"Expected DETERMINISTIC, got {route.tier}. Intent={route.intent_key}, "
            f"confidence={route.confidence:.2f}"
        )
        assert route.sql is not None
        assert "2025-06-01" in route.sql
        mock_client.models.generate_content.assert_not_called()

    def test_headcount_query_routes_deterministic(self):
        """A headcount query must also resolve via Tier 1 template."""
        from agent.hc_attrition.query_router import RouteTier

        mock_client = MagicMock()
        router, _ = self._make_router_and_orch(mock_client)

        route = router.route(
            "What is the total active headcount this month?",
            table_name="fact_headcount",
            table_columns=[
                "empid", "empstatus", "endofmonth",
                "department", "grade", "gender", "lob",
            ],
        )
        assert route.tier == RouteTier.DETERMINISTIC
        assert route.sql is not None
        mock_client.models.generate_content.assert_not_called()

    def test_novel_query_routes_to_tier3(self):
        """An unknown complex query must route to Tier 3 (full LLM)."""
        from agent.hc_attrition.query_router import RouteTier

        mock_client = MagicMock()
        router, _ = self._make_router_and_orch(mock_client)

        route = router.route(
            "What is the correlation between employee satisfaction scores and attrition?",
            table_name="custom_table",
            table_columns=["id", "score", "year"],
        )
        assert route.tier == RouteTier.LLM_TIER3

    def test_repeated_attrition_queries_no_cascade_429(self):
        """Repeated identical attrition queries must hit the cache — no LLM calls."""
        from agent.hc_attrition.llm_orchestrator import LLMOrchestrator

        mock_client = MagicMock()
        good_response = MagicMock()
        good_response.text = "SELECT 1"
        mock_client.models.generate_content.return_value = good_response

        orch = LLMOrchestrator(
            client=mock_client,
            fallback_chain=["model-a"],
            config={
                "max_retries": 1,
                "base_backoff_seconds": 0.001,
                "max_backoff_seconds": 0.01,
                "jitter": False,
                "rpm_per_session": 100,
                "rpm_global": 100,
                "circuit_breaker_failure_threshold": 5,
                "circuit_breaker_cooldown_seconds": 60,
                "circuit_breaker_half_open_max_calls": 1,
                "cache_ttl_seconds": 60,
                "cache_max_entries": 100,
            },
        )
        key = orch.cache_key_for("attrition june 2025", "fact_attrition", {})

        # First call — LLM
        orch.generate_content("prompt", cache_key=key)
        call_count_after_first = mock_client.models.generate_content.call_count

        # 9 more identical calls — all should hit cache
        for _ in range(9):
            _, m, events = orch.generate_content("prompt", cache_key=key)
            assert m == "cache"

        assert mock_client.models.generate_content.call_count == call_count_after_first


# ===========================================================================
# F) Catalogue loading
# ===========================================================================

class TestCatalogueLoader:
    def test_load_catalogue_from_csv(self, tmp_path):
        from agent.hc_attrition.catalogue import load_catalogue
        import csv

        csv_path = tmp_path / "test_catalogue.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id", "audience", "theme", "business_question",
                    "question_type", "time_grain", "key_filters_slicers",
                    "metric_kpi", "source_columns_raw", "fact_table",
                    "dimension_tables", "priority",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "id": "1",
                "audience": "Both",
                "theme": "Headcount",
                "business_question": "What is the total active headcount?",
                "question_type": "Snapshot",
                "time_grain": "Monthly",
                "key_filters_slicers": "Department|Grade",
                "metric_kpi": "Active HC",
                "source_columns_raw": "empid|empstatus|department",
                "fact_table": "fact_headcount",
                "dimension_tables": "dim_employee|dim_date",
                "priority": "P1",
            })

        catalogue = load_catalogue(str(csv_path))
        assert 1 in catalogue
        entry = catalogue[1]
        assert entry.theme == "Headcount"
        assert entry.priority == "P1"
        assert "Department" in entry.key_filters_slicers

    def test_load_catalogue_missing_file(self):
        from agent.hc_attrition.catalogue import load_catalogue
        catalogue = load_catalogue("/nonexistent/path/catalogue.csv")
        assert catalogue == {}

    def test_load_real_catalogue(self):
        """The bundled catalogue.csv must load all 25 entries."""
        from agent.hc_attrition.catalogue import load_catalogue, DEFAULT_CATALOGUE_PATH
        if not os.path.isfile(DEFAULT_CATALOGUE_PATH):
            pytest.skip("catalogue.csv not found at default path")
        catalogue = load_catalogue(DEFAULT_CATALOGUE_PATH)
        assert len(catalogue) == 25
        # Every entry must have a non-empty intent_key
        for entry in catalogue.values():
            assert entry.intent_key, f"Entry {entry.id} has empty intent_key"


# ===========================================================================
# G) QueryRouter tier transitions
# ===========================================================================

class TestQueryRouter:
    def _make_router(self, det_threshold=0.75, t2_threshold=0.40):
        from agent.hc_attrition.query_router import QueryRouter
        return QueryRouter(config={
            "deterministic_first": True,
            "confidence_threshold_deterministic": det_threshold,
            "confidence_threshold_tier2": t2_threshold,
            "fallback_chain": ["gemini-flash", "gemini-lite", "template-only"],
        })

    def test_deterministic_first_disabled(self):
        """With deterministic_first=False, always goes to LLM."""
        from agent.hc_attrition.query_router import QueryRouter, RouteTier
        router = QueryRouter(config={
            "deterministic_first": False,
            "confidence_threshold_deterministic": 0.75,
            "confidence_threshold_tier2": 0.40,
            "fallback_chain": ["gemini-flash"],
        })
        route = router.route(
            "What is the total active headcount?",
            table_name="fact_headcount",
            table_columns=["empid", "empstatus", "endofmonth", "department", "grade"],
        )
        # Even with a high-confidence match, deterministic should be skipped
        assert route.tier != RouteTier.DETERMINISTIC

    def test_attrition_columns_detected(self):
        from agent.hc_attrition.query_router import QueryRouter
        router = QueryRouter._table_looks_like_hc_or_attrition
        assert router(["empid", "empstatus", "lwd", "final_exit_type", "tenure_months", "department"])
        assert not router(["customer_id", "product", "price"])


# ===========================================================================
# H) Observability tracker
# ===========================================================================

class TestObservability:
    def test_trace_lifecycle(self):
        from agent.hc_attrition.observability import ObservabilityTracker

        tracker = ObservabilityTracker()
        trace = tracker.start_trace("test query")
        trace.route_tier = "deterministic"
        trace.intent_key = "attrition_overall"
        trace.confidence = 0.90
        trace.model_used = "deterministic"
        trace.finish("success")
        tracker.finish_trace(trace)

        metrics = tracker.summary()
        assert metrics["total_queries"] == 1
        assert metrics["deterministic_hits"] == 1
        assert metrics["deterministic_hit_rate_pct"] == 100.0

    def test_why_routed_summary(self):
        from agent.hc_attrition.observability import QueryTrace

        trace = QueryTrace(query="show me attrition for June 2025")
        trace.route_tier = "deterministic"
        trace.intent_key = "attrition_overall"
        trace.confidence = 0.92
        trace.model_used = "deterministic"
        trace.finish("success")

        summary = trace.why_routed_summary()
        assert "Tier 1" in summary
        assert "attrition_overall" in summary

    def test_counters_increment(self):
        from agent.hc_attrition.observability import ObservabilityTracker

        tracker = ObservabilityTracker()
        for _ in range(3):
            t = tracker.start_trace("q")
            t.route_tier = "llm_tier2"
            t.finish()
            tracker.finish_trace(t)

        m = tracker.summary()
        assert m["total_queries"] == 3
        assert m["llm_calls"] == 3
        assert m["deterministic_hits"] == 0
