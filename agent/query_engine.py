"""
Query Engine Module
===================
Handles natural language to SQL translation using Gemini, executes
SQL on DataFrames via DuckDB, and generates smart visualizations.

Integrates the Quota-Safe Query Orchestrator:
  * Deterministic-first routing via the 25-question HC/Attrition catalogue.
  * Per-model circuit breaker + exponential backoff for 429/503 errors.
  * Configurable fallback chain and response cache.
  * Per-query observability traces exposed in the UI "Show SQL & Reasoning".

Uses the new ``google-genai`` SDK (v1 API) for Streamlit Cloud compatibility.
"""

import logging
import re
import json
import time
import duckdb
import pandas as pd
from google import genai
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
MAX_BACKOFF_DELAY_SECONDS = 3

# Default orchestrator config (mirrors config.yaml defaults).
_DEFAULT_ORCH_CONFIG: Dict[str, Any] = {
    "deterministic_first": True,
    "max_llm_calls_per_query": 1,
    "default_model": "gemini-2.0-flash",
    "fallback_chain": ["gemini-2.0-flash", "gemini-2.0-flash-lite", "template-only"],
    "confidence_threshold_deterministic": 0.75,
    "confidence_threshold_tier2": 0.40,
    "rpm_per_session": 10,
    "rpm_global": 60,
    "token_budget_per_query": 8000,
    "max_retries": 3,
    "base_backoff_seconds": 1.0,
    "max_backoff_seconds": 60.0,
    "jitter": True,
    "circuit_breaker_failure_threshold": 3,
    "circuit_breaker_cooldown_seconds": 60,
    "circuit_breaker_half_open_max_calls": 1,
    "cache_ttl_seconds": 300,
    "cache_max_entries": 500,
    "catalogue_path": "catalogue.csv",
}


class QueryEngine:
    """Translates Natural Language to SQL, executes it,
    generates a plain-language summary, and suggests visualizations.

    Quota-safe additions
    --------------------
    * ``generate_sql()`` first attempts deterministic routing (Tier 1) for
      known HC/Attrition questions.  Only falls through to the LLM when
      no template matches or the table schema doesn't fit.
    * The ``LLMOrchestrator`` handles retry / backoff / circuit-breaker /
      fallback-chain transparently.
    * Each query produces a ``QueryTrace`` accessible via
      ``self.last_trace`` for display in "Show SQL & Reasoning".
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        model_names: Optional[List[str]] = None,
        orchestrator_config: Optional[Dict[str, Any]] = None,
    ):
        if not api_key:
            raise ValueError("Gemini API key is required for AI Query functionality.")

        self.client = genai.Client(api_key=api_key)
        default_models = [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash-001",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.5-pro-preview-03-25",
        ]
        requested_models = model_names or [model_name]
        combined_models = requested_models + default_models
        self.model_candidates: List[str] = []
        seen: set = set()
        for m in combined_models:
            if m and m not in seen:
                self.model_candidates.append(m)
                seen.add(m)
        self.model_name = self.model_candidates[0]
        self.current_model_index = 0
        self.con = duckdb.connect(database=':memory:')
        self.chat = None
        self.table_names: List[str] = []
        self._preamble = ""

        # ---- Quota-safe orchestrator ----------------------------------------
        orch_cfg = dict(_DEFAULT_ORCH_CONFIG)
        if orchestrator_config:
            orch_cfg.update(orchestrator_config)

        # Build fallback chain from user-selected models + config defaults
        orch_cfg["fallback_chain"] = self._build_fallback_chain(
            self.model_candidates, orch_cfg.get("fallback_chain", [])
        )

        try:
            from agent.hc_attrition.llm_orchestrator import LLMOrchestrator
            from agent.hc_attrition.query_router import QueryRouter
            from agent.hc_attrition.observability import ObservabilityTracker

            self._orchestrator = LLMOrchestrator(
                client=self.client,
                fallback_chain=orch_cfg["fallback_chain"],
                config=orch_cfg,
            )
            self._router = QueryRouter(config=orch_cfg)
            self._obs = ObservabilityTracker()
            self._orch_cfg = orch_cfg
            logger.info(
                "Quota-safe orchestrator initialised | fallback_chain=%s",
                orch_cfg["fallback_chain"],
            )
        except Exception as exc:
            logger.warning("Could not initialise orchestrator (non-fatal): %s", exc)
            self._orchestrator = None
            self._router = None
            self._obs = None
            self._orch_cfg = orch_cfg

        # Last trace (set by generate_sql for UI display)
        self.last_trace: Optional[Any] = None

    def start_chat(self, results_dict: dict):
        """
        Initializes a stateful chat session with the schema of ALL tables.
        Tries multiple models until one works.
        """
        all_schemas = []
        self.table_names = list(results_dict.keys())

        for table_name, result in results_dict.items():
            schema_desc = self._build_schema_context(result.get("semantic", {}))
            filename = result.get("metadata", {}).get("filename", table_name)
            all_schemas.append(f"CRITICAL: YOU MUST USE THIS EXACT TABLE NAME IN SQL: \"{table_name}\" (File: {filename})\n{schema_desc}\n")

        full_schema_context = "\n---\n".join(all_schemas)

        system_instruction = f"""
You are an expert Data Analyst. Your goal is to translate natural language into SQL for the following tables: {', '.join(self.table_names)}.

--- SCHEMA CONTEXT (ALL TABLES) ---
{full_schema_context}
---

--- CRITICAL RULES ---
1. **EXACT TABLE NAMES**: You MUST use the exact table names provided (e.g., "{self.table_names[0]}"). Do NOT look at the file name or try to "fix" the table name.
2. **QUOTING**: ALWAYS enclose all table names and column names in double quotes (e.g., `"table_name"`, `"column_name"`). This is mandatory to avoid errors with hyphens or spaces.
3. **STRICT COLUMN ADHERENCE**: Use ONLY the exact 'raw_column' names listed in the schema.
4. **NON-TECH FRIENDLY**: Your EXPLANATION should be simple. Avoid talking about JOINs or GROUP BYs in the explanation; instead, talk about what the data represents.
5. **SQL FORMATTING**: Wrap SQL code in a markdown block:
   ```sql
   SELECT ...
   ```
6. **SUGGESTIONS**: Provide 2-3 logical follow-up questions.
7. Use standard SQL compatible with DuckDB.
8. Follow this format:
   EXPLANATION: [Business-friendly reasoning]
   SQL: [The SQL code block]
   STATUS: [SUCCESS or VERIFICATION_REQUIRED]
   CORRECTION_PROMPT: [If needed]
   SUGGESTIONS: [Follow-up questions]
---
"""
        self._preamble = f"SYSTEM INSTRUCTIONS:\n{system_instruction}\nPlease acknowledge and wait for the first user question."

        last_error = None
        for idx, m_name in enumerate(self.model_candidates):
            for attempt in range(1, 4):
                try:
                    logger.info(f"Trying model: {m_name} (attempt {attempt}/3)")
                    chat = self.client.chats.create(model=m_name)
                    chat.send_message(self._preamble)
                    self.chat = chat
                    self.model_name = m_name
                    self.current_model_index = idx
                    logger.info(f"✅ Connected to model: {m_name}")
                    return
                except Exception as e:
                    last_error = e
                    logger.warning(f"Model {m_name} failed (attempt {attempt}/3): {e}")
                    if attempt < 3 and self._is_retryable_error(e):
                        time.sleep(self._get_backoff_delay(attempt))
                        continue
                    break

        # List what's actually available to help debug
        try:
            available = [m.name for m in self.client.models.list()]
            available_str = ", ".join(available[:10])
        except Exception:
            available_str = "Could not list models"

        raise RuntimeError(
            f"Could not connect to any Gemini model. "
            f"Models tried: {', '.join(self.model_candidates)}. "
            f"Available in your account: {available_str}. "
            f"Last error: {last_error}"
        )

    def generate_sql(self, query: str) -> dict:
        """Translate *query* to SQL using deterministic routing or LLM fallback.

        Routing tiers (when the orchestrator is available)
        ---------------------------------------------------
        Tier 1 — Deterministic template (no LLM call):
            High-confidence match in the 25-question catalogue + table
            columns look like HC/Attrition data.

        Tier 2 — Lightweight LLM (flash):
            Moderate-confidence match; sends schema-trimmed prompt to a
            lightweight model via the quota-safe orchestrator.

        Tier 3 — Full LLM (pro):
            Low/no confidence; sends full prompt to the configured pro model.

        The ``last_trace`` attribute is populated after every call for display
        in the UI "Show SQL & Reasoning" expander.
        """
        if not self.chat:
            return {
                "sql": None,
                "full_text": "Error: Chat not initialized",
                "status": "ERROR",
                "suggestions": [],
                "route_info": None,
            }

        # ---- Deterministic-first routing ------------------------------------
        if self._router and self._orchestrator and self._obs:
            trace = self._obs.start_trace(
                query, preferred_model=self.model_name
            )
            try:
                result = self._route_and_generate(query, trace)
                self.last_trace = trace
                return result
            finally:
                self._obs.finish_trace(trace)

        # ---- Fallback: classic LLM-only path --------------------------------
        try:
            response = self._send_message_with_fallback(query)
            full_text = response.text.strip()
        except Exception as e:
            return {
                "sql": None,
                "full_text": f"Error: {e}",
                "status": "ERROR",
                "suggestions": [],
                "route_info": None,
            }
        return self._parse_structured_response(full_text)

    def _route_and_generate(self, query: str, trace: Any) -> dict:
        """Core routing + generation logic (used when orchestrator is active).

        Mutates *trace* in-place with routing decisions and events.
        """
        from agent.hc_attrition.query_router import RouteTier

        # Collect table column names for schema-aware routing
        table_columns = self._get_all_registered_columns()
        primary_table = self.table_names[0] if self.table_names else ""

        route = self._router.route(
            query,
            table_name=primary_table,
            table_columns=table_columns,
        )
        trace.route_tier = route.tier.value
        trace.intent_key = route.intent_key
        trace.confidence = route.confidence
        trace.catalogue_id = route.catalogue_id

        # ---- Tier 1: Deterministic ------------------------------------------
        if route.is_deterministic() and route.sql:
            # Check cache first (deterministic results can also be cached)
            cache_key = self._orchestrator.cache_key_for(
                query.lower().strip(),
                primary_table,
                {k: v for k, v in route.filters.items() if v is not None},
            )
            hit, cached_result = self._orchestrator.get_cache(cache_key)
            if hit:
                trace.cache_hit = True
                trace.model_used = "cache"
                trace.status = "success"
                logger.info("Deterministic cache hit for query=%r", query[:60])
                cached_result["route_info"] = trace.why_routed_summary()
                return cached_result

            result = {
                "sql": route.sql,
                "full_text": (
                    f"EXPLANATION: {route.why}\n"
                    f"STATUS: SUCCESS\n"
                    f"SUGGESTIONS: "
                    f"Show trend over time, "
                    f"Break down by department, "
                    f"Compare with last period"
                ),
                "explanation": route.why,
                "status": "SUCCESS",
                "correction_prompt": None,
                "suggestions": [
                    "Show trend over time",
                    "Break down by department",
                    "Compare with last period",
                ],
                "route_info": route.why,
            }
            trace.model_used = "deterministic"
            trace.sql = route.sql
            self._orchestrator.put_cache(cache_key, result)
            return result

        # ---- Tier 2 / 3: LLM path ------------------------------------------
        preferred = route.preferred_model or self.model_name
        trace.preferred_model = preferred

        cache_key = self._orchestrator.cache_key_for(
            query.lower().strip(),
            primary_table,
            {k: v for k, v in route.filters.items() if v is not None},
        )
        hit, cached_result = self._orchestrator.get_cache(cache_key)
        if hit:
            trace.cache_hit = True
            trace.model_used = "cache"
            cached_result["route_info"] = trace.why_routed_summary()
            return cached_result

        if not self.chat:
            return {
                "sql": None,
                "full_text": "Error: Chat not initialized",
                "status": "ERROR",
                "suggestions": [],
                "route_info": route.why,
            }

        try:
            response = self._send_message_with_fallback(query)
            full_text = response.text.strip()
        except Exception as e:
            trace.error_codes.append(str(e)[:60])
            return {
                "sql": None,
                "full_text": f"Error: {e}",
                "status": "ERROR",
                "suggestions": [],
                "route_info": route.why,
            }

        parsed = self._parse_structured_response(full_text)
        parsed["route_info"] = route.why
        trace.model_used = self.model_name
        trace.sql = parsed.get("sql")
        self._orchestrator.put_cache(cache_key, parsed)
        return parsed

    def _get_all_registered_columns(self) -> List[str]:
        """Return a flat list of column names from all registered tables."""
        columns: List[str] = []
        for t in self.table_names:
            try:
                # Use a parameterized query to avoid SQL injection via table names
                info = self.con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = ?",
                    [t],
                ).fetchall()
                columns.extend(row[0] for row in info)
            except Exception:
                pass
        return columns

    @staticmethod
    def _build_fallback_chain(
        user_models: List[str], config_chain: List[str]
    ) -> List[str]:
        """Merge user-selected models with the config-defined fallback chain.

        User-selected models are prepended; ``template-only`` sentinel is
        preserved at the end.
        """
        sentinel = "template-only"
        has_sentinel = sentinel in config_chain
        # Start with user models, then add config chain (excluding duplicates)
        combined: List[str] = []
        seen: set = set()
        for m in user_models + config_chain:
            if m == sentinel:
                continue
            if m not in seen:
                combined.append(m)
                seen.add(m)
        if has_sentinel:
            combined.append(sentinel)
        return combined

    def _send_message_with_fallback(self, query: str):
        """Sends message with retry + model fallback."""
        if not self.chat:
            raise RuntimeError("Chat not initialized.")

        model_attempt_order = self._get_model_rotation_order()
        last_error = None
        for m_name in model_attempt_order:
            if m_name != self.model_name:
                self._reconnect_chat(m_name)
            for attempt in range(1, 4):
                try:
                    return self.chat.send_message(query)
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Query send failed on model {m_name} (attempt {attempt}/3): {e}"
                    )
                    if attempt < 3 and self._is_retryable_error(e):
                        time.sleep(self._get_backoff_delay(attempt))
                        continue
                    break

        raise RuntimeError(f"All configured models failed to answer. Last error: {last_error}")

    def _reconnect_chat(self, model_name: str):
        """Recreate chat context on a different model."""
        chat = self.client.chats.create(model=model_name)
        chat.send_message(self._preamble)
        self.chat = chat
        self.model_name = model_name
        self.current_model_index = self.model_candidates.index(model_name)
        logger.info(f"Reconnected chat using fallback model: {model_name}")

    def _is_retryable_error(self, err: Exception) -> bool:
        """Detect transient provider/network failures."""
        msg = str(err).lower()
        transient_markers = [
            "503",
            "unavailable",
            "timeout",
            "temporar",
            "rate limit",
            "deadline exceeded",
            "connection reset",
            "service unavailable",
        ]
        return any(marker in msg for marker in transient_markers)

    def _generate_content_with_fallback(self, prompt: str):
        """Calls generate_content with retry + model fallback."""
        model_attempt_order = self._get_model_rotation_order()
        last_error = None
        for m_name in model_attempt_order:
            for attempt in range(1, 4):
                try:
                    response = self.client.models.generate_content(model=m_name, contents=prompt)
                    self.model_name = m_name
                    self.current_model_index = self.model_candidates.index(m_name)
                    return response
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"generate_content failed on model {m_name} (attempt {attempt}/3): {e}"
                    )
                    if attempt < 3 and self._is_retryable_error(e):
                        time.sleep(self._get_backoff_delay(attempt))
                        continue
                    break
        raise RuntimeError(f"All configured models failed to generate content. Last error: {last_error}")

    def _get_model_rotation_order(self) -> list[str]:
        """Returns candidates starting with current active model."""
        return self.model_candidates[self.current_model_index:] + self.model_candidates[:self.current_model_index]

    def _get_backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with a short cap for UI responsiveness."""
        return min(2 ** (attempt - 1), MAX_BACKOFF_DELAY_SECONDS)

    def _parse_structured_response(self, text: str) -> dict:
        """Parses the LLM's structured response into a clean dictionary."""
        res = {
            "sql": self._extract_sql(text),
            "full_text": text,
            "explanation": "",
            "status": "SUCCESS",
            "correction_prompt": None,
            "suggestions": []
        }

        # Extract Explanation
        exp_match = re.search(r"EXPLANATION:\s*(.*?)(?=SQL:|STATUS:|CORRECTION_PROMPT:|SUGGESTIONS:|```|$)", text, re.DOTALL | re.IGNORECASE)
        if exp_match:
            res["explanation"] = exp_match.group(1).strip()

        # Extract Status
        status_match = re.search(r"STATUS:\s*(VERIFICATION_REQUIRED|SUCCESS)", text, re.IGNORECASE)
        if status_match:
            res["status"] = status_match.group(1).upper()

        # Extract Correction Prompt
        cp_match = re.search(r"CORRECTION_PROMPT:\s*(.*)", text, re.IGNORECASE)
        if cp_match:
            res["correction_prompt"] = cp_match.group(1).strip()

        # Extract Suggestions
        sug_match = re.search(r"SUGGESTIONS:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if sug_match:
            s_list = sug_match.group(1).strip().split("\n")
            res["suggestions"] = [s.strip("- 1234567890.").strip() for s in s_list if s.strip() and len(s.strip()) > 5]

        return res

    def _extract_sql(self, text: str) -> Optional[str]:
        """Extracts SQL from the LLM response. Tries multiple patterns."""
        # 1. Preferred: ```sql ... ```
        match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # 2. Any code block containing SELECT/WITH
        match = re.search(r"```\s*(SELECT|WITH)[\s\S]*?```", text, re.IGNORECASE)
        if match:
            return match.group(0).replace("```", "").strip()

        # 3. Raw SELECT/WITH statement (greedy until next section)
        match = re.search(r"(SELECT|WITH)[\s\S]+?(?=STATUS:|CORRECTION_PROMPT:|SUGGESTIONS:|$)", text, re.IGNORECASE)
        if match:
            return match.group(0).strip().rstrip(";")

        return None

    def execute_query(self, sql: str, results_dict: dict) -> pd.DataFrame:
        """
        Executes a SQL query against ALL provided DataFrames using DuckDB.
        If the query fails due to a wrong table name (e.g. hyphens vs underscores),
        it automatically corrects the table name and retries once.
        """
        registered_tables = []
        for table_name, result in results_dict.items():
            df = result.get("dataframe")
            if df is not None:
                self.con.register(table_name, df)
                registered_tables.append(table_name)

        try:
            return self.con.execute(sql).df()
        except Exception as e:
            error_str = str(e)
            logger.warning(f"SQL execution failed (attempt 1): {error_str}")

            # --- Auto-correct wrong table names ---
            # Normalise by collapsing all separators to nothing for fuzzy matching
            def _norm(s: str) -> str:
                return re.sub(r"[^a-z0-9]", "", s.lower())

            corrected_sql = sql
            made_correction = False
            for reg_table in registered_tables:
                # Find any quoted or unquoted token in the SQL that looks like
                # a table name but has wrong separators (hyphens vs underscores)
                pattern = re.compile(
                    r'"([^"]+)"|(?<!\w)(' + re.escape(reg_table.replace("_", "-")) + r')(?!\w)',
                    re.IGNORECASE
                )
                for m in pattern.finditer(sql):
                    candidate = m.group(1) or m.group(2)
                    if candidate and _norm(candidate) == _norm(reg_table) and candidate != reg_table:
                        corrected_sql = corrected_sql.replace(
                            f'"{candidate}"', f'"{reg_table}"'
                        ).replace(candidate, f'"{reg_table}"')
                        made_correction = True
                        logger.info(f"Auto-corrected table name: '{candidate}' → '{reg_table}'")

            # Also do a broader replacement: any quoted string whose normalised
            # form matches a registered table
            quoted_names = re.findall(r'"([^"]+)"', sql)
            for qname in quoted_names:
                for reg_table in registered_tables:
                    if _norm(qname) == _norm(reg_table) and qname != reg_table:
                        corrected_sql = corrected_sql.replace(f'"{qname}"', f'"{reg_table}"')
                        made_correction = True
                        logger.info(f"Auto-corrected quoted table name: '{qname}' → '{reg_table}'")

            if made_correction:
                try:
                    logger.info(f"Retrying with corrected SQL: {corrected_sql}")
                    return self.con.execute(corrected_sql).df()
                except Exception as e2:
                    logger.error(f"SQL execution failed after correction: {e2}")
                    raise RuntimeError(f"Error executing SQL: {e2}")

            raise RuntimeError(f"Error executing SQL: {error_str}")

    def summarize_results(self, df_result: pd.DataFrame, original_query: str) -> str:
        """
        Generates a plain-language summary of the data result for non-technical users.
        """
        if df_result is None or df_result.empty:
            return "I couldn't find any data matching your request."

        cols = df_result.columns.tolist()
        num_rows = len(df_result)

        prompt = f"""You are a senior data analyst. Answer the user's question based on the provided data.
Question: {original_query}

Data ({num_rows} rows):
{df_result.head(15).to_markdown()}

Guidelines:
1. Be concise (2-4 sentences).
2. Use a friendly, professional tone.
3. Highlight key findings, trends, or outliers.
4. If there's a single main takeaway, lead with it.
5. Do NOT mention SQL or technical column names if they are messy.
"""

        try:
            response = self._generate_content_with_fallback(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            # Fallback: generate a basic summary without AI
            if num_rows == 0:
                return "No records matched your request."
            return f"Found **{num_rows} records** matching your query."

    def generate_visualization(self, df_result: pd.DataFrame, original_query: str) -> dict:
        """
        Suggests and builds the best Plotly chart for the result data.
        Falls back to heuristics if AI suggestion fails.
        """
        if df_result is None or df_result.empty:
            return {"type": None, "fig": None, "insight": "No data to visualize."}

        cols = df_result.columns.tolist()
        num_rows = len(df_result)

        # Ask AI for best chart type
        prompt = f"""You are a data visualization expert.
Given this query and data, suggest the best chart type.

Query: {original_query}
Columns: {', '.join(cols)}
Row count: {num_rows}
Sample data (first 5 rows):
{df_result.head(5).to_markdown()}

AVAILABLE CHART TYPES: bar, line, scatter, pie, histogram, table

Return ONLY a valid JSON object (no extra text):
{{
  "chart_type": "one of the above",
  "x": "column name for X axis",
  "y": "column name for Y axis (null if not needed)",
  "title": "A descriptive chart title",
  "insight": "One sentence insight from the data"
}}"""

        try:
            response = self._generate_content_with_fallback(prompt)
            raw = response.text.strip()
            match = re.search(r"\{[\s\S]*?\}", raw)
            if match:
                spec = json.loads(match.group())
                return self._create_plotly_figure(df_result, spec)
        except Exception as e:
            logger.warning(f"AI visualization failed: {e}. Using heuristics.")

        return self._heuristic_visualization(df_result)

    def _build_schema_context(self, semantic_layer: dict) -> str:
        """Serializes the semantic layer for the system prompt."""
        context = []

        if semantic_layer.get("dimensions"):
            context.append("Dimensions (Categories):")
            for d in semantic_layer["dimensions"]:
                val_sample = ", ".join(d.get("values", [])[:3])
                context.append(f"  - {d['raw_column']}: {d['description']}. Examples: [{val_sample}]")

        if semantic_layer.get("measures"):
            context.append("\nMeasures (Numbers):")
            for m in semantic_layer["measures"]:
                context.append(f"  - {m['raw_column']}: {m['description']} (Default agg: {m.get('aggregation', 'sum')})")

        if semantic_layer.get("time_fields"):
            context.append("\nTime Fields:")
            for t in semantic_layer["time_fields"]:
                context.append(f"  - {t['raw_column']}: {t['description']}")

        if semantic_layer.get("kpis"):
            context.append("\nCalculated KPIs:")
            for k in semantic_layer["kpis"]:
                context.append(f"  - {k['name']}: {k['description']} (Formula: {k['formula']})")

        return "\n".join(context)

    def _create_plotly_figure(self, df: pd.DataFrame, spec: dict) -> dict:
        """Builds a Plotly figure from the AI-suggested spec."""
        import plotly.express as px

        chart_type = spec.get("chart_type", "table")
        x = spec.get("x")
        y = spec.get("y")
        title = spec.get("title", "Query Result")
        insight = spec.get("insight", "")

        # Validate columns exist
        if x and x not in df.columns:
            x = df.columns[0]
        if y and y not in df.columns:
            y = df.columns[1] if len(df.columns) > 1 else None

        fig = None
        try:
            if chart_type == "bar" and x and y:
                fig = px.bar(df, x=x, y=y, title=title, template="plotly_white", color_discrete_sequence=["#2d6a9f"])
            elif chart_type == "line" and x and y:
                fig = px.line(df, x=x, y=y, title=title, template="plotly_white")
            elif chart_type == "scatter" and x and y:
                fig = px.scatter(df, x=x, y=y, title=title, template="plotly_white")
            elif chart_type == "pie" and x and y:
                fig = px.pie(df, names=x, values=y, title=title, template="plotly_white")
            elif chart_type == "histogram" and x:
                fig = px.histogram(df, x=x, title=title, template="plotly_white", color_discrete_sequence=["#2d6a9f"])
        except Exception as e:
            logger.warning(f"Failed to create chart: {e}")
            fig = None

        if fig:
            fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=420)
            return {"type": chart_type, "fig": fig, "insight": insight}

        return self._heuristic_visualization(df)

    def _heuristic_visualization(self, df: pd.DataFrame) -> dict:
        """Simple rule-based chart when AI suggestion fails."""
        import plotly.express as px
        import plotly.graph_objects as go

        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        try:
            # Case 1: Single Number (Metric)
            if len(df) == 1 and num_cols:
                val = float(df[num_cols[0]].iloc[0])
                fig = go.Figure(go.Indicator(
                    mode="number",
                    value=val,
                    title={"text": num_cols[0].replace("_", " ").title()},
                    number={"font": {"size": 60}, "valueformat": ",.2f"}
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                return {"type": "kpi", "fig": fig, "insight": f"The calculated {num_cols[0]} is {val:,.2f}."}

            # Case 2: Categories and Numbers (Bar Chart)
            if cat_cols and num_cols:
                fig = px.bar(df.head(20), x=cat_cols[0], y=num_cols[0],
                             title=f"{num_cols[0]} by {cat_cols[0]}", template="plotly_white",
                             color_discrete_sequence=["#2d6a9f"])
                fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=420)
                return {"type": "bar", "fig": fig, "insight": f"Breakdown of {num_cols[0]} across {cat_cols[0]}."}

            # Case 3: List of numbers (Histogram or Bar by Index)
            elif num_cols:
                if len(df) <= 10:
                    # Small list (e.g. top 5 ages) -> Bar chart by index
                    fig = px.bar(df, x=df.index.astype(str), y=num_cols[0],
                                 title=f"Values for {num_cols[0]}", labels={"index": "Row", num_cols[0]: "Value"},
                                 template="plotly_white", color_discrete_sequence=["#2d6a9f"])
                else:
                    # Larger list -> Histogram
                    fig = px.histogram(df, x=num_cols[0], title=f"Distribution of {num_cols[0]}",
                                       template="plotly_white", color_discrete_sequence=["#2d6a9f"])
                
                fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=420)
                return {"type": "histogram", "fig": fig, "insight": f"Visualization of the values in {num_cols[0]}."}

        except Exception as e:
            logger.warning(f"Heuristic chart failed: {e}")

        return {"type": "table", "fig": None, "insight": "Best viewed as a table."}
