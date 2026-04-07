"""
Query Engine Module
===================
Handles natural language to SQL translation using Gemini, executes 
SQL on DataFrames via DuckDB, and generates smart visualizations.

Uses the new `google-genai` SDK (v1 API) for Streamlit Cloud compatibility.
"""

import logging
import re
import json
import duckdb
import pandas as pd
from google import genai
from typing import Optional

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Translates Natural Language to SQL, executes it, 
    generates a plain-language summary, and suggests visualizations.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        if not api_key:
            raise ValueError("Gemini API key is required for AI Query functionality.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.con = duckdb.connect(database=':memory:')
        self.chat = None
        self.table_names = []

    def start_chat(self, results_dict: dict):
        """
        Initializes a stateful chat session with the schema of ALL tables.
        Tries multiple models until one works.
        """
        all_schemas = []
        self.table_names = list(results_dict.keys())

        for table_name, result in results_dict.items():
            schema_desc = self._build_schema_context(result.get("semantic", {}))
            all_schemas.append(f"TABLE: '{table_name}'\n{schema_desc}\n")

        full_schema_context = "\n---\n".join(all_schemas)

        system_instruction = f"""
You are an expert Data Analyst and SQL Engineer. Your goal is to translate natural language into SQL for the following tables: {', '.join(self.table_names)}.

--- SCHEMA CONTEXT (ALL TABLES) ---
{full_schema_context}
---

--- CRITICAL RULES ---
1. **MULTI-TABLE JOINS**: If the query spans multiple tables, use standard SQL JOINs with the correct table names.
2. **STRICT COLUMN ADHERENCE**: Use ONLY the exact 'raw_column' names listed.
3. **VERIFICATION**: If you are unsure about a column mapping, STOP and output a 'VERIFICATION_REQUIRED' block.
4. **SQL FORMATTING**: ALWAYS wrap your SQL code in a markdown block like this:
   ```sql
   SELECT * FROM table...
   ```
5. **SUGGESTIONS**: After every successful SQL generation, provide 2-3 'SMART_SUGGESTIONS' for the next logical query.
6. Use standard SQL compatible with DuckDB.
7. The response MUST follow this structured format:
   EXPLANATION: [Brief technical reasoning]
   SQL: [The SQL code block]
   STATUS: [SUCCESS or VERIFICATION_REQUIRED]
   CORRECTION_PROMPT: [If VERIFICATION_REQUIRED, the question to ask the user]
   SUGGESTIONS: [List of 2-3 suggested follow-up questions]
---
"""
        preamble = f"SYSTEM INSTRUCTIONS:\n{system_instruction}\nPlease acknowledge and wait for the first user question."

        # Try the selected model first, then fallbacks
        models_to_try = [
            self.model_name,
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash-001",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ]
        # Deduplicate while preserving order
        seen = set()
        models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

        last_error = None
        for m_name in models_to_try:
            try:
                logger.info(f"Trying model: {m_name}")
                chat = self.client.chats.create(model=m_name)
                chat.send_message(preamble)
                self.chat = chat
                self.model_name = m_name
                logger.info(f"✅ Connected to model: {m_name}")
                return
            except Exception as e:
                last_error = e
                logger.warning(f"Model {m_name} failed: {e}")
                continue

        # List what's actually available to help debug
        try:
            available = [m.name for m in self.client.models.list()]
            available_str = ", ".join(available[:10])
        except Exception:
            available_str = "Could not list models"

        raise RuntimeError(
            f"Could not connect to any Gemini model. "
            f"Models tried: {', '.join(models_to_try)}. "
            f"Available in your account: {available_str}. "
            f"Last error: {last_error}"
        )

    def generate_sql(self, query: str) -> dict:
        """
        Sends a natural language query and returns a structured dict with SQL + metadata.
        """
        if not self.chat:
            return {"sql": None, "full_text": "Error: Chat not initialized", "status": "ERROR", "suggestions": []}

        try:
            response = self.chat.send_message(query)
            full_text = response.text.strip()
        except Exception as e:
            return {"sql": None, "full_text": f"Error: {e}", "status": "ERROR", "suggestions": []}

        return self._parse_structured_response(full_text)

    def _parse_structured_response(self, text: str) -> dict:
        """Parses the LLM's structured response into a clean dictionary."""
        res = {
            "sql": self._extract_sql(text),
            "full_text": text,
            "status": "SUCCESS",
            "correction_prompt": None,
            "suggestions": []
        }

        status_match = re.search(r"STATUS:\s*(VERIFICATION_REQUIRED|SUCCESS)", text, re.IGNORECASE)
        if status_match:
            res["status"] = status_match.group(1).upper()

        cp_match = re.search(r"CORRECTION_PROMPT:\s*(.*)", text, re.IGNORECASE)
        if cp_match:
            res["correction_prompt"] = cp_match.group(1).strip()

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
        """Executes a SQL query against ALL provided DataFrames using DuckDB."""
        for table_name, result in results_dict.items():
            df = result.get("dataframe")
            if df is not None:
                self.con.register(table_name, df)

        try:
            return self.con.execute(sql).df()
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            raise RuntimeError(f"Error executing SQL: {e}")

    def summarize_results(self, df_result: pd.DataFrame, original_query: str) -> str:
        """
        Generates a plain-language summary of the data result for non-technical users.
        """
        if df_result is None or df_result.empty:
            return "I couldn't find any data matching your request."

        cols = df_result.columns.tolist()
        num_rows = len(df_result)

        prompt = f"""You are a helpful Data Assistant speaking to a non-technical business user.
Answer their question clearly and concisely based ONLY on the data shown below.
Do NOT mention SQL, code, or technical jargon. Be direct and friendly.

User Question: {original_query}

Results ({num_rows} rows returned, Columns: {', '.join(cols)}):
{df_result.head(10).to_markdown()}

Write a short, clear answer in 1-3 sentences. Include key numbers or counts where relevant."""

        try:
            response = self.client.models.generate_content(model=self.model_name, contents=prompt)
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
            response = self.client.models.generate_content(model=self.model_name, contents=prompt)
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

        cols = df.columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        try:
            if cat_cols and num_cols:
                fig = px.bar(df.head(20), x=cat_cols[0], y=num_cols[0],
                             title=f"{num_cols[0]} by {cat_cols[0]}", template="plotly_white",
                             color_discrete_sequence=["#2d6a9f"])
                fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=420)
                return {"type": "bar", "fig": fig, "insight": "Auto-generated bar chart."}
            elif num_cols:
                fig = px.histogram(df, x=num_cols[0], title=f"Distribution of {num_cols[0]}",
                                   template="plotly_white", color_discrete_sequence=["#2d6a9f"])
                fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=420)
                return {"type": "histogram", "fig": fig, "insight": "Auto-generated histogram."}
        except Exception as e:
            logger.warning(f"Heuristic chart failed: {e}")

        return {"type": "table", "fig": None, "insight": "Best viewed as a table."}
