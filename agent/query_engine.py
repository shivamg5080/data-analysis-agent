"""
Query Engine Module
===================
Handles natural language to SQL translation using Gemini, executes 
SQL on DataFrames via DuckDB, and generates smart visualizations.
"""

import logging
import re
import json
import duckdb
import pandas as pd
import google.generativeai as genai
from typing import Any, Tuple, Optional

logger = logging.getLogger(__name__)

class QueryEngine:
    """
    Translates Natural Language to SQL, executes it, 
    and suggests visualizations.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        if not api_key:
            raise ValueError("Gemini API key is required for AI Query functionality.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.con = duckdb.connect(database=':memory:')
        self.chat = None
        self.table_names = []

    def start_chat(self, results_dict: dict[str, dict]):
        """
        Initializes a stateful chat session with the schema of ALL tables.
        
        results_dict: dict where keys are table names and values are 
                     the pipeline result dictionaries (containing 'semantic').
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
        # Start the chat session
        self.chat = None
        models_to_try = [self.model.model_name, "gemini-1.5-flash-latest", "gemini-1.5-pro", "gemini-pro"]
        
        last_error = None
        for m_name in models_to_try:
            try:
                # Remove 'models/' prefix if present for instantiation assdk will add it
                clean_name = m_name.replace("models/", "")
                self.model = genai.GenerativeModel(clean_name)
                self.chat = self.model.start_chat(history=[])
                self.chat.send_message(f"SYSTEM INSTRUCTIONS:\n{system_instruction}\nPlease acknowledge and wait for the first user question.")
                logger.info(f"Successfully initialized AI Assistant with model: {clean_name}")
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to initialize AI Assistant with model {m_name}: {e}")
                continue
        
        if self.chat is None:
            if "PermissionDenied" in str(last_error) or "403" in str(last_error):
                raise PermissionError(
                    "Google Generative AI: Permission Denied (403).\n"
                    "Possible Fixes:\n"
                    "1. Check if your Gemini API key is valid.\n"
                    "2. Ensure the 'Generative Language API' is ENABLED in your Google Cloud Console for the associated project.\n"
                    "3. Check for regional availability—some regions have limited access to certain models.\n"
                    "4. If you recently restricted the key to specific APIs, ensure the Generative Language API is included."
                ) from last_error
            raise RuntimeError(f"Failed to initialize AI Assistant across all fallback models: {last_error}") from last_error

    def generate_sql(self, query: str, results_dict: dict = None) -> dict:
        """
        Generates SQL and metadata. Returns a structured dictionary.
        """
        if self.chat:
            response = self.chat.send_message(query)
            full_text = response.text.strip()
        else:
            # Fallback (stateless) implementation for backward compatibility
            return {"sql": None, "full_text": "Error: Chat not initialized", "status": "ERROR"}
        
        return self._parse_structured_response(full_text)

    def get_correction(self, error_message: str, user_feedback: str = None) -> dict:
        """
        Asks the AI to correct the previous SQL. Returns a structured dictionary.
        """
        if not self.chat:
            return {"sql": None, "full_text": "Error: Chat session not initialized.", "status": "ERROR"}
            
        feedback_prompt = "The previous attempt failed or requires correction."
        if error_message:
            feedback_prompt += f"\nEXECUTION ERROR: {error_message}"
        if user_feedback:
            feedback_prompt += f"\nUSER FEEDBACK: {user_feedback}"
            
        feedback_prompt += "\nPlease analyze and provide the CORRECTED result using the structured format."
        
        response = self.chat.send_message(feedback_prompt)
        full_text = response.text.strip()
        
        return self._parse_structured_response(full_text)

    def _parse_structured_response(self, text: str) -> dict:
        """Parses the LLM's structured response into a dictionary."""
        res = {
            "sql": self._extract_sql(text),
            "full_text": text,
            "status": "SUCCESS",
            "correction_prompt": None,
            "suggestions": []
        }
        
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
            res["suggestions"] = [s.strip("- ").strip() for s in s_list if s.strip()]
            
        return res

    def _extract_sql(self, text: str) -> Optional[str]:
        """Extracts SQL from a potentially conversational response."""
        # 1. Try markdown blocks (primary)
        match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # 2. Try just code blocks without 'sql'
        match = re.search(r"```\s*(SELECT|WITH|UPDATE|DELETE|INSERT).*?```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(0).replace("```", "").strip()

        # 3. Try raw SELECT/WITH (Greedier search)
        # Look for SELECT until the next heading or end of text
        match = re.search(r"(SELECT|WITH)[\s\S]+?(?=STATUS:|CORRECTION_PROMPT:|SUGGESTIONS:|$|```)", text, re.IGNORECASE)
        if match:
            sql = match.group(0).strip()
            # Basic cleanup: remove trailing semicolons or markdown 
            sql = sql.rstrip(";").strip()
            return sql
            
        return None

    def summarize_results(self, df_result: pd.DataFrame, original_query: str) -> str:
        """
        Takes the resulting dataframe and generates a natural language summary.
        """
        if df_result is None or df_result.empty:
            return "I couldn't find any data matching your request."
            
        cols = df_result.columns.tolist()
        
        prompt = f"""
You are a helpful Data Assistant speaking to a non-technical user.
Answer their question clearly and concisely based ONLY on the provided data result. Do not mention SQL or technical jargon.

User Question: {original_query}

Data Result (Columns: {', '.join(cols)}):
{df_result.head(10).to_markdown()}

Provide a short, direct answer (1-3 sentences).
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            return "Here is the data you requested."

    def execute_query(self, sql: str, results_dict: dict[str, dict]) -> pd.DataFrame:
        """
        Executes a SQL query against ALL provided DataFrames using DuckDB.
        """
        # Register all dataframes
        for table_name, result in results_dict.items():
            df = result.get("dataframe")
            if df is not None:
                self.con.register(table_name, df)
                
        try:
            return self.con.execute(sql).df()
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            raise RuntimeError(f"Error executing SQL: {e}")

    def generate_visualization(self, df_result: pd.DataFrame, original_query: str) -> dict:
        """
        Suggests a visualization based on the query results.
        """
        if df_result.empty:
            return {"type": None, "fig": None, "insight": "No data found for this query."}

        cols = df_result.columns.tolist()
        num_rows = len(df_result)
        
        # Try to use LLM to decide the best chart
        prompt = f"""
You are a visualization expert. Given a sample of data and a query, suggest the best chart type.

Query: {original_query}
Columns: {', '.join(cols)}
Data Sample (first 3 rows):
{df_result.head(3).to_markdown()}

CHART TYPES: [bar, line, scatter, pie, table]

Return ONLY a JSON object with:
{{
  "chart_type": "one of the above",
  "x": "column name for X axis",
  "y": "column name for Y axis",
  "title": "A descriptive title for the chart",
  "insight": "A 1-sentence insight about the data"
}}
"""
        try:
            response = self.model.generate_content(prompt)
            # Find JSON in response
            match = re.search(r"\{.*\}", response.text.replace("\n", ""), re.DOTALL)
            if match:
                spec = json.loads(match.group())
                return self._create_plotly_figure(df_result, spec)
        except Exception as e:
            logger.warning(f"AI visualization suggestion failed: {e}. Falling back to defaults.")

        # Fallback to simple heuristics
        return self._heuristic_visualization(df_result)

    def _build_schema_context(self, semantic_layer: dict) -> str:
        """Serializes the semantic layer for the prompt."""
        context = []
        
        if semantic_layer.get("dimensions"):
            context.append("Dimensions (Categories):")
            for d in semantic_layer["dimensions"]:
                val_sample = ", ".join(d.get("values", [])[:3])
                context.append(f"- {d['raw_column']}: {d['description']}. Examples: [{val_sample}]")
        
        if semantic_layer.get("measures"):
            context.append("\nMeasures (Numbers):")
            for m in semantic_layer["measures"]:
                context.append(f"- {m['raw_column']}: {m['description']} (Default agg: {m.get('aggregation', 'sum')})")
        
        if semantic_layer.get("time_fields"):
            context.append("\nTime Fields:")
            for t in semantic_layer["time_fields"]:
                context.append(f"- {t['raw_column']}: {t['description']}")
                
        if semantic_layer.get("kpis"):
            context.append("\nCalculated KPIs:")
            for k in semantic_layer["kpis"]:
                context.append(f"- {k['name']}: {k['description']} (Formula: {k['formula']})")

        return "\n".join(context)

    def _create_plotly_figure(self, df: pd.DataFrame, spec: dict) -> dict:
        """Builds a Plotly figure from AI spec."""
        import plotly.express as px
        chart_type = spec.get("chart_type", "table")
        x = spec.get("x")
        y = spec.get("y")
        title = spec.get("title", "Query Result")
        
        fig = None
        if chart_type == "bar":
            fig = px.bar(df, x=x, y=y, title=title, template="plotly_white")
        elif chart_type == "line":
            fig = px.line(df, x=x, y=y, title=title, template="plotly_white")
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x, y=y, title=title, template="plotly_white")
        elif chart_type == "pie":
            fig = px.pie(df, names=x, values=y, title=title, template="plotly_white")
        
        if fig:
            fig.update_layout(
                margin=dict(l=20, r=20, t=50, b=20),
                height=450,
            )
            return {"type": chart_type, "fig": fig, "insight": spec.get("insight", "")}
        
        return {"type": "table", "fig": None, "insight": spec.get("insight", "")}

    def _heuristic_visualization(self, df: pd.DataFrame) -> dict:
        """Simple rule-based visualization when AI fails."""
        import plotly.express as px
        cols = df.columns.tolist()
        num_rows = len(df)
        
        if num_rows > 0 and len(cols) >= 2:
            # Assume first col is X, second is Y
            fig = px.bar(df, x=cols[0], y=cols[1], title=f"{cols[1]} by {cols[0]}")
            return {"type": "bar", "fig": fig, "insight": "Automatically generated bar chart."}
        
        return {"type": "table", "fig": None, "insight": "Best viewed as a table."}
