"""
Streamlit UI for the Automated Excel Data Analysis Agent
=========================================================
Provides a beautiful multi-tab interface for file upload,
analysis pipeline execution, and result exploration.
"""

import io
import logging
import os
import sys
import tempfile
import streamlit as st
import pandas as pd

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

# ---- Logging setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("app")

# -----------------------------------------------------------------------
# Page config + custom CSS
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  #MainMenu, footer { visibility: hidden; }
  .hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    border-radius: 14px;
    padding: 2.5rem 2rem;
    color: white;
    margin-bottom: 1.8rem;
  }
  .hero h1 { font-size: 2rem; font-weight: 700; margin: 0; }
  .hero p  { opacity: .85; margin-top: .4rem; font-size: 1rem; }
  div[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 6px rgba(0,0,0,.08);
    border-top: 3px solid #2d6a9f;
  }
  .insight-box {
    padding: 1rem 1.2rem;
    border-radius: 10px;
    border-left: 4px solid #2d6a9f;
    background: white;
    margin-bottom: .7rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
  }
  .insight-box.p1 { border-color:#E53E3E; background:#fff5f5; }
  .insight-box.p2 { border-color:#DD6B20; background:#fffaf0; }
  .insight-box.p3 { border-color:#2d6a9f; background:#ebf4ff; }
  .insight-box.p4 { border-color:#38A169; background:#f0fff4; }
  .log-entry { font-size:.78rem; color:#4A5568; padding:.2rem 0; border-bottom:1px solid #EDF2F7; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # 🗝️ API Key Management (Secrets + Sidebar)
    secret_key = st.secrets.get("GEMINI_API_KEY")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=secret_key if secret_key else "",
        type="password",
        help="Enter your Gemini API key. If already set in Streamlit Secrets, it will match automatically.",
        key="api_key_sidebar"
    )
    api_key = api_key_input if api_key_input else secret_key

    selected_model = st.selectbox(
        "Model Version",
        options=[
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.5-pro-preview-03-25",
            "gemini-2.0-flash-001",
        ],
        index=0,
        help="Choose the Gemini model. 'gemini-2.0-flash' is recommended — fast and widely available."
    )

    st.markdown("---")
    st.markdown("### 📋 Pipeline Log")
    log_placeholder = st.empty()

# -----------------------------------------------------------------------
# Hero header
# -----------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>📊 Automated Data Analysis Agent</h1>
  <p>Upload files — get schema, quality, semantic models, and conversational insights.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# File upload
# -----------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload your Data file(s)",
    type=["xlsx", "xls", "csv"],
    accept_multiple_files=True,
    help="Supports multiple files for cross-table analysis",
    key="file_uploader",
)

if "all_results" not in st.session_state:
    st.session_state.all_results = {}
if "logs" not in st.session_state:
    st.session_state.logs = []

def _update_log(logs: list[str]):
    html = "".join(f"<div class='log-entry'>{l}</div>" for l in logs[-30:])
    log_placeholder.markdown(html, unsafe_allow_html=True)

run_btn = st.button("🚀 Run Analysis on All Files", use_container_width=True, disabled=not uploaded_files)

if run_btn and uploaded_files:
    st.session_state.all_results = {}
    st.session_state.logs = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    def on_progress(step: int, total: int, message: str):
        pct = int(step / total * 100)
        progress_bar.progress(pct)
        status_text.info(f"**Step {step}/{total}** — {message}")
        st.session_state.logs.append(message)
        _update_log(st.session_state.logs)

    try:
        from agent.orchestrator import run_pipeline
        num_files = len(uploaded_files)
        for i, file in enumerate(uploaded_files):
            status_text.info(f"Processing {file.name} ({i+1}/{num_files})...")
            file_bytes = io.BytesIO(file.read())
            result = run_pipeline(
                file=file_bytes,
                filename=file.name,
                progress_callback=on_progress,
                api_key=api_key,
            )
            import re
            table_name = re.sub(r"[^a-zA-Z0-9]+", "_", os.path.splitext(file.name)[0]).lower().strip("_")
            result["table_name"] = table_name
            st.session_state.all_results[table_name] = result
            
        progress_bar.progress(100)
        status_text.success(f"✅ Analysis complete for {num_files} file(s)!")
        st.rerun()
    except Exception as e:
        status_text.error(f"❌ Pipeline failed: {e}")
        logger.exception("Pipeline error")

# -----------------------------------------------------------------------
# Results tabs
# -----------------------------------------------------------------------
if st.session_state.all_results:
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📂 Select Dataset to View")
        selected_table = st.selectbox(
            "Viewing Details for:",
            options=list(st.session_state.all_results.keys()),
            format_func=lambda x: f"📊 {st.session_state.all_results[x]['metadata']['filename']}"
        )
    
    result = st.session_state.all_results[selected_table]
    meta, schema, quality, semantic, analysis, charts, report_html = \
        result.get("metadata", {}), result.get("schema", {}), result.get("quality", {}), \
        result.get("semantic", {}), result.get("analysis", {}), result.get("charts", []), result.get("report_html", "")

    tabs = st.tabs(["🏠 Summary", "📈 Insights", "📊 Charts", "💬 AI Assistant", "📋 Quality", "🗂️ Schema", "📥 Report"])

    with tabs[0]:
        st.subheader("Executive Summary")
        narrative = result.get("analysis", {}).get("narrative_summary")
        if narrative:
            st.markdown(f"<div style='background:white; padding:1.5rem; border-radius:12px; border-left:5px solid #2d6a9f; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom:1.5rem;'>{narrative}</div>", unsafe_allow_html=True)
        else:
            st.info("AI Narrative summary is only available when a Gemini API key is provided.")
        
        # Mini metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{meta.get('rows_raw', 0):,}")
        col2.metric("Analysis Cols", len(schema.get("analysis_columns", [])))
        col3.metric("Quality Score", f"{int(quality.get('summary', {}).get('score', 0))}%")
        col4.metric("Insights Found", len(analysis.get("insights", [])))

    with tabs[1]:
        st.subheader("Key Insights")
        for ins in analysis.get("insights", []):
            st.markdown(f"<div class='insight-box p{ins.get('priority',3)}'><strong>{ins.get('title')}</strong><br/>{ins.get('insight', ins.get('detail', ''))}</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.subheader("Interactive Visualizations")
        for i, chart in enumerate(charts):
            with st.expander(f"📊 {chart.get('title', f'Chart {i+1}')}"):
                st.plotly_chart(chart["fig"], use_container_width=True, key=f"chart_{i}_{selected_table}")
                st.caption(f"💡 **Insight:** {chart.get('insight')}")

    # ---- Tab 4: AI Assistant ----------------------------------------------
    with tabs[3]:
        st.subheader("💬 AI Data Assistant")
        st.markdown(f"Ask plain-English questions about your data.")
        
        if "ai_history" not in st.session_state: st.session_state.ai_history = []
        if "qe_instance" not in st.session_state: st.session_state.qe_instance = None

        if st.button("🗑️ Clear Chat"):
            st.session_state.ai_history, st.session_state.qe_instance = [], None
            st.rerun()

        for i, turn in enumerate(st.session_state.ai_history):
            with st.chat_message("user"): st.write(turn["query"])
            with st.chat_message("assistant"):

                # 1. Plain-English Summary
                if turn.get("summary"):
                    st.markdown(f"<div style='background:#f0f7ff; padding:1rem; border-radius:10px; border:1px solid #d1e3ff; margin-bottom:1rem;'>{turn['summary']}</div>", unsafe_allow_html=True)
                elif turn.get("df_result") is not None:
                    n = len(turn["df_result"])
                    st.success(f"I found **{n} records** matching your request.")
                elif turn.get("error"):
                    st.error(f"❌ Error: {turn['error']}")
                elif turn.get("full_text") and not turn.get("sql"):
                    st.info(turn["full_text"])

                # 2. Visualization (Prominent)
                if turn.get("df_result") is not None:
                    if turn.get("viz") and turn["viz"].get("fig"):
                        st.plotly_chart(turn["viz"]["fig"], use_container_width=True, key=f"viz_{i}")
                        if turn["viz"].get("insight"):
                            st.markdown(f"<div style='margin-top:-1rem; margin-bottom:1rem;'><small>📌 <em>{turn['viz']['insight']}</em></small></div>", unsafe_allow_html=True)
                    elif len(turn["df_result"]) > 0:
                        st.dataframe(turn["df_result"].head(20), use_container_width=True)

                # 3. Technical Details (Collapsed)
                with st.expander("🛠️ Show SQL & Reasoning", expanded=False):
                    if turn.get("explanation"):
                        st.markdown(f"**Analysis Reasoning:** {turn['explanation']}")
                    if turn.get("sql"):
                        st.code(turn["sql"], language="sql")
                    if turn.get("df_result") is not None:
                        st.markdown("**Data Preview (first 5 rows):**")
                        st.dataframe(turn["df_result"].head(5))

                # Verification flow
                if turn.get("status") == "VERIFICATION_REQUIRED":
                    st.warning(turn.get("correction_prompt", "Please verify the column mapping."))
                    if st.button("✅ Yes, proceed", key=f"v_y_{i}"):
                        try:
                            qe = st.session_state.qe_instance
                            df_res = qe.execute_query(turn["sql"], st.session_state.all_results)
                            viz = qe.generate_visualization(df_res, turn["query"])
                            smry = qe.summarize_results(df_res, turn["query"])
                            turn.update({"df_result": df_res, "viz": viz, "summary": smry, "status": "SUCCESS"})
                            st.rerun()
                        except Exception as e:
                            turn["error"] = str(e)
                            st.rerun()

                # 4. Suggestion buttons (only for latest turn)
                sugs = turn.get("suggestions", [])
                if sugs and i == len(st.session_state.ai_history) - 1:
                    st.divider()
                    st.caption("**💬 Suggested follow-ups:**")
                    cols = st.columns(min(len(sugs), 3))
                    for j, s in enumerate(sugs):
                        if cols[j % 3].button(s, key=f"s_{i}_{j}", use_container_width=True):
                            st.session_state.pushed_query = s
                            st.rerun()

        if not api_key: st.warning("⚠️ Enter API key in sidebar.")
        else:
            default_q = st.session_state.get("pushed_query", "")
            q = st.text_input("Ask a question:", value=default_q, key="ai_q")
            if "pushed_query" in st.session_state: del st.session_state.pushed_query
            if st.button("🔍 Send") and q:
                if st.session_state.qe_instance is None:
                    try:
                        from agent.query_engine import QueryEngine
                        st.session_state.qe_instance = QueryEngine(api_key=api_key, model_name=selected_model)
                        st.session_state.qe_instance.start_chat(st.session_state.all_results)
                    except Exception as e:
                        st.error(f"❌ AI Assistant Initialization Error: {e}")
                        st.session_state.qe_instance = None
                        st.stop()
                
                res = st.session_state.qe_instance.generate_sql(q)
                new_turn = {
                    "query": q,
                    "sql": res.get("sql"),
                    "full_text": res.get("full_text", ""),
                    "explanation": res.get("explanation", ""),
                    "status": res.get("status", "SUCCESS"),
                    "correction_prompt": res.get("correction_prompt"),
                    "suggestions": res.get("suggestions", []),
                    "df_result": None,
                    "viz": None,
                    "error": None,
                    "summary": None,
                }

                if new_turn["status"] == "SUCCESS" and new_turn["sql"]:
                    try:
                        df_res = st.session_state.qe_instance.execute_query(new_turn["sql"], st.session_state.all_results)
                        new_turn["df_result"] = df_res
                        new_turn["summary"] = st.session_state.qe_instance.summarize_results(df_res, q)
                        new_turn["viz"] = st.session_state.qe_instance.generate_visualization(df_res, q)
                    except Exception as e:
                        new_turn["error"] = str(e)

                st.session_state.ai_history.append(new_turn)
                st.rerun()

    with tabs[4]:
        st.subheader("Data Quality Checks")
        for issue in quality.get("issues", [])[:10]:
            st.warning(f"⚠️ **{issue.get('column', 'N/A')}**: {issue.get('issue', 'Issue')}")

    with tabs[5]:
        st.subheader("Data Dictionary")
        st.dataframe(pd.DataFrame(schema.get("data_dictionary", [])), use_container_width=True)

    with tabs[6]:
        st.subheader("Download Report")
        st.download_button("⬇️ Download HTML", data=report_html.encode("utf-8"), file_name="report.html", mime="text/html")
else:
    st.markdown("<div style='text-align:center;padding:4rem;'>📁 Upload files to start</div>", unsafe_allow_html=True)
