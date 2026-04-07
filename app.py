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
            )
            table_name = os.path.splitext(file.name)[0].replace(" ", "_").replace(".", "_").lower()
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

    tabs = st.tabs(["📋 Quality", "🗂️ Schema", "🧠 Semantic", "📈 Insights", "📊 Charts", "💬 AI Assistant", "📥 Report"])

    with tabs[0]:
        st.subheader("Data Quality")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{meta.get('rows_raw', 0):,}")
        col2.metric("Columns", len(schema.get("analysis_columns", [])))
        col3.metric("Quality Score", f"{int(quality.get('summary', {}).get('score', 0))}%")
        
        for issue in quality.get("issues", [])[:5]:
            st.warning(f"⚠️ **{issue.get('column', 'N/A')}**: {issue.get('issue', 'Issue')}")

    with tabs[1]:
        st.subheader("Schema / Data Dictionary")
        st.dataframe(pd.DataFrame(schema.get("data_dictionary", [])), use_container_width=True)

    with tabs[2]:
        st.subheader("Business Semantic Layer")
        def _show_sem(title, items):
            if items:
                st.markdown(f"**{title}**")
                cols = st.columns(min(len(items), 4))
                for i, item in enumerate(items):
                    cols[i%4].info(item.get("raw_column", str(item)))
        _show_sem("📍 Dimensions", semantic.get("dimensions", []))
        _show_sem("🔢 Measures", semantic.get("measures", []))

    with tabs[3]:
        st.subheader("Insights")
        for ins in analysis.get("insights", []):
            st.markdown(f"<div class='insight-box p{ins.get('priority',3)}'><strong>{ins.get('title')}</strong><br/>{ins.get('insight', '')}</div>", unsafe_allow_html=True)

    with tabs[4]:
        st.subheader("Visualizations")
        for i, chart in enumerate(charts):
            with st.expander(f"📊 {chart.get('title', f'Chart {i+1}')}"):
                st.plotly_chart(chart["fig"], use_container_width=True, key=f"chart_{i}_{selected_table}")

    # ---- Tab 6: AI Assistant ----------------------------------------------
    with tabs[5]:
        st.subheader("💬 AI Data Assistant")
        st.markdown(f"Ask questions across **{len(st.session_state.all_results)} active datasets**.")
        
        if "ai_history" not in st.session_state: st.session_state.ai_history = []
        if "qe_instance" not in st.session_state: st.session_state.qe_instance = None

        if st.button("🗑️ Reset"):
            st.session_state.ai_history, st.session_state.qe_instance = [], None
            st.rerun()

        for i, turn in enumerate(st.session_state.ai_history):
            with st.chat_message("user"): st.write(turn["query"])
            with st.chat_message("assistant"):
                # 1. Display Non-Technical Summary
                if turn.get("summary"):
                    st.markdown(f"#### 💡 Answer\n{turn['summary']}")
                elif turn.get("full_text") and not turn.get("sql"):
                    # If it's just a conversational response without data
                    st.markdown(turn["full_text"])
                
                # 2. Display Visualization (if available)
                if turn.get("df_result") is not None:
                    if turn.get("viz") and turn["viz"].get("fig"):
                        st.plotly_chart(turn["viz"]["fig"], use_container_width=True, key=f"viz_{i}")
                        if turn["viz"].get("insight"):
                            st.info(f"**Insight:** {turn['viz']['insight']}")

                # 3. Technical Details (Collapsible)
                if turn.get("sql") or turn.get("full_text"):
                    with st.expander("🛠️ Technical Details (SQL & Reasoning)"):
                        if turn.get("full_text"):
                            # Filter out the SQL and Suggestions from full_text to avoid redundancy if possible, 
                            # or just show it all if preferred. Here we show it all but clean.
                            st.markdown(turn["full_text"])
                        if turn.get("sql"):
                            st.markdown("**Executed SQL:**")
                            st.code(turn["sql"], language="sql")
                        
                        if turn.get("df_result") is not None:
                            st.markdown("**Preview of Data:**")
                            st.dataframe(turn["df_result"].head(10), use_container_width=True)

                if turn.get("status") == "VERIFICATION_REQUIRED":
                    st.warning(turn.get("correction_prompt", "Verify mapping?"))
                    if st.button("✅ Yes", key=f"v_y_{i}"):
                        try:
                            qe = st.session_state.qe_instance
                            df_res = qe.execute_query(turn["sql"], st.session_state.all_results)
                            turn.update({"df_result": df_res, "viz": qe.generate_visualization(df_res, turn["query"]), "status": "SUCCESS"})
                            st.rerun()
                        except Exception as e: turn["error"] = str(e); st.rerun()
                
                # 4. Suggestions as buttons
                sugs = turn.get("suggestions", [])
                if sugs and i == len(st.session_state.ai_history) - 1:
                    st.markdown("---")
                    st.markdown("**Suggested next steps:**")
                    cols = st.columns(len(sugs))
                    for j, s in enumerate(sugs):
                        if cols[j].button(s, key=f"s_{i}_{j}"):
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
                        st.session_state.qe_instance = QueryEngine(api_key=api_key)
                        st.session_state.qe_instance.start_chat(st.session_state.all_results)
                    except Exception as e:
                        st.error(f"❌ AI Assistant Initialization Error: {e}")
                        st.session_state.qe_instance = None
                        st.stop()
                
                res = st.session_state.qe_instance.generate_sql(q)
                new_turn = {"query": q, "sql": res["sql"], "full_text": res["full_text"], "status": res["status"], "correction_prompt": res["correction_prompt"], "suggestions": res["suggestions"], "df_result": None, "viz": None, "error": None, "summary": None}
                
                if new_turn["status"] == "SUCCESS" and new_turn["sql"]:
                    try:
                        df_res = st.session_state.qe_instance.execute_query(new_turn["sql"], st.session_state.all_results)
                        viz = st.session_state.qe_instance.generate_visualization(df_res, q)
                        summary = st.session_state.qe_instance.summarize_results(df_res, q)
                        new_turn.update({"df_result": df_res, "viz": viz, "summary": summary})
                    except Exception as e: new_turn["error"] = str(e)
                st.session_state.ai_history.append(new_turn)
                st.rerun()

    with tabs[6]:
        st.subheader("Download Report")
        st.download_button("⬇️ Download HTML", data=report_html.encode("utf-8"), file_name="report.html", mime="text/html")
else:
    st.markdown("<div style='text-align:center;padding:4rem;'>📁 Upload files to start</div>", unsafe_allow_html=True)
