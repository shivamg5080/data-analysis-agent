"""
Report Generator Module
=======================
Generates a self-contained, stakeholder-friendly HTML report
with embedded Plotly charts, data quality tables, semantic layer
summary, key insights, and recommended next actions.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import plotly.io as pio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{report_title}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', sans-serif;
      background: #F4F6F9;
      color: #1A1D23;
      line-height: 1.6;
    }}
    /* ---- Header ---- */
    .report-header {{
      background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
      color: white;
      padding: 3rem 4rem;
      position: relative;
      overflow: hidden;
    }}
    .report-header::after {{
      content: '';
      position: absolute;
      right: -80px; top: -80px;
      width: 300px; height: 300px;
      border-radius: 50%;
      background: rgba(255,255,255,0.07);
    }}
    .report-header h1 {{ font-size: 2rem; font-weight: 700; margin-bottom: .5rem; }}
    .report-header .meta {{ font-size: .9rem; opacity: .85; }}
    /* ---- Nav ---- */
    .nav-bar {{
      background: white;
      border-bottom: 1px solid #E2E8F0;
      padding: .6rem 4rem;
      position: sticky;
      top: 0;
      z-index: 100;
      display: flex;
      gap: 1.5rem;
    }}
    .nav-bar a {{
      text-decoration: none;
      color: #4A5568;
      font-size: .9rem;
      font-weight: 500;
      padding: .3rem 0;
      border-bottom: 2px solid transparent;
      transition: all .2s;
    }}
    .nav-bar a:hover {{ color: #2d6a9f; border-bottom-color: #2d6a9f; }}
    /* ---- Main ---- */
    .main {{ max-width: 1200px; margin: 0 auto; padding: 2rem 2rem 4rem; }}
    /* ---- Section ---- */
    section {{ margin-bottom: 3rem; }}
    .section-title {{
      font-size: 1.25rem;
      font-weight: 700;
      color: #1e3a5f;
      border-left: 4px solid #2d6a9f;
      padding-left: .75rem;
      margin-bottom: 1.25rem;
    }}
    /* ---- Cards ---- */
    .card {{
      background: white;
      border-radius: 12px;
      box-shadow: 0 1px 6px rgba(0,0,0,.08);
      padding: 1.5rem;
      margin-bottom: 1rem;
    }}
    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 1rem;
      margin-bottom: 1rem;
    }}
    .stat-card {{
      background: white;
      border-radius: 10px;
      padding: 1.2rem 1rem;
      text-align: center;
      box-shadow: 0 1px 4px rgba(0,0,0,.07);
      border-top: 3px solid #2d6a9f;
    }}
    .stat-card .value {{ font-size: 1.7rem; font-weight: 700; color: #1e3a5f; }}
    .stat-card .label {{ font-size: .8rem; color: #718096; margin-top: .25rem; }}
    /* ---- Table ---- */
    table {{ width: 100%; border-collapse: collapse; font-size: .875rem; }}
    th {{
      background: #EBF4FF;
      color: #1e3a5f;
      font-weight: 600;
      text-align: left;
      padding: .6rem .9rem;
    }}
    td {{ padding: .55rem .9rem; border-bottom: 1px solid #F0F4F8; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #F7FBFF; }}
    .badge {{
      display: inline-block;
      border-radius: 5px;
      padding: .15rem .5rem;
      font-size: .75rem;
      font-weight: 600;
    }}
    .badge-numeric {{ background:#DBEAFE; color:#1D4ED8; }}
    .badge-categorical {{ background:#D1FAE5; color:#065F46; }}
    .badge-datetime {{ background:#FEF3C7; color:#92400E; }}
    .badge-boolean {{ background:#EDE9FE; color:#5B21B6; }}
    .badge-text {{ background:#FCE7F3; color:#9D174D; }}
    .badge-identifier {{ background:#F3F4F6; color:#374151; }}
    .badge-unknown {{ background:#F3F4F6; color:#374151; }}
    /* ---- Insight cards ---- */
    .insight-list {{ display: flex; flex-direction: column; gap: .8rem; }}
    .insight-card {{
      display: flex;
      gap: 1rem;
      align-items: flex-start;
      background: white;
      border-radius: 10px;
      padding: 1rem 1.25rem;
      box-shadow: 0 1px 4px rgba(0,0,0,.07);
      border-left: 4px solid #2d6a9f;
    }}
    .insight-card.p1 {{ border-color: #E53E3E; }}
    .insight-card.p2 {{ border-color: #DD6B20; }}
    .insight-card.p3 {{ border-color: #2d6a9f; }}
    .insight-card.p4 {{ border-color: #38A169; }}
    .insight-icon {{ font-size: 1.4rem; }}
    .insight-body .title {{ font-weight: 600; margin-bottom: .25rem; }}
    .insight-body .detail {{ font-size: .875rem; color: #4A5568; }}
    .insight-cat {{
      font-size: .7rem;
      font-weight: 600;
      color: #718096;
      text-transform: uppercase;
      letter-spacing: .05em;
      margin-bottom: .2rem;
    }}
    .confidence-bar {{
      height: 4px;
      border-radius: 2px;
      background: #E2E8F0;
      margin-top: .4rem;
    }}
    .confidence-bar-fill {{
      height: 100%;
      border-radius: 2px;
      background: linear-gradient(90deg, #2d6a9f, #4BB6EF);
    }}
    /* ---- Charts ---- */
    .chart-card {{
      background: white;
      border-radius: 12px;
      box-shadow: 0 1px 6px rgba(0,0,0,.08);
      padding: 1.25rem 1.5rem 1.5rem;
      margin-bottom: 1.5rem;
    }}
    .chart-card h3 {{
      font-size: 1rem;
      font-weight: 600;
      color: #1e3a5f;
      margin-bottom: .5rem;
    }}
    .chart-insight {{
      font-size: .82rem;
      color: #4A5568;
      margin-bottom: .75rem;
      padding: .5rem .75rem;
      background: #F0F7FF;
      border-radius: 6px;
    }}
    .chart-note {{
      font-size: .78rem;
      color: #718096;
      margin-top: .6rem;
      font-style: italic;
    }}
    /* ---- Semantic ---- */
    .tag-list {{ display: flex; flex-wrap: wrap; gap: .5rem; }}
    .tag {{
      display: inline-block;
      padding: .3rem .7rem;
      border-radius: 20px;
      font-size: .8rem;
      font-weight: 500;
    }}
    .tag-dim {{ background: #D1FAE5; color: #065F46; }}
    .tag-meas {{ background: #DBEAFE; color: #1D4ED8; }}
    .tag-time {{ background: #FEF3C7; color: #92400E; }}
    .tag-kpi {{ background: #EDE9FE; color: #5B21B6; }}
    .tag-ent {{ background: #FCE7F3; color: #9D174D; }}
    /* ---- Footer ---- */
    .footer {{
      text-align: center;
      padding: 2rem;
      font-size: .8rem;
      color: #A0AEC0;
      border-top: 1px solid #E2E8F0;
      margin-top: 3rem;
    }}
    /* ---- Responsive ---- */
    @media(max-width: 700px) {{
      .report-header {{ padding: 2rem 1.5rem; }}
      .nav-bar {{ padding: .6rem 1rem; flex-wrap: wrap; }}
      .main {{ padding: 1.5rem 1rem; }}
    }}
  </style>
</head>
<body>

<!-- HEADER -->
<header class="report-header">
  <h1>📊 {report_title}</h1>
  <p class="meta">Generated on {generated_at} &bull; File: <strong>{filename}</strong> &bull; Sheet: <strong>{sheet}</strong></p>
</header>

<!-- NAV -->
<nav class="nav-bar">
  <a href="#overview">Overview</a>
  <a href="#quality">Data Quality</a>
  <a href="#schema">Schema</a>
  <a href="#semantic">Semantic Layer</a>
  <a href="#insights">Insights</a>
  <a href="#charts">Charts</a>
</nav>

<!-- MAIN -->
<div class="main">

  <!-- 1. OVERVIEW -->
  <section id="overview">
    <h2 class="section-title">Dataset Overview</h2>
    <div class="stat-grid">
      <div class="stat-card"><div class="value">{rows:,}</div><div class="label">Total Rows</div></div>
      <div class="stat-card"><div class="value">{columns}</div><div class="label">Columns</div></div>
      <div class="stat-card"><div class="value">{complete_rows_pct}%</div><div class="label">Complete Rows</div></div>
      <div class="stat-card"><div class="value">{duplicate_rows}</div><div class="label">Duplicate Rows</div></div>
      <div class="stat-card"><div class="value">{memory_mb} MB</div><div class="label">Memory Size</div></div>
      <div class="stat-card"><div class="value">{analysis_columns}</div><div class="label">Analysed Columns</div></div>
    </div>
  </section>

  <!-- 2. DATA QUALITY -->
  <section id="quality">
    <h2 class="section-title">Data Quality Report</h2>
    {quality_summary_html}
    {quality_table_html}
  </section>

  <!-- 3. SCHEMA -->
  <section id="schema">
    <h2 class="section-title">Schema / Data Dictionary</h2>
    {schema_table_html}
  </section>

  <!-- 4. SEMANTIC LAYER -->
  <section id="semantic">
    <h2 class="section-title">Semantic Layer</h2>
    {semantic_html}
  </section>

  <!-- 5. KEY INSIGHTS -->
  <section id="insights">
    <h2 class="section-title">Key Insights & Recommended Actions</h2>
    <div class="insight-list">
      {insights_html}
    </div>
  </section>

  <!-- 6. CHARTS -->
  <section id="charts">
    <h2 class="section-title">Visualizations</h2>
    {charts_html}
  </section>

</div>

<footer class="footer">
  Auto-generated by Data Analysis Agent &bull; {generated_at}
</footer>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    pipeline_result: dict[str, Any],
    output_path: str | None = None,
) -> str:
    """
    Generate a self-contained HTML report from the pipeline result dict.

    Parameters
    ----------
    pipeline_result:
        The dict returned by ``orchestrator.run_pipeline()``.
    output_path:
        If provided, write the HTML to this file path.

    Returns
    -------
    str
        The complete HTML string of the report.
    """
    meta = pipeline_result.get("metadata", {})
    analysis = pipeline_result.get("analysis", {})
    charts = pipeline_result.get("charts", [])
    quality = pipeline_result.get("quality", {})
    schema = pipeline_result.get("schema", {})
    semantic = pipeline_result.get("semantic", {})

    dataset_summary = analysis.get("dataset_summary", {})
    insights = analysis.get("insights", [])
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    filename = meta.get("filename", "Unknown")
    sheet = meta.get("sheet", "Unknown")

    html = _HTML_TEMPLATE.format(
        report_title=f"Data Analysis Report — {filename}",
        generated_at=generated_at,
        filename=filename,
        sheet=sheet,
        rows=dataset_summary.get("rows", 0),
        columns=dataset_summary.get("columns", 0),
        analysis_columns=dataset_summary.get("analysis_columns", 0),
        complete_rows_pct=dataset_summary.get("complete_row_pct", 0),
        duplicate_rows=dataset_summary.get("duplicate_rows", 0),
        memory_mb=dataset_summary.get("memory_mb", 0),
        quality_summary_html=_render_quality_summary(quality),
        quality_table_html=_render_quality_table(quality),
        schema_table_html=_render_schema_table(schema),
        semantic_html=_render_semantic_layer(semantic),
        insights_html=_render_insights(insights),
        charts_html=_render_charts(charts),
    )

    if output_path:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info("Report written to %s", output_path)
        except Exception as e:
            logger.error("Failed to write report: %s", e)

    return html


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_quality_summary(quality: dict) -> str:
    overall = quality.get("overall_score", 1.0)
    issues = quality.get("issues", [])
    color = "#38A169" if overall >= 0.8 else "#DD6B20" if overall >= 0.6 else "#E53E3E"
    issues_html = "".join(
        f"<li style='color:#718096;font-size:.875rem;margin-bottom:.25rem;'>{iss}</li>"
        for iss in issues[:10]
    ) if issues else "<li style='color:#38A169;'>No major issues found.</li>"
    return f"""
    <div class="card" style="border-top:4px solid {color};">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
        <span style="font-weight:600;font-size:1rem;">Overall Quality Score</span>
        <span style="font-size:1.5rem;font-weight:700;color:{color};">{int(overall * 100)}%</span>
      </div>
      <ul style="padding-left:1.2rem;">{issues_html}</ul>
    </div>
    """


def _render_quality_table(quality: dict) -> str:
    col_quality = quality.get("column_quality", [])
    if not col_quality:
        return "<p style='color:#718096;font-size:.875rem;'>No column-level quality data available.</p>"

    rows_html = ""
    for cq in col_quality:
        null_pct = cq.get("null_pct", 0)
        null_color = "#E53E3E" if null_pct > 50 else "#DD6B20" if null_pct > 20 else "#38A169"
        outlier_flag = "⚠️" if cq.get("outlier_count", 0) > 0 else ""
        duplicate_flag = "⚠️" if cq.get("duplicate_count", 0) > 0 else ""
        rows_html += f"""
        <tr>
          <td><strong>{cq.get('column', '')}</strong></td>
          <td>{cq.get('total_rows', 0):,}</td>
          <td>{cq.get('non_null_count', 0):,}</td>
          <td style="color:{null_color};font-weight:600;">{null_pct:.1f}%</td>
          <td>{cq.get('unique_count', 0):,}</td>
          <td>{cq.get('duplicate_count', 0):,} {duplicate_flag}</td>
          <td>{cq.get('outlier_count', 0)} {outlier_flag}</td>
        </tr>"""

    return f"""
    <div class="card" style="padding:0;overflow-x:auto;">
      <table>
        <thead><tr>
          <th>Column</th><th>Total Rows</th><th>Non-Null</th><th>Null %</th>
          <th>Unique</th><th>Duplicates</th><th>Outliers</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""


def _render_schema_table(schema: dict) -> str:
    data_dict = schema.get("data_dictionary", [])
    if not data_dict:
        return "<p style='color:#718096;font-size:.875rem;'>Schema information not available.</p>"

    rows_html = ""
    for col_info in data_dict:
        col_type = col_info.get("inferred_type", "unknown")
        badge_cls = f"badge-{col_type}"
        samples = ", ".join(str(s) for s in col_info.get("sample_values", [])[:3])
        rows_html += f"""
        <tr>
          <td><strong>{col_info.get('column', '')}</strong></td>
          <td><span class="badge {badge_cls}">{col_type}</span></td>
          <td>{col_info.get('non_null_count', 0):,}</td>
          <td>{col_info.get('null_pct', 0):.1f}%</td>
          <td>{col_info.get('unique_count', 0):,}</td>
          <td style="color:#718096;font-size:.82rem;">{samples}</td>
          <td style="font-size:.82rem;color:#718096;">{col_info.get('notes', '')[:80]}</td>
        </tr>"""

    return f"""
    <div class="card" style="padding:0;overflow-x:auto;">
      <table>
        <thead><tr>
          <th>Column</th><th>Type</th><th>Non-Null</th><th>Null %</th>
          <th>Unique</th><th>Samples</th><th>Notes</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""


def _render_semantic_layer(semantic: dict) -> str:
    def tags(items: list, cls: str, label_key: str = "name") -> str:
        if not items:
            return "<span style='color:#A0AEC0;font-size:.85rem;'>None</span>"
        return "".join(f"<span class='tag {cls}'>{i.get(label_key, str(i))}</span>" for i in items)

    dims = semantic.get("dimensions", [])
    measures = semantic.get("measures", [])
    time_fields = semantic.get("time_fields", [])
    kpis = semantic.get("kpis", [])
    entities = semantic.get("entities", [])

    return f"""
    <div class="card">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;">
        <div>
          <p style="font-weight:600;color:#4A5568;margin-bottom:.5rem;">📐 Dimensions ({len(dims)})</p>
          <div class="tag-list">{tags(dims, 'tag-dim', 'raw_column')}</div>
        </div>
        <div>
          <p style="font-weight:600;color:#4A5568;margin-bottom:.5rem;">📏 Measures ({len(measures)})</p>
          <div class="tag-list">{tags(measures, 'tag-meas', 'raw_column')}</div>
        </div>
        <div>
          <p style="font-weight:600;color:#4A5568;margin-bottom:.5rem;">📅 Time Fields ({len(time_fields)})</p>
          <div class="tag-list">{tags(time_fields, 'tag-time', 'raw_column')}</div>
        </div>
        <div>
          <p style="font-weight:600;color:#4A5568;margin-bottom:.5rem;">🎯 KPIs ({len(kpis)})</p>
          <div class="tag-list">{tags(kpis, 'tag-kpi', 'name')}</div>
        </div>
        <div>
          <p style="font-weight:600;color:#4A5568;margin-bottom:.5rem;">🔑 Entities ({len(entities)})</p>
          <div class="tag-list">{tags(entities, 'tag-ent', 'raw_column')}</div>
        </div>
      </div>
    </div>"""


_PRIORITY_ICON = {1: "🔴", 2: "🟠", 3: "🔵", 4: "🟢"}
_PRIORITY_CLASS = {1: "p1", 2: "p2", 3: "p3", 4: "p4"}


def _render_insights(insights: list[dict]) -> str:
    if not insights:
        return "<p style='color:#718096;'>No insights generated.</p>"
    html_parts = []
    for ins in insights:
        p = ins.get("priority", 3)
        icon = _PRIORITY_ICON.get(p, "🔵")
        cls = _PRIORITY_CLASS.get(p, "p3")
        # TODO(next task): Re-enable insight confidence rendering (bar + percentage).
        # conf = ins.get("confidence", 0.8)
        # fill_width = int(conf * 100)
        html_parts.append(f"""
        <div class="insight-card {cls}">
          <div class="insight-icon">{icon}</div>
          <div class="insight-body" style="flex:1;">
            <div class="insight-cat">{ins.get('category', '')}</div>
            <div class="title">{ins.get('title', '')}</div>
            <div class="detail">{ins.get('detail', '')}</div>
            <!-- TODO(next task): confidence bar intentionally hidden for now -->
          </div>
        </div>""")
    return "\n".join(html_parts)


def _render_charts(charts: list[dict]) -> str:
    if not charts:
        return "<p style='color:#718096;'>No charts generated.</p>"
    html_parts = []
    for chart in charts:
        chart_html = chart.get("html", "")
        html_parts.append(f"""
        <div class="chart-card">
          <h3>{chart.get('title', 'Chart')}</h3>
          <div class="chart-insight">💡 {chart.get('insight', '')}</div>
          {chart_html}
          <div class="chart-note">📌 {chart.get('stakeholder_note', '')}</div>
        </div>""")
    return "\n".join(html_parts)
