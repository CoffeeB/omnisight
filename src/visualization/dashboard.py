"""
Interactive HTML Research Dashboard Generator (Plotly & HTML5)
"""

from typing import Dict, Any, List
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_research_dashboard(
    experiment_results: Dict[str, Any],
    output_html_path: str
) -> str:
    """
    Generates a standalone, rich HTML research dashboard with interactive figures.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "EXP-01: Viewpoint Robustness (mIoU)",
            "EXP-02: Altitude GSD Degradation (mIoU)",
            "EXP-03: Occlusion Tolerance Curve",
            "EXP-04: Atmospheric Degradation (RDI)"
        )
    )

    # 1. Viewpoint Chart
    v_data = experiment_results.get("viewpoint", {})
    if v_data:
        v_keys = list(v_data.get("single_view", {}).keys())
        sv_vals = list(v_data.get("single_view", {}).values())
        mv_vals = list(v_data.get("multi_view", {}).values())
        fig.add_trace(go.Bar(name="Single-View", x=v_keys, y=sv_vals, marker_color="#d9534f"), row=1, col=1)
        fig.add_trace(go.Bar(name="OmniSight Multi-View", x=v_keys, y=mv_vals, marker_color="#2b5c8f"), row=1, col=1)

    # 2. Altitude Chart
    alt_data = experiment_results.get("altitude", {})
    if alt_data:
        alt_keys = [f"{k}m" for k in alt_data.keys()]
        alt_vals = list(alt_data.values())
        fig.add_trace(go.Scatter(name="Altitude mIoU", x=alt_keys, y=alt_vals, mode="lines+markers", line=dict(color="#2ca02c", width=3)), row=1, col=2)

    # 3. Occlusion Chart
    occ_data = experiment_results.get("occlusion", {})
    if occ_data:
        occ_keys = [f"{int(float(k)*100)}%" for k in occ_data.keys()]
        occ_vals = list(occ_data.values())
        fig.add_trace(go.Scatter(name="Occlusion Retention", x=occ_keys, y=occ_vals, mode="lines+markers", line=dict(color="#ff7f0e", width=3)), row=2, col=1)

    # 4. Environmental RDI Chart
    env_data = experiment_results.get("environmental", {})
    if env_data:
        env_keys = list(env_data.keys())
        env_vals = list(env_data.values())
        fig.add_trace(go.Bar(name="Condition mIoU", x=env_keys, y=env_vals, marker_color="#9467bd"), row=2, col=2)

    fig.update_layout(
        title_text="OmniSight Research Suite: Multi-Modal Robustness Benchmarks",
        height=850,
        showlegend=False,
        template="plotly_white"
    )

    plotly_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>OmniSight Research Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 24px;
        }}
        .header {{
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border: 1px solid #334155;
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 24px;
        }}
        .header h1 {{ margin: 0 0 8px 0; color: #38bdf8; font-size: 28px; }}
        .header p {{ margin: 0; color: #94a3b8; font-size: 15px; }}
        .card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>OmniSight Laboratory Research Dashboard</h1>
        <p>Comprehensive Empirical Evaluation: Viewpoint, Altitude GSD, Occlusion, Atmospheric Scattering, & Calibration</p>
    </div>
    <div class="card">
        {plotly_html}
    </div>
</body>
</html>
"""
    with open(output_html_path, "w") as f:
        f.write(html_template)

    return output_html_path
