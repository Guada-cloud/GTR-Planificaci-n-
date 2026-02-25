# viz_helpers.py — componentes visuales mejorados (no barras “pesadas”)
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

PALETTE = {
    "text": "#E5E7EB",
    "plan": "#60A5FA",
    "real": "#F59E0B",
    "diff": "rgba(16,185,129,0.22)",
    "pos":  "#22C55E",
    "neg":  "#EF4444",
    "grid": "rgba(148,163,184,.25)"
}

def _sty(fig: go.Figure, title: str | None = None, y_pct: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", title=title,
        font=dict(family="Inter, system-ui, Segoe UI, Roboto", size=13, color=PALETTE["text"]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="", margin=dict(t=45, r=10, b=30, l=10)
    )
    if y_pct:
        fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=PALETTE["grid"])
    return fig

def kpi_indicator(title: str, value: float | None, fmt="%.1f%%"):
    txt = "—" if value is None or (isinstance(value,float) and np.isnan(value)) else (fmt % (value*100 if "%" in fmt else value))
    fig = go.Figure(go.Indicator(mode="number", value=0,
                                 number={"font":{"size":36,"color":PALETTE["text"]}},
                                 title={"text":f"<b>{title}</b>", "font":{"size":14,"color":PALETTE["text"]}}))
    fig.update_layout(annotations=[dict(text=f"<b style='font-size:28px;'>{txt}</b>", showarrow=False, x=0.5, y=0.4)])
    return _sty(fig)

def chart_plan_real_band(df: pd.DataFrame, x="HoraStr", y_plan="Servicios_Planificados", y_real="Servicios_Reales", title="Servicios — Plan vs Real"):
    d = df.copy().sort_values(x)
    y1 = d[y_plan].astype(float).values
    y2 = d[y_real].astype(float).values
    fig = go.Figure()
    # Banda |Real-Plan|
    fig.add_trace(go.Scatter(x=d[x], y=np.maximum(y1,y2), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d[x], y=np.minimum(y1,y2), mode="lines", line=dict(width=0), fill="tonexty", fillcolor=PALETTE["diff"], name="|Real−Plan|"))
    # Plan (escalera)
    fig.add_trace(go.Scatter(x=d[x], y=y1, name="Plan", mode="lines", line=dict(color=PALETTE["plan"], width=2), line_shape="hv"))
    # Real (línea + markers)
    fig.add_trace(go.Scatter(x=d[x], y=y2, name="Real", mode="lines+markers", line=dict(color=PALETTE["real"], width=2)))
    return _sty(fig, title)

def chart_waterfall_diff(df: pd.DataFrame, x="HoraStr", y_diff="Dif_Servicios", title="Desvío horario (Real − Plan)"):
    d = df.copy().groupby(x, as_index=False)[y_diff].sum().sort_values(x)
    fig = go.Figure(go.Waterfall(
        x=d[x], y=d[y_diff], measure=["relative"]*len(d),
        decreasing={"marker":{"color":PALETTE["neg"]}},
        increasing={"marker":{"color":PALETTE["pos"]}},
        connector={"line":{"color":PALETTE["grid"]}},
        text=[f"{v:+.0f}" for v in d[y_diff]], textposition="outside"
    ))
    return _sty(fig, title)

def chart_dumbbell_base(df: pd.DataFrame, top_n=15, y="Base", xp="Servicios_Planificados", xr="Servicios_Reales", title="Bases — Plan vs Real (Dumbbell)"):
    g = df.groupby(y, as_index=False)[[xp, xr]].sum()
    g["abs_diff"] = (g[xr]-g[xp]).abs()
    g = g.sort_values("abs_diff", ascending=False).head(top_n)
    fig = go.Figure()
    # líneas
    for _, r in g.iterrows():
        fig.add_trace(go.Scatter(x=[r[xp], r[xr]], y=[r[y], r[y]], mode="lines", line=dict(color=PALETTE["grid"], width=3), showlegend=False, hoverinfo="skip"))
    # puntos
    fig.add_trace(go.Scatter(x=g[xp], y=g[y], mode="markers", name="Plan", marker=dict(size=10, color=PALETTE["plan"])))
    fig.add_trace(go.Scatter(x=g[xr], y=g[y], mode="markers", name="Real", marker=dict(size=10, color=PALETTE["real"])))
    fig.update_yaxes(autorange="reversed")
    return _sty(fig, title)
