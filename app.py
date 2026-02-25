# app.py — Planificador robusto Plan vs Real con SEMÁFOROS, VARIACIONES y Centro de Alertas
from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Núcleo de datos
from utils_ops import (
    ALIAS_PLAN, ALIAS_REAL, guess_mapping, apply_mapping, enrich_time,
    merge_plan_real, compute_metrics, agg_error_metrics,
    aggregate_nacional, aggregate_bases, aggregate_franja, aggregate_cat,
    forecast_next_hours, to_excel_bytes,
    DATA_DIR, MERG_CSV, save_csv, load_csv
)

# Visuales mejorados
from viz_helpers import (
    kpi_indicator, chart_plan_real_band, chart_waterfall_diff, chart_dumbbell_base
)

# ==========================
# Configuración base
# ==========================
st.set_page_config(page_title="Planificación vs Realidad — Bases & Red", layout="wide")
if "ui_theme" not in st.session_state: st.session_state["ui_theme"]="dark"
if "ui_dense_tables" not in st.session_state: st.session_state["ui_dense_tables"]=True
if "tolerancia_pct" not in st.session_state: st.session_state["tolerancia_pct"]=0.0
for k in ("plan_df","real_df","merged","map_plan","map_real","last_tick"):
    if k not in st.session_state:
        st.session_state[k] = (pd.DataFrame() if k in ("plan_df","real_df","merged") else {} if "map" in k else None)

# ==========================
# Helpers de parsing / pegado
# ==========================
def _smart_sep(text: str) -> str:
    s = text[:1000]
    if "\t" in s: return "\t"
    if ";"  in s: return ";"
    return ","

@st.cache_data(show_spinner=False)
def parse_text_table(text: str) -> pd.DataFrame:
    if not text or not text.strip(): return pd.DataFrame()
    return pd.read_csv(StringIO(text), sep=_smart_sep(text), engine="python")

def parse_any(df_or_text, is_plan: bool) -> tuple[pd.DataFrame, Dict[str,str]]:
    if isinstance(df_or_text, str): df_raw = parse_text_table(df_or_text)
    else: df_raw = df_or_text
    if df_raw is None or df_raw.empty: return pd.DataFrame(), {}
    mapping = guess_mapping(df_raw, ALIAS_PLAN if is_plan else ALIAS_REAL)
    out = apply_mapping(df_raw, mapping, kind="plan" if is_plan else "real")
    out = enrich_time(out)
    return out, mapping

def _quality_report(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    ref = df["Fecha"].max() if "Fecha" in df else None
    d = df[df["Fecha"].eq(ref)].copy() if ref else df.copy()
    expected = {f"{h:02d}:00" for h in range(24)}
    present = set(d["HoraStr"].dropna().unique().tolist()) if "HoraStr" in d else set()
    faltantes = sorted(list(expected - present))
    dup = d["HoraStr"].value_counts() if "HoraStr" in d else pd.Series(dtype=int)
    duplicadas = dup[dup > 1].index.tolist() if not dup.empty else []
    nulos = int(d.isna().sum().sum())
    return pd.DataFrame([{
        "Fecha_ref": str(ref) if ref else "—",
        "Horas_sin_dato": len(faltantes),
        "Horas_duplicadas": len(duplicadas),
        "Celdas_nulas": nulos
    }])

# ==========================
# Sidebar — Menú & Config
# ==========================
st.sidebar.title("Menú")
menu = st.sidebar.radio(
    "Secciones",
    ["Cargar datos", "Dashboard", "Bases", "Red Nacional", "Franjas & CAT", "Analítica", "Predicción", "Auditoría / Excel"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Opciones de UI")
st.session_state["ui_theme"] = st.sidebar.selectbox("Tema", options=["dark","light"], index=0)
st.session_state["ui_dense_tables"] = st.sidebar.checkbox("Tablas densas", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Tolerancia & Semáforos")

# Banda de tolerancia ±% sobre Plan (para gráficos)
st.session_state["tolerancia_pct"] = st.sidebar.slider("Banda ±% sobre Plan", 0, 30, value=int(st.session_state["tolerancia_pct"]*100), step=1) / 100.0

# Umbrales semáforo (total del período filtrado)
with st.sidebar.expander("Umbrales globales (Dashboard)"):
    objetivo_efect = st.number_input("Objetivo Efectividad (%)", min_value=0.0, max_value=100.0, value=95.0, step=0.5)
    # Desvío Servicios/Agentes: usamos franjas de severidad por |desvío|
    green_ds  = st.number_input("Desvío Servicios (verde) ≤ %",   min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    yellow_ds = st.number_input("Desvío Servicios (amarillo) ≤ %", min_value=0.0, max_value=100.0, value=12.0, step=0.5)
    green_dm  = st.number_input("Desvío Agentes (verde) ≤ %",     min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    yellow_dm = st.number_input("Desvío Agentes (amarillo) ≤ %",   min_value=0.0, max_value=100.0, value=10.0, step=0.5)
    crit_abs  = st.number_input("Hora crítica (|Dif_Servicios| ≥)", min_value=0, max_value=999999, value=25, step=1)

def _semaforo_ratio(val_ratio: Optional[float], green_thr: float, yellow_thr: float) -> tuple[str,str]:
    """Devuelve (emoji, color_str) según |val_ratio| en % vs umbrales green/yellow; rojo por encima."""
    if val_ratio is None or (isinstance(val_ratio,float) and np.isnan(val_ratio)): return ("⚪", "neutral")
    v = abs(val_ratio)*100.0
    if v <= green_thr:  return ("🟢","green")
    if v <= yellow_thr: return ("🟡","yellow")
    return ("🔴","red")

def _semaforo_target(val_ratio: Optional[float], target_pct: float) -> tuple[str,str]:
    """Efectividad: verde si >= target; amarillo si a -2pp; rojo resto."""
    if val_ratio is None or (isinstance(val_ratio,float) and np.isnan(val_ratio)): return ("⚪","neutral")
    v = val_ratio*100.0
    if v >= target_pct:            return ("🟢","green")
    if v >= target_pct - 2.0:      return ("🟡","yellow")
    return ("🔴","red")

# ==========================
# 1) Cargar datos
# ==========================
if menu == "Cargar datos":
    st.title("Cargar datos")
    st.caption("Podés **subir archivos** (XLSX/CSV) o **pegar** tablas. El sistema reconoce **sinónimos/abreviaturas** y columnas en **cualquier orden**.")

    st.subheader("Plan")
    c1, c2 = st.columns(2)
    with c1:
        f_plan = st.file_uploader("Archivo Plan (XLSX/CSV)", type=["xlsx","csv"], key="plan_file")
        if st.button("Procesar archivo Plan"):
            try:
                if f_plan is None: st.warning("Subí un archivo o usá el pegado.")
                else:
                    df_raw = pd.read_excel(f_plan) if f_plan.name.lower().endswith(".xlsx") else pd.read_csv(f_plan)
                    plan_df, map_plan = parse_any(df_raw, is_plan=True)
                    st.session_state["plan_df"]=plan_df; st.session_state["map_plan"]=map_plan
                    st.success(f"Plan listo ({len(plan_df)} filas)")
            except Exception as e:
                st.error(f"Plan: {e}")
    with c2:
        t_plan = st.text_area("Pegado Plan (orden libre, sinónimos/abreviaturas)", height=140, key="plan_text")
        if st.button("Procesar pegado Plan"):
            try:
                plan_df, map_plan = parse_any(t_plan, is_plan=True)
                st.session_state["plan_df"]=plan_df; st.session_state["map_plan"]=map_plan
                st.success(f"Plan listo ({len(plan_df)} filas)")
            except Exception as e:
                st.error(f"Plan: {e}")

    st.subheader("Real")
    c3, c4 = st.columns(2)
    with c3:
        f_real = st.file_uploader("Archivo Real (XLSX/CSV)", type=["xlsx","csv"], key="real_file")
        if st.button("Procesar archivo Real"):
            try:
                if f_real is None: st.warning("Subí un archivo o usá el pegado.")
                else:
                    df_raw = pd.read_excel(f_real) if f_real.name.lower().endswith(".xlsx") else pd.read_csv(f_real)
                    real_df, map_real = parse_any(df_raw, is_plan=False)
                    st.session_state["real_df"]=real_df; st.session_state["map_real"]=map_real
                    st.success(f"Real listo ({len(real_df)} filas)")
            except Exception as e:
                st.error(f"Real: {e}")
    with c4:
        t_real = st.text_area("Pegado Real (orden libre, sinónimos/abreviaturas)", height=140, key="real_text")
        if st.button("Procesar pegado Real"):
            try:
                real_df, map_real = parse_any(t_real, is_plan=False)
                st.session_state["real_df"]=real_df; st.session_state["map_real"]=map_real
                st.success(f"Real listo ({len(real_df)} filas)")
            except Exception as e:
                st.error(f"Real: {e}")

    st.markdown("---")
    cA, cB, cC = st.columns(3)
    with cA:
        if st.button("Ver mapeo detectado"):
            col1, col2 = st.columns(2)
            with col1: st.caption("Plan"); st.json(st.session_state.get("map_plan") or {}, expanded=False)
            with col2: st.caption("Real"); st.json(st.session_state.get("map_real") or {}, expanded=False)
    with cB:
        if st.button("Calidad (último día)"):
            qa, qb = st.columns(2)
            with qa: st.write("**Plan**"); st.dataframe(_quality_report(st.session_state["plan_df"]), hide_index=True, use_container_width=True)
            with qb: st.write("**Real**"); st.dataframe(_quality_report(st.session_state["real_df"]), hide_index=True, use_container_width=True)
    with cC:
        if st.button("3) **Comparar ahora**"):
            p, r = st.session_state["plan_df"], st.session_state["real_df"]
            if p.empty and r.empty:
                st.error("Cargá al menos Plan o Real.")
            else:
                if p.empty:
                    base = r[["Fecha","Hora","Base","CAT"]].drop_duplicates().copy()
                    base["Moviles_Planificados"]=0.0; base["Servicios_Planificados"]=0.0
                    p = enrich_time(base)
                if r.empty:
                    base = p[["Fecha","Hora","Base","CAT"]].drop_duplicates().copy()
                    base["Moviles_Reales"]=0.0; base["Servicios_Reales"]=0.0
                    r = enrich_time(base)
                merged = compute_metrics(merge_plan_real(p, r))
                st.session_state["merged"]=merged
                st.session_state["last_tick"]=pd.Timestamp.now().strftime("%H:%M:%S")
                st.success(f"Comparación lista ({len(merged)} filas). Ir a **Dashboard**.")

    st.markdown("---")
    pv1, pv2 = st.columns(2)
    with pv1: st.markdown("#### Preview Plan"); st.dataframe(st.session_state["plan_df"].head(15), hide_index=True, use_container_width=True)
    with pv2: st.markdown("#### Preview Real"); st.dataframe(st.session_state["real_df"].head(15), hide_index=True, use_container_width=True)

# ==========================
# Resto de vistas: filtros y utilidades comunes
# ==========================
else:
    merged = st.session_state.get("merged", pd.DataFrame())
    if merged.empty:
        st.info("No hay datos comparados. Volvé a **Cargar datos**.")
        st.stop()

    with st.expander("Filtros"):
        fecha_fil = st.date_input("Fecha", value=merged["Fecha"].max())
        bases_opt = sorted(merged["Base"].dropna().unique().tolist())
        bases_sel = st.multiselect("Bases", options=bases_opt, default=bases_opt[: min(10, len(bases_opt))])

        franjas_opt = ["MADRUGADA","MAÑANA","TARDE","NOCHE"]
        franjas_sel = st.multiselect("Franja", options=franjas_opt, default=franjas_opt)

        horas_opt = sorted(merged["HoraStr"].dropna().unique().tolist())
        cH1, cH2 = st.columns(2)
        with cH1:
            horas_sel = st.multiselect("Horas (selección libre)", options=horas_opt, default=[])
        with cH2:
            hr_from, hr_to = st.select_slider("Rango horario", options=horas_opt, value=(horas_opt[0], horas_opt[-1]) if horas_opt else ("00:00","23:00"))

        cats_opt = sorted(merged["CAT"].dropna().unique().tolist()) if "CAT" in merged else []
        cats_sel = st.multiselect("CAT", options=cats_opt, default=[])

        desvio_range = st.slider("Filtrar por Desvío Servicios % (Real − Plan)", min_value=-200, max_value=200, value=(-200, 200), step=5)

    def _apply_filters(df: pd.DataFrame, override_date: Optional[pd.Timestamp]=None) -> pd.DataFrame:
        if df is None or df.empty: return pd.DataFrame()
        d = df.copy()
        tgt_date = override_date if override_date is not None else fecha_fil
        if "Fecha" in d and tgt_date is not None:
            d = d[d["Fecha"].eq(pd.to_datetime(tgt_date).date())]
        if bases_sel: d = d[d["Base"].isin(bases_sel)]
        if franjas_sel: d = d[d["Franja"].isin(franjas_sel)]
        if "HoraStr" in d and hr_from and hr_to:
            idx_from = horas_opt.index(hr_from) if hr_from in horas_opt else 0
            idx_to   = horas_opt.index(hr_to)   if hr_to   in horas_opt else len(horas_opt)-1
            rango = set(horas_opt[idx_from: idx_to+1])
            d = d[d["HoraStr"].isin(rango)]
        if horas_sel: d = d[d["HoraStr"].isin(horas_sel)]
        if cats_sel and "CAT" in d: d = d[d["CAT"].isin(cats_sel)]
        if "Desvio_Servicios_%" in d:
            d = d[(d["Desvio_Servicios_%"] >= desvio_range[0]) & (d["Desvio_Servicios_%"] <= desvio_range[1])]
        return d.reset_index(drop=True)

    df_f = _apply_filters(merged)

    # Utilidades de comparación temporal (día anterior / semana anterior)
    def _prev_dates(all_dates: List[pd.Timestamp], current_date: pd.Timestamp) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Devuelve (día anterior disponible, misma fecha-7 disponible) si existen."""
        if not all_dates: return None, None
        all_sorted = sorted(pd.to_datetime(pd.Series(all_dates)).dt.date.unique().tolist())
        cur = pd.to_datetime(current_date).date()
        prev_list = [d for d in all_sorted if d < cur]
        prev_day = prev_list[-1] if prev_list else None
        week_day = pd.to_datetime(cur) - pd.Timedelta(days=7)
        prev_week = None
        # elegir la fecha más cercana a cur-7
        diffs = [(abs((pd.to_datetime(d) - week_day).days), d) for d in all_sorted if d < cur]
        if diffs:
            diffs.sort(); prev_week = diffs[0][1]
        return prev_day, prev_week

    # ==========================
    # DASHBOARD (con SEMÁFOROS, VARIACIONES y ALERTAS)
    # ==========================
    if menu == "Dashboard":
        st.title("Dashboard — Centro de Alertas y Variaciones")

        if df_f.empty:
            st.info("No hay datos para los filtros aplicados."); st.stop()

        # Totales actuales
        tot_p_s = df_f["Servicios_Planificados"].sum()
        tot_r_s = df_f["Servicios_Reales"].sum()
        tot_p_m = df_f["Moviles_Planificados"].sum()
        tot_r_m = df_f["Moviles_Reales"].sum()

        efect   = 1 - (abs(tot_r_s - tot_p_s)/tot_p_s) if tot_p_s>0 else np.nan
        desv_s  = (tot_r_s - tot_p_s)/tot_p_s if tot_p_s>0 else np.nan
        desv_m  = (tot_r_m - tot_p_m)/tot_p_m if tot_p_m>0 else np.nan

        # Variaciones vs día anterior / semana anterior
        all_dates = merged["Fecha"].unique().tolist()
        prev_day, prev_week = _prev_dates(all_dates, fecha_fil)

        def _agg_for(date_):
            if date_ is None: return (np.nan, np.nan)
            d = _apply_filters(merged, override_date=date_)
            if d.empty: return (np.nan, np.nan)
            s = d["Servicios_Reales"].sum()
            m = d["Moviles_Reales"].sum()
            return (s, m)

        s_prev_day, m_prev_day   = _agg_for(prev_day)
        s_prev_week, m_prev_week = _agg_for(prev_week)

        var_s_day  = (tot_r_s/s_prev_day - 1.0) if s_prev_day and s_prev_day>0 else np.nan
        var_s_week = (tot_r_s/s_prev_week- 1.0) if s_prev_week and s_prev_week>0 else np.nan
        var_m_day  = (tot_r_m/m_prev_day - 1.0) if m_prev_day and m_prev_day>0 else np.nan
        var_m_week = (tot_r_m/m_prev_week- 1.0) if m_prev_week and m_prev_week>0 else np.nan

        # KPIs (cards) + deltas
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.plotly_chart(kpi_indicator("Efectividad", efect), use_container_width=True)
        with c2: st.plotly_chart(kpi_indicator("Desvío Servicios", desv_s), use_container_width=True)
        with c3: st.plotly_chart(kpi_indicator("Desvío Agentes",  desv_m), use_container_width=True)
        with c4:
            d_text = f"Δ día: {var_s_day*100:,.1f}% | Δ sem: {var_s_week*100:,.1f}%" if pd.notna(var_s_day) or pd.notna(var_s_week) else "—"
            st.metric("Derivaciones (total)", f"{df_f.get('Servicios_Derivados', pd.Series(dtype=float)).sum():,.0f}", d_text)

        # ===== Semáforos (según umbrales) =====
        sm_ef,  _ = _semaforo_target(efect, objetivo_efect)
        sm_ds,  _ = _semaforo_ratio(desv_s, green_ds, yellow_ds)
        sm_dm,  _ = _semaforo_ratio(desv_m, green_dm, yellow_dm)

        st.markdown("### Semáforos globales")
        b1,b2,b3 = st.columns(3)
        with b1: st.markdown(f"**Efectividad**: {sm_ef}  (objetivo {objetivo_efect:.1f}%)")
        with b2: st.markdown(f"**Desvío Servicios**: {sm_ds}  (verde ≤ {green_ds:.1f}%, amarillo ≤ {yellow_ds:.1f}%)")
        with b3: st.markdown(f"**Desvío Agentes**: {sm_dm}  (verde ≤ {green_dm:.1f}%, amarillo ≤ {yellow_dm:.1f}%)")

        st.caption(f"Último refresh: **{st.session_state.get('last_tick') or '—'}**  ·  Banda de tolerancia activa: **±{int((st.session_state['tolerancia_pct'] or 0)*100)}%**")

        # ===== Serie Plan vs Real con banda de diferencia + tolerancia =====
        g_srv = df_f.groupby("HoraStr", as_index=False)[["Servicios_Planificados","Servicios_Reales"]].sum().sort_values("HoraStr")
        fig_pr = chart_plan_real_band(g_srv, y_plan="Servicios_Planificados", y_real="Servicios_Reales",
                                      title="Servicios — Plan vs Real (banda de diferencia)")
        tol = st.session_state["tolerancia_pct"] or 0.0
        if tol > 0 and not g_srv.empty:
            xs = g_srv["HoraStr"].tolist()
            yplan = g_srv["Servicios_Planificados"].astype(float).values
            upper = yplan*(1+tol); lower = yplan*(1-tol)
            fig_pr.add_trace(go.Scatter(x=xs, y=upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_pr.add_trace(go.Scatter(x=xs, y=lower, mode="lines", line=dict(width=0), fill="tonexty",
                                        fillcolor="rgba(59,130,246,0.10)", name=f"Tolerancia ±{int(tol*100)}%"))
        st.plotly_chart(fig_pr, use_container_width=True)

        # ===== Waterfall de desvío horario =====
        g_diff = df_f.groupby("HoraStr", as_index=False)["Dif_Servicios"].sum().sort_values("HoraStr")
        st.plotly_chart(chart_waterfall_diff(g_diff, title="Aporte por hora al desvío (Real − Plan) — Servicios"),
                        use_container_width=True)

        # ===== Centro de Alertas (brechas) =====
        st.subheader("Centro de Alertas (brechas por umbral)")
        # A) Horas críticas (|Dif_Servicios| ≥ crit_abs)
        crit_hours = df_f.groupby("HoraStr", as_index=False)["Dif_Servicios"].sum()
        crit_hours["AbsDif"] = crit_hours["Dif_Servicios"].abs()
        crit_hours = crit_hours[crit_hours["AbsDif"] >= crit_abs].sort_values("AbsDif", ascending=False)

        # B) Bases con |Desvío Servicios %| > amarillo
        bases_alert = df_f.groupby("Base", as_index=False)[["Servicios_Planificados","Servicios_Reales"]].sum()
        bases_alert["Desvio_%"] = np.where(bases_alert["Servicios_Planificados"]>0,
                                           (bases_alert["Servicios_Reales"]-bases_alert["Servicios_Planificados"])/bases_alert["Servicios_Planificados"]*100,
                                           np.nan)
        bases_alert = bases_alert[bases_alert["Desvio_%"].abs() > yellow_ds].sort_values(bases_alert["Desvio_%"].abs().name, ascending=False)

        # C) CAT con mayor desvío absoluto (opcional)
        cat_alert = pd.DataFrame()
        if "CAT" in df_f.columns:
            cat_alert = df_f.groupby("CAT", as_index=False)["Dif_Servicios"].sum()
            cat_alert["AbsDif"] = cat_alert["Dif_Servicios"].abs()
            cat_alert = cat_alert.sort_values("AbsDif", ascending=False).head(10)

        a1,a2,a3 = st.columns(3)
        with a1:
            st.write(f"**Horas críticas** (|Dif_Servicios| ≥ {crit_abs})")
            st.dataframe(crit_hours, hide_index=True, use_container_width=True)
        with a2:
            st.write(f"**Bases fuera de rango** (|Desvío %| > {yellow_ds:.1f}%)")
            st.dataframe(bases_alert[["Base","Servicios_Planificados","Servicios_Reales","Desvio_%"]],
                         hide_index=True, use_container_width=True)
        with a3:
            st.write("**Top CAT por desvío absoluto**")
            st.dataframe(cat_alert, hide_index=True, use_container_width=True)

        # Export de alertas
        if st.button("⬇️ Exportar alertas (CSV)"):
            alerts = {
                "Horas_criticas": crit_hours,
                "Bases_fuera_rango": bases_alert,
                "CAT_top_desvio": cat_alert
            }
            # Concateno con etiqueta
            out = []
            for name, df_ in alerts.items():
                if df_ is None or df_.empty: continue
                dft = df_.copy(); dft.insert(0, "Alerta", name); out.append(dft)
            if out:
                final = pd.concat(out, ignore_index=True)
                csv = final.to_csv(index=False).encode("utf-8")
                st.download_button("Descargar CSV", data=csv, file_name="alertas_plan_vs_real.csv", mime="text/csv")
            else:
                st.info("No hay alertas para exportar.")

        # ===== Heatmap por Base x Hora (desvío) =====
        st.subheader("Heatmap — Desvío por Base x Hora (Real − Plan)")
        heap = df_f.pivot_table(values="Dif_Servicios", index="Base", columns="HoraStr", aggfunc="sum").fillna(0)
        if heap.empty:
            st.info("No hay datos para el heatmap.")
        else:
            fig_hm = px.imshow(heap, color_continuous_scale="RdYlGn", aspect="auto", origin="lower",
                               title="Desvío de servicios por Base x Hora (más verde = sobre plan)")
            st.plotly_chart(fig_hm, use_container_width=True)

    # ==========================
    # BASES
    # ==========================
    elif menu == "Bases":
        st.title("Comparación por Bases")
        if df_f.empty: st.info("No hay datos para los filtros."); st.stop()
        st.plotly_chart(chart_dumbbell_base(df_f, top_n=25,
                        xp="Servicios_Planificados", xr="Servicios_Reales",
                        title="Bases — Plan vs Real (Dumbbell)"),
                        use_container_width=True)

        g = aggregate_bases(df_f)
        g["Abs_Desvío"] = g["abs_diff"]
        st.dataframe(g[["Base","Servicios_Planificados","Servicios_Reales","Desvio_%","Abs_Desvío"]],
                     hide_index=True, use_container_width=True)

    # ==========================
    # RED NACIONAL
    # ==========================
    elif menu == "Red Nacional":
        st.title("Red Nacional")
        if df_f.empty: st.info("No hay datos."); st.stop()
        g = aggregate_nacional(df_f)
        st.plotly_chart(chart_plan_real_band(g, y_plan="Servicios_Planificados", y_real="Servicios_Reales",
                        title="Servicios — Plan vs Real (Nacional)"), use_container_width=True)
        g2 = g.rename(columns={"Moviles_Planificados":"Plan","Moviles_Reales":"Real"})
        figm = px.line(g2, x="HoraStr", y=["Plan","Real"], title="Agentes — Plan vs Real (Nacional)",
                       template="plotly_dark" if st.session_state["ui_theme"]=="dark" else "plotly")
        st.plotly_chart(figm, use_container_width=True)

    # ==========================
    # FRANJAS & CAT
    # ==========================
    elif menu == "Franjas & CAT":
        st.title("Franjas & CAT")
        if df_f.empty: st.info("No hay datos."); st.stop()
        gf = aggregate_franja(df_f)
        figf = px.area(gf, x="HoraStr", y="Servicios_Reales", facet_col="Franja", facet_col_wrap=2,
                       color_discrete_sequence=["#10B981"], title="Servicios Reales por franja",
                       template="plotly_dark" if st.session_state["ui_theme"]=="dark" else "plotly")
        for a in figf.layout.annotations: a.font.size = 12
        st.plotly_chart(figf, use_container_width=True)

        gc = aggregate_cat(df_f)
        if gc.empty:
            st.info("No hay columna CAT en los datos filtrados.")
        else:
            figc = px.bar(gc, x="CAT", y="Dif_Servicios", color="Dif_Servicios",
                          color_continuous_scale="RdYlGn", title="Desvío de Servicios por CAT",
                          template="plotly_dark" if st.session_state["ui_theme"]=="dark" else "plotly")
            st.plotly_chart(figc, use_container_width=True)
            st.dataframe(gc, hide_index=True, use_container_width=True)

    # ==========================
    # ANALÍTICA
    # ==========================
    elif menu == "Analítica":
        st.title("Analítica (detalle)")
        if df_f.empty: st.info("No hay datos."); st.stop()
        cols = ["Fecha","HoraStr","Base","CAT","Franja",
                "Servicios_Planificados","Servicios_Reales","Dif_Servicios","Desvio_Servicios_%",
                "Moviles_Planificados","Moviles_Reales","Dif_Moviles","Desvio_Moviles_%",
                "Efectividad","Clasificacion","Status","Semana","Mes","Año","Bias","AE","APE"]
        cols = [c for c in cols if c in df_f.columns]
        st.dataframe(df_f[cols].sort_values(["Fecha","HoraStr","Base"]),
                     hide_index=True, use_container_width=True)

    # ==========================
    # PREDICCIÓN
    # ==========================
    elif menu == "Predicción":
        st.title("Predicción próximas horas")
        if df_f.empty: st.info("No hay datos."); st.stop()
        modo = st.radio("Ámbito", ["Nacional","Por Base"], horizontal=True)
        horizonte = st.slider("Horizonte (horas)", min_value=1, max_value=8, value=6, step=1)

        if modo == "Nacional":
            scope = merged[merged["Fecha"] <= pd.to_datetime(fecha_fil).date()]
            preds = forecast_next_hours(scope, target_col="Servicios_Reales", horizon=horizonte, today=pd.to_datetime(fecha_fil))
            amb = "Nacional"
        else:
            base_sel = st.selectbox("Base", options=sorted(merged["Base"].dropna().unique().tolist()))
            scope = merged[(merged["Base"]==base_sel) & (merged["Fecha"] <= pd.to_datetime(fecha_fil).date())]
            preds = forecast_next_hours(scope, target_col="Servicios_Reales", horizon=horizonte, today=pd.to_datetime(fecha_fil))
            amb = base_sel

        if preds.empty: st.info("No hay suficiente información para estimar.")
        else:
            hoy = scope[scope["Fecha"].eq(pd.to_datetime(fecha_fil).date())]
            g = hoy.groupby("HoraStr", as_index=False)["Servicios_Reales"].sum(); g["tipo"]="Observado"
            p = preds.rename(columns={"Prediccion":"Servicios_Reales"}); p["tipo"]="Predicción"
            plot = pd.concat([g, p[["HoraStr","Servicios_Reales","tipo"]]], ignore_index=True)
            figp = px.line(plot, x="HoraStr", y="Servicios_Reales", color="tipo",
                           title=f"Servicios_Reales — Observado vs Predicción ({amb})",
                           template="plotly_dark" if st.session_state["ui_theme"]=="dark" else "plotly")
            st.plotly_chart(figp, use_container_width=True)
            st.dataframe(preds, hide_index=True, use_container_width=True)

    # ==========================
    # AUDITORÍA / EXCEL
    # ==========================
    else:
        st.title("Auditoría y Export")
        if df_f.empty: st.info("No hay datos."); st.stop()
        resumen = pd.DataFrame([agg_error_metrics(df_f)])
        book = {
            "Resumen": resumen,
            "Nacional": aggregate_nacional(df_f),
            "Por_Bases": aggregate_bases(df_f),
            "Por_Franja": aggregate_franja(df_f),
            "Por_CAT": aggregate_cat(df_f),
            "Detalle": df_f
        }
        xls, fname = to_excel_bytes(book, "reporte_plan_vs_real.xlsx")
        st.download_button("⬇️ Descargar Excel (todas las hojas)", data=xls, file_name=fname,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.markdown("---")
        s1, s2, s3 = st.columns(3)
        with s1:
            if st.button("Guardar snapshot filtrado en /data/merged.csv"):
                save_csv(df_f, MERG_CSV); st.success(f"Guardado: {MERG_CSV}")
        with s2:
            if st.button("Limpiar memoria (no borra /data)"):
                st.session_state["plan_df"]=pd.DataFrame()
                st.session_state["real_df"]=pd.DataFrame()
                st.session_state["merged"]=pd.DataFrame()
                st.session_state["map_plan"]={}
                st.session_state["map_real"]={}
                st.success("Memoria limpia")
        with s3:
            st.caption("Consejo: versioná /data en .gitignore para publicar sin datos locales.")
