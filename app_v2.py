# app_v2.py — Nueva versión de Streamlit con gestión de flota, tracking GPS y alertas en vivo
from __future__ import annotations
from typing import Optional, Dict, List
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Módulos locales
from database_models import init_db, Vehicle, GPSTracking, Alert, Trip, ForecastMetric
from fleet_ml_engine import FleetForecastModel, DelayPredictor, AnomalyDetector
from gps_realtime_engine import GPSRealTimeEngine, haversine_distance
from fleet_analytics import FleetAnalytics, KPICalculator
from utils_ops import (
    ALIAS_PLAN, ALIAS_REAL, guess_mapping, apply_mapping, enrich_time,
    merge_plan_real, compute_metrics, agg_error_metrics,
    aggregate_nacional, aggregate_bases, aggregate_franja, aggregate_cat,
    forecast_next_hours, to_excel_bytes
)
from viz_helpers import (
    kpi_indicator, chart_plan_real_band, chart_waterfall_diff, chart_dumbbell_base
)

# ==========================
# Configuración & Setup
# ==========================
st.set_page_config(page_title="Fleet Management — IA + GPS Real-Time", layout="wide")

if "db_session" not in st.session_state:
    Session = init_db()
    st.session_state["db_session"] = Session()

if "gps_engine" not in st.session_state:
    st.session_state["gps_engine"] = GPSRealTimeEngine()

if "forecast_model" not in st.session_state:
    st.session_state["forecast_model"] = FleetForecastModel()

if "delay_predictor" not in st.session_state:
    st.session_state["delay_predictor"] = DelayPredictor()

if "anomaly_detector" not in st.session_state:
    st.session_state["anomaly_detector"] = AnomalyDetector()

# Estado de sesión
for k in ["plan_df", "real_df", "merged", "gps_updates", "alerts_log"]:
    if k not in st.session_state:
        st.session_state[k] = pd.DataFrame()

# ==========================
# Sidebar — Navegación
# ==========================
st.sidebar.title("🚗 Fleet Management")
menu = st.sidebar.radio(
    "Secciones",
    [
        "📊 Dashboard",
        "📥 Cargar Datos",
        "🗺️ Tracking GPS",
        "🚨 Alertas en Vivo",
        "📈 Análisis",
        "🧠 ML & Forecasting",
        "📅 Planificación",
        "💾 Auditoría/Export"
    ],
    index=0
)

st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Configuración"):
    theme = st.selectbox("Tema", ["dark", "light"], index=0)
    offline_alert_min = st.number_input("Alerta offline (minutos)", min_value=1, max_value=120, value=15)
    speed_alert_kmh = st.number_input("Alerta velocidad (km/h)", min_value=1, max_value=200, value=120)
    st.session_state["gps_engine"].offline_timeout_min = offline_alert_min
    st.session_state["gps_engine"].speed_alert_kmh = speed_alert_kmh

st.sidebar.markdown("---")
st.sidebar.caption("v2.0 — AI + Real-Time GPS")

# ==========================
# 1. DASHBOARD (KPIs + Alertas Activas)
# ==========================
if menu == "📊 Dashboard":
    st.title("🎯 Dashboard Operacional")
    
    if st.session_state["merged"].empty:
        st.info("No hay datos comparados. Cargá primero en **📥 Cargar Datos**.")
    else:
        merged = st.session_state["merged"]
        
        # KPIs principales
        kpis = KPICalculator.compute_dashboard_kpis(merged)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Efectividad", f"{kpis['effectiveness']*100:.1f}%", 
                     delta=f"{(kpis['effectiveness']-0.95)*100:.1f}%" if kpis['effectiveness'] != 0 else None)
        with col2:
            st.metric("Utilización Flota", f"{kpis['fleet_utilization']*100:.1f}%")
        with col3:
            st.metric("Puntualidad", f"{kpis['on_time_rate']*100:.1f}%")
        with col4:
            st.metric("Coef. Operativo", f"{kpis['operational_coefficient']:.2f}")
        
        # Alertas activas
        st.subheader("🚨 Alertas Activas (últimas 24h)")
        if not st.session_state["alerts_log"].empty:
            alerts = st.session_state["alerts_log"]
            alerts_today = alerts[alerts['timestamp'] > (datetime.utcnow() - timedelta(hours=24))]
            
            col_info, col_warning, col_critical = st.columns(3)
            with col_info:
                st.metric("ℹ️ Info", (alerts_today['severity'] == 'info').sum())
            with col_warning:
                st.metric("⚠️ Warning", (alerts_today['severity'] == 'warning').sum())
            with col_critical:
                st.metric("🔴 Critical", (alerts_today['severity'] == 'critical').sum())
            
            st.dataframe(alerts_today[['timestamp', 'vehicle_id', 'alert_type', 'severity', 'message']].tail(20),
                        hide_index=True, use_container_width=True)
        
        # Gráficos
        st.markdown("---")
        col_left, col_right = st.columns(2)
        
        with col_left:
            g_srv = merged.groupby("HoraStr", as_index=False)[["Servicios_Planificados","Servicios_Reales"]].sum()
            fig_pr = chart_plan_real_band(g_srv, y_plan="Servicios_Planificados", y_real="Servicios_Reales",
                                         title="Servicios — Plan vs Real")
            st.plotly_chart(fig_pr, use_container_width=True)
        
        with col_right:
            g_diff = merged.groupby("HoraStr", as_index=False)["Dif_Servicios"].sum()
            st.plotly_chart(chart_waterfall_diff(g_diff, title="Desvío Horario"),
                           use_container_width=True)

# ==========================
# 2. CARGAR DATOS (Plan vs Real)
# ==========================
elif menu == "📥 Cargar Datos":
    st.title("📥 Cargar Datos — Plan vs Real")
    st.caption("Soporta XLSX, CSV o pegado. Detecta sinónimos y columnas en cualquier orden.")
    
    tab_plan, tab_real = st.tabs(["📋 Plan", "📊 Real"])
    
    with tab_plan:
        c1, c2 = st.columns(2)
        with c1:
            f_plan = st.file_uploader("Plan (XLSX/CSV)", type=["xlsx", "csv"], key="plan_file")
            if st.button("Procesar archivo Plan"):
                if f_plan:
                    try:
                        df_raw = pd.read_excel(f_plan) if f_plan.name.endswith(".xlsx") else pd.read_csv(f_plan)
                        plan_df, map_plan = _parse_any(df_raw, is_plan=True)
                        st.session_state["plan_df"] = plan_df
                        st.success(f"Plan: {len(plan_df)} filas")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with c2:
            t_plan = st.text_area("Pegado Plan", height=140, key="plan_text")
            if st.button("Procesar pegado Plan"):
                if t_plan.strip():
                    try:
                        df_raw = pd.read_csv(StringIO(t_plan), sep="\t" if "\t" in t_plan else ",")
                        plan_df, map_plan = _parse_any(df_raw, is_plan=True)
                        st.session_state["plan_df"] = plan_df
                        st.success(f"Plan: {len(plan_df)} filas")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    with tab_real:
        c1, c2 = st.columns(2)
        with c1:
            f_real = st.file_uploader("Real (XLSX/CSV)", type=["xlsx", "csv"], key="real_file")
            if st.button("Procesar archivo Real"):
                if f_real:
                    try:
                        df_raw = pd.read_excel(f_real) if f_real.name.endswith(".xlsx") else pd.read_csv(f_real)
                        real_df, map_real = _parse_any(df_raw, is_plan=False)
                        st.session_state["real_df"] = real_df
                        st.success(f"Real: {len(real_df)} filas")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with c2:
            t_real = st.text_area("Pegado Real", height=140, key="real_text")
            if st.button("Procesar pegado Real"):
                if t_real.strip():
                    try:
                        df_raw = pd.read_csv(StringIO(t_real), sep="\t" if "\t" in t_real else ",")
                        real_df, map_real = _parse_any(df_raw, is_plan=False)
                        st.session_state["real_df"] = real_df
                        st.success(f"Real: {len(real_df)} filas")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    st.markdown("---")
    if st.button("🔗 Comparar Ahora", use_container_width=True):
        try:
            plan_df = st.session_state["plan_df"]
            real_df = st.session_state["real_df"]
            
            if plan_df.empty and real_df.empty:
                st.error("Cargá al menos Plan o Real.")
            else:
                if plan_df.empty:
                    plan_df = real_df[["Fecha","Hora","Base","CAT"]].drop_duplicates().copy()
                    plan_df["Moviles_Planificados"] = 0.0
                    plan_df["Servicios_Planificados"] = 0.0
                    plan_df = enrich_time(plan_df)
                
                if real_df.empty:
                    real_df = plan_df[["Fecha","Hora","Base","CAT"]].drop_duplicates().copy()
                    real_df["Moviles_Reales"] = 0.0
                    real_df["Servicios_Reales"] = 0.0
                    real_df = enrich_time(real_df)
                
                merged = compute_metrics(merge_plan_real(plan_df, real_df))
                st.session_state["merged"] = merged
                st.success(f"✅ Datos comparados: {len(merged)} filas")
                st.info("→ Ir a **📊 Dashboard** para ver análisis")
        except Exception as e:
            st.error(f"Error: {e}")

# ==========================
# 3. TRACKING GPS EN TIEMPO REAL
# ==========================
elif menu == "🗺️ Tracking GPS":
    st.title("🗺️ Seguimiento GPS en Tiempo Real")
    
    col_upload, col_manual = st.columns(2)
    
    with col_upload:
        st.subheader("Cargar datos GPS (CSV)")
        gps_file = st.file_uploader("Archivo GPS", type=["csv"])
        if gps_file:
            try:
                gps_df = pd.read_csv(gps_file)
                st.session_state["gps_updates"] = gps_df
                st.success(f"GPS: {len(gps_df)} registros")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col_manual:
        st.subheader("Ingreso manual")
        vehicle_id = st.text_input("ID del vehículo")
        lat = st.number_input("Latitud")
        lon = st.number_input("Longitud")
        speed = st.number_input("Velocidad (km/h)", min_value=0.0)
        
        if st.button("Registrar GPS"):
            alerts = st.session_state["gps_engine"].process_gps_update(vehicle_id, lat, lon, speed)
            st.success(f"GPS registrado | {len(alerts)} alertas generadas")
            
            if alerts:
                for alert in alerts:
                    st.warning(f"{alert['severity'].upper()}: {alert['message']}")
    
    # Mostrar geofences
    st.subheader("⛔ Geofences")
    col_new, col_view = st.columns(2)
    
    with col_new:
        st.caption("Agregar nueva zona")
        fence_name = st.text_input("Nombre de zona")
        fence_lat = st.number_input("Latitud zona", key="fence_lat")
        fence_lon = st.number_input("Longitud zona", key="fence_lon")
        fence_radius = st.number_input("Radio (km)", min_value=0.1, value=2.0)
        
        if st.button("Crear Geofence"):
            st.session_state["gps_engine"].register_geofence(fence_name, fence_lat, fence_lon, fence_radius)
            st.success(f"Geofence '{fence_name}' creado")
    
    with col_view:
        if st.session_state["gps_engine"].geofences:
            st.caption("Zonas activas")
            for name, fence in st.session_state["gps_engine"].geofences.items():
                st.info(f"📍 {name}: ({fence['lat']:.4f}, {fence['lon']:.4f}), radius={fence['radius_km']}km")
    
    # Mapa (simulado con tabla)
    st.subheader("Historial de movimientos")
    if not st.session_state["gps_updates"].empty:
        st.dataframe(st.session_state["gps_updates"], hide_index=True, use_container_width=True)

# ==========================
# 4. ALERTAS EN VIVO
# ==========================
elif menu == "🚨 Alertas en Vivo":
    st.title("🚨 Centro de Control de Alertas")
    
    col_filter, col_action = st.columns([3, 1])
    
    with col_filter:
        severity_filter = st.multiselect("Filtrar por severidad", ["info", "warning", "critical"], default=["warning", "critical"])
        alert_type_filter = st.multiselect("Filtrar por tipo", ["offline", "delay", "speeding", "geofence", "maintenance"], default=[])
    
    with col_action:
        if st.button("🔄 Refrescar"):
            st.rerun()
    
    # Tabla de alertas (simulada)
    if st.session_state["alerts_log"].empty:
        st.info("Sin alertas activas.")
    else:
        alerts = st.session_state["alerts_log"]
        if severity_filter:
            alerts = alerts[alerts['severity'].isin(severity_filter)]
        
        st.dataframe(alerts.sort_values('timestamp', ascending=False).head(50),
                    hide_index=True, use_container_width=True)

# ==========================
# 5. ANÁLISIS AVANZADO
# ==========================
elif menu == "📈 Análisis":
    st.title("📈 Análisis Avanzado")
    
    if st.session_state["merged"].empty:
        st.info("Cargá datos primero.")
    else:
        tab1, tab2, tab3 = st.tabs(["Impacto Horario", "Escenarios", "Comparación Periódica"])
        
        with tab1:
            st.subheader("Impacto Hora a Hora")
            impact = FleetAnalytics.calculate_hourly_impact(st.session_state["merged"])
            st.dataframe(impact, hide_index=True, use_container_width=True)
            
            fig = px.bar(impact, x='HoraStr', y=['plan_srv', 'real_srv'],
                        title="Servicios Planificados vs Realizados",
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Análisis de Escenarios")
            missing = st.slider("¿Cuántos vehículos faltarían?", 0, 20, 1)
            scenario = FleetAnalytics.scenario_impact_missing_vehicles(st.session_state["merged"], missing)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Servicios Perdidos", scenario['projected_lost_services'])
            with col2:
                st.metric("% de Pérdida", f"{scenario['service_loss_pct']:.1f}%")
            with col3:
                st.metric("Nuevos Servicios", scenario['new_total_services'])
            with col4:
                st.metric("Coef. Operativo", f"{scenario['new_operational_coefficient']:.2f}")
        
        with tab3:
            st.subheader("Comparación Período vs Período")
            merged = st.session_state["merged"]
            if 'Fecha' in merged.columns:
                dates = sorted(merged['Fecha'].unique())
                if len(dates) >= 2:
                    date1 = st.date_input("Período 1", value=dates[-2])
                    date2 = st.date_input("Período 2", value=dates[-1])
                    
                    df1 = merged[merged['Fecha'] == pd.to_datetime(date1).date()]
                    df2 = merged[merged['Fecha'] == pd.to_datetime(date2).date()]
                    
                    comparison = KPICalculator.compare_periods(df2, df1)
                    
                    for key, values in comparison.items():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(key, f"{values['current']:.2f}")
                        with col2:
                            st.metric("Anterior", f"{values['previous']:.2f}")
                        with col3:
                            delta = f"{values['change_pct']:.1f}% ({values['trend'].upper()})"
                            st.metric("Cambio", delta)

# ==========================
# 6. ML & FORECASTING
# ==========================
elif menu == "🧠 ML & Forecasting":
    st.title("🧠 Predicciones con IA")
    
    col_train, col_pred = st.columns(2)
    
    with col_train:
        st.subheader("📚 Entrenar Modelos")
        if st.button("Entrenar Forecast"):
            if not st.session_state["merged"].empty:
                result = st.session_state["forecast_model"].train(st.session_state["merged"], "Servicios_Reales")
                st.json(result)
        
        if st.button("Entrenar Delay Predictor"):
            if not st.session_state["real_df"].empty:
                result = st.session_state["delay_predictor"].train(st.session_state["real_df"])
                st.json(result)
    
    with col_pred:
        st.subheader("🔮 Predicciones")
        horizonte = st.slider("Horizonte (horas)", 1, 24, 6)
        
        if st.button("Generar Forecast"):
            if not st.session_state["merged"].empty:
                preds = forecast_next_hours(st.session_state["merged"], "Servicios_Reales", horizonte)
                st.dataframe(preds, hide_index=True, use_container_width=True)

# ==========================
# 7. PLANIFICACIÓN DE TURNOS
# ==========================
elif menu == "📅 Planificación":
    st.title("📅 Planificación Semanal de Turnos")
    
    col_add, col_view = st.columns(2)
    
    with col_add:
        st.subheader("Agregar Turno")
        agent_id = st.text_input("ID Agente")
        week_start = st.date_input("Inicio de semana")
        day = st.selectbox("Día", ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"])
        shift_start = st.time_input("Hora inicio")
        shift_end = st.time_input("Hora fin")
        assigned_vehicle = st.text_input("Vehículo asignado (opcional)")
        
        if st.button("Crear Turno"):
            st.success(f"✅ Turno creado: {agent_id} - {day}")
    
    with col_view:
        st.subheader("Turnos Programados")
        st.info("Ver turnos aquí (integración BD)")

# ==========================
# 8. AUDITORÍA/EXPORT
# ==========================
else:  # menu == "💾 Auditoría/Export"
    st.title("💾 Auditoría y Exportación")
    
    if st.session_state["merged"].empty:
        st.info("Sin datos para exportar.")
    else:
        merged = st.session_state["merged"]
        
        # Excel export
        st.subheader("📊 Exportar Excel")
        resumen = pd.DataFrame([agg_error_metrics(merged)])
        book = {
            "Resumen": resumen,
            "Nacional": aggregate_nacional(merged),
            "Por_Bases": aggregate_bases(merged),
            "Por_Franja": aggregate_franja(merged),
            "Por_CAT": aggregate_cat(merged),
            "Detalle": merged
        }
        
        xls, fname = to_excel_bytes(book, "reporte_flota_completo.xlsx")
        st.download_button("⬇️ Descargar Excel", data=xls, file_name=fname,
                          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ==========================
# Helpers
# ==========================
def _parse_any(df_or_text, is_plan: bool):
    """Helper para parsear Plan o Real"""
    from utils_ops import parse_text_table, guess_mapping, apply_mapping, enrich_time
    
    if isinstance(df_or_text, str):
        df_raw = parse_text_table(df_or_text)
    else:
        df_raw = df_or_text
    
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(), {}
    
    mapping = guess_mapping(df_raw, ALIAS_PLAN if is_plan else ALIAS_REAL)
    out = apply_mapping(df_raw, mapping, kind="plan" if is_plan else "real")
    out = enrich_time(out)
    return out, mapping
