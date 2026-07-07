# Fleet Management Platform v2.0 — AI + Real-Time GPS

**Solución integral de gestión de flota con seguimiento GPS en tiempo real, predicción con IA, alertas inteligentes y análisis operacional avanzado.**

## 🎯 Características Principales

### 1. **Plan vs Real (heredado mejorado)**
- ✅ Carga flexible: XLSX, CSV, pegado directo
- ✅ Detección automática de sinónimos y abreviaturas en español
- ✅ Comparación temporal (Plan vs Real)
- ✅ Análisis de desvíos por servicio tipo

### 2. **Seguimiento GPS en Tiempo Real**
- 🗺️ Tracking de unidades con histórico
- 🛰️ Integración con datos GPS (lat/lon, velocidad, heading)
- ⛔ Geofencing: alertas al entrar/salir de zonas (bases, talleres)
- 📊 Visualización de rutas y trazabilidad

### 3. **Alertas en Vivo**
- 🚨 Múltiples niveles de severidad: INFO, WARNING, CRITICAL
- 📱 Tipos: offline (sin GPS), delay (retrasos), speeding (velocidad), geofence, maintenance
- ⏱️ Detección automática de móviles offline (configurable)
- 🔴 Centro de control centralizado de alertas

### 4. **Predicción con IA/ML**
- 🧠 **FleetForecastModel**: Proyecta demanda horaria de servicios y móviles
- 🎯 **DelayPredictor**: Estima retrasos en función de condiciones
- 🔍 **AnomalyDetector**: Identifica comportamientos GPS anómalos
- 📈 Modelos persistidos (joblib) y reutilizables

### 5. **Análisis Avanzado**
- 📊 Coeficiente operativo: `(Efectividad_Servicios × 0.7) + (Disponibilidad_Flota × 0.3)`
- ⏰ Análisis impacto hora-a-hora: servicios perdidos, retrasos acumulados
- 📉 Escenarios: impacto de X vehículos faltantes
- 📊 Comparación período vs período (KPIs deltas)

### 6. **Gestión de Turnos**
- 📅 Planificación semanal con asignación de vehículos
- 👥 Vinculación agente ↔ móvil ↔ servicio
- ✅ Estados: scheduled, confirmed, completed, cancelled

### 7. **Persistencia de Datos**
- 💾 BD SQLite con tablas: vehicles, gps_tracking, trips, alerts, forecast_metrics, turn_schedules, historical_indicators, ml_models
- 📦 Historial completo de indicadores (análisis por período)
- 🔐 Auditoría: quién, qué, cuándo

### 8. **Dashboards & KPIs**
- 📈 KPIs principales: Efectividad, Utilización, Puntualidad, Coef. Operativo
- 🎯 Gráficos interactivos con Plotly
- 🚨 Alertas activas con contador por severidad
- 📊 Heatmaps, waterfalls, dumbbells (bases comparadas)

## 🚀 Instalación

```bash
# 1. Clonar repositorio
git clone https://github.com/Guada-cloud/GTR-Planificacion-.git
cd GTR-Planificacion-

# 2. Rama de desarrollo
git checkout feature/ai-fleet-management-v2

# 3. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate  # Windows

# 4. Instalar dependencias
pip install -r requirements_v2.txt

# 5. Ejecutar app
streamlit run app_v2.py
```

## 📁 Estructura de Módulos

```
GTR-Planificacion-/
├── app_v2.py                    # Streamlit app principal (v2)
├── database_models.py           # Modelos SQLAlchemy + ORM
├── fleet_ml_engine.py          # ML: Forecast, Delay, Anomaly
├── gps_realtime_engine.py      # Tracking GPS, geofencing
├── fleet_analytics.py          # KPIs, análisis, escenarios
├── utils_ops.py                # Mapeo, merge, cálculos (heredado)
├── viz_helpers.py              # Componentes Plotly (heredado)
├── requirements_v2.txt         # Dependencias
├── data/                        # Datos locales
│   └── fleet.db               # Base de datos SQLite
└── models/                      # Modelos ML persistidos
    ├── forecast_model.pkl
    ├── delay_predictor.pkl
    └── anomaly_detector.pkl
```

## 🔧 Cómo Usar

### A. Cargar Datos Plan vs Real
1. **📥 Cargar Datos** → Subir XLSX/CSV o pegar datos
2. Sistema detecta automáticamente columnas (sinónimos: "srv", "servicios", "Svc plan", etc.)
3. Hacer clic **🔗 Comparar Ahora**

### B. Seguimiento GPS
1. **🗺️ Tracking GPS** → Cargar CSV con histórico o ingresar manualmente
2. Definir **geofences** (bases, talleres, etc.)
3. Ver alertas de geofence (entrada/salida)

### C. Entrenar Modelos ML
1. **🧠 ML & Forecasting** → Click en "Entrenar Forecast" o "Entrenar Delay Predictor"
2. Sistema usa datos cargados para entrenar (train/test 80/20)
3. Modelos se persisten automáticamente
4. Ver métricas de precisión (MAE, R², etc.)

### D. Análisis de Escenarios
1. **📈 Análisis** → Tab "Escenarios"
2. Mover slider "¿Cuántos vehículos faltarían?"
3. Ver impacto en servicios perdidos y coef. operativo

### E. Planificar Turnos
1. **📅 Planificación** → Agregar turno (agente, día, horario, vehículo)
2. Vincular con GPS para validar cobertura

## 📊 KPIs Principales

| KPI | Fórmula | Rango |
|-----|---------|-------|
| **Efectividad** | `1 - \|Real - Plan\| / Plan` | 0-100% |
| **Puntualidad** | `# Servicios on-time / Total` | 0-100% |
| **Utilización Flota** | `Móviles reales / Móviles planificados` | 0-100% |
| **Coef. Operativo** | `0.7 × Efectividad + 0.3 × Utilización` | 0-1 |
| **Desv. Servicios %** | `(Real - Plan) / Plan × 100` | -∞ a +∞ |
| **Delay Promedio** | `Media(delay_min)` | minutos |

## 🤖 Modelos ML

### FleetForecastModel (RandomForest)
- **Input**: hora, día, histórico (7d), temperatura
- **Output**: Servicios proyectados para próxima hora
- **Métricas**: MAE, R²
- **Entrenamiento**: Automático, reutilizable

### DelayPredictor (RandomForest)
- **Input**: hora, día, distancia, vehículos disponibles, histórico de retrasos
- **Output**: Delay estimado (minutos)
- **Uso**: Anticipar retrasos y alertar proactivamente

### AnomalyDetector (IsolationForest)
- **Input**: velocidad, rumbo, precisión GPS
- **Output**: 1 = normal, -1 = anómalo
- **Uso**: Detectar comportamientos inusuales (paradas inesperadas, rutas desviadas)

## 🚨 Tipos de Alertas

| Tipo | Severidad | Condición | Acción |
|------|-----------|-----------|--------|
| **offline** | WARNING/CRITICAL | Sin GPS > 15 min | Notificar supervisor |
| **speeding** | WARNING | Velocidad > 120 km/h | Log, historial |
| **delay** | WARNING | Retraso > 20 min | Redirigir si es posible |
| **geofence_exit** | INFO | Sale de zona | Notificar |
| **geofence_enter** | INFO | Entra a zona | Notificar |
| **maintenance** | CRITICAL | Vehículo en mantenimiento | Remover de operación |

## 📈 Análisis Período a Período

Comparación automática de KPIs entre dos fechas:
- **Cambio %**: `(Actual - Anterior) / Anterior × 100`
- **Trend**: UP ↑ / DOWN ↓ / STABLE →
- Útil para detectar degradación de desempeño

## 💾 Exportación

### Excel Completo
- Hoja 1: **Resumen** (MAPE, MAE, Bias)
- Hoja 2: **Nacional** (totales por hora)
- Hoja 3: **Por Bases** (desvío por base)
- Hoja 4: **Por Franja** (MADRUGADA/MAÑANA/TARDE/NOCHE)
- Hoja 5: **Por CAT** (categoría de servicio)
- Hoja 6: **Detalle** (registro completo)

## 🔄 Roadmap v2.1+

- [ ] Integración con GAP (API)
- [ ] Mapas interactivos (Leaflet/Folium)
- [ ] Notificaciones email/Slack
- [ ] Autoscaling de pronósticos
- [ ] Dashboard para móviles (React Native)
- [ ] API REST (FastAPI)
- [ ] Autenticación multi-usuario
- [ ] Reportes automáticos (scheduling)
- [ ] Análisis de costos por servicio

## 📝 Notas

- Los modelos ML se entrenan **on-demand** con datos cargados
- Geofences se almacenan en memoria (sesión actual)
- BD SQLite en `data/fleet.db` — considerar migrar a PostgreSQL en producción
- Alertas se generan automáticamente pero se pueden silenciar/acknowlegde

## 👥 Soporte

Para bugs, features o preguntas: crear issue en GitHub o contactar a [@Guada-cloud](https://github.com/Guada-cloud)

---

**v2.0** — Built with ❤️ using Python, Streamlit, Scikit-Learn, SQLAlchemy
