# database_models.py — Data persistence layer with SQLAlchemy + SQLite
from __future__ import annotations
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timedelta
import json
from pathlib import Path

Base = declarative_base()
DB_PATH = Path("data/fleet.db")
DB_PATH.parent.mkdir(exist_ok=True)

class Vehicle(Base):
    """Representa una unidad/móvil de flota"""
    __tablename__ = "vehicles"
    
    id = Column(Integer, primary_key=True)
    gps_id = Column(String(50), unique=True, nullable=False, index=True)
    base = Column(String(100), nullable=False, index=True)
    service_type = Column(String(50), nullable=False)  # '6001', 'Mecanica', etc.
    agent_id = Column(String(50), nullable=True)
    status = Column(String(20), default="available")  # available, in_service, maintenance, offline
    last_gps_update = Column(DateTime, nullable=True)
    last_lat = Column(Float, nullable=True)
    last_lon = Column(Float, nullable=True)
    total_km = Column(Float, default=0.0)
    operational_score = Column(Float, default=100.0)  # 0-100
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    gps_history = relationship("GPSTracking", back_populates="vehicle", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="vehicle", cascade="all, delete-orphan")
    trips = relationship("Trip", back_populates="vehicle", cascade="all, delete-orphan")

class GPSTracking(Base):
    """Historial de seguimiento GPS en tiempo real"""
    __tablename__ = "gps_tracking"
    
    id = Column(Integer, primary_key=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    speed = Column(Float, default=0.0)
    heading = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    vehicle = relationship("Vehicle", back_populates="gps_history")

class Trip(Base):
    """Registro de viajes/servicios prestados"""
    __tablename__ = "trips"
    
    id = Column(Integer, primary_key=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=False, index=True)
    service_type = Column(String(50), nullable=False)
    origin_lat = Column(Float, nullable=True)
    origin_lon = Column(Float, nullable=True)
    destination_lat = Column(Float, nullable=True)
    destination_lon = Column(Float, nullable=True)
    estimated_duration_min = Column(Integer, nullable=True)  # minutos
    actual_duration_min = Column(Integer, nullable=True)
    distance_km = Column(Float, nullable=True)
    status = Column(String(20), default="scheduled")  # scheduled, in_progress, completed, cancelled
    delay_min = Column(Integer, default=0)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    vehicle = relationship("Vehicle", back_populates="trips")

class Alert(Base):
    """Alertas en vivo con niveles de severidad"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)  # delay, offline, maintenance, low_fuel, etc.
    severity = Column(String(10), nullable=False)  # info, warning, critical
    message = Column(Text, nullable=False)
    data_json = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    vehicle = relationship("Vehicle", back_populates="alerts")

class ForecastMetric(Base):
    """Proyecciones de flota por hora y tipo de servicio"""
    __tablename__ = "forecast_metrics"
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False, index=True)
    hour = Column(Integer, nullable=False)  # 0-23
    service_type = Column(String(50), nullable=False, index=True)
    base = Column(String(100), nullable=False, index=True)
    projected_services = Column(Integer, nullable=False)  # servicios esperados
    actual_services = Column(Integer, nullable=True)  # servicios reales
    projected_vehicles = Column(Integer, nullable=False)  # móviles en nómina esperados
    actual_vehicles = Column(Integer, nullable=True)  # móviles en nómina reales
    operational_coefficient = Column(Float, nullable=True)  # KPI de eficiencia
    deviation_pct = Column(Float, nullable=True)  # desviación %
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = ('__table_args__', {'indexes': [('date', 'hour', 'service_type', 'base')]})

class TurnSchedule(Base):
    """Planificación semanal de turnos"""
    __tablename__ = "turn_schedules"
    
    id = Column(Integer, primary_key=True)
    week_start = Column(DateTime, nullable=False, index=True)
    agent_id = Column(String(50), nullable=False)
    day_of_week = Column(Integer, nullable=False)  # 0=Lunes, 6=Domingo
    shift_start = Column(String(5), nullable=False)  # HH:MM
    shift_end = Column(String(5), nullable=False)
    assigned_vehicle = Column(String(50), nullable=True)
    status = Column(String(20), default="scheduled")  # scheduled, confirmed, completed, cancelled
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = ('__table_args__', {'indexes': [('week_start', 'agent_id', 'day_of_week')]})

class HistoricalIndicator(Base):
    """Historial de indicadores para análisis período a período"""
    __tablename__ = "historical_indicators"
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False, index=True)
    period_type = Column(String(20), nullable=False)  # day, week, month
    base = Column(String(100), nullable=False, index=True)
    service_type = Column(String(50), nullable=False)
    total_services = Column(Integer, nullable=False)
    on_time_services = Column(Integer, nullable=False)
    delayed_services = Column(Integer, nullable=False)
    avg_delay_min = Column(Float, nullable=True)
    fleet_utilization_pct = Column(Float, nullable=True)  # % de móviles utilizados
    operational_coefficient = Column(Float, nullable=True)
    fuel_consumed_liters = Column(Float, nullable=True)
    total_km = Column(Float, nullable=True)
    avg_km_per_service = Column(Float, nullable=True)
    cost_per_service = Column(Float, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class MLModel(Base):
    """Gestión de modelos ML entrenados"""
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False, unique=True)
    model_type = Column(String(50), nullable=False)  # forecast, delay_predictor, anomaly_detector
    version = Column(String(20), nullable=False)
    accuracy = Column(Float, nullable=True)
    last_trained = Column(DateTime, nullable=True)
    model_path = Column(String(255), nullable=True)
    hyperparams_json = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def init_db() -> sessionmaker:
    """Inicializa BD y retorna session factory"""
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session
