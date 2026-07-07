# reconciliation_engine.py - Conciliacion de datos: km GPS vs km cobrado, alertas de desvio
from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReconciliationRecord:
    """Registro de conciliacion entre GPS y tarifas"""
    service_id: str
    vehicle_id: str
    client_id: str
    service_date: datetime
    estimated_distance_km: float
    gps_distance_km: float
    distance_variance_pct: float
    distance_variance_km: float
    estimated_fare: float
    actual_fare: float
    fare_adjustment: float
    detour_ratio: float
    alert_level: str
    alert_reason: str
    status: str
    notes: str = ""
    reconciled_at: Optional[datetime] = None
    reconciled_by: str = "SYSTEM"

class DetourAnalyzer:
    """Analiza desvios en rutas y genera alertas"""
    
    DETOUR_WARNING_THRESHOLD = 1.15
    DETOUR_CRITICAL_THRESHOLD = 1.30
    DISTANCE_VARIANCE_WARNING = 0.10
    DISTANCE_VARIANCE_CRITICAL = 0.20
    
    def analyze_detour(self, estimated_km: float, gps_km: float, detour_ratio: float) -> Tuple[str, str]:
        """Analiza desvio y retorna (nivel_alerta, razon)"""
        
        if estimated_km <= 0:
            return "OK", "Distancia estimada invalida"
        
        variance_pct = abs(gps_km - estimated_km) / estimated_km
        
        if detour_ratio > self.DETOUR_CRITICAL_THRESHOLD or variance_pct > self.DISTANCE_VARIANCE_CRITICAL:
            reason = f"Desvio CRITICO: ratio={detour_ratio:.2f}, varianza={variance_pct*100:.1f}%"
            return "CRITICAL", reason
        
        elif detour_ratio > self.DETOUR_WARNING_THRESHOLD or variance_pct > self.DISTANCE_VARIANCE_WARNING:
            reason = f"Desvio WARNING: ratio={detour_ratio:.2f}, varianza={variance_pct*100:.1f}%"
            return "WARNING", reason
        
        return "OK", "Desvio dentro de parametros normales"
    
    def detect_route_irregularities(self, gps_points: List[Dict]) -> List[Dict]:
        """Detecta irregularidades en la ruta"""
        irregularities = []
        
        if len(gps_points) < 3:
            return irregularities
        
        for i in range(1, len(gps_points) - 1):
            prev_point = gps_points[i - 1]
            current_point = gps_points[i]
            next_point = gps_points[i + 1]
            
            time_diff = (current_point['timestamp'] - prev_point['timestamp']).total_seconds() / 3600
            if time_diff > 0:
                implied_speed = current_point.get('speed', 0)
                
                if implied_speed < 1 and time_diff > 0.083:
                    irregularities.append({
                        'type': 'parada_prolongada',
                        'timestamp': current_point['timestamp'],
                        'lat': current_point['lat'],
                        'lon': current_point['lon'],
                        'duration_min': int(time_diff * 60)
                    })
        
        return irregularities

class ReconciliationEngine:
    """Motor de conciliacion: km GPS vs km cobrado"""
    
    def __init__(self):
        self.reconciliation_records: List[ReconciliationRecord] = []
        self.detour_analyzer = DetourAnalyzer()
    
    def reconcile_service(self, service_data: Dict, gps_distance_km: float,
                         detour_ratio: float, actual_fare: float) -> ReconciliationRecord:
        """Reconcilia un servicio completo"""
        
        estimated_distance_km = service_data['estimated_distance_km']
        estimated_fare = service_data['estimated_fare']
        
        distance_variance_km = gps_distance_km - estimated_distance_km
        distance_variance_pct = distance_variance_km / estimated_distance_km if estimated_distance_km > 0 else 0
        
        fare_adjustment = actual_fare - estimated_fare
        
        alert_level, alert_reason = self.detour_analyzer.analyze_detour(
            estimated_distance_km, gps_distance_km, detour_ratio
        )
        
        record = ReconciliationRecord(
            service_id=service_data['id'],
            vehicle_id=service_data['assigned_vehicle_id'],
            client_id=service_data['client_id'],
            service_date=service_data['service_date'],
            estimated_distance_km=estimated_distance_km,
            gps_distance_km=gps_distance_km,
            distance_variance_pct=distance_variance_pct,
            distance_variance_km=distance_variance_km,
            estimated_fare=estimated_fare,
            actual_fare=actual_fare,
            fare_adjustment=fare_adjustment,
            detour_ratio=detour_ratio,
            alert_level=alert_level,
            alert_reason=alert_reason,
            status="PENDING"
        )
        
        self.reconciliation_records.append(record)
        
        logger.info(f"[RECON] Servicio {service_data['id']} reconciliado: {alert_level}")
        
        return record
    
    def reconcile_batch(self, services: List[Dict], gps_data: Dict[str, float],
                       fare_data: Dict[str, float]) -> pd.DataFrame:
        """Reconcilia lote de servicios"""
        records = []
        
        for service in services:
            service_id = service['id']
            
            if service_id not in gps_data:
                logger.warning(f"[RECON] Sin datos GPS para {service_id}")
                continue
            
            gps_km = gps_data[service_id]
            actual_fare = fare_data.get(service_id, service['estimated_fare'])
            detour = gps_km / service['estimated_distance_km'] if service['estimated_distance_km'] > 0 else 1.0
            
            record = self.reconcile_service(service, gps_km, detour, actual_fare)
            records.append(record.__dict__)
        
        df = pd.DataFrame(records)
        logger.info(f"[RECON] {len(records)} servicios reconciliados")
        
        return df
    
    def get_alerts_by_level(self, level: str = "CRITICAL") -> List[ReconciliationRecord]:
        """Retorna alertas de conciliacion por nivel"""
        return [r for r in self.reconciliation_records if r.alert_level == level]
    
    def get_variance_summary(self, date_from: datetime, date_to: datetime) -> Dict:
        """Resume varianzas de distancia y tarifa en periodo"""
        records = [
            r for r in self.reconciliation_records
            if date_from <= r.service_date <= date_to
        ]
        
        if not records:
            return {}
        
        df = pd.DataFrame([r.__dict__ for r in records])
        
        return {
            'total_services': len(records),
            'avg_distance_variance_pct': float(df['distance_variance_pct'].mean()),
            'avg_detour_ratio': float(df['detour_ratio'].mean()),
            'critical_alerts': (df['alert_level'] == 'CRITICAL').sum(),
            'warning_alerts': (df['alert_level'] == 'WARNING').sum(),
            'total_fare_adjustment': float(df['fare_adjustment'].sum()),
            'vehicles_with_issues': df[df['alert_level'] != 'OK']['vehicle_id'].nunique()
        }
    
    def approve_reconciliation(self, service_id: str, notes: str = "", approved_by: str = "SYSTEM") -> bool:
        """Aprueba reconciliacion de un servicio"""
        for record in self.reconciliation_records:
            if record.service_id == service_id:
                record.status = "RECONCILED"
                record.reconciled_at = datetime.utcnow()
                record.reconciled_by = approved_by
                record.notes = notes
                
                logger.info(f"[RECON] Servicio {service_id} aprobado por {approved_by}")
                return True
        
        return False
    
    def dispute_reconciliation(self, service_id: str, reason: str, disputed_by: str) -> bool:
        """Marca reconciliacion como disputada"""
        for record in self.reconciliation_records:
            if record.service_id == service_id:
                record.status = "DISPUTED"
                record.notes = f"Disputado por {disputed_by}: {reason}"
                
                logger.info(f"[RECON] Servicio {service_id} disputado: {reason}")
                return True
        
        return False
    
    def get_reconciliation_report(self, date_from: datetime, date_to: datetime) -> pd.DataFrame:
        """Genera reporte completo de conciliaciones en periodo"""
        records = [
            r for r in self.reconciliation_records
            if date_from <= r.service_date <= date_to
        ]
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame([r.__dict__ for r in records])
        
        df['alert_priority'] = df['alert_level'].apply(
            lambda x: 3 if x == 'CRITICAL' else (2 if x == 'WARNING' else 1)
        )
        
        return df.sort_values('alert_priority', ascending=False)
