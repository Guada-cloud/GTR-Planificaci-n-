# gps_realtime_engine.py — Real-time GPS tracking, geofencing, alert generation
from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from math import radians, cos, sin, asin, sqrt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Location:
    """Representa una ubicación geográfica"""
    lat: float
    lon: float
    timestamp: datetime

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula distancia en km entre dos puntos GPS (Haversine formula)"""
    try:
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radio de la Tierra en km
        return c * r
    except:
        return 0.0

class GPSRealTimeEngine:
    """Motor de seguimiento GPS en tiempo real con alertas automáticas"""
    
    def __init__(self):
        self.vehicle_trails: Dict[str, List[Location]] = {}  # histórico por vehículo
        self.geofences: Dict[str, Dict] = {}  # bases/zonas definidas
        self.offline_timeout_min = 15  # alerta si no hay GPS en 15 min
        self.speed_alert_kmh = 120  # alerta si excede velocidad
        self.delay_alert_min = 20  # alerta si retraso > 20 min
    
    def register_geofence(self, name: str, lat: float, lon: float, radius_km: float):
        """Registra una zona geográfica (base, taller, etc.)"""
        self.geofences[name] = {
            'lat': lat,
            'lon': lon,
            'radius_km': radius_km,
            'created_at': datetime.utcnow()
        }
        logger.info(f"Geofence registered: {name} ({lat:.4f}, {lon:.4f}), radius={radius_km}km")
    
    def process_gps_update(self, vehicle_id: str, lat: float, lon: float, speed: float, heading: Optional[float] = None) -> List[Dict]:
        """Procesa actualización GPS y genera alertas"""
        alerts = []
        now = datetime.utcnow()
        location = Location(lat, lon, now)
        
        # Guardar en histórico
        if vehicle_id not in self.vehicle_trails:
            self.vehicle_trails[vehicle_id] = []
        self.vehicle_trails[vehicle_id].append(location)
        
        # Limpiar histórico (mantener últimas 24 horas)
        cutoff = now - timedelta(hours=24)
        self.vehicle_trails[vehicle_id] = [loc for loc in self.vehicle_trails[vehicle_id] if loc.timestamp > cutoff]
        
        # Alerta: velocidad excesiva
        if speed > self.speed_alert_kmh:
            alerts.append({
                'type': 'speeding',
                'severity': 'warning',
                'message': f"Vehicle {vehicle_id} speeding: {speed:.1f} km/h",
                'data': {'speed': speed, 'limit': self.speed_alert_kmh}
            })
        
        # Alerta: geofence (salida/entrada a zona)
        inside_geofences = self._check_geofences(lat, lon)
        last_location = self.vehicle_trails[vehicle_id][-2] if len(self.vehicle_trails[vehicle_id]) > 1 else None
        if last_location:
            was_inside = self._check_geofences(last_location.lat, last_location.lon)
            for fence in inside_geofences:
                if fence not in was_inside:
                    alerts.append({
                        'type': 'geofence_enter',
                        'severity': 'info',
                        'message': f"Vehicle {vehicle_id} entered zone {fence}",
                        'data': {'zone': fence, 'lat': lat, 'lon': lon}
                    })
            for fence in was_inside:
                if fence not in inside_geofences:
                    alerts.append({
                        'type': 'geofence_exit',
                        'severity': 'info',
                        'message': f"Vehicle {vehicle_id} exited zone {fence}",
                        'data': {'zone': fence, 'lat': lat, 'lon': lon}
                    })
        
        return alerts
    
    def detect_offline_vehicles(self, vehicle_updates: Dict[str, datetime]) -> List[Dict]:
        """Detecta móviles sin señal GPS"""
        alerts = []
        now = datetime.utcnow()
        
        for vehicle_id, last_update in vehicle_updates.items():
            minutes_offline = (now - last_update).total_seconds() / 60
            if minutes_offline > self.offline_timeout_min:
                severity = 'critical' if minutes_offline > 60 else 'warning'
                alerts.append({
                    'type': 'offline',
                    'severity': severity,
                    'message': f"Vehicle {vehicle_id} offline for {int(minutes_offline)} minutes",
                    'data': {'vehicle_id': vehicle_id, 'minutes_offline': int(minutes_offline)}
                })
        
        return alerts
    
    def calculate_route_deviation(self, vehicle_id: str, planned_route: List[Tuple[float, float]]) -> Tuple[float, bool]:
        """Calcula desviación de ruta respecto a lo planificado. Retorna (desviación_km, está_desviado)"""
        if vehicle_id not in self.vehicle_trails or not self.vehicle_trails[vehicle_id]:
            return 0.0, False
        
        current_loc = self.vehicle_trails[vehicle_id][-1]
        
        # Encontrar punto más cercano en ruta planeada
        min_distance = float('inf')
        for waypoint in planned_route:
            dist = haversine_distance(current_loc.lat, current_loc.lon, waypoint[0], waypoint[1])
            min_distance = min(min_distance, dist)
        
        # Alerta si desviación > 2 km
        is_deviated = min_distance > 2.0
        return min_distance, is_deviated
    
    def _check_geofences(self, lat: float, lon: float) -> List[str]:
        """Retorna lista de geofences donde está ubicado el punto"""
        inside = []
        for name, fence in self.geofences.items():
            dist = haversine_distance(lat, lon, fence['lat'], fence['lon'])
            if dist <= fence['radius_km']:
                inside.append(name)
        return inside
    
    def get_vehicle_trail(self, vehicle_id: str, hours: int = 24) -> List[Dict]:
        """Retorna historial GPS del vehículo (últimas N horas)"""
        if vehicle_id not in self.vehicle_trails:
            return []
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        trail = [
            {'lat': loc.lat, 'lon': loc.lon, 'timestamp': loc.timestamp.isoformat()}
            for loc in self.vehicle_trails[vehicle_id]
            if loc.timestamp > cutoff
        ]
        return trail
    
    def estimate_eta(self, vehicle_id: str, destination_lat: float, destination_lon: float, avg_speed_kmh: float = 40) -> Optional[timedelta]:
        """Estima ETA basado en distancia y velocidad promedio"""
        if vehicle_id not in self.vehicle_trails or not self.vehicle_trails[vehicle_id]:
            return None
        
        current_loc = self.vehicle_trails[vehicle_id][-1]
        distance_km = haversine_distance(current_loc.lat, current_loc.lon, destination_lat, destination_lon)
        
        if avg_speed_kmh <= 0:
            return None
        
        hours = distance_km / avg_speed_kmh
        return timedelta(hours=hours)
