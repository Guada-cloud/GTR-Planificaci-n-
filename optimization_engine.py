# optimization_engine.py - Optimizacion de rutas y eficiencia operacional
from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import numpy as np
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RouteOptimizer:
    """Optimiza rutas para conductores"""
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_route(self, waypoints: List[Tuple[float, float]],
                      vehicle_capacity: int = 1,
                      time_windows: Optional[List[Tuple[datetime, datetime]]] = None) -> Dict:
        """Optimiza ruta considerando waypoints, capacidad y ventanas de tiempo"""
        
        if len(waypoints) < 2:
            return {'error': 'Minimo 2 waypoints requeridos'}
        
        # Algoritmo greedy: nearest neighbor TSP
        current = waypoints[0]
        remaining = waypoints[1:]
        route = [current]
        total_distance = 0.0
        
        while remaining:
            nearest_idx = 0
            nearest_distance = float('inf')
            
            for idx, point in enumerate(remaining):
                distance = self._haversine(current[0], current[1], point[0], point[1])
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_idx = idx
            
            nearest_point = remaining.pop(nearest_idx)
            route.append(nearest_point)
            total_distance += nearest_distance
            current = nearest_point
        
        # Agregar distancia final a origen
        total_distance += self._haversine(route[-1][0], route[-1][1], route[0][0], route[0][1])
        
        optimization = {
            'route': route,
            'total_distance_km': total_distance,
            'estimated_duration_min': int(total_distance * 1.5),
            'waypoints_count': len(route),
            'created_at': datetime.utcnow().isoformat()
        }
        
        self.optimization_history.append(optimization)
        logger.info(f"[OPTIMIZE] Ruta optimizada: {len(route)} puntos, {total_distance:.2f} km")
        
        return optimization
    
    def calculate_vehicle_utilization(self, trips: List[Dict]) -> float:
        """Calcula tasa de utilizacion de vehiculo"""
        if not trips:
            return 0.0
        
        total_hours = 0
        active_hours = 0
        
        for trip in trips:
            start = trip.get('start_time')
            end = trip.get('end_time')
            
            if start and end:
                duration = (end - start).total_seconds() / 3600
                total_hours += duration
                active_hours += duration  # Simplificado
        
        return (active_hours / total_hours * 100) if total_hours > 0 else 0.0
    
    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcula distancia en km"""
        try:
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            return c * 6371
        except:
            return 0.0

class FuelConsumptionAnalyzer:
    """Analiza consumo de combustible"""
    
    def __init__(self, avg_consumption_km_per_liter: float = 7.0):
        self.avg_consumption = avg_consumption_km_per_liter
        self.fuel_log = []
    
    def estimate_fuel_consumption(self, distance_km: float) -> float:
        """Estima consumo de combustible"""
        return distance_km / self.avg_consumption
    
    def calculate_fuel_cost(self, distance_km: float, price_per_liter: float) -> float:
        """Calcula costo de combustible"""
        liters = self.estimate_fuel_consumption(distance_km)
        return liters * price_per_liter
    
    def log_fuel_fill(self, vehicle_id: str, liters: float, price_per_liter: float,
                     odometer_km: float):
        """Registra carga de combustible"""
        entry = {
            'vehicle_id': vehicle_id,
            'timestamp': datetime.utcnow(),
            'liters': liters,
            'total_cost': liters * price_per_liter,
            'odometer_km': odometer_km
        }
        self.fuel_log.append(entry)
        logger.info(f"[FUEL] {vehicle_id}: {liters}L a {price_per_liter}/L")
    
    def get_fuel_efficiency_report(self, vehicle_id: str, date_from: datetime,
                                   date_to: datetime) -> Dict:
        """Reporte de eficiencia de combustible"""
        
        records = [
            r for r in self.fuel_log
            if r['vehicle_id'] == vehicle_id and
               date_from <= r['timestamp'] <= date_to
        ]
        
        if len(records) < 2:
            return {'vehicle_id': vehicle_id, 'records': 0}
        
        total_liters = sum(r['liters'] for r in records)
        total_cost = sum(r['total_cost'] for r in records)
        km_range = records[-1]['odometer_km'] - records[0]['odometer_km']
        
        efficiency = km_range / total_liters if total_liters > 0 else 0
        
        return {
            'vehicle_id': vehicle_id,
            'period_start': date_from.date(),
            'period_end': date_to.date(),
            'total_liters': total_liters,
            'total_km': km_range,
            'efficiency_km_per_liter': efficiency,
            'total_cost': total_cost,
            'avg_cost_per_km': total_cost / km_range if km_range > 0 else 0
        }

class MaintenancePredictor:
    """Predice necesidad de mantenimiento basado en uso"""
    
    def __init__(self):
        self.maintenance_schedules = {}
        self.maintenance_history = []
    
    def predict_maintenance(self, vehicle_id: str, odometer_km: float,
                           engine_hours: float) -> List[Dict]:
        """Predice mantenimientos requeridos"""
        
        maintenance_tasks = []
        
        # Cambio de aceite cada 10000 km
        if odometer_km % 10000 < 500:
            maintenance_tasks.append({
                'type': 'oil_change',
                'description': 'Cambio de aceite',
                'urgency': 'high' if odometer_km % 10000 < 200 else 'medium',
                'km': odometer_km
            })
        
        # Rotacion de llantas cada 20000 km
        if odometer_km % 20000 < 500:
            maintenance_tasks.append({
                'type': 'tire_rotation',
                'description': 'Rotacion de llantas',
                'urgency': 'medium',
                'km': odometer_km
            })
        
        # Inspeccion general cada 30000 km
        if odometer_km % 30000 < 500:
            maintenance_tasks.append({
                'type': 'inspection',
                'description': 'Inspeccion general',
                'urgency': 'low',
                'km': odometer_km
            })
        
        logger.info(f"[MAINT] {len(maintenance_tasks)} tareas previstas para {vehicle_id}")
        
        return maintenance_tasks
    
    def log_maintenance(self, vehicle_id: str, maintenance_type: str,
                       description: str, cost: float, odometer_km: float):
        """Registra mantenimiento realizado"""
        entry = {
            'vehicle_id': vehicle_id,
            'type': maintenance_type,
            'description': description,
            'cost': cost,
            'odometer_km': odometer_km,
            'timestamp': datetime.utcnow()
        }
        self.maintenance_history.append(entry)
        logger.info(f"[MAINT] Mantenimiento registrado: {vehicle_id} - {maintenance_type}")
    
    def get_maintenance_history(self, vehicle_id: str) -> List[Dict]:
        """Obtiene historial de mantenimiento"""
        return [
            m for m in self.maintenance_history
            if m['vehicle_id'] == vehicle_id
        ]
