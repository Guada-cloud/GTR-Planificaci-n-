# service_management.py - Gestion de servicios: validacion, tarificacion, asignacion y seguimiento
from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
from math import radians, cos, sin, asin, sqrt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceStatus(str, Enum):
    """Estados del workflow de un servicio"""
    PENDIENTE = "PENDIENTE"
    ASIGNADO = "ASIGNADO"
    EN_CAMINO = "EN_CAMINO"
    EN_ORIGEN = "EN_ORIGEN"
    EN_DESTINO = "EN_DESTINO"
    FINALIZADO = "FINALIZADO"
    CANCELADO = "CANCELADO"

class AssignmentType(str, Enum):
    """Tipos de asignacion de servicios"""
    MANUAL = "MANUAL"
    AUTOMATICO = "AUTOMATICO"
    OFERTA = "OFERTA"

@dataclass
class ServiceRequest:
    """Solicitud de servicio"""
    id: str
    client_id: str
    service_type: str
    origin_lat: float
    origin_lon: float
    destination_lat: float
    destination_lon: float
    origin_address: str = ""
    destination_address: str = ""
    passenger_count: int = 1
    luggage_count: int = 0
    special_requirements: str = ""
    estimated_distance_km: float = 0.0
    estimated_duration_min: int = 0
    base_fare: float = 0.0
    surge_multiplier: float = 1.0
    total_fare: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: ServiceStatus = ServiceStatus.PENDIENTE
    assigned_vehicle_id: Optional[str] = None
    acceptance_time: Optional[datetime] = None
    pickup_time: Optional[datetime] = None
    dropoff_time: Optional[datetime] = None
    actual_distance_km: float = 0.0
    actual_duration_min: int = 0
    actual_fare: float = 0.0
    detour_ratio: float = 1.0
    notes: str = ""

class ServiceValidator:
    """Valida y procesa solicitudes de servicio"""
    
    MIN_PASSENGER = 1
    MAX_PASSENGER = 6
    MIN_DISTANCE_KM = 0.5
    
    @staticmethod
    def validate_request(request: ServiceRequest) -> Tuple[bool, List[str]]:
        """Valida solicitud de servicio. Retorna (valido, lista_errores)"""
        errors = []
        
        if not request.id or len(request.id.strip()) == 0:
            errors.append("ID de servicio requerido")
        
        if not request.client_id:
            errors.append("ID de cliente requerido")
        
        if request.passenger_count < ServiceValidator.MIN_PASSENGER or \
           request.passenger_count > ServiceValidator.MAX_PASSENGER:
            errors.append(f"Pasajeros debe estar entre {ServiceValidator.MIN_PASSENGER} y {ServiceValidator.MAX_PASSENGER}")
        
        if not (-90 <= request.origin_lat <= 90) or not (-180 <= request.origin_lon <= 180):
            errors.append("Coordenadas origen invalidas")
        
        if not (-90 <= request.destination_lat <= 90) or not (-180 <= request.destination_lon <= 180):
            errors.append("Coordenadas destino invalidas")
        
        if request.origin_lat == request.destination_lat and \
           request.origin_lon == request.destination_lon:
            errors.append("Origen y destino no pueden ser iguales")
        
        if request.base_fare < 0:
            errors.append("Tarifa base no puede ser negativa")
        
        if request.surge_multiplier < 0.5 or request.surge_multiplier > 5.0:
            errors.append("Multiplicador de demanda debe estar entre 0.5 y 5.0")
        
        return len(errors) == 0, errors

class TarificationEngine:
    """Motor de tarificacion automatica"""
    
    def __init__(self, base_rate_per_km: float = 1.5, base_fare: float = 2.0,
                 waiting_rate_per_min: float = 0.05):
        self.base_rate_per_km = base_rate_per_km
        self.base_fare = base_fare
        self.waiting_rate_per_min = waiting_rate_per_min
        self.surge_rules = {}
    
    def calculate_fare(self, request: ServiceRequest) -> float:
        """Calcula tarifa total basado en distancia, demanda y extras"""
        fare = self.base_fare
        
        if request.estimated_distance_km > 0:
            fare += request.estimated_distance_km * self.base_rate_per_km
        
        fare *= request.surge_multiplier
        fare += self._calculate_extras(request)
        
        return max(fare, self.base_fare)
    
    def _calculate_extras(self, request: ServiceRequest) -> float:
        """Calcula cargos adicionales por pasajeros, equipaje, etc."""
        extras = 0.0
        
        if request.passenger_count > 1:
            extras += (request.passenger_count - 1) * 0.5
        
        if request.luggage_count > 2:
            extras += (request.luggage_count - 2) * 0.25
        
        if 'discapacitado' in request.special_requirements.lower():
            extras += 2.0
        if 'mascotas' in request.special_requirements.lower():
            extras += 1.5
        
        return extras
    
    def calculate_final_fare(self, request: ServiceRequest, actual_distance_km: float,
                            actual_duration_min: int, waiting_time_min: int = 0) -> float:
        """Calcula tarifa final basada en valores reales"""
        fare = self.base_fare
        
        distance_for_calc = max(request.estimated_distance_km, actual_distance_km)
        fare += distance_for_calc * self.base_rate_per_km
        fare += waiting_time_min * self.waiting_rate_per_min
        fare *= request.surge_multiplier
        fare += self._calculate_extras(request)
        
        return max(fare, self.base_fare)
    
    def set_surge_multiplier(self, request_count: int, available_vehicles: int) -> float:
        """Calcula multiplicador dinamico basado en oferta/demanda"""
        if available_vehicles == 0:
            return 5.0
        
        ratio = request_count / available_vehicles
        
        if ratio <= 1.0:
            multiplier = 1.0
        elif ratio <= 2.0:
            multiplier = 1.25
        elif ratio <= 3.0:
            multiplier = 1.5
        elif ratio <= 4.0:
            multiplier = 2.0
        else:
            multiplier = 3.0
        
        return min(multiplier, 5.0)

class AssignmentEngine:
    """Motor de asignacion automatica de servicios a moviles"""
    
    def __init__(self):
        self.assignment_history = []
    
    def assign_automatic(self, request: ServiceRequest, 
                        available_vehicles: List[Dict]) -> Optional[str]:
        """Asigna servicio automaticamente al movil mas cercano disponible"""
        if not available_vehicles:
            logger.warning(f"[ASSIGN] Sin vehiculos disponibles para servicio {request.id}")
            return None
        
        best_vehicle = None
        min_distance = float('inf')
        
        for vehicle in available_vehicles:
            distance = self._haversine(
                request.origin_lat, request.origin_lon,
                vehicle.get('lat'), vehicle.get('lon')
            )
            
            if distance < min_distance:
                min_distance = distance
                best_vehicle = vehicle
        
        if best_vehicle:
            vehicle_id = best_vehicle.get('id')
            request.assigned_vehicle_id = vehicle_id
            request.status = ServiceStatus.ASIGNADO
            request.acceptance_time = None
            
            self.assignment_history.append({
                'service_id': request.id,
                'vehicle_id': vehicle_id,
                'assignment_type': 'AUTOMATICO',
                'distance_to_origin_km': min_distance,
                'timestamp': datetime.utcnow()
            })
            
            logger.info(f"[ASSIGN] Servicio {request.id} asignado a {vehicle_id} ({min_distance:.2f} km)")
            return vehicle_id
        
        return None
    
    def assign_manual(self, request: ServiceRequest, vehicle_id: str) -> bool:
        """Asigna servicio manualmente a un vehiculo especifico"""
        request.assigned_vehicle_id = vehicle_id
        request.status = ServiceStatus.ASIGNADO
        
        self.assignment_history.append({
            'service_id': request.id,
            'vehicle_id': vehicle_id,
            'assignment_type': 'MANUAL',
            'timestamp': datetime.utcnow()
        })
        
        logger.info(f"[ASSIGN] Servicio {request.id} asignado manualmente a {vehicle_id}")
        return True
    
    def create_offer(self, request: ServiceRequest, available_vehicles: List[Dict]) -> Dict:
        """Crea oferta para que conductores hagan oferta"""
        offer = {
            'service_id': request.id,
            'service_type': request.service_type,
            'origin': {'lat': request.origin_lat, 'lon': request.origin_lon},
            'destination': {'lat': request.destination_lat, 'lon': request.destination_lon},
            'estimated_distance_km': request.estimated_distance_km,
            'estimated_fare': request.total_fare,
            'surge_multiplier': request.surge_multiplier,
            'created_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(minutes=5),
            'nearby_vehicles': len(available_vehicles)
        }
        
        logger.info(f"[OFFER] Oferta creada para servicio {request.id} a {len(available_vehicles)} conductores")
        return offer
    
    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcula distancia en km entre dos puntos GPS"""
        try:
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371
            return c * r
        except:
            return float('inf')

class ServiceWorkflow:
    """Gestiona el workflow completo de un servicio"""
    
    def __init__(self):
        self.services: Dict[str, ServiceRequest] = {}
        self.gps_tracking: Dict[str, List[Dict]] = {}
    
    def create_service(self, request: ServiceRequest) -> Tuple[bool, str]:
        """Crea nuevo servicio si es valido"""
        valid, errors = ServiceValidator.validate_request(request)
        
        if not valid:
            error_msg = "; ".join(errors)
            logger.error(f"[SERVICE] Validacion fallida para {request.id}: {error_msg}")
            return False, error_msg
        
        self.services[request.id] = request
        self.gps_tracking[request.id] = []
        
        logger.info(f"[SERVICE] Servicio {request.id} creado - Estado: {request.status}")
        return True, "OK"
    
    def update_service_status(self, service_id: str, new_status: ServiceStatus) -> bool:
        """Actualiza estado del servicio con validaciones de transicion"""
        if service_id not in self.services:
            logger.error(f"[SERVICE] Servicio {service_id} no encontrado")
            return False
        
        service = self.services[service_id]
        old_status = service.status
        
        valid_transitions = {
            ServiceStatus.PENDIENTE: [ServiceStatus.ASIGNADO, ServiceStatus.CANCELADO],
            ServiceStatus.ASIGNADO: [ServiceStatus.EN_CAMINO, ServiceStatus.CANCELADO],
            ServiceStatus.EN_CAMINO: [ServiceStatus.EN_ORIGEN, ServiceStatus.CANCELADO],
            ServiceStatus.EN_ORIGEN: [ServiceStatus.EN_DESTINO, ServiceStatus.CANCELADO],
            ServiceStatus.EN_DESTINO: [ServiceStatus.FINALIZADO, ServiceStatus.CANCELADO],
            ServiceStatus.FINALIZADO: [],
            ServiceStatus.CANCELADO: []
        }
        
        if new_status not in valid_transitions.get(old_status, []):
            logger.error(f"[SERVICE] Transicion invalida: {old_status} -> {new_status}")
            return False
        
        service.status = new_status
        
        if new_status == ServiceStatus.EN_CAMINO:
            service.acceptance_time = datetime.utcnow()
        elif new_status == ServiceStatus.EN_ORIGEN:
            service.pickup_time = datetime.utcnow()
        elif new_status == ServiceStatus.FINALIZADO:
            service.dropoff_time = datetime.utcnow()
        
        logger.info(f"[SERVICE] Servicio {service_id} transicion: {old_status} -> {new_status}")
        return True
    
    def record_gps_point(self, service_id: str, lat: float, lon: float,
                        speed: float = 0.0) -> bool:
        """Registra punto GPS durante recorrido"""
        if service_id not in self.services:
            logger.error(f"[SERVICE] Servicio {service_id} no encontrado")
            return False
        
        gps_point = {
            'timestamp': datetime.utcnow(),
            'lat': lat,
            'lon': lon,
            'speed': speed
        }
        
        self.gps_tracking[service_id].append(gps_point)
        return True
    
    def calculate_detour_ratio(self, service_id: str, 
                              direct_distance_km: float) -> float:
        """Calcula detour ratio: distancia real GPS / distancia directa"""
        if service_id not in self.gps_tracking:
            logger.error(f"[SERVICE] Sin datos GPS para {service_id}")
            return 1.0
        
        gps_points = self.gps_tracking[service_id]
        if len(gps_points) < 2:
            return 1.0
        
        actual_distance = 0.0
        for i in range(len(gps_points) - 1):
            p1 = gps_points[i]
            p2 = gps_points[i + 1]
            
            dist = self._haversine(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
            actual_distance += dist
        
        if direct_distance_km <= 0:
            return 1.0
        
        detour_ratio = actual_distance / direct_distance_km
        
        service = self.services[service_id]
        service.actual_distance_km = actual_distance
        service.detour_ratio = detour_ratio
        
        logger.info(f"[SERVICE] Servicio {service_id} - Distancia real: {actual_distance:.2f} km, Detour: {detour_ratio:.2f}")
        
        return detour_ratio
    
    def get_service_summary(self, service_id: str) -> Optional[Dict]:
        """Retorna resumen completo del servicio"""
        if service_id not in self.services:
            return None
        
        service = self.services[service_id]
        
        return {
            'id': service.id,
            'client_id': service.client_id,
            'service_type': service.service_type,
            'status': service.status.value,
            'assigned_vehicle': service.assigned_vehicle_id,
            'origin': {'lat': service.origin_lat, 'lon': service.origin_lon, 'address': service.origin_address},
            'destination': {'lat': service.destination_lat, 'lon': service.destination_lon, 'address': service.destination_address},
            'passenger_count': service.passenger_count,
            'estimated_distance_km': service.estimated_distance_km,
            'actual_distance_km': service.actual_distance_km,
            'detour_ratio': service.detour_ratio,
            'estimated_fare': service.total_fare,
            'actual_fare': service.actual_fare,
            'base_fare': service.base_fare,
            'surge_multiplier': service.surge_multiplier,
            'created_at': service.created_at.isoformat(),
            'acceptance_time': service.acceptance_time.isoformat() if service.acceptance_time else None,
            'pickup_time': service.pickup_time.isoformat() if service.pickup_time else None,
            'dropoff_time': service.dropoff_time.isoformat() if service.dropoff_time else None,
            'gps_points_count': len(self.gps_tracking.get(service_id, []))
        }
    
    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcula distancia en km entre dos puntos GPS"""
        try:
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371
            return c * r
        except:
            return 0.0
