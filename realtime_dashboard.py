# realtime_dashboard.py - Dashboard en tiempo real con WebSockets simulados
from __future__ import annotations
from typing import Optional, Dict, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DashboardMetric:
    """Metrica del dashboard"""
    id: str
    name: str
    value: float
    unit: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    status: str = "OK"  # OK, WARNING, CRITICAL
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        self._update_status()
    
    def _update_status(self):
        """Actualiza estado basado en thresholds"""
        if self.threshold_critical and self.value >= self.threshold_critical:
            self.status = "CRITICAL"
        elif self.threshold_warning and self.value >= self.threshold_warning:
            self.status = "WARNING"
        else:
            self.status = "OK"

class RealtimeDashboardEngine:
    """Motor de dashboard en tiempo real"""
    
    def __init__(self):
        self.metrics: Dict[str, DashboardMetric] = {}
        self.subscribers: List[Callable] = []
        self.metric_history: List[Dict] = []
    
    def register_metric(self, metric: DashboardMetric):
        """Registra nueva metrica"""
        self.metrics[metric.id] = metric
        logger.info(f"[DASHBOARD] Metrica registrada: {metric.name}")
    
    def update_metric(self, metric_id: str, new_value: float):
        """Actualiza valor de metrica"""
        if metric_id not in self.metrics:
            logger.error(f"[DASHBOARD] Metrica {metric_id} no encontrada")
            return
        
        metric = self.metrics[metric_id]
        old_value = metric.value
        old_status = metric.status
        
        metric.value = new_value
        metric.updated_at = datetime.utcnow()
        metric._update_status()
        
        # Registrar cambio
        self.metric_history.append({
            'metric_id': metric_id,
            'metric_name': metric.name,
            'old_value': old_value,
            'new_value': new_value,
            'old_status': old_status,
            'new_status': metric.status,
            'timestamp': datetime.utcnow()
        })
        
        # Notificar suscriptores si cambio de estado
        if old_status != metric.status:
            self._notify_subscribers(metric)
    
    def subscribe(self, callback: Callable):
        """Suscribe callback para cambios de metricas"""
        self.subscribers.append(callback)
        logger.info(f"[DASHBOARD] Nuevo suscriptor registrado")
    
    def _notify_subscribers(self, metric: DashboardMetric):
        """Notifica a suscriptores cambio de metrica"""
        for callback in self.subscribers:
            try:
                callback(metric)
            except Exception as e:
                logger.error(f"[DASHBOARD] Error notificando suscriptor: {e}")
    
    def get_dashboard_snapshot(self) -> Dict:
        """Retorna snapshot actual del dashboard"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {
                mid: {
                    'name': m.name,
                    'value': m.value,
                    'unit': m.unit,
                    'status': m.status,
                    'updated_at': m.updated_at.isoformat()
                }
                for mid, m in self.metrics.items()
            },
            'summary': {
                'total_metrics': len(self.metrics),
                'critical_count': sum(1 for m in self.metrics.values() if m.status == 'CRITICAL'),
                'warning_count': sum(1 for m in self.metrics.values() if m.status == 'WARNING'),
                'ok_count': sum(1 for m in self.metrics.values() if m.status == 'OK')
            }
        }
    
    def get_metric_trend(self, metric_id: str, hours: int = 24) -> List[Dict]:
        """Obtiene tendencia de metrica en ultimas N horas"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            h for h in self.metric_history
            if h['metric_id'] == metric_id and h['timestamp'] > cutoff
        ]

class PerformanceMonitor:
    """Monitorea performance del sistema"""
    
    def __init__(self):
        self.performance_metrics = []
    
    def log_performance(self, operation: str, duration_ms: float,
                       success: bool, details: Dict = None):
        """Registra metrica de performance"""
        entry = {
            'timestamp': datetime.utcnow(),
            'operation': operation,
            'duration_ms': duration_ms,
            'success': success,
            'details': details or {}
        }
        self.performance_metrics.append(entry)
    
    def get_performance_report(self, date_from: datetime, date_to: datetime) -> Dict:
        """Reporte de performance en periodo"""
        
        metrics = [
            m for m in self.performance_metrics
            if date_from <= m['timestamp'] <= date_to
        ]
        
        if not metrics:
            return {}
        
        # Agrupar por operacion
        by_operation = {}
        for m in metrics:
            op = m['operation']
            if op not in by_operation:
                by_operation[op] = []
            by_operation[op].append(m)
        
        report = {
            'period_start': date_from.date(),
            'period_end': date_to.date(),
            'total_operations': len(metrics),
            'operations': {}
        }
        
        for op, op_metrics in by_operation.items():
            durations = [m['duration_ms'] for m in op_metrics]
            successes = sum(1 for m in op_metrics if m['success'])
            
            report['operations'][op] = {
                'count': len(op_metrics),
                'avg_duration_ms': sum(durations) / len(durations),
                'min_duration_ms': min(durations),
                'max_duration_ms': max(durations),
                'success_rate': successes / len(op_metrics) * 100
            }
        
        return report
