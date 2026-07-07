# advanced_reporting.py - Reportes avanzados y analisis de datos
from __future__ import annotations
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedReporting:
    """Motor de reportes avanzados"""
    
    def generate_executive_summary(self, services: List[Dict], payments: List[Dict],
                                  alerts: List[Dict], date_from: datetime,
                                  date_to: datetime) -> Dict:
        """Resumen ejecutivo de operaciones"""
        
        services_df = pd.DataFrame(services)
        payments_df = pd.DataFrame(payments)
        alerts_df = pd.DataFrame(alerts)
        
        # Servicios
        total_services = len(services_df)
        total_revenue = services_df['actual_fare'].sum() if not services_df.empty else 0
        avg_fare = services_df['actual_fare'].mean() if not services_df.empty else 0
        
        # Calidad
        on_time = (services_df['delay_min'] <= 0).sum() if 'delay_min' in services_df.columns else 0
        on_time_pct = (on_time / total_services * 100) if total_services > 0 else 0
        
        # Alertas
        critical_alerts = (alerts_df['severity'] == 'critical').sum() if not alerts_df.empty else 0
        warning_alerts = (alerts_df['severity'] == 'warning').sum() if not alerts_df.empty else 0
        
        # Rentabilidad
        driver_costs = payments_df['driver_amount'].sum() if not payments_df.empty else 0
        margin = total_revenue - driver_costs
        margin_pct = (margin / total_revenue * 100) if total_revenue > 0 else 0
        
        return {
            'period': {
                'start': date_from.date(),
                'end': date_to.date()
            },
            'operations': {
                'total_services': int(total_services),
                'total_revenue': float(total_revenue),
                'average_fare': float(avg_fare),
                'on_time_rate': float(on_time_pct)
            },
            'quality': {
                'on_time_services': int(on_time),
                'delayed_services': int(total_services - on_time)
            },
            'alerts': {
                'critical': int(critical_alerts),
                'warnings': int(warning_alerts)
            },
            'profitability': {
                'total_revenue': float(total_revenue),
                'driver_costs': float(driver_costs),
                'gross_margin': float(margin),
                'margin_percentage': float(margin_pct)
            }
        }
    
    def generate_driver_leaderboard(self, services: List[Dict]) -> List[Dict]:
        """Ranking de conductores por productividad"""
        
        df = pd.DataFrame(services)
        
        if 'assigned_vehicle_id' not in df.columns:
            return []
        
        leaderboard = df.groupby('assigned_vehicle_id').agg({
            'actual_fare': 'sum',
            'id': 'count',
            'actual_distance_km': 'sum'
        }).rename(columns={'id': 'services'})
        
        leaderboard['avg_fare'] = leaderboard['actual_fare'] / leaderboard['services']
        leaderboard['avg_km_per_service'] = leaderboard['actual_distance_km'] / leaderboard['services']
        leaderboard = leaderboard.sort_values('actual_fare', ascending=False)
        
        result = []
        for idx, (driver_id, row) in enumerate(leaderboard.iterrows(), 1):
            result.append({
                'rank': idx,
                'driver_id': driver_id,
                'total_revenue': float(row['actual_fare']),
                'services_completed': int(row['services']),
                'avg_fare': float(row['avg_fare']),
                'total_km': float(row['actual_distance_km']),
                'avg_km_per_service': float(row['avg_km_per_service'])
            })
        
        return result
    
    def generate_trend_analysis(self, metrics: List[Dict], metric_name: str,
                              days: int = 30) -> Dict:
        """Analiza tendencia de una metrica"""
        
        df = pd.DataFrame(metrics)
        df['date'] = pd.to_datetime(df.get('timestamp', df.get('created_at')))
        df = df[df['date'] >= datetime.utcnow() - timedelta(days=days)]
        
        if metric_name not in df.columns:
            return {}
        
        daily = df.groupby(df['date'].dt.date)[metric_name].agg(['mean', 'min', 'max', 'sum'])
        
        # Calcular tendencia (regresion lineal simple)
        x = np.arange(len(daily))
        y = daily['mean'].values
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            trend = 'UP' if slope > 0 else 'DOWN' if slope < 0 else 'STABLE'
        else:
            slope = 0
            trend = 'STABLE'
        
        return {
            'metric': metric_name,
            'period_days': days,
            'data_points': len(daily),
            'current_value': float(daily['mean'].iloc[-1]) if len(daily) > 0 else 0,
            'avg_value': float(daily['mean'].mean()),
            'min_value': float(daily['min'].min()),
            'max_value': float(daily['max'].max()),
            'trend': trend,
            'slope': float(slope)
        }
    
    def generate_anomaly_report(self, data: List[Dict], metric_col: str,
                              std_devs: float = 2.0) -> Dict:
        """Detecta anomalias usando desviacion estandar"""
        
        df = pd.DataFrame(data)
        
        if metric_col not in df.columns:
            return {'error': f'Columna {metric_col} no existe'}
        
        values = df[metric_col].astype(float)
        mean = values.mean()
        std = values.std()
        
        upper_bound = mean + (std_devs * std)
        lower_bound = mean - (std_devs * std)
        
        anomalies = df[
            (df[metric_col] > upper_bound) | (df[metric_col] < lower_bound)
        ].copy()
        
        return {
            'metric': metric_col,
            'total_records': len(df),
            'anomalies_detected': len(anomalies),
            'anomaly_percentage': (len(anomalies) / len(df) * 100) if len(df) > 0 else 0,
            'mean': float(mean),
            'std_dev': float(std),
            'upper_bound': float(upper_bound),
            'lower_bound': float(lower_bound),
            'anomalies': anomalies.to_dict('records')[:10]  # Top 10
        }
    
    def generate_comparison_report(self, current_period: List[Dict],
                                  previous_period: List[Dict]) -> Dict:
        """Compara metricas entre periodos"""
        
        current_df = pd.DataFrame(current_period)
        previous_df = pd.DataFrame(previous_period)
        
        metrics_to_compare = [
            ('actual_fare', 'Revenue'),
            ('actual_distance_km', 'Distance'),
            ('delay_min', 'Avg Delay')
        ]
        
        comparison = {}
        
        for metric, label in metrics_to_compare:
            if metric in current_df.columns and metric in previous_df.columns:
                current_value = current_df[metric].sum()
                previous_value = previous_df[metric].sum()
                
                change = current_value - previous_value
                change_pct = (change / previous_value * 100) if previous_value != 0 else 0
                
                comparison[label] = {
                    'current': float(current_value),
                    'previous': float(previous_value),
                    'change': float(change),
                    'change_percentage': float(change_pct),
                    'trend': 'UP' if change > 0 else 'DOWN' if change < 0 else 'STABLE'
                }
        
        return comparison
