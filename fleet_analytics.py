# fleet_analytics.py — Análisis avanzados: desvíos, coeficientes operativos, impacto horario
from __future__ import annotations
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FleetAnalytics:
    """Análisis hora a hora, impacto de faltantes, coeficientes operativos"""
    
    @staticmethod
    def calculate_operational_coefficient(actual_services: int, projected_services: int, 
                                         actual_vehicles: int, projected_vehicles: int) -> float:
        """Coeficiente operativo = Efectividad de servicios × Disponibilidad de flota"""
        if projected_services == 0 or projected_vehicles == 0:
            return 0.0
        
        service_efficiency = min(actual_services / projected_services, 1.0) if projected_services > 0 else 0.0
        fleet_availability = min(actual_vehicles / projected_vehicles, 1.0) if projected_vehicles > 0 else 0.0
        
        coef = (service_efficiency * 0.7) + (fleet_availability * 0.3)
        return float(coef)
    
    @staticmethod
    def calculate_hourly_impact(df: pd.DataFrame) -> pd.DataFrame:
        """Analiza impacto hora a hora (servicios no ejecutados, retrasos acumulados)"""
        if df.empty:
            return pd.DataFrame()
        
        result = df.groupby('HoraStr', as_index=False).agg({
            'Servicios_Planificados': 'sum',
            'Servicios_Reales': 'sum',
            'Moviles_Planificados': 'sum',
            'Moviles_Reales': 'sum',
            'delay_min': ['mean', 'max', 'sum']
        }).fillna(0)
        
        result.columns = ['HoraStr', 'plan_srv', 'real_srv', 'plan_mov', 'real_mov', 
                         'avg_delay_min', 'max_delay_min', 'total_delay_min']
        
        result['servicios_no_ejecutados'] = result['plan_srv'] - result['real_srv']
        result['impacto_retrasos_pct'] = (result['total_delay_min'] / (result['real_srv'] * 30)) * 100 if result['real_srv'].sum() > 0 else 0
        
        return result
    
    @staticmethod
    def scenario_impact_missing_vehicles(df: pd.DataFrame, missing_vehicles: int, 
                                        avg_services_per_vehicle: float = 8.0) -> Dict:
        """Proyecta impacto de X vehículos faltantes"""
        total_services = df['Servicios_Reales'].sum()
        total_vehicles = df['Moviles_Reales'].sum()
        
        projected_lost_services = missing_vehicles * avg_services_per_vehicle
        new_total_services = max(total_services - projected_lost_services, 0)
        
        service_loss_pct = (projected_lost_services / total_services * 100) if total_services > 0 else 0
        new_operational_coef = FleetAnalytics.calculate_operational_coefficient(
            int(new_total_services), int(df['Servicios_Planificados'].sum()),
            int(total_vehicles - missing_vehicles), int(df['Moviles_Planificados'].sum())
        )
        
        return {
            'missing_vehicles': missing_vehicles,
            'current_services': int(total_services),
            'projected_lost_services': int(projected_lost_services),
            'new_total_services': int(new_total_services),
            'service_loss_pct': float(service_loss_pct),
            'new_operational_coefficient': float(new_operational_coef),
            'revenue_impact': float(projected_lost_services * 100)  # Asume costo por servicio
        }
    
    @staticmethod
    def analyze_period(df: pd.DataFrame, period_type: str = 'day') -> Dict:
        """Análisis agregado por período (día, semana, mes)"""
        if df.empty:
            return {}
        
        total_services = df['Servicios_Reales'].sum()
        planned_services = df['Servicios_Planificados'].sum()
        on_time = ((df['delay_min'] <= 0).sum() if 'delay_min' in df.columns else 0)
        delayed = df.shape[0] - on_time if 'delay_min' in df.columns else 0
        
        return {
            'period_type': period_type,
            'total_services_executed': int(total_services),
            'total_services_planned': int(planned_services),
            'service_fulfillment_pct': (total_services / planned_services * 100) if planned_services > 0 else 0,
            'on_time_count': int(on_time),
            'delayed_count': int(delayed),
            'on_time_pct': (on_time / df.shape[0] * 100) if df.shape[0] > 0 else 0,
            'avg_delay_min': float(df['delay_min'].mean()) if 'delay_min' in df.columns else 0,
            'max_delay_min': float(df['delay_min'].max()) if 'delay_min' in df.columns else 0,
            'fleet_utilization_pct': (df['Moviles_Reales'].sum() / df['Moviles_Planificados'].sum() * 100) if df['Moviles_Planificados'].sum() > 0 else 0
        }

class KPICalculator:
    """Calcula KPIs operacionales agregados"""
    
    @staticmethod
    def compute_dashboard_kpis(merged_df: pd.DataFrame) -> Dict[str, float]:
        """KPIs principales para dashboard"""
        if merged_df.empty:
            return {}
        
        total_planned = merged_df['Servicios_Planificados'].sum()
        total_actual = merged_df['Servicios_Reales'].sum()
        
        kpis = {
            'effectiveness': float(1 - abs(total_actual - total_planned) / total_planned) if total_planned > 0 else 0,
            'on_time_rate': 0.0,
            'fleet_utilization': float(merged_df['Moviles_Reales'].sum() / merged_df['Moviles_Planificados'].sum()) if merged_df['Moviles_Planificados'].sum() > 0 else 0,
            'avg_delay_minutes': float(merged_df.get('delay_min', pd.Series([0])).mean()),
            'service_deviation_pct': float((total_actual - total_planned) / total_planned * 100) if total_planned > 0 else 0,
            'operational_coefficient': 0.0
        }
        
        # On-time rate
        if 'delay_min' in merged_df.columns:
            on_time = (merged_df['delay_min'] <= 0).sum()
            kpis['on_time_rate'] = float(on_time / len(merged_df)) if len(merged_df) > 0 else 0
        
        # Operational coefficient
        kpis['operational_coefficient'] = FleetAnalytics.calculate_operational_coefficient(
            int(total_actual), int(total_planned),
            int(merged_df['Moviles_Reales'].sum()), int(merged_df['Moviles_Planificados'].sum())
        )
        
        return kpis
    
    @staticmethod
    def compare_periods(current_df: pd.DataFrame, previous_df: pd.DataFrame) -> Dict:
        """Compara KPIs entre períodos"""
        current_kpis = KPICalculator.compute_dashboard_kpis(current_df)
        previous_kpis = KPICalculator.compute_dashboard_kpis(previous_df)
        
        comparison = {}
        for key in current_kpis:
            current = current_kpis[key]
            previous = previous_kpis.get(key, 0)
            change = ((current - previous) / previous * 100) if previous != 0 else 0
            comparison[key] = {
                'current': current,
                'previous': previous,
                'change_pct': float(change),
                'trend': 'up' if change > 0 else 'down' if change < 0 else 'stable'
            }
        
        return comparison
