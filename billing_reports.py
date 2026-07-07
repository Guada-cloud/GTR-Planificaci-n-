# billing_reports.py - Facturacion a clientes y generacion de reportes
from __future__ import annotations
from typing import Optional, Dict, List
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClientBill:
    """Factura a cliente"""
    id: str
    client_id: str
    period_start: datetime
    period_end: datetime
    total_services: int
    subtotal: float
    iva_pct: float
    iva_amount: float
    total_amount: float
    status: str
    issue_date: datetime
    due_date: datetime
    payment_date: Optional[datetime] = None
    payment_method: str = ""
    reference_number: str = ""
    notes: str = ""

class BillingEngine:
    """Motor de facturacion a clientes"""
    
    def __init__(self, default_iva_pct: float = 21.0):
        self.default_iva_pct = default_iva_pct
        self.bills: Dict[str, ClientBill] = {}
    
    def generate_client_bill(self, client_id: str, services: List[Dict],
                            period_start: datetime, period_end: datetime,
                            payment_terms_days: int = 30) -> ClientBill:
        """Genera factura para cliente por servicios en periodo"""
        
        subtotal = sum(s['actual_fare'] for s in services)
        
        extras = 0.0
        for service in services:
            if 'extras' in service:
                extras += service['extras']
        
        subtotal += extras
        
        iva_amount = subtotal * (self.default_iva_pct / 100.0)
        total_amount = subtotal + iva_amount
        
        bill = ClientBill(
            id=f"BILL-{client_id}-{period_start.strftime('%Y%m%d')}",
            client_id=client_id,
            period_start=period_start,
            period_end=period_end,
            total_services=len(services),
            subtotal=subtotal,
            iva_pct=self.default_iva_pct,
            iva_amount=iva_amount,
            total_amount=total_amount,
            status="DRAFT",
            issue_date=datetime.utcnow(),
            due_date=datetime.utcnow() + timedelta(days=payment_terms_days)
        )
        
        self.bills[bill.id] = bill
        
        logger.info(f"[BILLING] Factura {bill.id} generada - Monto: {total_amount:.2f}")
        
        return bill
    
    def issue_bill(self, bill_id: str) -> bool:
        """Emite factura (DRAFT -> ISSUED)"""
        if bill_id not in self.bills:
            return False
        
        bill = self.bills[bill_id]
        bill.status = "ISSUED"
        bill.reference_number = f"REF-{datetime.utcnow().timestamp()}"
        
        logger.info(f"[BILLING] Factura {bill_id} emitida")
        return True
    
    def record_payment(self, bill_id: str, amount: float,
                      payment_method: str, payment_date: Optional[datetime] = None) -> bool:
        """Registra pago de factura"""
        if bill_id not in self.bills:
            return False
        
        bill = self.bills[bill_id]
        
        if amount < bill.total_amount:
            logger.warning(f"[BILLING] Pago parcial para {bill_id}")
        
        bill.status = "PAID" if amount >= bill.total_amount else "PARTIALLY_PAID"
        bill.payment_date = payment_date or datetime.utcnow()
        bill.payment_method = payment_method
        
        logger.info(f"[BILLING] Pago registrado para {bill_id} - Monto: {amount:.2f}")
        return True
    
    def get_overdue_bills(self, as_of_date: Optional[datetime] = None) -> List[ClientBill]:
        """Retorna facturas vencidas"""
        as_of_date = as_of_date or datetime.utcnow()
        
        return [
            bill for bill in self.bills.values()
            if bill.status == "ISSUED" and bill.due_date < as_of_date
        ]
    
    def export_bill_to_excel(self, bill_id: str) -> Optional[bytes]:
        """Exporta factura a Excel"""
        if bill_id not in self.bills:
            return None
        
        bill = self.bills[bill_id]
        
        data = {
            'Concepto': ['Subtotal', f'IVA ({bill.iva_pct}%)', 'TOTAL'],
            'Monto': [bill.subtotal, bill.iva_amount, bill.total_amount]
        }
        
        df = pd.DataFrame(data)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Factura', index=False)
        
        return output.getvalue()

class ProductivityReporter:
    """Genera reportes de productividad por movil"""
    
    def generate_driver_productivity(self, driver_id: str, services: List[Dict],
                                   date_from: datetime, date_to: datetime) -> Dict:
        """Genera reporte de productividad para conductor"""
        
        if not services:
            return {
                'driver_id': driver_id,
                'period': f"{date_from.date()} a {date_to.date()}",
                'total_services': 0,
                'total_earnings': 0.0,
                'avg_fare': 0.0,
                'total_km': 0.0
            }
        
        df = pd.DataFrame(services)
        
        report = {
            'driver_id': driver_id,
            'period_start': date_from.date(),
            'period_end': date_to.date(),
            'total_services': len(df),
            'total_earnings': float(df['actual_fare'].sum()),
            'avg_fare': float(df['actual_fare'].mean()),
            'min_fare': float(df['actual_fare'].min()),
            'max_fare': float(df['actual_fare'].max()),
            'total_km': float(df['actual_distance_km'].sum()),
            'avg_km_per_service': float(df['actual_distance_km'].mean()),
            'on_time_percentage': float((df['delay_min'] <= 0).sum() / len(df) * 100),
            'avg_detour_ratio': float(df['detour_ratio'].mean()),
            'daily_breakdown': df.groupby(df.get('service_date', pd.Series()).dt.date).agg({
                'actual_fare': 'sum',
                'id': 'count',
                'actual_distance_km': 'sum'
            }).to_dict()
        }
        
        return report
    
    def generate_fleet_productivity(self, services: List[Dict],
                                  date_from: datetime, date_to: datetime) -> Dict:
        """Genera reporte agregado de productividad de flota"""
        
        if not services:
            return {'total_services': 0}
        
        df = pd.DataFrame(services)
        
        report = {
            'period_start': date_from.date(),
            'period_end': date_to.date(),
            'total_services': len(df),
            'total_earnings': float(df['actual_fare'].sum()),
            'avg_fare_per_service': float(df['actual_fare'].mean()),
            'total_km': float(df['actual_distance_km'].sum()),
            'total_drivers': df['assigned_vehicle_id'].nunique(),
            'services_per_driver': float(len(df) / df['assigned_vehicle_id'].nunique()),
            'on_time_percentage': float((df.get('delay_min', pd.Series()) <= 0).sum() / len(df) * 100) if 'delay_min' in df.columns else 0,
            'avg_detour_ratio': float(df['detour_ratio'].mean()),
            'top_earner': df.groupby('assigned_vehicle_id')['actual_fare'].sum().idxmax() if len(df) > 0 else None
        }
        
        return report
    
    def generate_deviation_report(self, reconciliation_records: List[Dict],
                                 date_from: datetime, date_to: datetime) -> pd.DataFrame:
        """Reporte de desvios detectados"""
        
        filtered = [
            r for r in reconciliation_records
            if date_from <= r['service_date'] <= date_to
        ]
        
        if not filtered:
            return pd.DataFrame()
        
        df = pd.DataFrame(filtered)
        
        deviation_summary = df.groupby('alert_level').agg({
            'service_id': 'count',
            'detour_ratio': 'mean',
            'distance_variance_pct': 'mean',
            'fare_adjustment': 'sum'
        }).rename(columns={'service_id': 'count'})
        
        logger.info(f"[REPORTS] Reporte de desvios generado - {len(filtered)} registros")
        
        return df.sort_values('detour_ratio', ascending=False)
    
    def generate_profitability_report(self, services: List[Dict],
                                     payments: List[Dict],
                                     date_from: datetime, date_to: datetime) -> Dict:
        """Reporte de rentabilidad"""
        
        services_df = pd.DataFrame(services)
        payments_df = pd.DataFrame(payments)
        
        total_revenue = services_df['actual_fare'].sum()
        total_driver_payments = payments_df['driver_amount'].sum()
        platform_revenue = payments_df['platform_fee'].sum()
        margin = total_revenue - total_driver_payments
        margin_pct = (margin / total_revenue * 100) if total_revenue > 0 else 0
        
        report = {
            'period_start': date_from.date(),
            'period_end': date_to.date(),
            'total_revenue': float(total_revenue),
            'driver_payments': float(total_driver_payments),
            'platform_commission': float(platform_revenue),
            'gross_margin': float(margin),
            'margin_percentage': float(margin_pct),
            'total_services': len(services_df),
            'revenue_per_service': float(total_revenue / len(services_df)) if len(services_df) > 0 else 0,
            'avg_km': float(services_df['actual_distance_km'].mean()) if 'actual_distance_km' in services_df.columns else 0,
            'revenue_per_km': float(total_revenue / services_df['actual_distance_km'].sum()) if 'actual_distance_km' in services_df.columns and services_df['actual_distance_km'].sum() > 0 else 0
        }
        
        logger.info(f"[REPORTS] Reporte de rentabilidad: Margen {margin_pct:.1f}%")
        
        return report
