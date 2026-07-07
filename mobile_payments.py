# mobile_payments.py - Gestion de pagos a conductores y liquidaciones
from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaymentStatus(str, Enum):
    """Estados de pago a conductores"""
    PENDIENTE = "PENDIENTE"
    PROCESANDO = "PROCESANDO"
    PAGADO = "PAGADO"
    RECHAZADO = "RECHAZADO"
    REVERTIDO = "REVERTIDO"

@dataclass
class DriverPayment:
    """Pago a un conductor por servicio"""
    id: str
    driver_id: str
    service_id: str
    service_fare: float
    driver_commission_pct: float
    driver_amount: float
    platform_fee: float
    taxes: float
    status: PaymentStatus = PaymentStatus.PENDIENTE
    payment_method: str = "BANK_TRANSFER"
    created_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    reference_number: str = ""
    notes: str = ""

@dataclass
class DriverLiquidation:
    """Liquidacion periodica para un conductor"""
    id: str
    driver_id: str
    period_start: datetime
    period_end: datetime
    total_services: int
    total_earnings: float
    total_commission: float
    total_fees: float
    net_payment: float
    status: PaymentStatus = PaymentStatus.PENDIENTE
    payment_date: Optional[datetime] = None
    payment_method: str = "BANK_TRANSFER"
    created_at: datetime = field(default_factory=datetime.utcnow)

class DriverPaymentEngine:
    """Motor de gestion de pagos a conductores"""
    
    def __init__(self, platform_commission_pct: float = 25.0):
        self.platform_commission_pct = platform_commission_pct
        self.driver_commission_pct = 100 - platform_commission_pct
        self.payments: Dict[str, DriverPayment] = {}
        self.liquidations: Dict[str, DriverLiquidation] = {}
    
    def calculate_driver_earnings(self, service_fare: float) -> Tuple[float, float]:
        """Calcula cuanto recibe el conductor y cuanto la plataforma
        Retorna: (monto_conductor, monto_plataforma)
        """
        driver_amount = service_fare * (self.driver_commission_pct / 100.0)
        platform_amount = service_fare * (self.platform_commission_pct / 100.0)
        return driver_amount, platform_amount
    
    def create_payment(self, driver_id: str, service_id: str, service_fare: float,
                      payment_method: str = "BANK_TRANSFER") -> DriverPayment:
        """Crea registro de pago para un servicio completado"""
        
        driver_amount, platform_fee = self.calculate_driver_earnings(service_fare)
        taxes = driver_amount * 0.05
        net_driver_amount = driver_amount - taxes
        
        payment = DriverPayment(
            id=f"PAY-{driver_id}-{service_id}-{datetime.utcnow().timestamp()}",
            driver_id=driver_id,
            service_id=service_id,
            service_fare=service_fare,
            driver_commission_pct=self.driver_commission_pct,
            driver_amount=net_driver_amount,
            platform_fee=platform_fee,
            taxes=taxes,
            payment_method=payment_method
        )
        
        self.payments[payment.id] = payment
        
        logger.info(f"[PAYMENT] Pago creado: {driver_id} - Servicio {service_id} - Monto: {net_driver_amount:.2f}")
        
        return payment
    
    def process_payment(self, payment_id: str, reference_number: str = "") -> bool:
        """Procesa un pago (marca como pagado)"""
        if payment_id not in self.payments:
            logger.error(f"[PAYMENT] Pago {payment_id} no encontrado")
            return False
        
        payment = self.payments[payment_id]
        payment.status = PaymentStatus.PAGADO
        payment.processed_at = datetime.utcnow()
        payment.reference_number = reference_number or f"REF-{datetime.utcnow().timestamp()}"
        
        logger.info(f"[PAYMENT] Pago {payment_id} procesado - Ref: {payment.reference_number}")
        
        return True
    
    def create_liquidation(self, driver_id: str, period_start: datetime,
                          period_end: datetime) -> Optional[DriverLiquidation]:
        """Crea liquidacion periodica para un conductor"""
        
        period_payments = [
            p for p in self.payments.values()
            if p.driver_id == driver_id and 
               period_start <= p.created_at <= period_end and
               p.status == PaymentStatus.PAGADO
        ]
        
        if not period_payments:
            logger.warning(f"[LIQUID] Sin pagos para {driver_id} en periodo")
            return None
        
        total_services = len(period_payments)
        total_earnings = sum(p.driver_amount for p in period_payments)
        total_commission = sum(p.driver_commission_pct / 100 * p.service_fare for p in period_payments)
        total_fees = sum(p.platform_fee for p in period_payments)
        net_payment = total_earnings - sum(p.taxes for p in period_payments)
        
        liquidation = DriverLiquidation(
            id=f"LIQ-{driver_id}-{period_start.strftime('%Y%m%d')}-{period_end.strftime('%Y%m%d')}",
            driver_id=driver_id,
            period_start=period_start,
            period_end=period_end,
            total_services=total_services,
            total_earnings=total_earnings,
            total_commission=total_commission,
            total_fees=total_fees,
            net_payment=net_payment
        )
        
        self.liquidations[liquidation.id] = liquidation
        
        logger.info(f"[LIQUID] Liquidacion creada: {driver_id} - Monto: {net_payment:.2f} - Servicios: {total_services}")
        
        return liquidation
    
    def process_liquidation(self, liquidation_id: str, payment_date: datetime,
                           payment_method: str = "BANK_TRANSFER") -> bool:
        """Procesa liquidacion (marca como pagada)"""
        if liquidation_id not in self.liquidations:
            logger.error(f"[LIQUID] Liquidacion {liquidation_id} no encontrada")
            return False
        
        liquidation = self.liquidations[liquidation_id]
        liquidation.status = PaymentStatus.PAGADO
        liquidation.payment_date = payment_date
        liquidation.payment_method = payment_method
        
        logger.info(f"[LIQUID] Liquidacion {liquidation_id} pagada el {payment_date.date()}")
        
        return True
    
    def get_driver_balance(self, driver_id: str, up_to_date: Optional[datetime] = None) -> float:
        """Obtiene saldo pendiente de pago para un conductor"""
        up_to_date = up_to_date or datetime.utcnow()
        
        payments = [
            p for p in self.payments.values()
            if p.driver_id == driver_id and
               p.created_at <= up_to_date and
               p.status == PaymentStatus.PENDIENTE
        ]
        
        balance = sum(p.driver_amount for p in payments)
        
        return balance
    
    def get_driver_earnings_report(self, driver_id: str, date_from: datetime,
                                  date_to: datetime) -> Dict:
        """Genera reporte de ingresos para un conductor"""
        
        payments = [
            p for p in self.payments.values()
            if p.driver_id == driver_id and
               date_from <= p.created_at <= date_to
        ]
        
        if not payments:
            return {'total_services': 0, 'total_earnings': 0.0}
        
        df_payments = pd.DataFrame([{
            'service_fare': p.service_fare,
            'driver_amount': p.driver_amount,
            'status': p.status.value,
            'date': p.created_at.date()
        } for p in payments])
        
        return {
            'driver_id': driver_id,
            'period_start': date_from.date(),
            'period_end': date_to.date(),
            'total_services': len(payments),
            'total_fares': float(df_payments['service_fare'].sum()),
            'total_earnings': float(df_payments['driver_amount'].sum()),
            'avg_fare': float(df_payments['service_fare'].mean()),
            'avg_earnings_per_service': float(df_payments['driver_amount'].mean()),
            'paid_services': (df_payments['status'] == 'PAGADO').sum(),
            'pending_services': (df_payments['status'] == 'PENDIENTE').sum(),
            'daily_breakdown': df_payments.groupby('date')['driver_amount'].sum().to_dict()
        }
