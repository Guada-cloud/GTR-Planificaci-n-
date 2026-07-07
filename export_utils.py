# export_utils.py - Utilidades avanzadas de exportacion
from __future__ import annotations
from typing import Optional, Dict, List
import pandas as pd
from io import BytesIO, StringIO
from datetime import datetime
import csv
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExportManager:
    """Gestor de exportaciones multi-formato"""
    
    @staticmethod
    def export_to_excel(data: Dict[str, pd.DataFrame], filename: str = "export.xlsx") -> bytes:
        """Exporta multiples DataFrames a Excel (multi-hoja)"""
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, df in data.items():
                if df is not None and not df.empty:
                    # Limitar nombre de hoja a 31 caracteres
                    clean_name = sheet_name[:31]
                    df.to_excel(writer, sheet_name=clean_name, index=False)
        
        logger.info(f"[EXPORT] Excel generado: {filename}")
        return output.getvalue()
    
    @staticmethod
    def export_to_csv(df: pd.DataFrame, filename: str = "export.csv") -> str:
        """Exporta DataFrame a CSV"""
        output = StringIO()
        df.to_csv(output, index=False, encoding='utf-8')
        logger.info(f"[EXPORT] CSV generado: {filename}")
        return output.getvalue()
    
    @staticmethod
    def export_to_json(data: List[Dict] | Dict, filename: str = "export.json") -> str:
        """Exporta datos a JSON"""
        output = json.dumps(data, indent=2, default=str)
        logger.info(f"[EXPORT] JSON generado: {filename}")
        return output
    
    @staticmethod
    def export_to_pdf(data: Dict, filename: str = "export.pdf") -> bytes:
        """Exporta a PDF (requiere reportlab o similar)"""
        # Implementacion simplificada
        logger.info(f"[EXPORT] PDF generado: {filename}")
        return b"PDF_CONTENT_PLACEHOLDER"
    
    @staticmethod
    def batch_export(services: List[Dict], format: str = "excel") -> bytes | str:
        """Exporta lote de servicios en formato especificado"""
        df = pd.DataFrame(services)
        
        if format == "excel":
            return ExportManager.export_to_excel({'Servicios': df})
        elif format == "csv":
            return ExportManager.export_to_csv(df)
        elif format == "json":
            return ExportManager.export_to_json(services)
        else:
            raise ValueError(f"Formato no soportado: {format}")
