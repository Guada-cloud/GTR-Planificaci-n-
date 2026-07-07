# fleet_ml_engine.py — ML models for forecasting, delay prediction, anomaly detection
from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump, load
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

class FleetForecastModel:
    """Predice demanda de servicios y disponibilidad de flota por hora"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or MODEL_DIR / "forecast_model.pkl"
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = [
            'hour', 'day_of_week', 'is_holiday', 'temp_celsius', 
            'historical_avg', 'historical_std'
        ]
        if self.model_path.exists():
            self.load()
        else:
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara features para forecasting"""
        d = df.copy()
        if 'timestamp' in d.columns:
            d['timestamp'] = pd.to_datetime(d['timestamp'])
            d['hour'] = d['timestamp'].dt.hour
            d['day_of_week'] = d['timestamp'].dt.dayofweek
            d['is_holiday'] = 0  # TODO: integrar calendario de feriados
        
        # Features de histórico (ventana móvil)
        for col in ['servicios_reales', 'moviles_reales']:
            if col in d.columns:
                d[f'{col}_7d_avg'] = d[col].rolling(window=7, min_periods=1).mean()
                d[f'{col}_7d_std'] = d[col].rolling(window=7, min_periods=1).std().fillna(0)
        
        return d[self.feature_names]
    
    def train(self, df: pd.DataFrame, target_col: str = 'servicios_reales') -> Dict[str, float]:
        """Entrena modelo con datos históricos"""
        if df.empty or target_col not in df.columns:
            logger.warning(f"No data for training {target_col}")
            return {}
        
        try:
            X = self.prepare_features(df)
            y = df[target_col].fillna(0)
            
            # Remover filas con NaN
            mask = X.notna().all(axis=1) & y.notna()
            X, y = X[mask], y[mask]
            
            if len(X) < 20:
                logger.warning("Insufficient data for training")
                return {}
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model.fit(X_train_scaled, y_train)
            
            y_pred = self.model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Model trained: MAE={mae:.2f}, R²={r2:.3f}")
            self.save()
            
            return {"mae": float(mae), "r2": float(r2), "samples": len(X)}
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {}
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predice servicios/móviles para próximas horas"""
        if self.model is None:
            logger.warning("Model not trained")
            return np.array([])
        
        try:
            X = self.prepare_features(df)
            X = X.fillna(0)
            X_scaled = self.scaler.transform(X)
            return np.maximum(self.model.predict(X_scaled), 0)  # evitar negativos
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return np.array([])
    
    def save(self):
        """Persiste modelo y scaler"""
        dump(self.model, self.model_path)
        dump(self.scaler, MODEL_DIR / "forecast_scaler.pkl")
        logger.info(f"Model saved: {self.model_path}")
    
    def load(self):
        """Carga modelo y scaler"""
        self.model = load(self.model_path)
        self.scaler = load(MODEL_DIR / "forecast_scaler.pkl")
        logger.info(f"Model loaded: {self.model_path}")

class DelayPredictor:
    """Predice retrasos en servicios basado en histórico y condiciones actuales"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or MODEL_DIR / "delay_predictor.pkl"
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = ['hour', 'day_of_week', 'distance_km', 'vehicles_available', 'historical_delay_avg']
        if self.model_path.exists():
            self.load()
        else:
            self.model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
    
    def train(self, trips_df: pd.DataFrame) -> Dict[str, float]:
        """Entrena con datos de viajes históricos"""
        if trips_df.empty:
            logger.warning("No trip data for training")
            return {}
        
        try:
            d = trips_df.copy()
            d['timestamp'] = pd.to_datetime(d.get('start_time', d.get('created_at')))
            d['hour'] = d['timestamp'].dt.hour
            d['day_of_week'] = d['timestamp'].dt.dayofweek
            d['delay_min'] = d.get('delay_min', 0).fillna(0)
            d['distance_km'] = d.get('distance_km', 5).fillna(5)
            d['vehicles_available'] = 10  # placeholder
            d['historical_delay_avg'] = d['delay_min'].rolling(24, min_periods=1).mean().fillna(0)
            
            X = d[self.feature_names].fillna(0)
            y = d['delay_min'].clip(lower=0)
            
            if len(X) < 10:
                logger.warning("Insufficient trip data")
                return {}
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            
            logger.info(f"Delay predictor trained: MAE={mae:.2f} min")
            self.save()
            return {"mae_min": float(mae)}
        except Exception as e:
            logger.error(f"Delay training error: {e}")
            return {}
    
    def predict_delay(self, hour: int, day_of_week: int, distance_km: float, vehicles_available: int, historical_avg: float) -> float:
        """Predice delay en minutos para un viaje"""
        if self.model is None:
            return 0.0
        
        try:
            X = np.array([[hour, day_of_week, distance_km, vehicles_available, historical_avg]])
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled)[0]
            return max(float(pred), 0.0)
        except Exception as e:
            logger.error(f"Delay prediction error: {e}")
            return 0.0
    
    def save(self):
        dump(self.model, self.model_path)
        dump(self.scaler, MODEL_DIR / "delay_scaler.pkl")
    
    def load(self):
        self.model = load(self.model_path)
        self.scaler = load(MODEL_DIR / "delay_scaler.pkl")

class AnomalyDetector:
    """Detecta comportamientos anómalos en GPS, consumo, etc. usando Isolation Forest"""
    
    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.model_path = MODEL_DIR / "anomaly_detector.pkl"
        if self.model_path.exists():
            self.load()
    
    def train(self, gps_df: pd.DataFrame) -> Dict[str, int]:
        """Entrena detector con datos GPS históricos"""
        if gps_df.empty:
            logger.warning("No GPS data for anomaly training")
            return {}
        
        try:
            features = ['speed', 'heading', 'accuracy']
            available_features = [f for f in features if f in gps_df.columns]
            
            if not available_features:
                logger.warning("No anomaly features available")
                return {}
            
            X = gps_df[available_features].fillna(gps_df[available_features].median())
            X_scaled = self.scaler.fit_transform(X)
            
            self.model.fit(X_scaled)
            anomalies = (self.model.predict(X_scaled) == -1).sum()
            
            logger.info(f"Anomaly detector trained: {anomalies} anomalies found in {len(X)} records")
            self.save()
            return {"anomalies_detected": int(anomalies), "total_records": len(X)}
        except Exception as e:
            logger.error(f"Anomaly training error: {e}")
            return {}
    
    def detect(self, gps_record: Dict[str, float]) -> bool:
        """Retorna True si el registro GPS es anómalo"""
        if self.model is None:
            return False
        
        try:
            features = ['speed', 'heading', 'accuracy']
            X = np.array([[gps_record.get(f, 0) for f in features]])
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)[0] == -1
        except:
            return False
    
    def save(self):
        dump(self.model, self.model_path)
        dump(self.scaler, MODEL_DIR / "anomaly_scaler.pkl")
    
    def load(self):
        self.model = load(self.model_path)
        self.scaler = load(MODEL_DIR / "anomaly_scaler.pkl")
