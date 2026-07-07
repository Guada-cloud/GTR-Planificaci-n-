# cache_manager.py - Sistema de cache para mejorar performance
from __future__ import annotations
from typing import Optional, Any, Callable, Dict
from datetime import datetime, timedelta
import logging
import hashlib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheManager:
    """Administra cache en memoria"""
    
    def __init__(self, default_ttl_seconds: int = 3600):
        self.default_ttl = default_ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_count: Dict[str, int] = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Genera clave unica para cache"""
        content = f"{prefix}:{json.dumps(args)}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache"""
        if key not in self.cache:
            self.miss_count += 1
            return None
        
        entry = self.cache[key]
        
        if entry['expiry'] < datetime.utcnow():
            del self.cache[key]
            self.miss_count += 1
            return None
        
        self.hit_count += 1
        self.access_count[key] = self.access_count.get(key, 0) + 1
        
        return entry['value']
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Almacena valor en cache"""
        ttl = ttl_seconds or self.default_ttl
        expiry = datetime.utcnow() + timedelta(seconds=ttl)
        
        self.cache[key] = {
            'value': value,
            'expiry': expiry,
            'created_at': datetime.utcnow()
        }
        
        logger.debug(f"[CACHE] Set {key} (TTL: {ttl}s)")
    
    def delete(self, key: str) -> bool:
        """Elimina entrada del cache"""
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"[CACHE] Deleted {key}")
            return True
        return False
    
    def clear(self):
        """Limpia todo el cache"""
        count = len(self.cache)
        self.cache.clear()
        self.access_count.clear()
        logger.info(f"[CACHE] Limpiado: {count} entradas")
    
    def cleanup_expired(self):
        """Elimina entradas expiradas"""
        now = datetime.utcnow()
        expired_keys = [
            k for k, v in self.cache.items()
            if v['expiry'] < now
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        logger.info(f"[CACHE] Limpieza: {len(expired_keys)} entradas expiradas removidas")
    
    def get_stats(self) -> Dict:
        """Retorna estadisticas del cache"""
        total_accesses = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_accesses * 100) if total_accesses > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'total_accesses': total_accesses
        }

class CacheDecorator:
    """Decorador para cachear resultados de funciones"""
    
    def __init__(self, cache_manager: CacheManager, ttl_seconds: int = 3600):
        self.cache_manager = cache_manager
        self.ttl = ttl_seconds
    
    def __call__(self, func: Callable) -> Callable:
        """Decora una funcion para usar cache"""
        def wrapper(*args, **kwargs):
            key = self.cache_manager._generate_key(func.__name__, *args, **kwargs)
            
            # Intentar obtener del cache
            cached_result = self.cache_manager.get(key)
            if cached_result is not None:
                logger.debug(f"[CACHE] Hit para {func.__name__}")
                return cached_result
            
            # Ejecutar funcion
            result = func(*args, **kwargs)
            
            # Almacenar en cache
            self.cache_manager.set(key, result, self.ttl)
            
            return result
        
        return wrapper
