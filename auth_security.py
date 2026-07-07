# auth_security.py - Autenticacion, autorizacion y seguridad
from __future__ import annotations
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import hashlib
import secrets
import jwt
import logging
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserRole(str, Enum):
    """Roles de usuario en el sistema"""
    ADMIN = "ADMIN"
    MANAGER = "MANAGER"
    OPERATOR = "OPERATOR"
    DRIVER = "DRIVER"
    CLIENT = "CLIENT"
    ANALYST = "ANALYST"

class Permission(str, Enum):
    """Permisos del sistema"""
    VIEW_DASHBOARD = "view_dashboard"
    MANAGE_SERVICES = "manage_services"
    MANAGE_USERS = "manage_users"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_PAYMENTS = "manage_payments"
    MANAGE_BILLING = "manage_billing"
    VIEW_GPS = "view_gps"
    MANAGE_ALERTS = "manage_alerts"
    EXPORT_DATA = "export_data"
    MANAGE_SETTINGS = "manage_settings"

@dataclass
class User:
    """Usuario del sistema"""
    id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    full_name: str = ""
    phone: str = ""
    is_active: bool = True
    last_login: Optional[datetime] = None
    created_at: datetime = None
    updated_at: datetime = None
    permissions: List[Permission] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.permissions is None:
            self.permissions = self._default_permissions()
    
    def _default_permissions(self) -> List[Permission]:
        """Retorna permisos por defecto segun rol"""
        role_permissions = {
            UserRole.ADMIN: [
                Permission.VIEW_DASHBOARD, Permission.MANAGE_SERVICES,
                Permission.MANAGE_USERS, Permission.VIEW_ANALYTICS,
                Permission.MANAGE_PAYMENTS, Permission.MANAGE_BILLING,
                Permission.VIEW_GPS, Permission.MANAGE_ALERTS,
                Permission.EXPORT_DATA, Permission.MANAGE_SETTINGS
            ],
            UserRole.MANAGER: [
                Permission.VIEW_DASHBOARD, Permission.MANAGE_SERVICES,
                Permission.VIEW_ANALYTICS, Permission.MANAGE_PAYMENTS,
                Permission.VIEW_GPS, Permission.MANAGE_ALERTS, Permission.EXPORT_DATA
            ],
            UserRole.OPERATOR: [
                Permission.VIEW_DASHBOARD, Permission.MANAGE_SERVICES,
                Permission.VIEW_GPS, Permission.MANAGE_ALERTS
            ],
            UserRole.DRIVER: [
                Permission.VIEW_DASHBOARD
            ],
            UserRole.CLIENT: [
                Permission.VIEW_DASHBOARD
            ],
            UserRole.ANALYST: [
                Permission.VIEW_DASHBOARD, Permission.VIEW_ANALYTICS,
                Permission.EXPORT_DATA
            ]
        }
        return role_permissions.get(self.role, [])
    
    def has_permission(self, permission: Permission) -> bool:
        """Verifica si usuario tiene permiso"""
        return permission in self.permissions

class AuthenticationService:
    """Servicio de autenticacion"""
    
    def __init__(self, secret_key: str = "secret-key-change-in-prod",
                 token_expiry_hours: int = 24):
        self.secret_key = secret_key
        self.token_expiry_hours = token_expiry_hours
        self.users: Dict[str, User] = {}
        self.tokens: Dict[str, Dict] = {}
        self.failed_login_attempts: Dict[str, int] = {}
        self.MAX_FAILED_ATTEMPTS = 5
    
    def hash_password(self, password: str) -> str:
        """Hashea password de forma segura"""
        salt = secrets.token_hex(32)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}${password_hash.hex()}"
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verifica password contra hash"""
        try:
            salt, stored_hash = password_hash.split('$')
            password_hash_check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return password_hash_check.hex() == stored_hash
        except:
            return False
    
    def register_user(self, username: str, email: str, password: str,
                     full_name: str, role: UserRole) -> Tuple[bool, str]:
        """Registra nuevo usuario"""
        
        if username in self.users:
            return False, "Usuario ya existe"
        
        if len(password) < 8:
            return False, "Contrasena debe tener minimo 8 caracteres"
        
        user = User(
            id=f"USR-{secrets.token_hex(8)}",
            username=username,
            email=email,
            password_hash=self.hash_password(password),
            role=role,
            full_name=full_name
        )
        
        self.users[username] = user
        logger.info(f"[AUTH] Usuario {username} registrado con rol {role}")
        
        return True, f"Usuario {username} creado exitosamente"
    
    def login(self, username: str, password: str) -> Tuple[bool, str]:
        """Autentica usuario y retorna token"""
        
        if username not in self.users:
            self.failed_login_attempts[username] = self.failed_login_attempts.get(username, 0) + 1
            logger.warning(f"[AUTH] Intento fallido para usuario inexistente: {username}")
            return False, "Usuario o contrasena incorrectos"
        
        if self.failed_login_attempts.get(username, 0) >= self.MAX_FAILED_ATTEMPTS:
            logger.warning(f"[AUTH] Usuario {username} bloqueado por multiples intentos fallidos")
            return False, "Usuario bloqueado temporalmente"
        
        user = self.users[username]
        
        if not user.is_active:
            return False, "Usuario inactivo"
        
        if not self.verify_password(password, user.password_hash):
            self.failed_login_attempts[username] = self.failed_login_attempts.get(username, 0) + 1
            logger.warning(f"[AUTH] Contrasena incorrecta para {username}")
            return False, "Usuario o contrasena incorrectos"
        
        # Reset failed attempts
        self.failed_login_attempts[username] = 0
        
        # Generate token
        expiry = datetime.utcnow() + timedelta(hours=self.token_expiry_hours)
        token_data = {
            'user_id': user.id,
            'username': username,
            'role': user.role.value,
            'exp': expiry.timestamp()
        }
        
        token = jwt.encode(token_data, self.secret_key, algorithm='HS256')
        self.tokens[token] = token_data
        user.last_login = datetime.utcnow()
        
        logger.info(f"[AUTH] Login exitoso para {username} ({user.role})")
        
        return True, token
    
    def verify_token(self, token: str) -> Tuple[bool, Optional[Dict]]:
        """Verifica token JWT"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return True, payload
        except jwt.ExpiredSignatureError:
            logger.warning("[AUTH] Token expirado")
            return False, None
        except jwt.InvalidTokenError:
            logger.warning("[AUTH] Token invalido")
            return False, None
    
    def logout(self, token: str) -> bool:
        """Invalida token"""
        if token in self.tokens:
            del self.tokens[token]
            logger.info("[AUTH] Logout exitoso")
            return True
        return False
    
    def get_user(self, username: str) -> Optional[User]:
        """Obtiene usuario por username"""
        return self.users.get(username)
    
    def update_user_role(self, username: str, new_role: UserRole) -> bool:
        """Actualiza rol de usuario"""
        if username not in self.users:
            return False
        
        user = self.users[username]
        user.role = new_role
        user.permissions = user._default_permissions()
        user.updated_at = datetime.utcnow()
        
        logger.info(f"[AUTH] Rol de {username} actualizado a {new_role}")
        return True

class AuditLogger:
    """Registra todas las acciones para auditoria"""
    
    def __init__(self):
        self.audit_log: List[Dict] = []
    
    def log_action(self, user_id: str, action: str, resource: str,
                  resource_id: str, details: Dict = None, status: str = "SUCCESS"):
        """Registra una accion para auditoria"""
        
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'resource_id': resource_id,
            'status': status,
            'details': details or {}
        }
        
        self.audit_log.append(entry)
        logger.info(f"[AUDIT] {action} on {resource}:{resource_id} by {user_id} - {status}")
    
    def get_user_activity(self, user_id: str, days: int = 7) -> List[Dict]:
        """Obtiene actividad de usuario en ultimos N dias"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        return [
            entry for entry in self.audit_log
            if entry['user_id'] == user_id and 
               datetime.fromisoformat(entry['timestamp']) > cutoff
        ]
    
    def get_resource_history(self, resource: str, resource_id: str) -> List[Dict]:
        """Obtiene historial de cambios de un recurso"""
        
        return [
            entry for entry in self.audit_log
            if entry['resource'] == resource and entry['resource_id'] == resource_id
        ]
