"""
Domain Entities - Alerts Feature

Este módulo contém as entidades de domínio para o sistema de alertas de cheias.
Representa os objetos centrais de negócio relacionados a alertas, níveis de risco
e condições do rio.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List


class AlertLevel(Enum):
    """Níveis de alerta do sistema"""
    LOW = "baixo"
    MODERATE = "moderado"
    HIGH = "alto"
    CRITICAL = "critico"


class RiskLevel(Enum):
    """Níveis de risco calculado"""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class AlertAction(Enum):
    """Ações recomendadas por nível de alerta"""
    MONITORING = "monitoramento"
    ATTENTION = "atencao"
    ALERT = "alerta"
    EMERGENCY = "emergencia"


@dataclass
class RiverLevel:
    """
    Entidade que representa o nível atual do Rio Guaíba
    
    Attributes:
        timestamp: Momento da medição
        level_meters: Nível em metros
        trend: Tendência (rising, falling, stable)
        source: Fonte dos dados (api, manual)
    """
    timestamp: datetime
    level_meters: float
    trend: Optional[str] = None
    source: str = "api"
    
    def __post_init__(self):
        """Validação após inicialização"""
        if self.level_meters < 0:
            raise ValueError("Nível do rio não pode ser negativo")
        if self.level_meters > 10:
            raise ValueError("Nível do rio muito alto (>10m)")
    
    def get_level_category(self) -> str:
        """Categoriza o nível do rio"""
        if self.level_meters <= 1.5:
            return "normal"
        elif self.level_meters <= 2.5:
            return "atencao"
        elif self.level_meters <= 3.5:
            return "alerta"
        else:
            return "emergencia"
    
    def is_flood_risk(self) -> bool:
        """Verifica se há risco de enchente"""
        return self.level_meters > 2.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level_meters": self.level_meters,
            "trend": self.trend,
            "source": self.source,
            "category": self.get_level_category(),
            "flood_risk": self.is_flood_risk()
        }


@dataclass
class WeatherAlert:
    """
    Entidade que representa alertas meteorológicos
    
    Attributes:
        timestamp: Momento do alerta
        precipitation_forecast_mm: Precipitação prevista em mm
        confidence_score: Confiança da previsão (0.0-1.0)
        forecast_horizon_hours: Horizonte de previsão em horas
        model_version: Versão do modelo usado
    """
    timestamp: datetime
    precipitation_forecast_mm: float
    confidence_score: float
    forecast_horizon_hours: int = 24
    model_version: str = "unknown"
    
    def __post_init__(self):
        """Validação após inicialização"""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confiança deve estar entre 0.0 e 1.0")
        if self.precipitation_forecast_mm < 0:
            raise ValueError("Precipitação não pode ser negativa")
        if self.forecast_horizon_hours <= 0:
            raise ValueError("Horizonte de previsão deve ser positivo")
    
    def get_precipitation_level(self) -> str:
        """Categoriza o nível de precipitação"""
        if self.precipitation_forecast_mm < 1.0:
            return "leve"
        elif self.precipitation_forecast_mm < 10.0:
            return "moderada"
        elif self.precipitation_forecast_mm < 50.0:
            return "forte"
        else:
            return "intensa"
    
    def is_high_confidence(self) -> bool:
        """Verifica se é uma previsão de alta confiança"""
        return self.confidence_score >= 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "precipitation_forecast_mm": self.precipitation_forecast_mm,
            "confidence_score": self.confidence_score,
            "forecast_horizon_hours": self.forecast_horizon_hours,
            "model_version": self.model_version,
            "precipitation_level": self.get_precipitation_level(),
            "high_confidence": self.is_high_confidence()
        }


@dataclass
class FloodAlert:
    """
    Entidade principal que representa um alerta de cheia
    
    Combina informações de nível do rio e previsão meteorológica
    para gerar um alerta com nível de risco e ações recomendadas.
    """
    timestamp: datetime
    alert_id: str
    alert_level: AlertLevel
    risk_level: RiskLevel
    recommended_action: AlertAction
    river_level: RiverLevel
    weather_alert: WeatherAlert
    risk_score: float
    message: str
    valid_until: datetime
    source: str = "system"
    
    def __post_init__(self):
        """Validação após inicialização"""
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("Risk score deve estar entre 0.0 e 1.0")
        if self.valid_until <= self.timestamp:
            raise ValueError("valid_until deve ser posterior ao timestamp")
    
    def is_active(self, current_time: Optional[datetime] = None) -> bool:
        """Verifica se o alerta ainda está ativo"""
        if current_time is None:
            current_time = datetime.now()
        return current_time <= self.valid_until
    
    def get_severity_score(self) -> int:
        """Retorna score numérico de severidade (1-4)"""
        severity_map = {
            AlertLevel.LOW: 1,
            AlertLevel.MODERATE: 2,
            AlertLevel.HIGH: 3,
            AlertLevel.CRITICAL: 4
        }
        return severity_map[self.alert_level]
    
    def get_urgency_level(self) -> str:
        """Determina urgência baseada no nível e tempo restante"""
        time_remaining = (self.valid_until - datetime.now()).total_seconds() / 3600  # horas
        
        if self.alert_level == AlertLevel.CRITICAL:
            return "immediate"
        elif self.alert_level == AlertLevel.HIGH and time_remaining < 6:
            return "urgent"
        elif self.alert_level in [AlertLevel.HIGH, AlertLevel.MODERATE] and time_remaining < 12:
            return "soon"
        else:
            return "routine"
    
    def should_notify_emergency_services(self) -> bool:
        """Determina se deve notificar serviços de emergência"""
        return (
            self.alert_level == AlertLevel.CRITICAL or
            (self.alert_level == AlertLevel.HIGH and self.risk_score > 0.8)
        )
    
    def get_recommended_preparations(self) -> List[str]:
        """Lista de preparações recomendadas baseadas no nível"""
        preparations = {
            AlertLevel.LOW: [
                "Monitorar condições meteorológicas",
                "Verificar sistemas de drenagem"
            ],
            AlertLevel.MODERATE: [
                "Remover veículos de áreas baixas",
                "Preparar kits de emergência",
                "Monitorar níveis do rio"
            ],
            AlertLevel.HIGH: [
                "Evacuar áreas de risco",
                "Fechar pontes se necessário",
                "Ativar abrigos temporários",
                "Alertar população ribeirinha"
            ],
            AlertLevel.CRITICAL: [
                "Evacuação imediata obrigatória",
                "Fechar todas as pontes",
                "Ativar estado de emergência",
                "Mobilizar defesa civil"
            ]
        }
        return preparations.get(self.alert_level, [])
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário completo"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "alert_id": self.alert_id,
            "alert_level": self.alert_level.value,
            "risk_level": self.risk_level.value,
            "recommended_action": self.recommended_action.value,
            "risk_score": self.risk_score,
            "message": self.message,
            "valid_until": self.valid_until.isoformat(),
            "source": self.source,
            "river_level": self.river_level.to_dict(),
            "weather_alert": self.weather_alert.to_dict(),
            "is_active": self.is_active(),
            "severity_score": self.get_severity_score(),
            "urgency_level": self.get_urgency_level(),
            "notify_emergency": self.should_notify_emergency_services(),
            "recommended_preparations": self.get_recommended_preparations()
        }


@dataclass
class AlertHistory:
    """
    Entidade para histórico de alertas
    
    Permite rastrear padrões e tendências de alertas ao longo do tempo.
    """
    period_start: datetime
    period_end: datetime
    total_alerts: int
    alerts_by_level: Dict[AlertLevel, int]
    average_risk_score: float
    max_river_level: float
    total_precipitation: float
    false_positive_rate: Optional[float] = None
    
    def get_alert_frequency(self) -> float:
        """Calcula frequência de alertas por dia"""
        days = (self.period_end - self.period_start).days
        return self.total_alerts / max(days, 1)
    
    def get_most_common_level(self) -> AlertLevel:
        """Retorna o nível de alerta mais comum"""
        return max(self.alerts_by_level, key=self.alerts_by_level.get)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_alerts": self.total_alerts,
            "alerts_by_level": {level.value: count for level, count in self.alerts_by_level.items()},
            "average_risk_score": self.average_risk_score,
            "max_river_level": self.max_river_level,
            "total_precipitation": self.total_precipitation,
            "false_positive_rate": self.false_positive_rate,
            "alert_frequency_per_day": self.get_alert_frequency(),
            "most_common_level": self.get_most_common_level().value
        } 