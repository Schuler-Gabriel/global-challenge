"""
Pydantic Schemas - Alerts Feature

Este módulo define os schemas de entrada e saída para a API de alertas.
Utiliza Pydantic para validação automática e documentação da API.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class RiverLevelRequest(BaseModel):
    """Schema para dados de entrada do nível do rio"""
    level_meters: float = Field(..., ge=0.0, le=10.0, description="Nível do rio em metros")
    trend: Optional[str] = Field(None, description="Tendência do nível (rising, falling, stable)")
    source: str = Field("api", description="Fonte dos dados")
    
    @validator('trend')
    def validate_trend(cls, v):
        if v and v not in ['rising', 'falling', 'stable']:
            raise ValueError('Trend deve ser: rising, falling ou stable')
        return v


class WeatherAlertRequest(BaseModel):
    """Schema para dados de entrada de alerta meteorológico"""
    precipitation_forecast_mm: float = Field(..., ge=0.0, description="Precipitação prevista em mm")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confiança da previsão (0.0-1.0)")
    forecast_horizon_hours: int = Field(24, ge=1, le=72, description="Horizonte de previsão em horas")
    model_version: str = Field("latest", description="Versão do modelo usado")


class GenerateAlertRequest(BaseModel):
    """Schema para requisição de geração de alerta"""
    river_level_meters: float = Field(..., ge=0.0, le=10.0, description="Nível atual do rio em metros")
    precipitation_forecast_mm: float = Field(..., ge=0.0, description="Precipitação prevista em mm")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confiança da previsão")
    forecast_horizon_hours: int = Field(24, ge=1, le=72, description="Horizonte de previsão")
    model_version: str = Field("latest", description="Versão do modelo")


class RiverLevelResponse(BaseModel):
    """Schema de resposta para nível do rio"""
    timestamp: datetime
    level_meters: float
    trend: Optional[str]
    source: str
    category: str
    flood_risk: bool


class WeatherAlertResponse(BaseModel):
    """Schema de resposta para alerta meteorológico"""
    timestamp: datetime
    precipitation_forecast_mm: float
    confidence_score: float
    forecast_horizon_hours: int
    model_version: str
    precipitation_level: str
    high_confidence: bool


class FloodAlertResponse(BaseModel):
    """Schema de resposta para alerta de cheia"""
    timestamp: datetime
    alert_id: str
    alert_level: str
    risk_level: str
    recommended_action: str
    risk_score: float
    message: str
    valid_until: datetime
    source: str
    river_level: RiverLevelResponse
    weather_alert: WeatherAlertResponse
    is_active: bool
    severity_score: int
    urgency_level: str
    notify_emergency: bool
    recommended_preparations: List[str]


class AlertHistoryResponse(BaseModel):
    """Schema de resposta para histórico de alertas"""
    period_start: datetime
    period_end: datetime
    total_alerts: int
    alerts_by_level: Dict[str, int]
    average_risk_score: float
    max_river_level: float
    total_precipitation: float
    false_positive_rate: Optional[float]
    alert_frequency_per_day: float
    most_common_level: str


class AlertHistoryAnalysisResponse(BaseModel):
    """Schema de resposta para análise de histórico"""
    period: Dict[str, Any]
    summary: AlertHistoryResponse
    insights: Dict[str, Any]


class AlertStatusUpdateRequest(BaseModel):
    """Schema para atualização de status de alerta"""
    new_river_level: float = Field(..., ge=0.0, le=10.0, description="Novo nível do rio")
    new_precipitation: float = Field(..., ge=0.0, description="Nova precipitação prevista")
    new_confidence: float = Field(..., ge=0.0, le=1.0, description="Nova confiança")


class AlertListResponse(BaseModel):
    """Schema de resposta para lista de alertas"""
    alerts: List[FloodAlertResponse]
    total_count: int
    active_count: int
    critical_count: int


class AlertSummaryResponse(BaseModel):
    """Schema de resposta para resumo de alertas"""
    current_status: str
    active_alerts_count: int
    highest_alert_level: str
    latest_alert: Optional[FloodAlertResponse]
    river_status: RiverLevelResponse
    weather_status: WeatherAlertResponse


class ErrorResponse(BaseModel):
    """Schema de resposta para erros"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now) 