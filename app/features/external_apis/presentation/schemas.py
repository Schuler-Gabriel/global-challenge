"""
Schemas for External APIs - Presentation Layer

Esquemas Pydantic para validação e documentação dos endpoints
de APIs externas (CPTEC e Rio Guaíba).
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class RiskLevelEnum(str, Enum):
    """Enum para níveis de risco"""
    BAIXO = "baixo"
    MODERADO = "moderado"
    ALTO = "alto"
    CRITICO = "critico"


class ApiStatusEnum(str, Enum):
    """Enum para status de APIs"""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class WeatherConditionSchema(BaseModel):
    """Schema para condições meteorológicas"""
    timestamp: datetime = Field(..., description="Timestamp dos dados")
    temperature: float = Field(..., ge=-50, le=60, description="Temperatura em °C")
    humidity: float = Field(..., ge=0, le=100, description="Umidade relativa em %")
    pressure: float = Field(..., ge=800, le=1200, description="Pressão atmosférica em hPa")
    wind_speed: float = Field(..., ge=0, le=200, description="Velocidade do vento em m/s")
    wind_direction: float = Field(..., ge=0, le=360, description="Direção do vento em graus")
    description: str = Field(..., max_length=200, description="Descrição das condições")
    
    # Dados de precipitação
    precipitation_current: Optional[float] = Field(None, ge=0, le=1000, description="Precipitação atual em mm/h")
    precipitation_forecast_1h: Optional[float] = Field(None, ge=0, le=1000, description="Previsão 1h em mm")
    precipitation_forecast_6h: Optional[float] = Field(None, ge=0, le=1000, description="Previsão 6h em mm")
    precipitation_forecast_24h: Optional[float] = Field(None, ge=0, le=1000, description="Previsão 24h em mm")
    
    # Metadados
    station_id: Optional[str] = Field(None, description="ID da estação meteorológica")
    data_source: str = Field("CPTEC", description="Fonte dos dados")
    forecast_confidence: Optional[float] = Field(None, ge=0, le=1, description="Confiança da previsão (0-1)")
    
    # Campos calculados
    is_storm_conditions: bool = Field(..., description="Se há condições de tempestade")
    is_rain_expected: bool = Field(..., description="Se chuva é esperada")
    comfort_index: str = Field(..., description="Índice de conforto térmico")

    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-01-15T14:30:00",
                "temperature": 25.5,
                "humidity": 65.0,
                "pressure": 1013.2,
                "wind_speed": 5.2,
                "wind_direction": 180.0,
                "description": "Parcialmente nublado",
                "precipitation_current": 0.0,
                "precipitation_forecast_1h": 0.5,
                "precipitation_forecast_6h": 2.5,
                "precipitation_forecast_24h": 8.0,
                "station_id": "POA001",
                "data_source": "CPTEC",
                "forecast_confidence": 0.85,
                "is_storm_conditions": False,
                "is_rain_expected": True,
                "comfort_index": "confortável"
            }
        }


class RiverLevelSchema(BaseModel):
    """Schema para nível do rio"""
    timestamp: datetime = Field(..., description="Timestamp da medição")
    level_meters: float = Field(..., ge=-5, le=10, description="Nível do rio em metros")
    station_name: str = Field(..., max_length=100, description="Nome da estação")
    station_id: Optional[str] = Field(None, description="ID da estação")
    
    # Metadados da coleta
    source_timestamp: Optional[datetime] = Field(None, description="Timestamp da fonte")
    data_quality: Optional[str] = Field(None, description="Qualidade dos dados")
    measurement_uncertainty: Optional[float] = Field(None, ge=0, description="Incerteza da medição em cm")
    
    # Campos calculados
    risk_level: RiskLevelEnum = Field(..., description="Nível de risco")
    is_flood_risk: bool = Field(..., description="Se há risco de inundação")
    is_critical_level: bool = Field(..., description="Se está em nível crítico")
    distance_to_flood_level: float = Field(..., description="Distância até cota de inundação em metros")

    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-01-15T14:30:00",
                "level_meters": 2.45,
                "station_name": "Porto Alegre",
                "station_id": "POA001",
                "source_timestamp": "2024-01-15T14:25:00",
                "data_quality": "good",
                "measurement_uncertainty": 0.02,
                "risk_level": "baixo",
                "is_flood_risk": False,
                "is_critical_level": False,
                "distance_to_flood_level": 1.15
            }
        }


class TrendAnalysisSchema(BaseModel):
    """Schema para análise de tendência"""
    trend: str = Field(..., description="Tendência: 'rising', 'falling', 'stable'")
    rate_change_cm_per_hour: float = Field(..., description="Taxa de mudança em cm/h")
    confidence: float = Field(..., ge=0, le=1, description="Confiança da análise")
    first_level: float = Field(..., description="Primeiro nível da janela")
    last_level: float = Field(..., description="Último nível da janela")
    time_window_hours: float = Field(..., description="Janela de tempo em horas")
    data_points: int = Field(..., description="Número de pontos de dados")

    class Config:
        schema_extra = {
            "example": {
                "trend": "stable",
                "rate_change_cm_per_hour": 0.2,
                "confidence": 0.85,
                "first_level": 2.43,
                "last_level": 2.45,
                "time_window_hours": 6.0,
                "data_points": 24
            }
        }


class ExternalApiResponseSchema(BaseModel):
    """Schema para resposta de API externa"""
    success: bool = Field(..., description="Se a operação foi bem-sucedida")
    timestamp: datetime = Field(..., description="Timestamp da resposta")
    api_name: str = Field(..., description="Nome da API")
    response_time_ms: float = Field(..., ge=0, description="Tempo de resposta em ms")
    
    # Metadados opcionais
    status_code: Optional[int] = Field(None, description="Código de status HTTP")
    error_message: Optional[str] = Field(None, description="Mensagem de erro")
    retry_count: int = Field(0, ge=0, description="Número de tentativas")
    from_cache: bool = Field(False, description="Se veio do cache")
    cache_ttl_seconds: Optional[int] = Field(None, description="TTL do cache em segundos")


# Response Schemas
class WeatherResponse(BaseModel):
    """Resposta para condições meteorológicas"""
    data: Optional[WeatherConditionSchema] = Field(None, description="Dados meteorológicos")
    api_response: ExternalApiResponseSchema = Field(..., description="Metadados da resposta")

    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "timestamp": "2024-01-15T14:30:00",
                    "temperature": 25.5,
                    "humidity": 65.0,
                    "pressure": 1013.2,
                    "wind_speed": 5.2,
                    "wind_direction": 180.0,
                    "description": "Parcialmente nublado",
                    "is_storm_conditions": False,
                    "is_rain_expected": True,
                    "comfort_index": "confortável"
                },
                "api_response": {
                    "success": True,
                    "timestamp": "2024-01-15T14:30:15",
                    "api_name": "cptec_weather",
                    "response_time_ms": 245.5,
                    "status_code": 200,
                    "from_cache": False
                }
            }
        }


class RiverLevelResponse(BaseModel):
    """Resposta para nível do rio"""
    data: Optional[Union[RiverLevelSchema, Dict[str, Any]]] = Field(None, description="Dados do nível do rio")
    api_response: ExternalApiResponseSchema = Field(..., description="Metadados da resposta")

    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "timestamp": "2024-01-15T14:30:00",
                    "level_meters": 2.45,
                    "station_name": "Porto Alegre",
                    "risk_level": "baixo",
                    "is_flood_risk": False,
                    "distance_to_flood_level": 1.15
                },
                "api_response": {
                    "success": True,
                    "timestamp": "2024-01-15T14:30:15",
                    "api_name": "guaiba_river",
                    "response_time_ms": 189.2,
                    "status_code": 200,
                    "from_cache": False
                }
            }
        }


class CurrentConditionsResponse(BaseModel):
    """Resposta para condições completas (tempo + rio)"""
    weather: WeatherResponse = Field(..., description="Dados meteorológicos")
    river: RiverLevelResponse = Field(..., description="Dados do nível do rio")
    combined_analysis: Dict[str, Any] = Field(..., description="Análise combinada")

    class Config:
        schema_extra = {
            "example": {
                "weather": {
                    "data": {
                        "temperature": 25.5,
                        "description": "Parcialmente nublado",
                        "is_rain_expected": True
                    },
                    "api_response": {"success": True, "api_name": "cptec_weather"}
                },
                "river": {
                    "data": {
                        "level_meters": 2.45,
                        "risk_level": "baixo",
                        "is_flood_risk": False
                    },
                    "api_response": {"success": True, "api_name": "guaiba_river"}
                },
                "combined_analysis": {
                    "flood_risk_level": "baixo",
                    "weather_impact": "minimal",
                    "recommendations": ["Monitor conditions"]
                }
            }
        }


class ApiHealthSchema(BaseModel):
    """Schema para saúde de uma API"""
    api_name: str = Field(..., description="Nome da API")
    status: ApiStatusEnum = Field(..., description="Status atual")
    response_time_ms: float = Field(..., ge=0, description="Tempo de resposta médio")
    success_rate: float = Field(..., ge=0, le=1, description="Taxa de sucesso")
    last_check: datetime = Field(..., description="Último health check")
    error_message: Optional[str] = Field(None, description="Última mensagem de erro")


class ApiHealthResponse(BaseModel):
    """Resposta para health check das APIs"""
    timestamp: datetime = Field(..., description="Timestamp do health check")
    overall_status: str = Field(..., description="Status geral: 'healthy', 'degraded', 'unhealthy'")
    apis: List[ApiHealthSchema] = Field(..., description="Status de cada API")
    circuit_breaker: Dict[str, Any] = Field(..., description="Status dos circuit breakers")
    response_time_ms: float = Field(..., description="Tempo do health check completo")
    
    # Métricas detalhadas (opcionais)
    metrics: Optional[Dict[str, Any]] = Field(None, description="Métricas detalhadas")

    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-01-15T14:30:00",
                "overall_status": "healthy",
                "apis": [
                    {
                        "api_name": "cptec_weather",
                        "status": "online",
                        "response_time_ms": 250.0,
                        "success_rate": 0.98,
                        "last_check": "2024-01-15T14:30:00"
                    },
                    {
                        "api_name": "guaiba_river",
                        "status": "online", 
                        "response_time_ms": 180.0,
                        "success_rate": 0.99,
                        "last_check": "2024-01-15T14:30:00"
                    }
                ],
                "circuit_breaker": {
                    "weather_api": {"status": "closed", "failure_count": 0},
                    "river_api": {"status": "closed", "failure_count": 0}
                },
                "response_time_ms": 45.2
            }
        }


# Request Schemas
class WeatherRequestSchema(BaseModel):
    """Schema para requisição de dados meteorológicos"""
    city: str = Field("Porto Alegre", max_length=100, description="Nome da cidade")
    use_cache: bool = Field(True, description="Se deve usar cache")
    max_cache_age_minutes: int = Field(5, ge=1, le=60, description="Idade máxima do cache em minutos")

    @validator('city')
    def validate_city(cls, v):
        if not v or not v.strip():
            raise ValueError('Cidade não pode estar vazia')
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "city": "Porto Alegre",
                "use_cache": True,
                "max_cache_age_minutes": 5
            }
        }


class RiverLevelRequestSchema(BaseModel):
    """Schema para requisição de nível do rio"""
    station: str = Field("Porto Alegre", max_length=100, description="Nome da estação")
    use_cache: bool = Field(True, description="Se deve usar cache")
    max_cache_age_minutes: int = Field(10, ge=1, le=120, description="Idade máxima do cache em minutos")
    include_trend: bool = Field(True, description="Se deve incluir análise de tendência")

    @validator('station')
    def validate_station(cls, v):
        if not v or not v.strip():
            raise ValueError('Estação não pode estar vazia')
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "station": "Porto Alegre",
                "use_cache": True,
                "max_cache_age_minutes": 10,
                "include_trend": True
            }
        }


class ForecastRequestSchema(BaseModel):
    """Schema para requisição de previsão"""
    city: str = Field("Porto Alegre", max_length=100, description="Nome da cidade")
    hours_ahead: int = Field(24, ge=1, le=168, description="Horas à frente (máx 7 dias)")
    use_cache: bool = Field(True, description="Se deve usar cache")

    @validator('city')
    def validate_city(cls, v):
        if not v or not v.strip():
            raise ValueError('Cidade não pode estar vazia')
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "city": "Porto Alegre", 
                "hours_ahead": 24,
                "use_cache": True
            }
        }


class HealthCheckRequestSchema(BaseModel):
    """Schema para requisição de health check"""
    detailed: bool = Field(False, description="Se deve incluir métricas detalhadas")
    api_name: Optional[str] = Field(None, description="API específica (opcional)")

    class Config:
        schema_extra = {
            "example": {
                "detailed": True,
                "api_name": "cptec_weather"
            }
        }


# Error Response Schema
class ErrorResponseSchema(BaseModel):
    """Schema para respostas de erro"""
    error: str = Field(..., description="Tipo do erro")
    message: str = Field(..., description="Mensagem de erro")
    timestamp: datetime = Field(..., description="Timestamp do erro")
    request_id: Optional[str] = Field(None, description="ID da requisição")
    details: Optional[Dict[str, Any]] = Field(None, description="Detalhes adicionais do erro")

    class Config:
        schema_extra = {
            "example": {
                "error": "ExternalApiError",
                "message": "API CPTEC indisponível",
                "timestamp": "2024-01-15T14:30:00",
                "request_id": "req_123456",
                "details": {
                    "api_name": "cptec_weather",
                    "retry_count": 3,
                    "last_error": "Connection timeout"
                }
            }
        } 