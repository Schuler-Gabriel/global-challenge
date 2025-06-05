"""
Presentation Schemas - Forecast Feature

Este módulo define os schemas de entrada e saída (DTOs) para os endpoints
da API de previsão meteorológica. Utiliza Pydantic para validação,
serialização e documentação.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field, validator


class PrecipitationLevel(str, Enum):
    """Níveis de precipitação"""
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    EXTREME = "extreme"


class WeatherCondition(str, Enum):
    """Condições meteorológicas"""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    STORMY = "stormy"
    UNKNOWN = "unknown"


class WeatherDataRequest(BaseModel):
    """DTO para entrada de dados meteorológicos"""
    timestamp: datetime = Field(..., description="Data e hora da medição")
    precipitation: float = Field(..., description="Precipitação (mm/h)", ge=0, le=200)
    pressure: float = Field(..., description="Pressão atmosférica (mB)", ge=900, le=1100)
    temperature: float = Field(..., description="Temperatura do ar (°C)", ge=-10, le=50)
    dew_point: float = Field(..., description="Temperatura do ponto de orvalho (°C)", ge=-20, le=40)
    humidity: float = Field(..., description="Umidade relativa do ar (%)", ge=0, le=100)
    wind_speed: float = Field(..., description="Velocidade do vento (m/s)", ge=0, le=50)
    wind_direction: float = Field(..., description="Direção do vento (graus)", ge=0, le=360)
    radiation: Optional[float] = Field(None, description="Radiação global (Kj/m²)", ge=0)
    
    # Campos opcionais (valores extremos)
    pressure_max: Optional[float] = Field(None, description="Pressão máxima na hora anterior (mB)")
    pressure_min: Optional[float] = Field(None, description="Pressão mínima na hora anterior (mB)")
    temperature_max: Optional[float] = Field(None, description="Temperatura máxima na hora anterior (°C)")
    temperature_min: Optional[float] = Field(None, description="Temperatura mínima na hora anterior (°C)")
    humidity_max: Optional[float] = Field(None, description="Umidade máxima na hora anterior (%)")
    humidity_min: Optional[float] = Field(None, description="Umidade mínima na hora anterior (%)")
    dew_point_max: Optional[float] = Field(None, description="Ponto de orvalho máximo na hora anterior (°C)")
    dew_point_min: Optional[float] = Field(None, description="Ponto de orvalho mínimo na hora anterior (°C)")
    
    # Metadados
    station_id: Optional[str] = Field(None, description="Identificador da estação meteorológica")
    quality_flag: Optional[str] = Field(None, description="Flag de qualidade do dado")
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2025-06-15T14:00:00",
                "precipitation": 0.2,
                "pressure": 1013.2,
                "temperature": 25.7,
                "dew_point": 17.8,
                "humidity": 65.0,
                "wind_speed": 2.3,
                "wind_direction": 145.0,
                "radiation": 756.2,
                "station_id": "A801"
            }
        }


class WeatherDataResponse(BaseModel):
    """DTO para saída de dados meteorológicos"""
    timestamp: datetime
    precipitation: float
    pressure: float
    temperature: float
    dew_point: float
    humidity: float
    wind_speed: float
    wind_direction: float
    radiation: Optional[float] = None
    
    # Campos calculados
    precipitation_level: PrecipitationLevel
    weather_condition: WeatherCondition
    is_extreme: bool
    
    # Metadados
    station_id: Optional[str] = None
    quality_flag: Optional[str] = None


class ForecastRequest(BaseModel):
    """DTO para requisição de previsão"""
    use_cache: bool = Field(True, description="Se deve usar cache de previsões recentes")
    model_version: Optional[str] = Field(None, description="Versão específica do modelo (opcional)")
    
    class Config:
        schema_extra = {
            "example": {
                "use_cache": True,
                "model_version": None
            }
        }


class ForecastResponse(BaseModel):
    """DTO para resposta de previsão"""
    timestamp: datetime = Field(..., description="Data e hora da previsão")
    precipitation_mm: float = Field(..., description="Precipitação prevista (mm/h)")
    confidence_score: float = Field(..., description="Score de confiança (0.0-1.0)")
    precipitation_level: PrecipitationLevel = Field(..., description="Nível de precipitação categorizado")
    is_rain_expected: bool = Field(..., description="Se chuva é esperada (>0.1mm)")
    
    # Metadados do modelo
    model_version: str = Field(..., description="Versão do modelo")
    inference_time_ms: float = Field(..., description="Tempo de inferência (ms)")
    
    # Informações adicionais
    forecast_horizon_hours: Optional[int] = Field(None, description="Horizonte da previsão (horas)")
    input_sequence_length: Optional[int] = Field(None, description="Tamanho da sequência de entrada")
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2025-06-15T18:00:00",
                "precipitation_mm": 2.5,
                "confidence_score": 0.85,
                "precipitation_level": "moderate",
                "is_rain_expected": True,
                "model_version": "v2.1.0",
                "inference_time_ms": 45.3,
                "forecast_horizon_hours": 24,
                "input_sequence_length": 24
            }
        }


class ForecastHourlyResponse(BaseModel):
    """DTO para previsão horária"""
    hours: List[Dict[str, Any]] = Field(..., description="Previsões horárias")
    summary: Dict[str, Any] = Field(..., description="Resumo das previsões")
    model_info: Dict[str, Any] = Field(..., description="Informações do modelo")
    generated_at: datetime = Field(..., description="Data/hora de geração")


class ModelMetricsResponse(BaseModel):
    """DTO para métricas do modelo"""
    model_version: str = Field(..., description="Versão do modelo")
    training_date: datetime = Field(..., description="Data de treinamento")
    
    # Métricas principais
    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Square Error")
    accuracy: float = Field(..., description="Accuracy para classificação de eventos")
    
    # Métricas adicionais
    r2_score: Optional[float] = Field(None, description="R² Score")
    precision: Optional[float] = Field(None, description="Precision")
    recall: Optional[float] = Field(None, description="Recall")
    f1_score: Optional[float] = Field(None, description="F1-Score")
    skill_score: Optional[float] = Field(None, description="Skill Score")
    
    # Avaliação dos critérios
    meets_mae_criteria: bool = Field(..., description="Se MAE atende ao critério (<2.0)")
    meets_rmse_criteria: bool = Field(..., description="Se RMSE atende ao critério (<3.0)")
    meets_accuracy_criteria: bool = Field(..., description="Se accuracy atende ao critério (>75%)")
    meets_all_criteria: bool = Field(..., description="Se atende a todos os critérios")
    performance_grade: str = Field(..., description="Nota de performance (A-F)")
    
    # Informações do dataset
    train_samples: Optional[int] = Field(None, description="Amostras de treino")
    validation_samples: Optional[int] = Field(None, description="Amostras de validação")
    test_samples: Optional[int] = Field(None, description="Amostras de teste")


class RefreshModelRequest(BaseModel):
    """DTO para atualização de modelo"""
    new_model_version: Optional[str] = Field(None, description="Versão do novo modelo (opcional)")
    force_update: bool = Field(False, description="Se deve forçar atualização mesmo com performance inferior")
    
    class Config:
        schema_extra = {
            "example": {
                "new_model_version": "v2.2.0",
                "force_update": False
            }
        }


class RefreshModelResponse(BaseModel):
    """DTO para resposta de atualização de modelo"""
    success: bool = Field(..., description="Se atualização foi bem-sucedida")
    message: str = Field(..., description="Mensagem descritiva")
    previous_version: Optional[str] = Field(None, description="Versão anterior do modelo")
    new_version: Optional[str] = Field(None, description="Nova versão do modelo")
    comparison: Optional[Dict[str, Any]] = Field(None, description="Comparação entre versões")
    recommendation: Optional[Dict[str, Any]] = Field(None, description="Recomendação de atualização")
    validation: Optional[Dict[str, Any]] = Field(None, description="Validação do novo modelo")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Modelo atualizado com sucesso para v2.2.0",
                "previous_version": "v2.1.0",
                "new_version": "v2.2.0",
                "comparison": {
                    "mae_change": -0.15,
                    "rmse_change": -0.22,
                    "accuracy_change": 0.03
                },
                "recommendation": {
                    "should_update": True,
                    "confidence": "high"
                }
            }
        } 