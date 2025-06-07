#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast Presentation Layer - Schemas (DTOs)

Data Transfer Objects usando Pydantic para validação e serialização
de dados da API de previsão meteorológica híbrida.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator


class ForecastHorizon(str, Enum):
    """Horizontes de previsão disponíveis"""
    HOUR_6 = "6h"
    HOUR_12 = "12h"
    HOUR_24 = "24h"
    HOUR_48 = "48h"
    HOUR_72 = "72h"


class ModelType(str, Enum):
    """Tipos de modelo disponíveis"""
    HYBRID = "hybrid"
    ATMOSPHERIC_ONLY = "atmospheric_only"
    SURFACE_ONLY = "surface_only"


class SynopticPattern(str, Enum):
    """Padrões sinóticos identificados"""
    COLD_FRONT = "cold_front_approaching"
    WARM_FRONT = "warm_front_approaching"
    HIGH_PRESSURE = "high_pressure_system"
    LOW_PRESSURE = "low_pressure_system"
    STABLE = "stable"
    STRONG_VORTEX = "strong_vortex"
    MODERATE_VORTEX = "moderate_vortex"
    WEAK_PATTERN = "weak_pattern"


# === REQUEST SCHEMAS ===

class ForecastRequest(BaseModel):
    """
    Request para geração de previsão meteorológica
    """
    forecast_horizon: ForecastHorizon = Field(
        default=ForecastHorizon.HOUR_24,
        description="Horizonte de previsão desejado"
    )
    
    model_type: ModelType = Field(
        default=ModelType.HYBRID,
        description="Tipo de modelo a utilizar"
    )
    
    use_cache: bool = Field(
        default=True,
        description="Se deve usar cache de previsões"
    )
    
    force_refresh: bool = Field(
        default=False,
        description="Força nova previsão ignorando cache"
    )
    
    include_details: bool = Field(
        default=True,
        description="Se deve incluir detalhes do ensemble na resposta"
    )
    
    include_synoptic_analysis: bool = Field(
        default=True,
        description="Se deve incluir análise sinótica"
    )

    @validator('forecast_horizon')
    def validate_horizon(cls, v):
        if v not in ForecastHorizon:
            raise ValueError(f"Horizonte deve ser um dos: {list(ForecastHorizon)}")
        return v

    @root_validator
    def validate_request(cls, values):
        # Se force_refresh é True, use_cache deve ser False
        if values.get('force_refresh') and values.get('use_cache'):
            values['use_cache'] = False
        return values


class ModelMetricsRequest(BaseModel):
    """
    Request para obtenção de métricas do modelo
    """
    include_validation: bool = Field(
        default=True,
        description="Se deve incluir validação em tempo real"
    )
    
    use_cache: bool = Field(
        default=True,
        description="Se deve usar cache de métricas"
    )
    
    model_type: Optional[ModelType] = Field(
        default=None,
        description="Tipo específico de modelo (None = todos)"
    )


class RefreshModelRequest(BaseModel):
    """
    Request para refresh/reload do modelo
    """
    model_version: Optional[str] = Field(
        default=None,
        description="Versão específica do modelo (None = latest)"
    )
    
    validate_after_load: bool = Field(
        default=True,
        description="Se deve validar modelo após carregamento"
    )
    
    force_reload: bool = Field(
        default=False,
        description="Força reload mesmo se modelo já carregado"
    )


class HistoryRequest(BaseModel):
    """
    Request para histórico de previsões
    """
    start_date: datetime = Field(
        description="Data inicial para busca"
    )
    
    end_date: datetime = Field(
        description="Data final para busca"
    )
    
    include_accuracy: bool = Field(
        default=True,
        description="Se deve calcular accuracy vs dados reais"
    )
    
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=1000,
        description="Limite de registros (máximo 1000)"
    )
    
    model_type: Optional[ModelType] = Field(
        default=None,
        description="Filtrar por tipo de modelo"
    )

    @validator('end_date')
    def validate_dates(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError("Data final deve ser posterior à data inicial")
        return v

    @validator('start_date', 'end_date')
    def validate_date_range(cls, v):
        # Não permitir datas futuras para histórico
        if v > datetime.now():
            raise ValueError("Não é possível buscar histórico para datas futuras")
        return v


# === RESPONSE SCHEMAS ===

class SynopticAnalysis(BaseModel):
    """
    Análise sinótica dos dados atmosféricos
    """
    pressure_trend: Optional[float] = Field(
        description="Tendência de pressão nas últimas 24h (hPa)"
    )
    
    pressure_current: Optional[float] = Field(
        description="Pressão atmosférica atual (hPa)"
    )
    
    frontal_activity: Optional[SynopticPattern] = Field(
        description="Atividade frontal detectada"
    )
    
    temp_850_gradient: Optional[float] = Field(
        description="Gradiente de temperatura 850hPa (°C)"
    )
    
    upper_level_activity: Optional[SynopticPattern] = Field(
        description="Atividade em níveis superiores"
    )
    
    wind_500_max: Optional[float] = Field(
        description="Vento máximo 500hPa (km/h)"
    )


class EnsembleDetails(BaseModel):
    """
    Detalhes do ensemble híbrido
    """
    atmospheric_prediction: float = Field(
        description="Predição do componente atmosférico (mm)"
    )
    
    surface_prediction: float = Field(
        description="Predição do componente de superfície (mm)"
    )
    
    weighted_average: float = Field(
        description="Média ponderada dos componentes (mm)"
    )
    
    stacking_prediction: Optional[float] = Field(
        description="Predição do modelo de stacking (mm)"
    )
    
    ensemble_method: str = Field(
        description="Método de ensemble utilizado"
    )
    
    atmospheric_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Peso do componente atmosférico"
    )
    
    surface_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Peso do componente de superfície"
    )


class DataValidation(BaseModel):
    """
    Informações de validação dos dados de entrada
    """
    atmospheric_quality: float = Field(
        ge=0.0,
        le=1.0,
        description="Qualidade dos dados atmosféricos (0-1)"
    )
    
    surface_quality: float = Field(
        ge=0.0,
        le=1.0,
        description="Qualidade dos dados de superfície (0-1)"
    )
    
    sequence_completeness: float = Field(
        ge=0.0,
        le=1.0,
        description="Completude da sequência temporal (0-1)"
    )
    
    temporal_consistency: float = Field(
        ge=0.0,
        le=1.0,
        description="Consistência temporal dos dados (0-1)"
    )
    
    overall_quality: float = Field(
        ge=0.0,
        le=1.0,
        description="Qualidade geral dos dados (0-1)"
    )


class ForecastResponse(BaseModel):
    """
    Resposta da previsão meteorológica
    """
    timestamp: datetime = Field(
        description="Timestamp da previsão gerada"
    )
    
    forecast_timestamp: datetime = Field(
        description="Timestamp para o qual a previsão se aplica"
    )
    
    precipitation_mm: float = Field(
        ge=0.0,
        description="Precipitação prevista (mm)"
    )
    
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score de confiança da previsão (0-1)"
    )
    
    model_version: str = Field(
        description="Versão do modelo utilizado"
    )
    
    inference_time_ms: float = Field(
        ge=0.0,
        description="Tempo de inferência (milissegundos)"
    )
    
    forecast_horizon: str = Field(
        description="Horizonte de previsão aplicado"
    )
    
    # Dados opcionais
    ensemble_details: Optional[EnsembleDetails] = Field(
        description="Detalhes do ensemble híbrido"
    )
    
    synoptic_analysis: Optional[SynopticAnalysis] = Field(
        description="Análise sinótica dos dados atmosféricos"
    )
    
    data_validation: Optional[DataValidation] = Field(
        description="Validação dos dados de entrada"
    )
    
    # Metadados
    cached: bool = Field(
        description="Se a resposta veio do cache"
    )
    
    api_version: str = Field(
        default="3.2",
        description="Versão da API"
    )

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class ModelMetricsResponse(BaseModel):
    """
    Resposta das métricas do modelo
    """
    mae: float = Field(
        ge=0.0,
        description="Mean Absolute Error (mm)"
    )
    
    rmse: float = Field(
        ge=0.0,
        description="Root Mean Square Error (mm)"
    )
    
    accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Accuracy (0-1)"
    )
    
    model_version: str = Field(
        description="Versão do modelo"
    )
    
    training_date: datetime = Field(
        description="Data de treinamento do modelo"
    )
    
    # Métricas adicionais
    additional_metrics: Optional[Dict[str, float]] = Field(
        description="Métricas adicionais (MAE por componente, etc.)"
    )
    
    # Métricas ao vivo (se disponíveis)
    live_accuracy: Optional[float] = Field(
        description="Accuracy em tempo real"
    )
    
    live_mae: Optional[float] = Field(
        description="MAE em tempo real"
    )
    
    live_rmse: Optional[float] = Field(
        description="RMSE em tempo real"
    )
    
    data_freshness: Optional[float] = Field(
        description="Índice de frescor dos dados (0-1)"
    )
    
    model_drift: Optional[float] = Field(
        description="Índice de drift do modelo (0-1)"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp da coleta das métricas"
    )

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class ComponentInfo(BaseModel):
    """
    Informações de um componente do modelo híbrido
    """
    parameters: int = Field(
        ge=0,
        description="Número de parâmetros"
    )
    
    features: int = Field(
        ge=0,
        description="Número de features"
    )
    
    sequence_length: int = Field(
        ge=0,
        description="Comprimento da sequência temporal"
    )
    
    status: str = Field(
        description="Status do componente (loaded/not_loaded/error)"
    )


class ModelInfoResponse(BaseModel):
    """
    Resposta com informações detalhadas do modelo
    """
    status: str = Field(
        description="Status do modelo (loaded/not_loaded/error)"
    )
    
    model_type: str = Field(
        description="Tipo do modelo (hybrid_ensemble_lstm)"
    )
    
    phase: str = Field(
        description="Fase de implementação (3.2)"
    )
    
    target_accuracy: str = Field(
        description="Accuracy alvo (82-87%)"
    )
    
    # Componentes
    atmospheric_component: ComponentInfo = Field(
        description="Informações do componente atmosférico"
    )
    
    surface_component: ComponentInfo = Field(
        description="Informações do componente de superfície"
    )
    
    ensemble_info: Dict[str, Any] = Field(
        description="Informações do ensemble"
    )
    
    # Totais
    total_parameters: int = Field(
        ge=0,
        description="Total de parâmetros do modelo híbrido"
    )
    
    tensorflow_version: str = Field(
        description="Versão do TensorFlow"
    )
    
    # Metadados
    model_version: str = Field(
        description="Versão do modelo"
    )
    
    load_timestamp: Optional[datetime] = Field(
        description="Timestamp do carregamento"
    )
    
    cache_stats: Optional[Dict[str, Any]] = Field(
        description="Estatísticas do cache"
    )

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class RefreshModelResponse(BaseModel):
    """
    Resposta do refresh/reload do modelo
    """
    success: bool = Field(
        description="Se o refresh foi bem-sucedido"
    )
    
    model_version: str = Field(
        description="Versão do modelo carregada"
    )
    
    load_time_ms: float = Field(
        ge=0.0,
        description="Tempo de carregamento (milissegundos)"
    )
    
    components_loaded: List[str] = Field(
        description="Lista de componentes carregados"
    )
    
    validation: Optional[Dict[str, Any]] = Field(
        description="Resultado da validação pós-carregamento"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp da operação"
    )

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class HistoricalAccuracy(BaseModel):
    """
    Métricas de accuracy histórica
    """
    accuracy_percentage: float = Field(
        ge=0.0,
        le=100.0,
        description="Percentual de accuracy"
    )
    
    mae_mm: float = Field(
        ge=0.0,
        description="MAE histórico (mm)"
    )
    
    rmse_mm: float = Field(
        ge=0.0,
        description="RMSE histórico (mm)"
    )
    
    total_comparisons: int = Field(
        ge=0,
        description="Total de comparações realizadas"
    )
    
    accurate_forecasts: int = Field(
        ge=0,
        description="Número de previsões consideradas acuradas"
    )


class ForecastHistoryItem(BaseModel):
    """
    Item do histórico de previsões
    """
    timestamp: datetime = Field(
        description="Timestamp da previsão"
    )
    
    precipitation_mm: float = Field(
        ge=0.0,
        description="Precipitação prevista (mm)"
    )
    
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score de confiança"
    )
    
    model_version: str = Field(
        description="Versão do modelo"
    )
    
    actual_precipitation: Optional[float] = Field(
        description="Precipitação real observada (mm)"
    )
    
    error_mm: Optional[float] = Field(
        description="Erro absoluto (mm)"
    )

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class HistoryResponse(BaseModel):
    """
    Resposta do histórico de previsões
    """
    period: Dict[str, datetime] = Field(
        description="Período consultado (start/end)"
    )
    
    total_forecasts: int = Field(
        ge=0,
        description="Total de previsões no período"
    )
    
    forecasts: List[ForecastHistoryItem] = Field(
        description="Lista de previsões"
    )
    
    accuracy_analysis: Optional[HistoricalAccuracy] = Field(
        description="Análise de accuracy histórica"
    )

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


# === STATUS E HEALTH ===

class HealthStatus(BaseModel):
    """
    Status de saúde do sistema de previsão
    """
    healthy: bool = Field(
        description="Se o sistema está saudável"
    )
    
    model_loaded: bool = Field(
        description="Se o modelo está carregado"
    )
    
    test_prediction: Optional[float] = Field(
        description="Resultado de teste de predição (mm)"
    )
    
    test_inference_time_ms: Optional[float] = Field(
        description="Tempo do teste de inferência (ms)"
    )
    
    reason: Optional[str] = Field(
        description="Razão se não saudável"
    )
    
    timestamp: datetime = Field(
        description="Timestamp da verificação"
    )

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


# === ERROR RESPONSES ===

class ErrorDetail(BaseModel):
    """
    Detalhe de erro
    """
    type: str = Field(description="Tipo do erro")
    message: str = Field(description="Mensagem do erro")
    field: Optional[str] = Field(description="Campo relacionado ao erro")


class ErrorResponse(BaseModel):
    """
    Resposta de erro padronizada
    """
    error: bool = Field(default=True, description="Indica que é uma resposta de erro")
    error_type: str = Field(description="Tipo do erro")
    message: str = Field(description="Mensagem principal do erro")
    details: Optional[List[ErrorDetail]] = Field(description="Detalhes adicionais")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp do erro")
    request_id: Optional[str] = Field(description="ID da requisição para rastreamento")

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        } 