"""
Dependency Injection - Forecast Feature

Este módulo configura a injeção de dependências para a feature de previsão
meteorológica, conectando implementações concretas às interfaces abstratas
e fornecendo instâncias prontas para uso nos endpoints da API.
"""

import os
from typing import Dict, Any, Optional
from functools import lru_cache

from fastapi import Depends

from .domain.repositories import (
    WeatherDataRepository, 
    ForecastRepository, 
    ModelRepository, 
    CacheRepository
)
from .domain.services import (
    ForecastService, 
    ModelValidationService, 
    ForecastConfiguration
)
from .application.usecases import (
    GenerateForecastUseCase,
    GetModelMetricsUseCase,
    RefreshModelUseCase
)
from .infra import (
    FileWeatherDataRepository,
    FileForecastRepository,
    FileModelRepository,
    MemoryCacheRepository,
    RedisCacheRepository,
    ModelLoader,
    ForecastModel,
    DataProcessor
)


# Configurações
@lru_cache
def get_forecast_config() -> ForecastConfiguration:
    """Obtém configuração para previsão meteorológica"""
    # Em produção, poderia ler de variáveis de ambiente ou arquivo de configuração
    return ForecastConfiguration(
        sequence_length=24,
        forecast_horizon=24,
        confidence_threshold=0.7,
        max_inference_time_ms=100.0,
        features_count=16
    )


# Repositories
@lru_cache
def get_weather_data_repository() -> WeatherDataRepository:
    """Obtém implementação de WeatherDataRepository"""
    data_dir = os.environ.get("WEATHER_DATA_DIR", "data/processed")
    return FileWeatherDataRepository(data_dir=data_dir)


@lru_cache
def get_forecast_repository() -> ForecastRepository:
    """Obtém implementação de ForecastRepository"""
    data_dir = os.environ.get("FORECAST_DATA_DIR", "data/processed")
    return FileForecastRepository(data_dir=data_dir)


@lru_cache
def get_model_repository() -> ModelRepository:
    """Obtém implementação de ModelRepository"""
    models_dir = os.environ.get("MODELS_DIR", "models")
    return FileModelRepository(models_dir=models_dir)


@lru_cache
def get_cache_repository() -> CacheRepository:
    """Obtém implementação de CacheRepository"""
    # Verificar se Redis está configurado
    redis_url = os.environ.get("REDIS_URL")
    
    if redis_url:
        # Usar Redis se disponível
        return RedisCacheRepository(redis_url=redis_url)
    else:
        # Fallback para cache em memória
        return MemoryCacheRepository()


# Serviços de domínio
@lru_cache
def get_forecast_service(
    config: ForecastConfiguration = Depends(get_forecast_config)
) -> ForecastService:
    """Obtém serviço de previsão"""
    return ForecastService(config=config)


@lru_cache
def get_model_validation_service() -> ModelValidationService:
    """Obtém serviço de validação de modelo"""
    return ModelValidationService()


# Componentes de infraestrutura
@lru_cache
def get_model_loader() -> ModelLoader:
    """Obtém loader de modelos"""
    models_dir = os.environ.get("MODELS_DIR", "models")
    return ModelLoader(models_dir=models_dir)


@lru_cache
def get_forecast_model() -> ForecastModel:
    """Obtém modelo de previsão"""
    return ForecastModel()


@lru_cache
def get_data_processor() -> DataProcessor:
    """Obtém processador de dados"""
    data_dir = os.environ.get("PROCESSED_DATA_DIR", "data/processed")
    return DataProcessor(data_dir=data_dir)


# Use Cases
@lru_cache
def get_generate_forecast_usecase(
    weather_data_repository: WeatherDataRepository = Depends(get_weather_data_repository),
    forecast_repository: ForecastRepository = Depends(get_forecast_repository),
    model_repository: ModelRepository = Depends(get_model_repository),
    cache_repository: CacheRepository = Depends(get_cache_repository),
    forecast_service: ForecastService = Depends(get_forecast_service)
) -> GenerateForecastUseCase:
    """Obtém use case para geração de previsão"""
    return GenerateForecastUseCase(
        weather_data_repository=weather_data_repository,
        forecast_repository=forecast_repository,
        model_repository=model_repository,
        cache_repository=cache_repository,
        forecast_service=forecast_service
    )


@lru_cache
def get_model_metrics_usecase(
    model_repository: ModelRepository = Depends(get_model_repository),
    cache_repository: CacheRepository = Depends(get_cache_repository),
    model_validation_service: ModelValidationService = Depends(get_model_validation_service)
) -> GetModelMetricsUseCase:
    """Obtém use case para métricas de modelo"""
    return GetModelMetricsUseCase(
        model_repository=model_repository,
        cache_repository=cache_repository,
        model_validation_service=model_validation_service
    )


@lru_cache
def get_refresh_model_usecase(
    model_repository: ModelRepository = Depends(get_model_repository),
    cache_repository: CacheRepository = Depends(get_cache_repository),
    model_validation_service: ModelValidationService = Depends(get_model_validation_service)
) -> RefreshModelUseCase:
    """Obtém use case para atualização de modelo"""
    return RefreshModelUseCase(
        model_repository=model_repository,
        cache_repository=cache_repository,
        model_validation_service=model_validation_service
    ) 