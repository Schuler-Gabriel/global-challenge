"""
Dependency Injection - External APIs Feature

Este módulo configura a injeção de dependências para a feature de APIs externas,
conectando implementações concretas às interfaces abstratas.
"""

import os
from typing import Dict, Any
from functools import lru_cache

from fastapi import Depends
import httpx

from app.core.dependencies import get_http_client

from .domain.services import (
    WeatherApiService,
    RiverApiService, 
    CircuitBreakerService,
    ExternalApiAggregatorService,
    ApiRetryService,
    MonitoringService
)
from .application.usecases import (
    GetCurrentConditionsUseCase,
    GetWeatherDataUseCase,
    GetRiverDataUseCase,
    HealthCheckUseCase
)
from .infra import (
    CptecWeatherClient,
    GuaibaRiverClient,
    CircuitBreakerImpl,
    ApiMonitoringService,
    ExternalApiCacheService
)


# Configurações
@lru_cache
def get_external_apis_config() -> Dict[str, Any]:
    """Obtém configuração para APIs externas"""
    return {
        # URLs das APIs
        "guaiba_api_url": os.environ.get(
            "GUAIBA_API_URL", 
            "https://nivelguaiba.com.br/portoalegre.1day.json"
        ),
        "cptec_api_url": os.environ.get(
            "CPTEC_API_URL",
            "https://www.cptec.inpe.br/api/forecast-input"
        ),
        
        # Timeouts
        "default_timeout": float(os.environ.get("API_TIMEOUT", "10.0")),
        "long_timeout": float(os.environ.get("API_LONG_TIMEOUT", "30.0")),
        
        # Retry
        "max_retries": int(os.environ.get("API_MAX_RETRIES", "3")),
        "backoff_factor": float(os.environ.get("API_BACKOFF_FACTOR", "2.0")),
        
        # Circuit Breaker
        "failure_threshold": int(os.environ.get("CIRCUIT_FAILURE_THRESHOLD", "5")),
        "success_threshold": int(os.environ.get("CIRCUIT_SUCCESS_THRESHOLD", "3")),
        "timeout_threshold": int(os.environ.get("CIRCUIT_TIMEOUT_THRESHOLD", "5000")),
        "circuit_open_duration": int(os.environ.get("CIRCUIT_OPEN_DURATION", "60")),
        
        # Cache TTL (segundos)
        "weather_cache_ttl": int(os.environ.get("WEATHER_CACHE_TTL", "300")),
        "river_cache_ttl": int(os.environ.get("RIVER_CACHE_TTL", "180")),
        "forecast_cache_ttl": int(os.environ.get("FORECAST_CACHE_TTL", "600")),
        "health_cache_ttl": int(os.environ.get("HEALTH_CACHE_TTL", "60")),
        
        # Redis
        "redis_url": os.environ.get("REDIS_URL"),
    }


# Cache Service
@lru_cache
def get_cache_service() -> ExternalApiCacheService:
    """Obtém serviço de cache"""
    return ExternalApiCacheService()


# Circuit Breaker Service  
@lru_cache
def get_circuit_breaker_service() -> CircuitBreakerImpl:
    """Obtém serviço de circuit breaker"""
    return CircuitBreakerImpl()


# Weather API Service
@lru_cache
def get_weather_api_service(
    http_client: httpx.AsyncClient = Depends(get_http_client)
) -> CptecWeatherClient:
    """Obtém serviço de API meteorológica"""
    return CptecWeatherClient(http_client=http_client)


# River API Service
@lru_cache
def get_river_api_service(
    http_client: httpx.AsyncClient = Depends(get_http_client)
) -> GuaibaRiverClient:
    """Obtém serviço de API do rio"""
    return GuaibaRiverClient(http_client=http_client)


# Monitoring Service
@lru_cache
def get_monitoring_service(
    weather_service: CptecWeatherClient = Depends(get_weather_api_service),
    river_service: GuaibaRiverClient = Depends(get_river_api_service)
) -> ApiMonitoringService:
    """Obtém serviço de monitoramento de APIs externas"""
    return ApiMonitoringService(
        weather_service=weather_service,
        river_service=river_service
    )


# Use Cases
@lru_cache
def get_current_conditions_usecase(
    weather_service: CptecWeatherClient = Depends(get_weather_api_service),
    river_service: GuaibaRiverClient = Depends(get_river_api_service)
) -> GetCurrentConditionsUseCase:
    """Obtém use case para condições atuais consolidadas"""
    return GetCurrentConditionsUseCase(
        weather_service=weather_service,
        river_service=river_service
    )


@lru_cache
def get_weather_data_usecase(
    weather_service: CptecWeatherClient = Depends(get_weather_api_service)
) -> GetWeatherDataUseCase:
    """Obtém use case para dados meteorológicos"""
    return GetWeatherDataUseCase(weather_service=weather_service)


@lru_cache
def get_river_data_usecase(
    river_service: GuaibaRiverClient = Depends(get_river_api_service)
) -> GetRiverDataUseCase:
    """Obtém use case para dados do rio"""
    return GetRiverDataUseCase(river_service=river_service)


@lru_cache
def get_health_check_usecase(
    weather_service: CptecWeatherClient = Depends(get_weather_api_service),
    river_service: GuaibaRiverClient = Depends(get_river_api_service),
    circuit_breaker: CircuitBreakerImpl = Depends(get_circuit_breaker_service)
) -> HealthCheckUseCase:
    """Obtém use case para health check das APIs externas"""
    return HealthCheckUseCase(
        weather_service=weather_service,
        river_service=river_service,
        circuit_breaker=circuit_breaker
    ) 