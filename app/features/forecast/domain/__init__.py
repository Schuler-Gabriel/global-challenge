"""
Forecast Domain Layer

Este módulo exporta todas as entidades, serviços e interfaces de repository
da camada de domínio da feature Forecast.

A Domain Layer contém:
- Entities: Objetos de negócio puros (WeatherData, Forecast, ModelMetrics)
- Services: Lógica de negócio complexa (ForecastService, WeatherAnalysisService)
- Repositories: Interfaces abstratas para acesso a dados
"""

# Entities
from .entities import (
    Forecast,
    ModelMetrics,
    PrecipitationLevel,
    WeatherCondition,
    WeatherData,
)

# Repository Interfaces
from .repositories import (  # Exceptions; Utilities
    CacheError,
    CacheRepository,
    ConfigurableRepository,
    ConnectionError,
    DataNotFoundError,
    DataValidationError,
    ForecastQuery,
    ForecastRepository,
    HealthCheckRepository,
    ModelNotFoundError,
    ModelRepository,
    RepositoryError,
    WeatherDataQuery,
    WeatherDataRepository,
    create_cache_key,
    validate_date_range,
    validate_limit,
)

# Services
from .services import (
    ForecastConfiguration,
    ForecastService,
    ModelValidationService,
    WeatherAnalysisService,
)

__all__ = [
    # Entities
    "WeatherData",
    "Forecast",
    "ModelMetrics",
    "WeatherCondition",
    "PrecipitationLevel",
    # Services
    "ForecastService",
    "WeatherAnalysisService",
    "ModelValidationService",
    "ForecastConfiguration",
    # Repository Interfaces
    "WeatherDataRepository",
    "ForecastRepository",
    "ModelRepository",
    "CacheRepository",
    # Query Objects
    "WeatherDataQuery",
    "ForecastQuery",
    # Protocols
    "ConfigurableRepository",
    "HealthCheckRepository",
    # Exceptions
    "RepositoryError",
    "DataNotFoundError",
    "DataValidationError",
    "ConnectionError",
    "ModelNotFoundError",
    "CacheError",
    # Utilities
    "create_cache_key",
    "validate_date_range",
    "validate_limit",
]
