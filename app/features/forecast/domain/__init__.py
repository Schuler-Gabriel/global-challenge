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
    WeatherData,
    Forecast,
    ModelMetrics,
    WeatherCondition,
    PrecipitationLevel
)

# Services
from .services import (
    ForecastService,
    WeatherAnalysisService,
    ModelValidationService,
    ForecastConfiguration
)

# Repository Interfaces
from .repositories import (
    WeatherDataRepository,
    ForecastRepository,
    ModelRepository,
    CacheRepository,
    WeatherDataQuery,
    ForecastQuery,
    ConfigurableRepository,
    HealthCheckRepository,
    # Exceptions
    RepositoryError,
    DataNotFoundError,
    DataValidationError,
    ConnectionError,
    ModelNotFoundError,
    CacheError,
    # Utilities
    create_cache_key,
    validate_date_range,
    validate_limit
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
    "validate_limit"
]
