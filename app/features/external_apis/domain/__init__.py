"""
Domain Layer - External APIs

Entidades de domínio e interfaces para APIs externas.
"""

from .entities import (
    RiverLevel, WeatherCondition, ExternalApiResponse, CircuitBreakerState, ApiHealthStatus,
    RiverStatus, WeatherSeverity, ApiStatus, RiskLevel, CircuitBreakerStatus
)
from .services import (
    WeatherApiService, RiverApiService, CircuitBreakerService,
    ExternalApiAggregatorService, ApiRetryService, ApiCacheService,
    ApiServiceConfig, ExternalApiError, ApiTimeoutError, ApiUnavailableError
)

__all__ = [
    # Entities
    "RiverLevel",
    "WeatherCondition", 
    "ExternalApiResponse",
    "CircuitBreakerState",
    "ApiHealthStatus",
    # Enums
    "RiverStatus",
    "WeatherSeverity", 
    "ApiStatus",
    "RiskLevel",
    "CircuitBreakerStatus",
    # Services
    "WeatherApiService",
    "RiverApiService",
    "CircuitBreakerService",
    "ExternalApiAggregatorService",
    "ApiRetryService",
    "ApiCacheService",
    # Config & Exceptions
    "ApiServiceConfig",
    "ExternalApiError",
    "ApiTimeoutError", 
    "ApiUnavailableError"
] 