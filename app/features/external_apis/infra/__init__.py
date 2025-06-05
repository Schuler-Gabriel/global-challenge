"""
Infrastructure Layer - External APIs

Implementações concretas dos serviços para integração com APIs externas.
"""

from .clients import CptecWeatherClient, GuaibaRiverClient
from .circuit_breaker import CircuitBreakerImpl
from .monitoring import ApiMonitoringService
from .cache import ExternalApiCacheService

__all__ = [
    "CptecWeatherClient",
    "GuaibaRiverClient",
    "CircuitBreakerImpl",
    "ApiMonitoringService",
    "ExternalApiCacheService"
] 