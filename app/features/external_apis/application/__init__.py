"""
Application Layer - External APIs

Use cases para integração com APIs externas.
"""

from .usecases import (
    GetCurrentConditionsUseCase,
    GetWeatherDataUseCase,
    GetRiverDataUseCase,
    HealthCheckUseCase
)

__all__ = [
    "GetCurrentConditionsUseCase",
    "GetWeatherDataUseCase",
    "GetRiverDataUseCase",
    "HealthCheckUseCase"
] 