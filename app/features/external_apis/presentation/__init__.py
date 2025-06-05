"""
Presentation Layer - External APIs

Camada de apresentação para APIs externas.
"""

from .routes import external_apis_router
from .schemas import (
    WeatherResponse,
    RiverLevelResponse,
    CurrentConditionsResponse,
    ApiHealthResponse
)

__all__ = [
    "external_apis_router",
    "WeatherResponse",
    "RiverLevelResponse", 
    "CurrentConditionsResponse",
    "ApiHealthResponse"
] 