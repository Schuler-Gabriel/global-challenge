"""
Forecast Presentation Layer

Este módulo exporta os componentes da camada de apresentação da feature Forecast,
incluindo rotas da API, schemas e utilitários de apresentação.
"""

from .routes import router as forecast_router
from . import schemas

__all__ = [
    "forecast_router",
    "schemas"
]

