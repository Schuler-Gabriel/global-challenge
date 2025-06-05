"""
Forecast Application Layer

Este módulo exporta os use cases da feature Forecast, que coordenam
entre a camada de domínio e infraestrutura.

Use Cases:
- GenerateForecastUseCase: Gera nova previsão meteorológica
- GetModelMetricsUseCase: Recupera métricas do modelo ML
- RefreshModelUseCase: Atualiza o modelo para nova versão
"""

from .usecases import (
    GenerateForecastUseCase,
    GetModelMetricsUseCase,
    RefreshModelUseCase
)

__all__ = [
    "GenerateForecastUseCase",
    "GetModelMetricsUseCase",
    "RefreshModelUseCase"
]

