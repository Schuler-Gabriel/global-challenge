#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast Application Layer

Camada de aplicação da feature de previsão meteorológica.
Contém use cases que orquestram a lógica de negócio.
"""

from .usecases import (
    GenerateForecastUseCase,
    GetModelMetricsUseCase,
    RefreshModelUseCase,
    GetForecastHistoryUseCase
)

__all__ = [
    "GenerateForecastUseCase",
    "GetModelMetricsUseCase", 
    "RefreshModelUseCase",
    "GetForecastHistoryUseCase"
]
