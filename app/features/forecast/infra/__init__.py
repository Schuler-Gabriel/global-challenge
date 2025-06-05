"""
Forecast Infrastructure Layer

Este módulo exporta as implementações concretas da camada de infraestrutura
para a feature de previsão meteorológica.

Classes:
- ModelLoader: Carregamento e gerenciamento de modelos TensorFlow
- ForecastModel: Wrapper para o modelo LSTM de previsão
- DataProcessor: Processamento de dados para o modelo
- FileWeatherDataRepository: Implementação de repository para dados meteorológicos
- FileForecastRepository: Implementação de repository para previsões
- FileModelRepository: Implementação de repository para modelos ML
- MemoryCacheRepository: Implementação de cache em memória
- RedisCacheRepository: Implementação de cache com Redis
"""

from .model_loader import ModelLoader
from .forecast_model import ForecastModel
from .data_processor import DataProcessor
from .repositories import (
    FileWeatherDataRepository,
    FileForecastRepository,
    FileModelRepository,
    MemoryCacheRepository,
    RedisCacheRepository
)

__all__ = [
    # Componentes de modelo
    "ModelLoader",
    "ForecastModel",
    "DataProcessor",
    
    # Implementações de repositories
    "FileWeatherDataRepository",
    "FileForecastRepository",
    "FileModelRepository",
    "MemoryCacheRepository",
    "RedisCacheRepository"
]
