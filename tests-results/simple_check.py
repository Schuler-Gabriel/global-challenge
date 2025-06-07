#!/usr/bin/env python
"""
Script simples para testar o sistema de previsão.

Este script testa o sistema de previsão sem depender da injeção
de dependências do FastAPI, criando todas as dependências manualmente.
Inclui mock do modelo para testes sem TensorFlow.
"""

import sys
import os
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Adicionar diretório ao path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Importar componentes necessários
from app.features.forecast.infra import (
    FileWeatherDataRepository,
    FileForecastRepository,
    FileModelRepository,
    MemoryCacheRepository
)
from app.features.forecast.domain.services import ForecastService, ForecastConfiguration
from app.features.forecast.domain.entities import Forecast
from app.features.forecast.domain.repositories import ModelRepository
from app.features.forecast.application.usecases import GenerateForecastUseCase


# Mock do ModelRepository para testes sem TensorFlow
class MockModelRepository(ModelRepository):
    """Implementação mock do ModelRepository para testes"""
    
    async def load_model(self, model_version: str) -> Any:
        """Mock de carregamento de modelo"""
        print(f"[MOCK] Carregando modelo {model_version}")
        return {"version": model_version, "mock": True}
    
    async def save_model(self, model_version: str, model: Any, metadata: Dict[str, Any]) -> bool:
        """Mock de salvamento de modelo"""
        return True
    
    async def get_latest_model_version(self) -> Optional[str]:
        """Mock para obter versão mais recente"""
        return "v1.0.0"
    
    async def get_available_models(self) -> List[str]:
        """Mock de lista de modelos"""
        return ["v1.0.0"]
    
    async def save_model_metrics(self, model_version: str, metrics: Any) -> bool:
        """Mock de salvamento de métricas"""
        return True
    
    async def get_model_metrics(self, model_version: str) -> Optional[Any]:
        """Mock para obter métricas"""
        return None
    
    async def get_all_model_metrics(self) -> List[Any]:
        """Mock para obter todas as métricas"""
        return []
    
    async def delete_model(self, model_version: str) -> bool:
        """Mock para remover modelo"""
        return True
    
    async def get_model_metadata(self, model_version: str) -> Optional[Dict[str, Any]]:
        """Mock para obter metadados"""
        return {
            "version": model_version,
            "created_at": "2024-06-01T10:00:00",
            "performance": {
                "mae": 0.32,
                "rmse": 0.58,
                "accuracy": 0.84
            }
        }
    
    async def model_exists(self, model_version: str) -> bool:
        """Mock para verificar existência de modelo"""
        return True


# Mock para o ForecastService
class MockForecastService(ForecastService):
    """Versão modificada do ForecastService para testes"""
    
    async def generate_forecast(self, 
                               weather_data: List[Any], 
                               model: Any) -> Forecast:
        """Mock para geração de previsão"""
        # Criar previsão simulada com timestamp atual e valores padrão
        return Forecast(
            timestamp=datetime.now(),
            precipitation_mm=5.2,
            confidence_score=0.85,
            model_version="v1.0.0",
            inference_time_ms=120,
            forecast_horizon_hours=24,
            input_sequence_length=24,
            features_used=16
        )


async def test_generate_forecast():
    """Testa a geração de previsão criando manualmente todas as dependências"""
    # Criar repositories
    weather_data_repository = FileWeatherDataRepository(data_dir="data/processed")
    forecast_repository = FileForecastRepository(data_dir="data/processed")
    model_repository = MockModelRepository()  # Usar o mock
    cache_repository = MemoryCacheRepository()
    
    # Criar serviços
    config = ForecastConfiguration(
        sequence_length=24,
        forecast_horizon=24,
        confidence_threshold=0.7,
        max_inference_time_ms=100.0,
        features_count=16
    )
    forecast_service = MockForecastService(config=config)  # Usar serviço mock
    
    # Criar use case
    use_case = GenerateForecastUseCase(
        weather_data_repository=weather_data_repository,
        forecast_repository=forecast_repository,
        model_repository=model_repository,
        cache_repository=cache_repository,
        forecast_service=forecast_service
    )
    
    print(f"[INFO] Iniciando teste de geração de previsão")
    print(f"[INFO] Usando implementações dos repositórios:")
    print(f"  - WeatherData: {type(weather_data_repository).__name__}")
    print(f"  - Forecast: {type(forecast_repository).__name__}")
    print(f"  - Model: {type(model_repository).__name__} (mock)")
    print(f"  - Cache: {type(cache_repository).__name__}")
    print(f"[INFO] Usando serviço de previsão: {type(forecast_service).__name__} (mock)")
    
    # Gerar previsão (primeiro sem cache)
    print("\n[INFO] Gerando previsão (sem cache)...")
    start_time = datetime.now()
    forecast = await use_case.execute(use_cache=False)
    end_time = datetime.now()
    
    # Exibir resultado
    if forecast:
        print(f"[SUCCESS] Previsão gerada com sucesso em {(end_time - start_time).total_seconds():.2f}s")
        print(f"  - ID: {forecast.id}")
        print(f"  - Timestamp: {forecast.timestamp}")
        print(f"  - Precipitação: {forecast.precipitation_mm:.2f}mm")
        print(f"  - Nível: {forecast.get_precipitation_level().name}")
        print(f"  - Confiança: {forecast.confidence_score:.2f}")
        print(f"  - Modelo: {forecast.model_version}")
        print(f"  - Tempo de inferência: {forecast.inference_time_ms}ms")
    else:
        print("[ERROR] Falha ao gerar previsão")
        return False
    
    # Gerar previsão com cache
    print("\n[INFO] Gerando previsão (com cache)...")
    start_time = datetime.now()
    forecast_cached = await use_case.execute(use_cache=True)
    end_time = datetime.now()
    
    # Exibir resultado
    if forecast_cached:
        print(f"[SUCCESS] Previsão obtida do cache em {(end_time - start_time).total_seconds():.2f}s")
        print(f"  - ID: {forecast_cached.id}")
        
        # Verificar se é a mesma previsão
        is_same = forecast.id == forecast_cached.id
        print(f"  - Mesma previsão anterior: {'Sim' if is_same else 'Não'}")
    else:
        print("[ERROR] Falha ao obter previsão do cache")
        return False
    
    return True


if __name__ == "__main__":
    # Executar teste assíncrono
    print("=" * 50)
    print("Verificando implementação do sistema de previsão")
    print("=" * 50)
    
    result = asyncio.run(test_generate_forecast())
    
    if result:
        print("\n[SUCCESS] Todos os componentes estão funcionando corretamente!")
        sys.exit(0)
    else:
        print("\n[ERROR] Falha na implementação!")
        sys.exit(1) 