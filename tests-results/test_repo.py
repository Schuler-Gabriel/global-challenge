#!/usr/bin/env python
"""
Script para testar diretamente um repository
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta

# Adicionar diretório ao path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Importar implementações diretamente
from app.features.forecast.infra.repositories import (
    FileWeatherDataRepository,
    FileForecastRepository,
    FileModelRepository,
    MemoryCacheRepository
)


async def test_weather_repository():
    """Testa o repository de dados meteorológicos"""
    # Criar repository
    repo = FileWeatherDataRepository(data_dir="data/processed")
    print(f"[INFO] Testando {repo.__class__.__name__}")
    
    # Buscar dados recentes
    data = await repo.get_latest_data(count=10)
    print(f"[INFO] Dados recentes: {len(data)} registros")
    
    if data:
        print(f"[INFO] Primeiro registro: {data[0].timestamp}, {data[0].temperature}°C, {data[0].precipitation}mm")
    else:
        print("[WARNING] Nenhum dado encontrado")
    
    return len(data) > 0


async def test_forecast_repository():
    """Testa o repository de previsões"""
    # Criar repository
    repo = FileForecastRepository(data_dir="data/processed")
    print(f"[INFO] Testando {repo.__class__.__name__}")
    
    # Buscar previsão mais recente
    forecast = await repo.get_latest_forecast()
    
    if forecast:
        print(f"[INFO] Previsão mais recente: {forecast.timestamp}, {forecast.precipitation_mm}mm")
        print(f"[INFO] Confiança: {forecast.confidence_score}, Modelo: {forecast.model_version}")
    else:
        print("[WARNING] Nenhuma previsão encontrada")
    
    return forecast is not None


async def test_model_repository():
    """Testa o repository de modelos"""
    # Criar repository
    repo = FileModelRepository(models_dir="models")
    print(f"[INFO] Testando {repo.__class__.__name__}")
    
    # Listar modelos disponíveis
    models = await repo.get_available_models()
    print(f"[INFO] Modelos disponíveis: {models}")
    
    # Buscar versão mais recente
    latest = await repo.get_latest_model_version()
    
    if latest:
        print(f"[INFO] Modelo mais recente: {latest}")
        
        # Buscar metadados
        metadata = await repo.get_model_info(latest)
        print(f"[INFO] Metadados: {metadata}")
    else:
        print("[WARNING] Nenhum modelo encontrado")
    
    return latest is not None


async def test_cache_repository():
    """Testa o repository de cache"""
    # Criar repository
    repo = MemoryCacheRepository()
    print(f"[INFO] Testando {repo.__class__.__name__}")
    
    # Testar armazenamento e recuperação
    key = f"test_{datetime.now().timestamp()}"
    value = {"timestamp": datetime.now().isoformat(), "value": 42}
    
    print(f"[INFO] Armazenando em cache: {key}")
    await repo.set(key, value, ttl_seconds=60)
    
    # Recuperar
    retrieved = await repo.get(key)
    
    if retrieved:
        print(f"[INFO] Recuperado do cache: {retrieved}")
        print(f"[INFO] Valores iguais: {retrieved == value}")
    else:
        print("[ERROR] Falha ao recuperar do cache")
    
    return retrieved == value


async def main():
    """Função principal"""
    print("=" * 50)
    print("Testando repositories")
    print("=" * 50)
    
    # Testar repositories
    weather_ok = await test_weather_repository()
    print("\n" + "-" * 40 + "\n")
    
    forecast_ok = await test_forecast_repository()
    print("\n" + "-" * 40 + "\n")
    
    model_ok = await test_model_repository()
    print("\n" + "-" * 40 + "\n")
    
    cache_ok = await test_cache_repository()
    
    # Resumo
    print("\n" + "=" * 50)
    print("Resumo dos testes:")
    print(f"- WeatherRepository: {'✓' if weather_ok else '✗'}")
    print(f"- ForecastRepository: {'✓' if forecast_ok else '✗'}")
    print(f"- ModelRepository: {'✓' if model_ok else '✗'}")
    print(f"- CacheRepository: {'✓' if cache_ok else '✗'}")
    
    return all([weather_ok, forecast_ok, model_ok, cache_ok])


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1) 