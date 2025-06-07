#!/usr/bin/env python
"""
Script para testar a implementação do sistema de previsão.

Este script testa diretamente o use case de geração de previsão, sem
passar pela API, verificando se todos os componentes estão funcionando
corretamente juntos.
"""

import sys
import os
import asyncio
from datetime import datetime

# Adicionar diretório ao path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Importar componentes
from app.features.forecast.dependencies import get_generate_forecast_usecase


async def test_generate_forecast():
    """Testa a geração de previsão usando o use case"""
    # Obter use case via injeção de dependências
    use_case = get_generate_forecast_usecase()
    
    print(f"[INFO] Iniciando teste de geração de previsão")
    print(f"[INFO] Usando implementações dos repositórios:")
    print(f"  - WeatherData: {type(use_case.weather_data_repository).__name__}")
    print(f"  - Forecast: {type(use_case.forecast_repository).__name__}")
    print(f"  - Model: {type(use_case.model_repository).__name__}")
    print(f"  - Cache: {type(use_case.cache_repository).__name__}")
    print(f"[INFO] Usando serviço de previsão: {type(use_case.forecast_service).__name__}")
    
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