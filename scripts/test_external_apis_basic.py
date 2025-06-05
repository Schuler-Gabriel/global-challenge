#!/usr/bin/env python3
"""
Teste Básico - External APIs Feature

Script para verificar se a estrutura básica da Fase 5 está funcionando.
"""

import sys
import os

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_external_apis_structure():
    """Testa se a estrutura básica da external_apis está funcionando"""
    print("🧪 Testando Estrutura da External APIs...")
    
    try:
        # Testar importações básicas do domain
        from app.features.external_apis.domain.entities import (
            RiverLevel, WeatherCondition, ExternalApiResponse, ApiHealthStatus
        )
        print("   ✅ Domain entities: OK")
        
        from app.features.external_apis.domain.services import (
            WeatherApiService, RiverApiService, CircuitBreakerService
        )
        print("   ✅ Domain services: OK")
        
        # Testar criação de entidades
        from datetime import datetime
        
        river_level = RiverLevel(
            timestamp=datetime.now(),
            level_meters=2.5,
            station_name="Porto Alegre"
        )
        print(f"   ✅ RiverLevel criado: {river_level.level_meters}m - Status: {river_level.get_status().value}")
        
        weather = WeatherCondition(
            timestamp=datetime.now(),
            temperature=25.0,
            humidity=70.0,
            pressure=1013.0,
            wind_speed=5.0,
            wind_direction=180.0,
            description="Parcialmente nublado"
        )
        print(f"   ✅ WeatherCondition criado: {weather.temperature}°C - {weather.description}")
        
        api_response = ExternalApiResponse(
            success=True,
            data=river_level,
            timestamp=datetime.now(),
            api_name="GuaibaAPI",
            response_time_ms=150.0
        )
        print(f"   ✅ ExternalApiResponse criado: {api_response.api_name} - {api_response.response_time_ms}ms")
        
        health_status = ApiHealthStatus(
            api_name="CPTEC",
            status="healthy",
            response_time_ms=200.0,
            last_check_time=datetime.now()
        )
        print(f"   ✅ ApiHealthStatus criado: {health_status.api_name} - {health_status.status}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro na estrutura: {e}")
        return False

def test_external_apis_presentation():
    """Testa se a presentation layer está funcionando"""
    print("🧪 Testando Presentation Layer...")
    
    try:
        from app.features.external_apis.presentation.routes import router
        print("   ✅ Router importado: OK")
        
        from app.features.external_apis.presentation.schemas import (
            WeatherResponse, RiverLevelResponse
        )
        print("   ✅ Schemas importados: OK")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro na presentation: {e}")
        return False

def test_external_apis_application():
    """Testa se a application layer está funcionando"""
    print("🧪 Testando Application Layer...")
    
    try:
        from app.features.external_apis.application.usecases import (
            GetCurrentConditionsUseCase, GetWeatherDataUseCase, 
            GetRiverDataUseCase, HealthCheckUseCase
        )
        print("   ✅ Use cases importados: OK")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro na application: {e}")
        return False

def main():
    """Executa todos os testes básicos"""
    print("=" * 60)
    print("🚀 TESTE BÁSICO - FASE 5: EXTERNAL APIS")
    print("=" * 60)
    
    tests = [
        test_external_apis_structure,
        test_external_apis_presentation,
        test_external_apis_application
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"   ❌ Erro inesperado: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 RESULTADO: {passed}/{total} testes passaram")
    
    if passed == total:
        print("✅ FASE 5 - EXTERNAL APIS: ESTRUTURA BÁSICA COMPLETA!")
        print("🎯 Próximos passos:")
        print("   • Implementar integração com APIs reais")
        print("   • Configurar circuit breaker")
        print("   • Adicionar cache e retry logic")
        print("   • Testes de integração")
    else:
        print("❌ Alguns testes falharam. Verificar implementação.")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 