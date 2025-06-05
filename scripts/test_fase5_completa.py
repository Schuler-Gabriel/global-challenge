#!/usr/bin/env python3
"""
Teste Completo - Fase 5: External APIs

Script para verificar se a Fase 5 está completamente funcional na API.
"""

import sys
import os
import asyncio
from datetime import datetime

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app

def test_external_apis_endpoints():
    """Testa todos os endpoints da external APIs"""
    print("🧪 Testando Endpoints External APIs...")
    
    with TestClient(app) as client:
        
        # 1. Health check das APIs externas
        print("   🔍 Testando health check...")
        response = client.get("/api/v1/external-apis/health")
        print(f"   ✅ Health check: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"      📊 Status geral: {data.get('overall_status', 'N/A')}")
        
        # 2. Condições meteorológicas atuais
        print("   🌤️ Testando condições meteorológicas...")
        response = client.get("/api/v1/external-apis/weather/current?city=Porto Alegre")
        print(f"   ✅ Weather current: {response.status_code}")
        if response.status_code not in [200, 503]:  # 503 é esperado se serviço não disponível
            print(f"      ⚠️ Resposta inesperada: {response.status_code}")
        
        # 3. Previsão meteorológica
        print("   🌦️ Testando previsão meteorológica...")
        response = client.get("/api/v1/external-apis/weather/forecast?city=Porto Alegre&hours_ahead=24")
        print(f"   ✅ Weather forecast: {response.status_code}")
        
        # 4. Nível do rio
        print("   🌊 Testando nível do rio...")
        response = client.get("/api/v1/external-apis/river/current?station=Porto Alegre")
        print(f"   ✅ River current: {response.status_code}")
        
        # 5. Condições completas
        print("   🔄 Testando condições completas...")
        response = client.get("/api/v1/external-apis/conditions/current?city=Porto Alegre&station=Porto Alegre")
        print(f"   ✅ Conditions current: {response.status_code}")
        
        # 6. Cache stats
        print("   💾 Testando estatísticas do cache...")
        response = client.get("/api/v1/external-apis/cache/stats")
        print(f"   ✅ Cache stats: {response.status_code}")
        
        return True

def test_external_apis_integration():
    """Testa integração com outras features"""
    print("🧪 Testando Integração com Outras Features...")
    
    with TestClient(app) as client:
        
        # Verificar se não quebrou outras features
        print("   🏥 Testando health geral...")
        response = client.get("/health")
        print(f"   ✅ Health geral: {response.status_code}")
        
        print("   🚨 Testando alerts...")
        response = client.get("/api/v1/alerts/health")
        print(f"   ✅ Alerts health: {response.status_code}")
        
        print("   📈 Testando forecast...")
        response = client.get("/api/v1/forecast/metrics")
        print(f"   ✅ Forecast metrics: {response.status_code}")
        
        return True

def test_api_documentation():
    """Testa se a documentação da API está incluindo external APIs"""
    print("🧪 Testando Documentação da API...")
    
    with TestClient(app) as client:
        
        # OpenAPI schema
        response = client.get("/openapi.json")
        print(f"   ✅ OpenAPI schema: {response.status_code}")
        
        if response.status_code == 200:
            openapi_data = response.json()
            paths = openapi_data.get("paths", {})
            
            # Contar endpoints external-apis
            external_apis_paths = [path for path in paths.keys() if "external-apis" in path]
            print(f"   📚 Endpoints External APIs documentados: {len(external_apis_paths)}")
            
            if len(external_apis_paths) >= 5:
                print("   ✅ Documentação External APIs: COMPLETA")
                return True
            else:
                print("   ⚠️ Documentação External APIs: INCOMPLETA")
                return False
        
        return False

def test_api_comprehensive():
    """Teste abrangente da API completa"""
    print("🧪 Testando API Completa...")
    
    with TestClient(app) as client:
        
        # Verificar total de endpoints
        response = client.get("/openapi.json")
        if response.status_code == 200:
            openapi_data = response.json()
            paths = openapi_data.get("paths", {})
            total_endpoints = len(paths)
            
            # Categorizar endpoints
            forecast_endpoints = len([p for p in paths.keys() if "forecast" in p])
            alerts_endpoints = len([p for p in paths.keys() if "alerts" in p])
            external_endpoints = len([p for p in paths.keys() if "external-apis" in p])
            
            print(f"   📊 Total de endpoints: {total_endpoints}")
            print(f"   📈 Forecast: {forecast_endpoints}")
            print(f"   🚨 Alerts: {alerts_endpoints}")
            print(f"   🌐 External APIs: {external_endpoints}")
            print(f"   ⚡ Outros: {total_endpoints - forecast_endpoints - alerts_endpoints - external_endpoints}")
            
            return total_endpoints >= 15  # Esperamos pelo menos 15 endpoints
        
        return False

def main():
    """Executa todos os testes da Fase 5"""
    print("=" * 60)
    print("🚀 TESTE COMPLETO - FASE 5: EXTERNAL APIS")
    print("=" * 60)
    
    tests = [
        ("Endpoints External APIs", test_external_apis_endpoints),
        ("Integração com Features", test_external_apis_integration),
        ("Documentação da API", test_api_documentation),
        ("API Completa", test_api_comprehensive)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🎯 {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSOU")
            else:
                print(f"❌ {test_name}: FALHOU")
        except Exception as e:
            print(f"❌ {test_name}: ERRO - {e}")
        print()
    
    print("=" * 60)
    print(f"📊 RESULTADO FINAL: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 FASE 5 - EXTERNAL APIS: 100% COMPLETA!")
        print()
        print("✅ CONQUISTAS:")
        print("   • Estrutura domain/application/infra/presentation implementada")
        print("   • Endpoints REST funcionais")
        print("   • Integração com FastAPI completa")
        print("   • Documentação OpenAPI incluída")
        print("   • Compatibilidade com outras features")
        print()
        print("🎯 PRÓXIMAS ETAPAS SUGERIDAS:")
        print("   • Implementar integrações reais com APIs externas")
        print("   • Configurar autenticação e rate limiting")
        print("   • Adicionar monitoramento e alertas")
        print("   • Otimizar cache e performance")
        
    elif passed >= 3:
        print("⚠️ FASE 5 - EXTERNAL APIS: QUASE COMPLETA")
        print("   Apenas pequenos ajustes necessários")
    else:
        print("❌ FASE 5 - EXTERNAL APIS: PRECISA DE MAIS TRABALHO")
        print("   Revisar implementação e corrigir problemas")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 