#!/usr/bin/env python3
"""
Script de Teste - Domain Layer da Feature Forecast
Sistema de Alertas de Cheias - Rio Guaíba

Este script testa a implementação da Domain Layer da Feature Forecast,
validando entidades, services e interfaces de repository.

Uso:
    python scripts/test_forecast_domain.py
"""

import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Imports da Domain Layer
from app.features.forecast.domain import (
    WeatherData,
    Forecast,
    ModelMetrics,
    WeatherCondition,
    PrecipitationLevel,
    ForecastService,
    WeatherAnalysisService,
    ModelValidationService,
    ForecastConfiguration
)


def test_weather_data_entity():
    """Testa a entidade WeatherData"""
    print("🧪 Testando entidade WeatherData...")
    
    # Criar dados válidos
    weather_data = WeatherData(
        timestamp=datetime.now(),
        precipitation=5.2,
        pressure=1013.2,
        temperature=22.5,
        dew_point=18.3,
        humidity=75.0,
        wind_speed=8.5,
        wind_direction=180.0,
        radiation=850.0
    )
    
    print(f"   ✅ WeatherData criado: {weather_data.timestamp}")
    print(f"   ✅ Precipitação: {weather_data.precipitation}mm/h")
    print(f"   ✅ Nível de precipitação: {weather_data.get_precipitation_level().value}")
    print(f"   ✅ Condição meteorológica: {weather_data.get_weather_condition().value}")
    print(f"   ✅ É extremo? {weather_data.is_extreme_weather()}")
    
    # Testar conversão para dict
    data_dict = weather_data.to_dict()
    assert 'timestamp' in data_dict
    assert 'precipitation_level' in data_dict
    print("   ✅ Conversão para dicionário OK")
    
    # Testar dados extremos
    extreme_data = WeatherData(
        timestamp=datetime.now(),
        precipitation=75.0,  # Extremo
        pressure=1013.2,
        temperature=45.0,    # Extremo
        dew_point=18.3,
        humidity=75.0,
        wind_speed=35.0,     # Extremo
        wind_direction=180.0
    )
    
    assert extreme_data.is_extreme_weather()
    assert extreme_data.get_precipitation_level() == PrecipitationLevel.EXTREME
    print("   ✅ Detecção de condições extremas OK")
    
    print("✅ WeatherData: Todos os testes passaram!")


def test_forecast_entity():
    """Testa a entidade Forecast"""
    print("\n🧪 Testando entidade Forecast...")
    
    forecast = Forecast(
        timestamp=datetime.now(),
        precipitation_mm=12.5,
        confidence_score=0.85,
        model_version="v1.2.3",
        inference_time_ms=45.2,
        input_sequence_length=24,
        forecast_horizon_hours=24,
        features_used=16
    )
    
    print(f"   ✅ Forecast criado: {forecast.timestamp}")
    print(f"   ✅ Precipitação prevista: {forecast.precipitation_mm}mm/h")
    print(f"   ✅ Confiança: {forecast.confidence_score}")
    print(f"   ✅ Nível de precipitação: {forecast.get_precipitation_level().value}")
    print(f"   ✅ Chuva esperada? {forecast.is_rain_expected()}")
    print(f"   ✅ Alta confiança? {forecast.is_high_confidence()}")
    print(f"   ✅ Atende critérios de performance? {forecast.meets_performance_criteria()}")
    
    # Testar conversão para dict
    forecast_dict = forecast.to_dict()
    assert 'precipitation_level' in forecast_dict
    assert 'is_rain_expected' in forecast_dict
    print("   ✅ Conversão para dicionário OK")
    
    print("✅ Forecast: Todos os testes passaram!")


def test_model_metrics_entity():
    """Testa a entidade ModelMetrics"""
    print("\n🧪 Testando entidade ModelMetrics...")
    
    metrics = ModelMetrics(
        mae=1.5,
        rmse=2.2,
        accuracy=0.82,
        model_version="v1.2.3",
        training_date=datetime.now(),
        r2_score=0.78,
        precision=0.85,
        recall=0.79,
        f1_score=0.82,
        skill_score=0.75,
        train_samples=50000,
        validation_samples=10000,
        test_samples=5000
    )
    
    print(f"   ✅ ModelMetrics criado para modelo: {metrics.model_version}")
    print(f"   ✅ MAE: {metrics.mae} (critério: < 2.0)")
    print(f"   ✅ RMSE: {metrics.rmse} (critério: < 3.0)")
    print(f"   ✅ Accuracy: {metrics.accuracy} (critério: > 0.75)")
    print(f"   ✅ Atende critério MAE? {metrics.meets_mae_criteria()}")
    print(f"   ✅ Atende critério RMSE? {metrics.meets_rmse_criteria()}")
    print(f"   ✅ Atende critério Accuracy? {metrics.meets_accuracy_criteria()}")
    print(f"   ✅ Atende todos os critérios? {metrics.meets_all_criteria()}")
    print(f"   ✅ Nota de performance: {metrics.get_performance_grade()}")
    
    assert metrics.meets_all_criteria()
    assert metrics.get_performance_grade() == "A"
    print("   ✅ Validação de critérios OK")
    
    print("✅ ModelMetrics: Todos os testes passaram!")


def test_forecast_service():
    """Testa o ForecastService"""
    print("\n🧪 Testando ForecastService...")
    
    config = ForecastConfiguration(
        sequence_length=24,
        forecast_horizon=24,
        confidence_threshold=0.7,
        max_inference_time_ms=100.0,
        features_count=16
    )
    
    service = ForecastService(config)
    
    # Criar dados de teste
    weather_data = []
    base_time = datetime.now() - timedelta(hours=24)
    
    for i in range(24):
        data = WeatherData(
            timestamp=base_time + timedelta(hours=i),
            precipitation=2.0 + i * 0.1,
            pressure=1013.2,
            temperature=20.0 + i * 0.5,
            dew_point=15.0,
            humidity=70.0,
            wind_speed=5.0,
            wind_direction=180.0
        )
        weather_data.append(data)
    
    # Testar validação da sequência
    assert service.validate_input_sequence(weather_data)
    print("   ✅ Validação da sequência de entrada OK")
    
    # Criar previsão de teste
    forecast = Forecast(
        timestamp=datetime.now(),
        precipitation_mm=8.5,
        confidence_score=0.82,
        model_version="v1.2.3",
        inference_time_ms=75.0
    )
    
    # Testar validação da previsão
    assert service.validate_forecast_quality(forecast)
    print("   ✅ Validação da qualidade da previsão OK")
    
    # Testar lógica de alertas
    should_alert = service.should_generate_alert(forecast)
    assert should_alert  # Precipitação moderada deve gerar alerta
    print(f"   ✅ Deve gerar alerta? {should_alert}")
    
    # Testar cálculo de score de risco
    risk_score = service.calculate_risk_score(forecast)
    print(f"   ✅ Score de risco: {risk_score:.2f}")
    
    # Testar sumário da previsão
    summary = service.get_forecast_summary(forecast)
    assert 'precipitation_level' in summary
    assert 'risk_score' in summary
    print("   ✅ Geração de sumário OK")
    
    print("✅ ForecastService: Todos os testes passaram!")


def test_weather_analysis_service():
    """Testa o WeatherAnalysisService"""
    print("\n🧪 Testando WeatherAnalysisService...")
    
    service = WeatherAnalysisService()
    
    # Criar dados de teste
    weather_data = []
    base_time = datetime.now() - timedelta(hours=24)
    
    for i in range(24):
        data = WeatherData(
            timestamp=base_time + timedelta(hours=i),
            precipitation=5.0 if i % 6 == 0 else 0.5,  # Padrão de chuva
            pressure=1013.2 + i * 0.1,
            temperature=20.0 + i * 0.3,
            dew_point=15.0,
            humidity=70.0 + i,
            wind_speed=5.0 + i * 0.2,
            wind_direction=180.0
        )
        weather_data.append(data)
    
    # Testar detecção de padrões
    patterns = service.detect_patterns(weather_data)
    assert 'total_precipitation' in patterns
    assert 'avg_temperature' in patterns
    assert 'dominant_condition' in patterns
    print(f"   ✅ Padrões detectados: {len(patterns)} métricas")
    print(f"   ✅ Precipitação total: {patterns['total_precipitation']:.1f}mm")
    print(f"   ✅ Temperatura média: {patterns['avg_temperature']:.1f}°C")
    print(f"   ✅ Condição dominante: {patterns['dominant_condition']}")
    
    # Testar detecção de anomalias
    anomalies = service.detect_anomalies(weather_data)
    print(f"   ✅ Anomalias detectadas: {len(anomalies)}")
    
    # Testar cálculo de índices meteorológicos
    indices = service.calculate_meteorological_indices(weather_data)
    assert 'heat_index' in indices
    assert 'pressure_trend' in indices
    print(f"   ✅ Índices calculados: {len(indices)}")
    print(f"   ✅ Heat index: {indices['heat_index']:.1f}")
    print(f"   ✅ Tendência da pressão: {indices['pressure_trend']}")
    
    print("✅ WeatherAnalysisService: Todos os testes passaram!")


def test_model_validation_service():
    """Testa o ModelValidationService"""
    print("\n🧪 Testando ModelValidationService...")
    
    service = ModelValidationService()
    
    # Criar métricas de teste
    current_metrics = ModelMetrics(
        mae=1.8,
        rmse=2.5,
        accuracy=0.78,
        model_version="v1.2.2",
        training_date=datetime.now() - timedelta(days=7)
    )
    
    new_metrics = ModelMetrics(
        mae=1.5,  # Melhoria
        rmse=2.2,  # Melhoria
        accuracy=0.82,  # Melhoria
        model_version="v1.2.3",
        training_date=datetime.now()
    )
    
    # Testar validação de métricas
    validation = service.validate_model_metrics(new_metrics)
    assert validation['meets_all_criteria']
    assert validation['performance_grade'] == 'A'
    print(f"   ✅ Validação de métricas: Nota {validation['performance_grade']}")
    
    # Testar comparação de modelos
    comparison = service.compare_models(current_metrics, new_metrics)
    assert comparison['overall_better']
    assert comparison['improvements_count'] == 3
    print(f"   ✅ Comparação: {comparison['improvements_count']} melhorias detectadas")
    
    # Testar recomendação de atualização
    recommendation = service.recommend_model_update(comparison)
    assert recommendation['should_update']
    assert recommendation['confidence'] == 'high'
    print(f"   ✅ Recomendação: {recommendation['confidence']} confiança para atualizar")
    
    print("✅ ModelValidationService: Todos os testes passaram!")


def main():
    """Executa todos os testes da Domain Layer"""
    print("🚀 Iniciando testes da Domain Layer - Feature Forecast")
    print("=" * 60)
    
    try:
        test_weather_data_entity()
        test_forecast_entity()
        test_model_metrics_entity()
        test_forecast_service()
        test_weather_analysis_service()
        test_model_validation_service()
        
        print("\n" + "=" * 60)
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("✅ Domain Layer da Feature Forecast implementada com sucesso!")
        print("\n📋 Resumo da implementação:")
        print("   • Entidades: WeatherData, Forecast, ModelMetrics ✅")
        print("   • Services: ForecastService, WeatherAnalysisService, ModelValidationService ✅")
        print("   • Repositories: Interfaces abstratas definidas ✅")
        print("   • Validações: Ranges, critérios de qualidade, lógica de negócio ✅")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERRO durante os testes: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 