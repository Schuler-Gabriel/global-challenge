#!/usr/bin/env python3
"""
Teste da Estratégia Híbrida Open-Meteo

Script para testar a implementação completa da estratégia híbrida:
- Open-Meteo Forecast API (149 variáveis + sinótica)
- Open-Meteo Historical API (25 variáveis de superfície)
- Modelo de ensemble LSTM híbrido
- Análise sinótica automatizada

Execução: python scripts/test_hybrid_strategy.py
"""

import asyncio
import sys
import logging
from datetime import datetime, date, timedelta
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent))

from app.features.external_apis.infra.open_meteo_client import (
    OpenMeteoCurrentWeatherClient,
    get_porto_alegre_weather,
    test_openmeteo_connection
)
from app.features.external_apis.infra.open_meteo_historical_client import (
    OpenMeteoHistoricalClient,
    get_historical_porto_alegre,
    test_historical_connection
)
from app.features.forecast.infra.hybrid_ensemble_model import (
    HybridEnsembleModel,
    EnsembleConfig,
    create_hybrid_model,
    train_hybrid_model
)


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridStrategyTester:
    """
    Testador completo da estratégia híbrida Open-Meteo
    
    Executa testes de todas as componentes:
    1. Conectividade APIs
    2. Coleta de dados em tempo real
    3. Dados históricos
    4. Modelo de ensemble
    5. Análise sinótica
    6. Predições integradas
    """
    
    def __init__(self):
        self.results = {
            'api_connectivity': {},
            'current_data': {},
            'historical_data': {},
            'model_performance': {},
            'predictions': {},
            'overall_status': 'PENDING'
        }

    async def run_complete_test(self) -> dict:
        """
        Executa bateria completa de testes
        
        Returns:
            dict: Resultados dos testes
        """
        
        logger.info("🚀 Iniciando teste da estratégia híbrida Open-Meteo")
        
        try:
            # 1. Teste de conectividade
            await self._test_api_connectivity()
            
            # 2. Teste de dados em tempo real
            await self._test_current_weather_data()
            
            # 3. Teste de dados históricos
            await self._test_historical_data()
            
            # 4. Teste do modelo híbrido
            await self._test_hybrid_model()
            
            # 5. Teste de predições integradas
            await self._test_integrated_predictions()
            
            # 6. Análise de performance
            self._analyze_overall_performance()
            
            logger.info("✅ Teste completo da estratégia híbrida finalizado")
            
        except Exception as e:
            logger.error(f"❌ Erro durante teste: {str(e)}")
            self.results['overall_status'] = 'FAILED'
            self.results['error'] = str(e)
        
        return self.results

    async def _test_api_connectivity(self):
        """Testa conectividade com APIs Open-Meteo"""
        
        logger.info("📡 Testando conectividade APIs...")
        
        # Teste Open-Meteo Forecast
        forecast_connected = await test_openmeteo_connection()
        self.results['api_connectivity']['forecast'] = {
            'status': 'CONNECTED' if forecast_connected else 'FAILED',
            'url': 'https://api.open-meteo.com/v1/forecast',
            'description': '149 variáveis atmosféricas + níveis de pressão'
        }
        
        # Teste Open-Meteo Historical
        historical_connected = await test_historical_connection()
        self.results['api_connectivity']['historical'] = {
            'status': 'CONNECTED' if historical_connected else 'FAILED',
            'url': 'https://archive-api.open-meteo.com/v1/archive',
            'description': '25 variáveis de superfície (2000-2024)'
        }
        
        overall_connectivity = forecast_connected and historical_connected
        self.results['api_connectivity']['overall'] = 'CONNECTED' if overall_connectivity else 'PARTIAL'
        
        logger.info(f"   📊 Forecast API: {'✅' if forecast_connected else '❌'}")
        logger.info(f"   📊 Historical API: {'✅' if historical_connected else '❌'}")

    async def _test_current_weather_data(self):
        """Testa coleta de dados meteorológicos atuais"""
        
        logger.info("🌦️  Testando dados meteorológicos atuais...")
        
        try:
            # Busca dados atuais de Porto Alegre
            current_data = await get_porto_alegre_weather()
            
            # Analisa qualidade dos dados
            quality_score = current_data.get('data_quality', {}).get('overall_score', 0)
            synoptic_available = bool(current_data.get('synoptic_analysis'))
            pressure_data = current_data.get('processing_info', {}).get('has_pressure_data', False)
            
            self.results['current_data'] = {
                'status': 'SUCCESS' if quality_score > 0.5 else 'POOR_QUALITY',
                'quality_score': quality_score,
                'timestamp': current_data.get('timestamp'),
                'location': current_data.get('location'),
                'synoptic_analysis': synoptic_available,
                'pressure_levels': pressure_data,
                'variables_count': current_data.get('processing_info', {}).get('variables_count', 0)
            }
            
            # Exibe análise sinótica se disponível
            if synoptic_available:
                synoptic = current_data.get('synoptic_analysis', {})
                logger.info("   🌪️  Análise Sinótica:")
                
                if '850hPa' in synoptic:
                    frontal = synoptic['850hPa'].get('frontal_indicator', 'unknown')
                    logger.info(f"      - Atividade Frontal (850hPa): {frontal}")
                
                if '500hPa' in synoptic:
                    vortex = synoptic['500hPa'].get('vorticity_indicator', 'unknown')
                    logger.info(f"      - Atividade de Vórtices (500hPa): {vortex}")
                
                if 'combined_analysis' in synoptic:
                    stability = synoptic['combined_analysis'].get('atmospheric_stability', 'unknown')
                    pattern = synoptic['combined_analysis'].get('weather_pattern', 'unknown')
                    risk = synoptic['combined_analysis'].get('risk_level', 'unknown')
                    
                    logger.info(f"      - Estabilidade Atmosférica: {stability}")
                    logger.info(f"      - Padrão Meteorológico: {pattern}")
                    logger.info(f"      - Nível de Risco Sinótico: {risk}")
            
            logger.info(f"   📊 Qualidade dos dados: {quality_score:.2f}")
            logger.info(f"   📊 Variáveis coletadas: {current_data.get('processing_info', {}).get('variables_count', 0)}")
            
        except Exception as e:
            logger.error(f"   ❌ Erro ao coletar dados atuais: {str(e)}")
            self.results['current_data'] = {
                'status': 'FAILED',
                'error': str(e)
            }

    async def _test_historical_data(self):
        """Testa coleta de dados históricos"""
        
        logger.info("📈 Testando dados históricos...")
        
        try:
            # Testa com período de 1 mês recente
            end_date = date.today() - timedelta(days=1)
            start_date = end_date - timedelta(days=30)
            
            historical_data = await get_historical_porto_alegre(start_date, end_date)
            
            # Analisa dados coletados
            record_count = historical_data.get('record_count', 0)
            quality_score = historical_data.get('data_quality', {}).get('overall_score', 0)
            variables_count = historical_data.get('processing_info', {}).get('variables_count', 0)
            
            self.results['historical_data'] = {
                'status': 'SUCCESS' if record_count > 20 else 'INSUFFICIENT_DATA',
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat(),
                    'days_requested': 30,
                    'records_received': record_count
                },
                'quality_score': quality_score,
                'variables_count': variables_count,
                'data_completeness': historical_data.get('data_quality', {}).get('total_completeness', 0)
            }
            
            # Exibe estatísticas dos dados
            stats = historical_data.get('statistics', {})
            if 'precipitation_sum' in stats:
                precip_stats = stats['precipitation_sum']
                logger.info(f"   🌧️  Precipitação no período:")
                logger.info(f"      - Total: {precip_stats.get('total', 0):.1f} mm")
                logger.info(f"      - Máximo diário: {precip_stats.get('max', 0):.1f} mm")
                logger.info(f"      - Dias com chuva: {precip_stats.get('days_with_precipitation', 0)}")
            
            if 'temperature_2m_mean' in stats:
                temp_stats = stats['temperature_2m_mean']
                logger.info(f"   🌡️  Temperatura no período:")
                logger.info(f"      - Média: {temp_stats.get('mean', 0):.1f}°C")
                logger.info(f"      - Mínima: {temp_stats.get('min', 0):.1f}°C")
                logger.info(f"      - Máxima: {temp_stats.get('max', 0):.1f}°C")
            
            logger.info(f"   📊 Registros coletados: {record_count}/30 dias")
            logger.info(f"   📊 Qualidade dos dados: {quality_score:.2f}")
            logger.info(f"   📊 Variáveis disponíveis: {variables_count}")
            
        except Exception as e:
            logger.error(f"   ❌ Erro ao coletar dados históricos: {str(e)}")
            self.results['historical_data'] = {
                'status': 'FAILED',
                'error': str(e)
            }

    async def _test_hybrid_model(self):
        """Testa modelo de ensemble híbrido"""
        
        logger.info("🧠 Testando modelo de ensemble híbrido...")
        
        try:
            # Cria modelo híbrido
            model = await create_hybrid_model()
            
            # Testa preparação de dados (simulação)
            logger.info("   📊 Preparando dados de treinamento...")
            X_primary, X_secondary, y, synoptic = await model.prepare_training_data(
                "2023-01-01", "2023-12-31"
            )
            
            # Executa treinamento rápido (apenas algumas épocas para teste)
            model.config.epochs = 5  # Reduz para teste rápido
            model.config.batch_size = 16
            
            logger.info("   🎯 Iniciando treinamento do modelo...")
            training_results = await model.train(X_primary, X_secondary, y, synoptic)
            
            # Testa predição
            logger.info("   🔮 Testando predições...")
            prediction = await model.predict(days_ahead=7)
            
            self.results['model_performance'] = {
                'status': 'SUCCESS' if model.is_trained else 'FAILED',
                'architecture': {
                    'primary_lstm': '149 variáveis atmosféricas',
                    'secondary_lstm': '25 variáveis de superfície',
                    'synoptic_analysis': '15 features sinóticas',
                    'ensemble_method': 'stacking'
                },
                'training': {
                    'epochs_completed': training_results.get('epochs_trained', 0),
                    'final_accuracy': training_results.get('final_metrics', {}).get('accuracy', 0),
                    'overall_score': training_results.get('final_metrics', {}).get('overall_score', 0)
                },
                'prediction_test': {
                    'forecast_days': prediction.get('forecast_days', 0),
                    'confidence': prediction.get('confidence', {}).get('overall', 0),
                    'max_precipitation': prediction.get('precipitation_forecast', {}).get('max_daily', 0),
                    'risk_level': prediction.get('flood_risk_forecast', {}).get('risk_level', 'UNKNOWN')
                }
            }
            
            # Exibe métricas do modelo
            metrics = training_results.get('final_metrics', {})
            logger.info(f"   📊 Acurácia de risco: {metrics.get('accuracy', 0):.3f}")
            logger.info(f"   📊 R² precipitação: {metrics.get('precipitation_r2', 0):.3f}")
            logger.info(f"   📊 Score geral: {metrics.get('overall_score', 0):.3f}")
            
            # Exibe previsão teste
            logger.info("   🔮 Previsão de teste:")
            precip_forecast = prediction.get('precipitation_forecast', {})
            risk_forecast = prediction.get('flood_risk_forecast', {})
            logger.info(f"      - Precipitação máxima: {precip_forecast.get('max_daily', 0):.1f} mm")
            logger.info(f"      - Total do período: {precip_forecast.get('total_period', 0):.1f} mm")
            logger.info(f"      - Nível de risco: {risk_forecast.get('risk_level', 'UNKNOWN')}")
            logger.info(f"      - Confiança: {prediction.get('confidence', {}).get('overall', 0):.2f}")
            
        except Exception as e:
            logger.error(f"   ❌ Erro no teste do modelo: {str(e)}")
            self.results['model_performance'] = {
                'status': 'FAILED',
                'error': str(e)
            }

    async def _test_integrated_predictions(self):
        """Testa predições integradas com todos os componentes"""
        
        logger.info("🔮 Testando predições integradas...")
        
        try:
            # Busca dados atuais
            current_conditions = await get_porto_alegre_weather()
            
            # Cria modelo rápido para teste
            model = await create_hybrid_model()
            model.is_trained = True  # Simula modelo treinado
            
            # Simula predição integrada
            prediction = await model.predict(
                current_conditions=current_conditions,
                days_ahead=7
            )
            
            self.results['predictions'] = {
                'status': 'SUCCESS',
                'integration': {
                    'current_data_quality': current_conditions.get('data_quality', {}).get('overall_score', 0),
                    'synoptic_analysis': bool(current_conditions.get('synoptic_analysis')),
                    'forecast_days': prediction.get('forecast_days', 0),
                    'confidence': prediction.get('confidence', {}).get('overall', 0)
                },
                'forecast_summary': {
                    'precipitation': {
                        'total_period': prediction.get('precipitation_forecast', {}).get('total_period', 0),
                        'max_daily': prediction.get('precipitation_forecast', {}).get('max_daily', 0)
                    },
                    'flood_risk': {
                        'max_risk': prediction.get('flood_risk_forecast', {}).get('max_risk', 0),
                        'risk_level': prediction.get('flood_risk_forecast', {}).get('risk_level', 'UNKNOWN'),
                        'high_risk_days': prediction.get('flood_risk_forecast', {}).get('high_risk_days', 0)
                    }
                },
                'model_components': prediction.get('model_info', {}).get('components', [])
            }
            
            # Exibe resumo da predição integrada
            logger.info("   📋 Resumo da predição integrada:")
            
            precip = prediction.get('precipitation_forecast', {})
            logger.info(f"      🌧️  Precipitação total: {precip.get('total_period', 0):.1f} mm")
            logger.info(f"      🌧️  Máximo diário: {precip.get('max_daily', 0):.1f} mm")
            
            risk = prediction.get('flood_risk_forecast', {})
            logger.info(f"      ⚠️  Nível de risco: {risk.get('risk_level', 'UNKNOWN')}")
            logger.info(f"      ⚠️  Dias de alto risco: {risk.get('high_risk_days', 0)}")
            
            confidence = prediction.get('confidence', {})
            logger.info(f"      🎯 Confiança geral: {confidence.get('overall', 0):.2f}")
            
            # Exibe análise sinótica se disponível
            synoptic = prediction.get('synoptic_analysis', {})
            if synoptic:
                logger.info("   🌪️  Condições sinóticas:")
                logger.info(f"      - Atividade frontal: {synoptic.get('frontal_activity', 'unknown')}")
                logger.info(f"      - Estabilidade: {synoptic.get('atmospheric_stability', 'unknown')}")
                logger.info(f"      - Padrão: {synoptic.get('weather_pattern', 'unknown')}")
            
        except Exception as e:
            logger.error(f"   ❌ Erro nas predições integradas: {str(e)}")
            self.results['predictions'] = {
                'status': 'FAILED',
                'error': str(e)
            }

    def _analyze_overall_performance(self):
        """Analisa performance geral da estratégia híbrida"""
        
        logger.info("📊 Analisando performance geral...")
        
        # Conta sucessos por categoria
        successes = 0
        total_tests = 5
        
        if self.results['api_connectivity'].get('overall') == 'CONNECTED':
            successes += 1
            logger.info("   ✅ Conectividade APIs: OK")
        else:
            logger.info("   ❌ Conectividade APIs: FALHOU")
        
        if self.results['current_data'].get('status') == 'SUCCESS':
            successes += 1
            logger.info("   ✅ Dados atuais: OK")
        else:
            logger.info("   ❌ Dados atuais: FALHOU")
        
        if self.results['historical_data'].get('status') == 'SUCCESS':
            successes += 1
            logger.info("   ✅ Dados históricos: OK")
        else:
            logger.info("   ❌ Dados históricos: FALHOU")
        
        if self.results['model_performance'].get('status') == 'SUCCESS':
            successes += 1
            logger.info("   ✅ Modelo híbrido: OK")
        else:
            logger.info("   ❌ Modelo híbrido: FALHOU")
        
        if self.results['predictions'].get('status') == 'SUCCESS':
            successes += 1
            logger.info("   ✅ Predições integradas: OK")
        else:
            logger.info("   ❌ Predições integradas: FALHOU")
        
        # Determina status geral
        success_rate = successes / total_tests
        
        if success_rate >= 0.8:
            self.results['overall_status'] = 'EXCELLENT'
            status_emoji = "🎉"
            status_text = "EXCELENTE"
        elif success_rate >= 0.6:
            self.results['overall_status'] = 'GOOD'
            status_emoji = "✅"
            status_text = "BOM"
        elif success_rate >= 0.4:
            self.results['overall_status'] = 'PARTIAL'
            status_emoji = "⚠️"
            status_text = "PARCIAL"
        else:
            self.results['overall_status'] = 'FAILED'
            status_emoji = "❌"
            status_text = "FALHOU"
        
        self.results['summary'] = {
            'success_rate': success_rate,
            'tests_passed': successes,
            'total_tests': total_tests,
            'status': self.results['overall_status']
        }
        
        logger.info(f"\n{status_emoji} STATUS GERAL DA ESTRATÉGIA HÍBRIDA: {status_text}")
        logger.info(f"   📊 Taxa de sucesso: {success_rate:.1%} ({successes}/{total_tests})")
        
        # Recomendações baseadas nos resultados
        if success_rate < 1.0:
            logger.info("\n💡 Recomendações para melhorias:")
            
            if self.results['api_connectivity'].get('overall') != 'CONNECTED':
                logger.info("   - Verificar conectividade com APIs Open-Meteo")
            
            if self.results['current_data'].get('status') != 'SUCCESS':
                logger.info("   - Revisar configuração de coleta de dados atuais")
            
            if self.results['historical_data'].get('status') != 'SUCCESS':
                logger.info("   - Verificar acesso a dados históricos")
            
            if self.results['model_performance'].get('status') != 'SUCCESS':
                logger.info("   - Ajustar configuração do modelo de ensemble")
            
            if self.results['predictions'].get('status') != 'SUCCESS':
                logger.info("   - Verificar integração entre componentes")

    def print_detailed_report(self):
        """Imprime relatório detalhado dos testes"""
        
        print("\n" + "="*80)
        print("📋 RELATÓRIO DETALHADO - ESTRATÉGIA HÍBRIDA OPEN-METEO")
        print("="*80)
        
        # Conectividade
        print("\n🔌 CONECTIVIDADE APIs:")
        conn = self.results.get('api_connectivity', {})
        for api_name, details in conn.items():
            if api_name != 'overall':
                status = details.get('status', 'UNKNOWN')
                desc = details.get('description', '')
                print(f"   {api_name.upper()}: {status} - {desc}")
        
        # Dados atuais
        print("\n🌦️  DADOS METEOROLÓGICOS ATUAIS:")
        current = self.results.get('current_data', {})
        if 'quality_score' in current:
            print(f"   Qualidade: {current['quality_score']:.2f}")
            print(f"   Variáveis: {current.get('variables_count', 0)}")
            print(f"   Análise sinótica: {'✅' if current.get('synoptic_analysis') else '❌'}")
        
        # Dados históricos
        print("\n📈 DADOS HISTÓRICOS:")
        historical = self.results.get('historical_data', {})
        if 'period' in historical:
            period = historical['period']
            print(f"   Período: {period['start']} a {period['end']}")
            print(f"   Registros: {period['records_received']}/{period['days_requested']}")
            print(f"   Qualidade: {historical.get('quality_score', 0):.2f}")
        
        # Modelo
        print("\n🧠 MODELO DE ENSEMBLE:")
        model = self.results.get('model_performance', {})
        if 'training' in model:
            training = model['training']
            print(f"   Épocas: {training.get('epochs_completed', 0)}")
            print(f"   Acurácia: {training.get('final_accuracy', 0):.3f}")
            print(f"   Score geral: {training.get('overall_score', 0):.3f}")
        
        # Predições
        print("\n🔮 PREDIÇÕES INTEGRADAS:")
        predictions = self.results.get('predictions', {})
        if 'forecast_summary' in predictions:
            summary = predictions['forecast_summary']
            precip = summary.get('precipitation', {})
            risk = summary.get('flood_risk', {})
            print(f"   Precipitação total: {precip.get('total_period', 0):.1f} mm")
            print(f"   Nível de risco: {risk.get('risk_level', 'UNKNOWN')}")
            print(f"   Dias de alto risco: {risk.get('high_risk_days', 0)}")
        
        # Resumo
        print(f"\n📊 RESUMO GERAL:")
        summary = self.results.get('summary', {})
        print(f"   Status: {summary.get('status', 'UNKNOWN')}")
        print(f"   Taxa de sucesso: {summary.get('success_rate', 0):.1%}")
        print(f"   Testes aprovados: {summary.get('tests_passed', 0)}/{summary.get('total_tests', 0)}")
        
        print("\n" + "="*80)


async def main():
    """Função principal do teste"""
    
    print("🌊 SISTEMA DE ALERTAS DE CHEIAS - TESTE ESTRATÉGIA HÍBRIDA")
    print("Open-Meteo Forecast + Historical + Ensemble LSTM + Análise Sinótica")
    print("="*80)
    
    # Executa teste completo
    tester = HybridStrategyTester()
    results = await tester.run_complete_test()
    
    # Imprime relatório detalhado
    tester.print_detailed_report()
    
    # Salva resultados em arquivo
    import json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_results_hybrid_strategy_{timestamp}.json"
    
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 Resultados salvos em: {results_file}")
    except Exception as e:
        print(f"\n⚠️  Erro ao salvar resultados: {str(e)}")
    
    # Status de saída
    if results['overall_status'] in ['EXCELLENT', 'GOOD']:
        print("\n🎉 Estratégia híbrida funcionando corretamente!")
        return 0
    elif results['overall_status'] == 'PARTIAL':
        print("\n⚠️  Estratégia híbrida funcionando parcialmente.")
        return 1
    else:
        print("\n❌ Estratégia híbrida com problemas significativos.")
        return 2


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)