#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast Application Layer - Use Cases

Casos de uso da feature de previsão meteorológica integrada com modelo híbrido.
Implementação seguindo Clean Architecture com inversão de dependências.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.core.exceptions import (
    ModelLoadError, 
    PredictionError, 
    ValidationError,
    InfrastructureError
)
from app.features.forecast.domain.entities import (
    Forecast, 
    ModelMetrics, 
    WeatherData,
    ForecastConfiguration
)
from app.features.forecast.domain.repositories import (
    WeatherDataRepository,
    ForecastRepository, 
    ModelRepository,
    CacheRepository
)
from app.features.forecast.domain.services import (
    ForecastService,
    WeatherAnalysisService,
    ModelValidationService
)

logger = logging.getLogger(__name__)


class GenerateForecastUseCase:
    """
    Use Case principal para geração de previsões meteorológicas
    
    Integra modelo híbrido LSTM com dados Open-Meteo para previsão de 24h.
    Implementa cache inteligente e validação de qualidade dos dados.
    """

    def __init__(
        self,
        weather_data_repository: WeatherDataRepository,
        forecast_repository: ForecastRepository,
        model_repository: ModelRepository,
        cache_repository: CacheRepository,
        forecast_service: ForecastService,
        weather_analysis_service: WeatherAnalysisService,
        config: ForecastConfiguration
    ):
        self.weather_data_repo = weather_data_repository
        self.forecast_repo = forecast_repository
        self.model_repo = model_repository
        self.cache_repo = cache_repository
        self.forecast_service = forecast_service
        self.weather_analysis_service = weather_analysis_service
        self.config = config
        
        self._model_loaded = False
        self._last_model_check = None

    async def execute(
        self, 
        use_cache: bool = True,
        force_refresh: bool = False,
        forecast_hours: int = 24
    ) -> Forecast:
        """
        Gera previsão meteorológica usando modelo híbrido
        
        Args:
            use_cache: Se deve usar cache de previsões
            force_refresh: Força nova previsão ignorando cache
            forecast_hours: Horas para previsão (padrão 24h)
            
        Returns:
            Forecast: Previsão com dados atmosféricos e métricas
            
        Raises:
            PredictionError: Erro na geração da previsão
            ValidationError: Dados de entrada inválidos
        """
        try:
            logger.info(f"Iniciando geração de previsão para {forecast_hours}h")
            
            # 1. Verificar cache se permitido
            if use_cache and not force_refresh:
                cached_forecast = await self._check_cache(forecast_hours)
                if cached_forecast:
                    logger.info("Retornando previsão do cache")
                    return cached_forecast
            
            # 2. Garantir que modelo está carregado
            await self._ensure_model_loaded()
            
            # 3. Coletar dados atmosféricos necessários
            atmospheric_data, surface_data = await self._collect_weather_data()
            
            # 4. Validar qualidade dos dados
            validation_result = await self._validate_input_data(
                atmospheric_data, surface_data
            )
            
            # 5. Gerar previsão usando modelo híbrido
            forecast = await self._generate_hybrid_forecast(
                atmospheric_data, surface_data, forecast_hours, validation_result
            )
            
            # 6. Validar qualidade da previsão
            await self._validate_forecast_quality(forecast)
            
            # 7. Salvar no cache e repositório
            await self._save_forecast(forecast, use_cache)
            
            logger.info(
                f"Previsão gerada: {forecast.precipitation_mm:.2f}mm, "
                f"confiança: {forecast.confidence_score:.3f}"
            )
            
            return forecast
            
        except Exception as e:
            logger.error(f"Erro na geração de previsão: {e}")
            raise PredictionError(f"Falha ao gerar previsão: {str(e)}")

    async def _check_cache(self, forecast_hours: int) -> Optional[Forecast]:
        """Verifica cache de previsões"""
        try:
            cache_key = f"forecast_{forecast_hours}h_{datetime.now().strftime('%Y%m%d_%H')}"
            
            cached_data = await self.cache_repo.get(cache_key)
            if cached_data:
                return Forecast.from_dict(cached_data)
            
            return None
            
        except Exception as e:
            logger.warning(f"Erro ao verificar cache: {e}")
            return None

    async def _ensure_model_loaded(self):
        """Garante que modelo híbrido está carregado"""
        try:
            current_time = datetime.now()
            
            # Verificar se precisa recarregar modelo (a cada hora)
            if (self._last_model_check is None or 
                (current_time - self._last_model_check).total_seconds() > 3600):
                
                logger.info("Verificando status do modelo híbrido")
                
                if not await self.model_repo.is_model_loaded():
                    logger.info("Carregando modelo híbrido")
                    await self.model_repo.load_hybrid_model()
                
                self._model_loaded = True
                self._last_model_check = current_time
                
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise ModelLoadError(f"Falha ao carregar modelo híbrido: {str(e)}")

    async def _collect_weather_data(self) -> Tuple[Dict, List[Dict]]:
        """Coleta dados meteorológicos para o modelo híbrido"""
        try:
            # Buscar últimas 72h de dados históricos para contexto
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=72)
            
            # Dados atmosféricos (149 features) para componente principal
            atmospheric_data = await self.weather_data_repo.get_atmospheric_data(
                start_time=start_time,
                end_time=end_time,
                include_pressure_levels=True
            )
            
            # Dados de superfície (25 features) para componente secundário  
            surface_data = await self.weather_data_repo.get_surface_data(
                start_time=start_time,
                end_time=end_time
            )
            
            logger.debug(
                f"Coletados {len(atmospheric_data.get('temperature_2m', []))} "
                f"registros atmosféricos e {len(surface_data)} registros de superfície"
            )
            
            return atmospheric_data, surface_data
            
        except Exception as e:
            logger.error(f"Erro na coleta de dados: {e}")
            raise ValidationError(f"Falha na coleta de dados meteorológicos: {str(e)}")

    async def _validate_input_data(
        self, 
        atmospheric_data: Dict, 
        surface_data: List[Dict]
    ) -> Dict[str, float]:
        """Valida qualidade dos dados de entrada"""
        try:
            # Validar dados atmosféricos
            atmospheric_quality = await self.weather_analysis_service.assess_data_quality(
                atmospheric_data
            )
            
            # Validar dados de superfície
            surface_quality = await self.weather_analysis_service.assess_surface_data_quality(
                surface_data
            )
            
            # Validar sequência temporal para LSTM
            sequence_validation = self.forecast_service.validate_sequence_for_prediction(
                atmospheric_data, surface_data
            )
            
            validation_result = {
                "atmospheric_quality": atmospheric_quality,
                "surface_quality": surface_quality,
                "sequence_completeness": sequence_validation["completeness"],
                "temporal_consistency": sequence_validation["temporal_consistency"],
                "overall_quality": (atmospheric_quality + surface_quality) / 2.0
            }
            
            # Verificar se qualidade mínima é atendida
            if validation_result["overall_quality"] < self.config.min_data_quality:
                raise ValidationError(
                    f"Qualidade dos dados insuficiente: "
                    f"{validation_result['overall_quality']:.3f} < "
                    f"{self.config.min_data_quality}"
                )
            
            logger.debug(f"Validação de dados: {validation_result}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Erro na validação de dados: {e}")
            raise ValidationError(f"Falha na validação de dados: {str(e)}")

    async def _generate_hybrid_forecast(
        self,
        atmospheric_data: Dict,
        surface_data: List[Dict],
        forecast_hours: int,
        validation_result: Dict[str, float]
    ) -> Forecast:
        """Gera previsão usando modelo híbrido"""
        try:
            # Usar modelo híbrido via repository
            forecast_result = await self.model_repo.predict_hybrid(
                atmospheric_data=atmospheric_data,
                surface_data=surface_data,
                forecast_hours=forecast_hours
            )
            
            # Calcular timestamp da previsão
            forecast_timestamp = datetime.now() + timedelta(hours=forecast_hours)
            
            # Aplicar pós-processamento baseado na qualidade dos dados
            adjusted_precipitation = self._adjust_prediction_by_quality(
                forecast_result["precipitation_mm"],
                validation_result["overall_quality"]
            )
            
            # Calcular confiança baseada no ensemble e qualidade dos dados
            confidence_score = self._calculate_ensemble_confidence(
                forecast_result, validation_result
            )
            
            # Criar entidade Forecast
            forecast = Forecast(
                timestamp=forecast_timestamp,
                precipitation_mm=adjusted_precipitation,
                confidence_score=confidence_score,
                model_version=forecast_result.get("model_version", "hybrid_v1.0"),
                inference_time_ms=forecast_result.get("inference_time_ms", 0),
                details={
                    "ensemble_details": forecast_result.get("ensemble_details", {}),
                    "data_validation": validation_result,
                    "atmospheric_contribution": forecast_result.get("atmospheric_prediction", 0),
                    "surface_contribution": forecast_result.get("surface_prediction", 0),
                    "synoptic_analysis": forecast_result.get("synoptic_analysis", {}),
                    "forecast_horizon_hours": forecast_hours
                }
            )
            
            return forecast
            
        except Exception as e:
            logger.error(f"Erro na geração da previsão híbrida: {e}")
            raise PredictionError(f"Falha na predição híbrida: {str(e)}")

    def _adjust_prediction_by_quality(
        self, 
        raw_prediction: float, 
        data_quality: float
    ) -> float:
        """Ajusta previsão baseada na qualidade dos dados"""
        # Aplicar fator de ajuste conservador para dados de baixa qualidade
        if data_quality < 0.7:
            # Reduzir confiança em valores extremos para dados de baixa qualidade
            if raw_prediction > 10.0:
                adjustment_factor = 0.8 + (data_quality - 0.5) * 0.4
                return raw_prediction * adjustment_factor
        
        return max(0.0, raw_prediction)

    def _calculate_ensemble_confidence(
        self, 
        forecast_result: Dict, 
        validation_result: Dict[str, float]
    ) -> float:
        """Calcula confiança do ensemble considerando múltiplos fatores"""
        # Confiança base do modelo ensemble
        ensemble_confidence = forecast_result.get("ensemble_confidence", 0.8)
        
        # Ajuste baseado na qualidade dos dados
        data_quality_factor = validation_result["overall_quality"]
        
        # Ajuste baseado na concordância entre componentes
        atmospheric_pred = forecast_result.get("atmospheric_prediction", 0)
        surface_pred = forecast_result.get("surface_prediction", 0)
        
        if atmospheric_pred > 0 or surface_pred > 0:
            max_pred = max(atmospheric_pred, surface_pred, 0.1)
            agreement = 1.0 - min(abs(atmospheric_pred - surface_pred) / max_pred, 1.0)
        else:
            agreement = 1.0
        
        # Combinar fatores
        final_confidence = (
            0.5 * ensemble_confidence +
            0.3 * data_quality_factor +
            0.2 * agreement
        )
        
        return min(max(final_confidence, 0.1), 1.0)

    async def _validate_forecast_quality(self, forecast: Forecast):
        """Valida qualidade da previsão gerada"""
        try:
            # Usar ForecastService para validação
            quality_check = self.forecast_service.validate_forecast_quality(forecast)
            
            if not quality_check["is_valid"]:
                logger.warning(
                    f"Previsão com qualidade questionável: {quality_check['issues']}"
                )
                
                # Para valores extremos, pode rejeitar a previsão
                if quality_check["severity"] == "critical":
                    raise PredictionError(
                        f"Previsão rejeitada por qualidade crítica: {quality_check['issues']}"
                    )
            
        except Exception as e:
            logger.error(f"Erro na validação de qualidade: {e}")
            # Não bloquear por erro de validação, apenas logar

    async def _save_forecast(self, forecast: Forecast, use_cache: bool):
        """Salva previsão no repositório e cache"""
        try:
            # Salvar no repositório principal
            await self.forecast_repo.save_forecast(forecast)
            
            # Salvar no cache se habilitado
            if use_cache:
                cache_key = f"forecast_{24}h_{datetime.now().strftime('%Y%m%d_%H')}"
                await self.cache_repo.set(
                    cache_key, 
                    forecast.to_dict(), 
                    ttl=self.config.cache_ttl_seconds
                )
            
        except Exception as e:
            logger.warning(f"Erro ao salvar previsão: {e}")
            # Não bloquear por erro de salvamento


class GetModelMetricsUseCase:
    """
    Use Case para obter métricas do modelo híbrido
    
    Retorna métricas detalhadas de performance dos componentes ensemble.
    """

    def __init__(
        self,
        model_repository: ModelRepository,
        model_validation_service: ModelValidationService,
        cache_repository: CacheRepository
    ):
        self.model_repo = model_repository
        self.validation_service = model_validation_service
        self.cache_repo = cache_repository

    async def execute(
        self, 
        include_validation: bool = True,
        use_cache: bool = True
    ) -> ModelMetrics:
        """
        Obtém métricas detalhadas do modelo híbrido
        
        Args:
            include_validation: Se deve incluir validação em tempo real
            use_cache: Se deve usar cache de métricas
            
        Returns:
            ModelMetrics: Métricas completas do modelo
        """
        try:
            logger.info("Obtendo métricas do modelo híbrido")
            
            # Verificar cache
            if use_cache:
                cached_metrics = await self._check_metrics_cache()
                if cached_metrics:
                    logger.info("Retornando métricas do cache")
                    return cached_metrics
            
            # Obter métricas base do modelo
            base_metrics = await self.model_repo.get_model_metrics()
            
            # Validação adicional se solicitada
            if include_validation:
                validation_metrics = await self._perform_live_validation()
                base_metrics = self._combine_metrics(base_metrics, validation_metrics)
            
            # Salvar no cache
            if use_cache:
                await self._cache_metrics(base_metrics)
            
            return base_metrics
            
        except Exception as e:
            logger.error(f"Erro ao obter métricas: {e}")
            raise InfrastructureError(f"Falha ao obter métricas do modelo: {str(e)}")

    async def _check_metrics_cache(self) -> Optional[ModelMetrics]:
        """Verifica cache de métricas"""
        try:
            cache_key = "model_metrics_latest"
            cached_data = await self.cache_repo.get(cache_key)
            
            if cached_data:
                return ModelMetrics.from_dict(cached_data)
            return None
            
        except Exception:
            return None

    async def _perform_live_validation(self) -> Dict[str, float]:
        """Realiza validação em tempo real do modelo"""
        try:
            # Usar ModelValidationService para validação
            validation_result = await self.validation_service.validate_model_performance()
            
            return {
                "live_accuracy": validation_result.get("accuracy", 0.0),
                "live_mae": validation_result.get("mae", 0.0),
                "live_rmse": validation_result.get("rmse", 0.0),
                "data_freshness": validation_result.get("data_freshness", 0.0),
                "model_drift": validation_result.get("model_drift", 0.0)
            }
            
        except Exception as e:
            logger.warning(f"Erro na validação ao vivo: {e}")
            return {}

    def _combine_metrics(
        self, 
        base_metrics: ModelMetrics, 
        validation_metrics: Dict[str, float]
    ) -> ModelMetrics:
        """Combina métricas base com validação ao vivo"""
        # Atualizar métricas se validação disponível
        if validation_metrics:
            # Criar nova instância com métricas atualizadas
            return ModelMetrics(
                mae=validation_metrics.get("live_mae", base_metrics.mae),
                rmse=validation_metrics.get("live_rmse", base_metrics.rmse),
                accuracy=validation_metrics.get("live_accuracy", base_metrics.accuracy),
                model_version=base_metrics.model_version,
                training_date=base_metrics.training_date,
                additional_metrics={
                    **base_metrics.additional_metrics,
                    **validation_metrics
                }
            )
        
        return base_metrics

    async def _cache_metrics(self, metrics: ModelMetrics):
        """Salva métricas no cache"""
        try:
            cache_key = "model_metrics_latest"
            await self.cache_repo.set(
                cache_key,
                metrics.to_dict(),
                ttl=1800  # 30 minutos
            )
            
        except Exception as e:
            logger.warning(f"Erro ao cachear métricas: {e}")


class RefreshModelUseCase:
    """
    Use Case para atualização/recarregamento do modelo híbrido
    
    Permite recarregar modelo sem restart da aplicação.
    """

    def __init__(
        self,
        model_repository: ModelRepository,
        cache_repository: CacheRepository,
        model_validation_service: ModelValidationService
    ):
        self.model_repo = model_repository
        self.cache_repo = cache_repository
        self.validation_service = model_validation_service

    async def execute(
        self, 
        model_version: Optional[str] = None,
        validate_after_load: bool = True
    ) -> Dict[str, any]:
        """
        Recarrega modelo híbrido
        
        Args:
            model_version: Versão específica do modelo (None = latest)
            validate_after_load: Se deve validar após carregamento
            
        Returns:
            Dict com resultado da operação
        """
        try:
            logger.info(f"Iniciando refresh do modelo (versão: {model_version})")
            
            # Limpar cache de previsões e métricas
            await self._clear_related_cache()
            
            # Recarregar modelo
            load_result = await self.model_repo.reload_hybrid_model(model_version)
            
            result = {
                "success": True,
                "model_version": load_result.get("model_version"),
                "load_time_ms": load_result.get("load_time_ms"),
                "components_loaded": load_result.get("components_loaded", [])
            }
            
            # Validar modelo se solicitado
            if validate_after_load:
                validation_result = await self._validate_loaded_model()
                result["validation"] = validation_result
            
            logger.info(f"Modelo recarregado com sucesso: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Erro no refresh do modelo: {e}")
            raise ModelLoadError(f"Falha ao recarregar modelo: {str(e)}")

    async def _clear_related_cache(self):
        """Limpa cache relacionado ao modelo"""
        try:
            cache_keys = [
                "model_metrics_latest",
                "forecast_24h_*",  # Pattern para previsões
                "atmospheric_data_*"  # Pattern para dados atmosféricos
            ]
            
            for key in cache_keys:
                await self.cache_repo.delete_pattern(key)
            
            logger.info("Cache do modelo limpo")
            
        except Exception as e:
            logger.warning(f"Erro ao limpar cache: {e}")

    async def _validate_loaded_model(self) -> Dict[str, any]:
        """Valida modelo após carregamento"""
        try:
            return await self.validation_service.quick_model_validation()
            
        except Exception as e:
            logger.warning(f"Erro na validação do modelo carregado: {e}")
            return {"validation_error": str(e)}


class GetForecastHistoryUseCase:
    """
    Use Case para obter histórico de previsões
    
    Permite análise de performance histórica e comparação com dados reais.
    """

    def __init__(
        self,
        forecast_repository: ForecastRepository,
        weather_data_repository: WeatherDataRepository
    ):
        self.forecast_repo = forecast_repository
        self.weather_data_repo = weather_data_repository

    async def execute(
        self,
        start_date: datetime,
        end_date: datetime,
        include_accuracy: bool = True,
        limit: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Obtém histórico de previsões com análise de accuracy
        
        Args:
            start_date: Data inicial
            end_date: Data final  
            include_accuracy: Se deve calcular accuracy vs dados reais
            limit: Limite de registros
            
        Returns:
            Dict com histórico e métricas
        """
        try:
            logger.info(f"Obtendo histórico de previsões: {start_date} - {end_date}")
            
            # Buscar previsões
            forecasts = await self.forecast_repo.get_forecasts_by_period(
                start_date, end_date, limit
            )
            
            result = {
                "period": {"start": start_date, "end": end_date},
                "total_forecasts": len(forecasts),
                "forecasts": [f.to_dict() for f in forecasts]
            }
            
            # Calcular accuracy se solicitado
            if include_accuracy and forecasts:
                accuracy_metrics = await self._calculate_historical_accuracy(forecasts)
                result["accuracy_analysis"] = accuracy_metrics
            
            return result
            
        except Exception as e:
            logger.error(f"Erro ao obter histórico: {e}")
            raise InfrastructureError(f"Falha ao obter histórico de previsões: {str(e)}")

    async def _calculate_historical_accuracy(
        self, 
        forecasts: List[Forecast]
    ) -> Dict[str, float]:
        """Calcula métricas de accuracy histórica"""
        try:
            total_forecasts = len(forecasts)
            accurate_forecasts = 0
            total_mae = 0.0
            total_rmse = 0.0
            
            for forecast in forecasts:
                # Buscar dados reais para comparação
                actual_data = await self.weather_data_repo.get_actual_precipitation(
                    forecast.timestamp
                )
                
                if actual_data is not None:
                    # Calcular erro
                    error = abs(forecast.precipitation_mm - actual_data)
                    squared_error = error ** 2
                    
                    total_mae += error
                    total_rmse += squared_error
                    
                    # Considerar acurado se erro < 2mm
                    if error < 2.0:
                        accurate_forecasts += 1
            
            if total_forecasts > 0:
                accuracy = accurate_forecasts / total_forecasts
                mae = total_mae / total_forecasts
                rmse = (total_rmse / total_forecasts) ** 0.5
                
                return {
                    "accuracy_percentage": accuracy * 100,
                    "mae_mm": mae,
                    "rmse_mm": rmse,
                    "total_comparisons": total_forecasts,
                    "accurate_forecasts": accurate_forecasts
                }
            
            return {}
            
        except Exception as e:
            logger.warning(f"Erro no cálculo de accuracy: {e}")
            return {} 