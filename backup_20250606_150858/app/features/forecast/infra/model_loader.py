#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast Infrastructure Layer - Model Loader

Carregamento e gerenciamento do modelo híbrido LSTM.
Implementa Repository pattern para acesso ao modelo ML.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

from app.core.exceptions import ModelLoadError, PredictionError
from app.features.forecast.domain.entities import ModelMetrics
from app.features.forecast.domain.repositories import ModelRepository
from app.features.forecast.infra.hybrid_model import HybridLSTMForecastModel

logger = logging.getLogger(__name__)


class HybridModelLoader(ModelRepository):
    """
    Implementação concreta do repository para carregamento do modelo híbrido
    
    Gerencia carregamento, cache e operações do modelo LSTM híbrido integrado
    com dados atmosféricos Open-Meteo.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path("data/modelos_treinados")
        self.hybrid_model: Optional[HybridLSTMForecastModel] = None
        self._model_loaded = False
        self._load_timestamp: Optional[datetime] = None
        self._model_version: Optional[str] = None
        
        # Configurações de cache
        self._prediction_cache: Dict[str, Tuple[float, datetime]] = {}
        self._cache_ttl_seconds = 3600  # 1 hora
        
        logger.info(f"HybridModelLoader inicializado: {self.model_path}")

    async def load_hybrid_model(
        self, 
        atmospheric_model: str = "atmospheric_lstm",
        surface_model: str = "surface_lstm",
        stacking_model: str = "stacking_model"
    ) -> Dict[str, Any]:
        """
        Carrega modelo híbrido completo

        Args:
            atmospheric_model: Nome do modelo atmosférico
            surface_model: Nome do modelo de superfície  
            stacking_model: Nome do modelo de stacking

        Returns:
            Dict com informações do carregamento
            
        Raises:
            ModelLoadError: Erro no carregamento
        """
        try:
            start_time = datetime.now()
            logger.info("Iniciando carregamento do modelo híbrido")

            # Inicializar modelo híbrido
            self.hybrid_model = HybridLSTMForecastModel(self.model_path)
            
            # Carregar todos os componentes
            success = self.hybrid_model.load_models(
                atmospheric_model=atmospheric_model,
                surface_model=surface_model,
                stacking_model=stacking_model
            )
            
            if not success:
                raise ModelLoadError("Falha ao carregar componentes do modelo híbrido")
            
            # Atualizar estado
            self._model_loaded = True
            self._load_timestamp = datetime.now()
            self._model_version = self.hybrid_model.metadata.get("model_version", "hybrid_v1.0")
            
            load_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = {
                "success": True,
                "model_version": self._model_version,
                "load_time_ms": load_time,
                "components_loaded": [
                    "atmospheric_lstm" if self.hybrid_model.atmospheric_component.model else None,
                    "surface_lstm" if self.hybrid_model.surface_component.model else None,
                    "stacking_model" if self.hybrid_model.ensemble_predictor.stacking_model else None
                ],
                "total_parameters": self._get_total_parameters(),
                "model_info": self.hybrid_model.get_model_info()
            }
            
            # Filtrar None values
            result["components_loaded"] = [c for c in result["components_loaded"] if c]
            
            logger.info(
                f"✓ Modelo híbrido carregado em {load_time:.1f}ms: "
                f"{result['total_parameters']:,} parâmetros"
            )
            
            return result
            
        except Exception as e:
            self._model_loaded = False
            logger.error(f"Erro ao carregar modelo híbrido: {e}")
            raise ModelLoadError(f"Falha no carregamento: {str(e)}")

    async def reload_hybrid_model(self, model_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Recarrega modelo híbrido (útil para atualizações sem restart)
        
        Args:
            model_version: Versão específica (None = latest)
            
        Returns:
            Dict com resultado do recarregamento
        """
        try:
            logger.info(f"Recarregando modelo híbrido (versão: {model_version})")
            
            # Limpar modelo atual
            await self._unload_model()
            
            # Recarregar com nova versão
            atmospheric_name = f"atmospheric_lstm_{model_version}" if model_version else "atmospheric_lstm"
            surface_name = f"surface_lstm_{model_version}" if model_version else "surface_lstm"
            stacking_name = f"stacking_model_{model_version}" if model_version else "stacking_model"
            
            return await self.load_hybrid_model(
                atmospheric_model=atmospheric_name,
                surface_model=surface_name,
                stacking_model=stacking_name
            )
            
        except Exception as e:
            logger.error(f"Erro no reload do modelo: {e}")
            raise ModelLoadError(f"Falha no reload: {str(e)}")

    async def predict_hybrid(
        self,
        atmospheric_data: Dict[str, List[float]],
        surface_data: List[Dict[str, float]],
        forecast_hours: int = 24,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Gera predição usando modelo híbrido
        
        Args:
            atmospheric_data: Dados atmosféricos (149 features)
            surface_data: Dados de superfície (25 features)
            forecast_hours: Horizonte de previsão em horas
            use_cache: Se deve usar cache de predições
            
        Returns:
            Dict com resultado da predição
            
        Raises:
            PredictionError: Erro na predição
        """
        try:
            if not self._model_loaded or not self.hybrid_model:
                raise PredictionError("Modelo híbrido não carregado")
            
            # Verificar cache se habilitado
            if use_cache:
                cached_result = self._check_prediction_cache(atmospheric_data, surface_data)
                if cached_result:
                    logger.debug("Retornando predição do cache")
                    return cached_result
            
            logger.debug("Executando predição híbrida")
            start_time = datetime.now()
            
            # Usar modelo híbrido para predição
            forecast = self.hybrid_model.predict(
                atmospheric_data=atmospheric_data,
                surface_data=surface_data,
                use_stacking=True
            )
            
            inference_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Preparar resultado
            result = {
                "precipitation_mm": forecast.precipitation_mm,
                "confidence_score": forecast.confidence_score,
                "model_version": forecast.model_version,
                "inference_time_ms": inference_time,
                "ensemble_details": forecast.details,
                "atmospheric_prediction": forecast.details.get("atmospheric_prediction", 0),
                "surface_prediction": forecast.details.get("surface_prediction", 0),
                "ensemble_method": forecast.details.get("ensemble_method", "weighted_average"),
                "forecast_timestamp": forecast.timestamp.isoformat(),
                "synoptic_analysis": self._extract_synoptic_analysis(atmospheric_data)
            }
            
            # Salvar no cache se habilitado
            if use_cache:
                self._cache_prediction(atmospheric_data, surface_data, result)
            
            logger.debug(
                f"Predição híbrida: {result['precipitation_mm']:.2f}mm "
                f"em {inference_time:.1f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na predição híbrida: {e}")
            raise PredictionError(f"Falha na predição: {str(e)}")

    async def get_model_metrics(self) -> ModelMetrics:
        """
        Obtém métricas do modelo híbrido
        
        Returns:
            ModelMetrics: Métricas de performance
        """
        try:
            if not self._model_loaded or not self.hybrid_model:
                raise ModelLoadError("Modelo não carregado")
            
            return self.hybrid_model.get_model_metrics()
            
        except Exception as e:
            logger.error(f"Erro ao obter métricas: {e}")
            raise ModelLoadError(f"Falha ao obter métricas: {str(e)}")

    async def get_model_info(self) -> Dict[str, Any]:
        """
        Obtém informações detalhadas do modelo

        Returns:
            Dict com informações completas
        """
        try:
            if not self._model_loaded or not self.hybrid_model:
                return {
                    "status": "not_loaded",
                    "model_version": None,
                    "load_timestamp": None
                }
            
            model_info = self.hybrid_model.get_model_info()
            model_info.update({
                "load_timestamp": self._load_timestamp.isoformat() if self._load_timestamp else None,
                "cache_stats": self._get_cache_stats(),
                "is_loaded": self._model_loaded
            })
            
            return model_info

        except Exception as e:
            logger.error(f"Erro ao obter info do modelo: {e}")
            return {"status": "error", "error": str(e)}

    async def is_model_loaded(self) -> bool:
        """
        Verifica se modelo está carregado

        Returns:
            bool: True se carregado
        """
        return self._model_loaded and self.hybrid_model is not None

    async def validate_model_health(self) -> Dict[str, Any]:
        """
        Validação de saúde do modelo

        Returns:
            Dict com status de saúde
        """
        try:
            if not self._model_loaded:
                return {
                    "healthy": False,
                    "reason": "Model not loaded",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Teste rápido com dados sintéticos
            test_atmospheric = self._generate_test_atmospheric_data()
            test_surface = self._generate_test_surface_data()
            
            start_time = datetime.now()
            test_result = await self.predict_hybrid(
                test_atmospheric, test_surface, use_cache=False
            )
            test_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "healthy": True,
                "test_prediction": test_result["precipitation_mm"],
                "test_inference_time_ms": test_time,
                "model_version": self._model_version,
                "load_timestamp": self._load_timestamp.isoformat() if self._load_timestamp else None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check falhou: {e}")
            return {
                "healthy": False,
                "reason": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _check_prediction_cache(
        self, 
        atmospheric_data: Dict[str, List[float]], 
        surface_data: List[Dict[str, float]]
    ) -> Optional[Dict[str, Any]]:
        """Verifica cache de predições"""
        try:
            # Gerar chave de cache baseada nos dados
            cache_key = self._generate_cache_key(atmospheric_data, surface_data)
            
            if cache_key in self._prediction_cache:
                cached_result, cache_time = self._prediction_cache[cache_key]
                
                # Verificar TTL
                if (datetime.now() - cache_time).total_seconds() < self._cache_ttl_seconds:
                    return cached_result
                else:
                    # Remove cache expirado
                    del self._prediction_cache[cache_key]
            
            return None
            
        except Exception:
            return None

    def _cache_prediction(
        self,
        atmospheric_data: Dict[str, List[float]],
        surface_data: List[Dict[str, float]],
        result: Dict[str, Any]
    ):
        """Salva predição no cache"""
        try:
            cache_key = self._generate_cache_key(atmospheric_data, surface_data)
            self._prediction_cache[cache_key] = (result, datetime.now())
            
            # Limpar cache antigo se muito grande
            if len(self._prediction_cache) > 100:
                self._cleanup_cache()
                
        except Exception as e:
            logger.warning(f"Erro ao cachear predição: {e}")

    def _generate_cache_key(
        self, 
        atmospheric_data: Dict[str, List[float]], 
        surface_data: List[Dict[str, float]]
    ) -> str:
        """Gera chave de cache para os dados"""
        try:
            # Hash simples baseado nos últimos valores
            atm_hash = hash(str(atmospheric_data.get("temperature_2m", [])[-5:]))
            surf_hash = hash(str(surface_data[-3:] if surface_data else []))
            return f"pred_{atm_hash}_{surf_hash}"
            
        except Exception:
            return f"pred_{datetime.now().strftime('%Y%m%d_%H%M')}"

    def _cleanup_cache(self):
        """Remove entradas antigas do cache"""
        try:
            current_time = datetime.now()
            keys_to_remove = []
            
            for key, (_, cache_time) in self._prediction_cache.items():
                if (current_time - cache_time).total_seconds() > self._cache_ttl_seconds:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._prediction_cache[key]
                
            logger.debug(f"Cache limpo: {len(keys_to_remove)} entradas removidas")
            
        except Exception as e:
            logger.warning(f"Erro na limpeza do cache: {e}")

    def _extract_synoptic_analysis(self, atmospheric_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Extrai análise sinótica dos dados atmosféricos"""
        try:
            analysis = {}
            
            # Análise de pressão
            if "pressure_msl" in atmospheric_data:
                pressure_data = atmospheric_data["pressure_msl"][-24:]  # Últimas 24h
                if pressure_data:
                    analysis["pressure_trend"] = pressure_data[-1] - pressure_data[0]
                    analysis["pressure_current"] = pressure_data[-1]
            
            # Análise de temperatura 850hPa (frentes frias)
            if "temperature_850hPa" in atmospheric_data:
                temp_850 = atmospheric_data["temperature_850hPa"][-6:]  # Últimas 6h
                if len(temp_850) >= 2:
                    gradient = temp_850[-1] - temp_850[0]
                    if gradient < -3:
                        analysis["frontal_activity"] = "cold_front_approaching"
                    elif gradient > 3:
                        analysis["frontal_activity"] = "warm_front_approaching"
                    else:
                        analysis["frontal_activity"] = "stable"
                    analysis["temp_850_gradient"] = gradient
            
            # Análise de vento 500hPa (vórtices)
            if "wind_speed_500hPa" in atmospheric_data:
                wind_500 = atmospheric_data["wind_speed_500hPa"][-12:]  # Últimas 12h
                if wind_500:
                    wind_max = max(wind_500)
                    if wind_max > 60:
                        analysis["upper_level_activity"] = "strong_vortex"
                    elif wind_max > 40:
                        analysis["upper_level_activity"] = "moderate_vortex"
            else:
                        analysis["upper_level_activity"] = "weak_pattern"
                    analysis["wind_500_max"] = wind_max

            return analysis

        except Exception as e:
            logger.warning(f"Erro na análise sinótica: {e}")
            return {}

    def _get_total_parameters(self) -> int:
        """Obtém total de parâmetros do modelo híbrido"""
        try:
            if not self.hybrid_model:
                return 0
            
            total_params = 0
            
            if self.hybrid_model.atmospheric_component.model:
                total_params += self.hybrid_model.atmospheric_component.model.count_params()
            
            if self.hybrid_model.surface_component.model:
                total_params += self.hybrid_model.surface_component.model.count_params()
            
            return total_params
            
        except Exception:
            return 0

    def _get_cache_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas do cache"""
        try:
            current_time = datetime.now()
            valid_entries = 0
            
            for _, cache_time in self._prediction_cache.values():
                if (current_time - cache_time).total_seconds() < self._cache_ttl_seconds:
                    valid_entries += 1
            
            return {
                "total_entries": len(self._prediction_cache),
                "valid_entries": valid_entries,
                "cache_ttl_seconds": self._cache_ttl_seconds
            }
            
        except Exception:
            return {"error": "Failed to get cache stats"}

    def _generate_test_atmospheric_data(self) -> Dict[str, List[float]]:
        """Gera dados atmosféricos sintéticos para teste"""
        return {
            "temperature_2m": [25.0] * 72,
            "relative_humidity_2m": [65.0] * 72,
            "precipitation": [0.0] * 72,
            "pressure_msl": [1013.2] * 72,
            "wind_speed_10m": [10.0] * 72,
            "temperature_850hPa": [15.0] * 72,
            "wind_speed_500hPa": [45.0] * 72,
            "geopotential_height_500hPa": [5820.0] * 72
        }

    def _generate_test_surface_data(self) -> List[Dict[str, float]]:
        """Gera dados de superfície sintéticos para teste"""
        return [
            {
                "temperature_2m_mean": 25.0,
                "relativehumidity_2m_mean": 65.0,
                "precipitation_sum": 0.0,
                "pressure_msl_mean": 1013.2,
                "windspeed_10m_mean": 10.0
            }
        ] * 48

    async def _unload_model(self):
        """Descarrega modelo da memória"""
        try:
            if self.hybrid_model:
                # Limpar referências
                self.hybrid_model.atmospheric_component.model = None
                self.hybrid_model.surface_component.model = None
                self.hybrid_model.ensemble_predictor.stacking_model = None
                self.hybrid_model = None
            
            # Limpar cache
            self._prediction_cache.clear()
            
            # Atualizar estado
            self._model_loaded = False
            self._load_timestamp = None
            self._model_version = None
            
            # Forçar garbage collection do TensorFlow
            tf.keras.backend.clear_session()
            
            logger.info("Modelo híbrido descarregado")

        except Exception as e:
            logger.warning(f"Erro ao descarregar modelo: {e}")


# Factory function para facilitar injeção de dependência
def create_hybrid_model_loader(model_path: Optional[Path] = None) -> HybridModelLoader:
    """
    Factory function para criar HybridModelLoader
    
    Args:
        model_path: Caminho para modelos (None = padrão)
        
    Returns:
        HybridModelLoader: Instância configurada
    """
    return HybridModelLoader(model_path)
