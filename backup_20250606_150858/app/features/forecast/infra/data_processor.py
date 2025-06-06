#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast Infrastructure Layer - Data Processor

Processamento e preparação de dados meteorológicos para o modelo híbrido.
Implementa Repository pattern para acesso a dados históricos e em tempo real.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import httpx
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import asyncio

from app.core.exceptions import ValidationError, InfrastructureError
from app.features.forecast.domain.entities import WeatherData
from app.features.forecast.domain.repositories import WeatherDataRepository

logger = logging.getLogger(__name__)


class OpenMeteoDataProcessor(WeatherDataRepository):
    """
    Implementação concreta do repository para dados meteorológicos Open-Meteo
    
    Integra dados históricos e em tempo real das APIs Open-Meteo:
    - Historical Forecast API (149 features atmosféricas)
    - Historical Weather API (25 features de superfície)  
    - Current Weather API (dados em tempo real)
    """

    def __init__(self):
        self.base_urls = {
            "historical_forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
            "historical_weather": "https://archive-api.open-meteo.com/v1/archive", 
            "current_weather": "https://api.open-meteo.com/v1/forecast"
        }
        
        # Coordenadas Porto Alegre
        self.coordinates = {
            "latitude": -30.0331,
            "longitude": -51.2300,
            "timezone": "America/Sao_Paulo"
        }
        
        # Configurações HTTP
        self.http_timeout = 10
        self.max_retries = 3
        
        # Cache interno
        self._data_cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_ttl = 3600  # 1 hora
        
        logger.info("OpenMeteoDataProcessor inicializado")

    async def get_atmospheric_data(
        self,
        start_time: datetime,
        end_time: datetime,
        include_pressure_levels: bool = True
    ) -> Dict[str, List[float]]:
        """
        Obtém dados atmosféricos (149 features) da Historical Forecast API
        
        Args:
            start_time: Data/hora inicial
            end_time: Data/hora final
            include_pressure_levels: Se deve incluir dados de níveis de pressão
            
        Returns:
            Dict com dados atmosféricos formatados para o modelo
        """
        try:
            logger.info(
                f"Coletando dados atmosféricos: {start_time} - {end_time} "
                f"(níveis de pressão: {include_pressure_levels})"
            )
            
            # Verificar cache
            cache_key = f"atmospheric_{start_time.date()}_{end_time.date()}_{include_pressure_levels}"
            cached_data = self._check_cache(cache_key)
            if cached_data:
                logger.debug("Retornando dados atmosféricos do cache")
                return cached_data
            
            # Parâmetros base para Historical Forecast API
            params = {
                **self.coordinates,
                "start_date": start_time.strftime("%Y-%m-%d"),
                "end_date": end_time.strftime("%Y-%m-%d"),
                "hourly": [
                    # Surface variables (21)
                    "temperature_2m", "relative_humidity_2m", "dewpoint_2m",
                    "apparent_temperature", "precipitation_probability", "precipitation",
                    "rain", "showers", "pressure_msl", "surface_pressure",
                    "cloudcover", "cloudcover_low", "cloudcover_mid", "cloudcover_high",
                    "windspeed_10m", "winddirection_10m", "windgusts_10m",
                    "cape", "lifted_index", "vapour_pressure_deficit",
                    "soil_temperature_0cm"
                ]
            }
            
            # Adicionar níveis de pressão se solicitado
            if include_pressure_levels:
                params["pressure_level"] = [1000, 850, 700, 500, 300]
                params["pressure_level_variables"] = [
                    "temperature", "relative_humidity", "wind_speed",
                    "wind_direction", "geopotential_height"
                ]
            
            # Fazer requisição
            atmospheric_data = await self._make_request(
                self.base_urls["historical_forecast"], 
                params
            )
            
            # Processar dados
            processed_data = self._process_atmospheric_response(atmospheric_data, include_pressure_levels)
            
            # Salvar no cache
            self._save_cache(cache_key, processed_data)
            
            logger.info(f"✓ Coletados dados atmosféricos: {len(processed_data)} variáveis")
            return processed_data
            
        except Exception as e:
            logger.error(f"Erro ao obter dados atmosféricos: {e}")
            raise InfrastructureError(f"Falha na coleta de dados atmosféricos: {str(e)}")

    async def get_surface_data(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, float]]:
        """
        Obtém dados de superfície (25 features) da Historical Weather API
        
        Args:
            start_time: Data/hora inicial
            end_time: Data/hora final
            
        Returns:
            List de dicionários com dados agregados diários
        """
        try:
            logger.info(f"Coletando dados de superfície: {start_time} - {end_time}")
            
            # Verificar cache
            cache_key = f"surface_{start_time.date()}_{end_time.date()}"
            cached_data = self._check_cache(cache_key)
            if cached_data:
                logger.debug("Retornando dados de superfície do cache")
                return cached_data
            
            # Parâmetros para Historical Weather API (dados diários agregados)
            params = {
                **self.coordinates,
                "start_date": start_time.strftime("%Y-%m-%d"),
                "end_date": end_time.strftime("%Y-%m-%d"),
                "daily": [
                    # Temperature (6)
                    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
                    "apparent_temperature_mean", "apparent_temperature_max", "apparent_temperature_min",
                    
                    # Humidity (4)
                    "relativehumidity_2m_mean", "relativehumidity_2m_max",
                    "relativehumidity_2m_min", "dewpoint_2m_mean",
                    
                    # Precipitation (3)
                    "precipitation_sum", "rain_sum", "showers_sum",
                    
                    # Wind (4)
                    "windspeed_10m_mean", "windspeed_10m_max",
                    "winddirection_10m_dominant", "windgusts_10m_max",
                    
                    # Pressure (3)
                    "pressure_msl_mean", "surface_pressure_mean", "pressure_msl_min",
                    
                    # Cloud (3)
                    "cloudcover_mean", "cloudcover_low_mean", "cloudcover_high_mean",
                    
                    # Solar/Weather (2)
                    "shortwave_radiation_sum", "weathercode_mode"
                ]
            }
            
            # Fazer requisição
            surface_data = await self._make_request(
                self.base_urls["historical_weather"],
                params
            )
            
            # Processar dados
            processed_data = self._process_surface_response(surface_data)
            
            # Salvar no cache
            self._save_cache(cache_key, processed_data)
            
            logger.info(f"✓ Coletados dados de superfície: {len(processed_data)} registros")
            return processed_data
            
        except Exception as e:
            logger.error(f"Erro ao obter dados de superfície: {e}")
            raise InfrastructureError(f"Falha na coleta de dados de superfície: {str(e)}")

    async def get_current_weather_data(self) -> Dict[str, Any]:
        """
        Obtém dados meteorológicos atuais da Current Weather API
        
        Returns:
            Dict com dados meteorológicos atuais
        """
        try:
            logger.debug("Coletando dados meteorológicos atuais")
            
            # Verificar cache (TTL curto para dados atuais)
            cache_key = f"current_{datetime.now().strftime('%Y%m%d_%H')}"
            cached_data = self._check_cache(cache_key, ttl=900)  # 15 minutos
            if cached_data:
                return cached_data
            
            # Parâmetros para Current Weather API
            params = {
                **self.coordinates,
                "current": [
                    "temperature_2m", "relative_humidity_2m", "precipitation",
                    "pressure_msl", "windspeed_10m", "winddirection_10m",
                    "weather_code"
                ],
                "hourly": [
                    "temperature_2m", "relative_humidity_2m", "precipitation",
                    "pressure_msl", "windspeed_10m", "winddirection_10m"
                ],
                "past_days": 1,  # Últimas 24h para contexto
                "forecast_days": 1,  # Próximas 24h
                # Dados sinóticos atuais
                "pressure_level": [850, 500],
                "pressure_level_variables": [
                    "temperature", "wind_speed", "wind_direction", "geopotential_height"
                ]
            }
            
            # Fazer requisição
            current_data = await self._make_request(
                self.base_urls["current_weather"],
                params
            )
            
            # Processar dados
            processed_data = self._process_current_response(current_data)
            
            # Salvar no cache
            self._save_cache(cache_key, processed_data, ttl=900)
            
            logger.debug("✓ Dados meteorológicos atuais coletados")
            return processed_data
            
        except Exception as e:
            logger.error(f"Erro ao obter dados atuais: {e}")
            raise InfrastructureError(f"Falha na coleta de dados atuais: {str(e)}")

    async def get_actual_precipitation(self, timestamp: datetime) -> Optional[float]:
        """
        Obtém precipitação real para comparação com previsões
        
        Args:
            timestamp: Data/hora para buscar precipitação
            
        Returns:
            float: Precipitação em mm ou None se não disponível
        """
        try:
            # Buscar dados do dia específico
            start_date = timestamp.date()
            end_date = start_date
            
            params = {
                **self.coordinates,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "hourly": ["precipitation"]
            }
            
            data = await self._make_request(
                self.base_urls["historical_weather"],
                params
            )
            
            if "hourly" in data and "precipitation" in data["hourly"]:
                hourly_data = data["hourly"]
                timestamps = pd.to_datetime(hourly_data["time"])
                precipitation = hourly_data["precipitation"]
                
                # Encontrar timestamp mais próximo
                target_hour = timestamp.replace(minute=0, second=0, microsecond=0)
                closest_idx = np.argmin(np.abs(timestamps - target_hour))
                
                return float(precipitation[closest_idx])
            
            return None
            
        except Exception as e:
            logger.warning(f"Erro ao buscar precipitação real para {timestamp}: {e}")
            return None

    async def _make_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Faz requisição HTTP com retry automático
        
        Args:
            url: URL da API
            params: Parâmetros da requisição
            
        Returns:
            Dict com resposta da API
        """
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    return response.json()
                    
            except httpx.RequestError as e:
                logger.warning(f"Tentativa {attempt + 1} falhou: {e}")
                if attempt == self.max_retries - 1:
                    raise InfrastructureError(f"Falha após {self.max_retries} tentativas: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Backoff exponencial
            except httpx.HTTPStatusError as e:
                logger.error(f"Erro HTTP {e.response.status_code}: {e.response.text}")
                raise InfrastructureError(f"Erro HTTP {e.response.status_code}")

    def _process_atmospheric_response(
        self, 
        data: Dict[str, Any], 
        include_pressure_levels: bool
    ) -> Dict[str, List[float]]:
        """
        Processa resposta da Historical Forecast API
        
        Args:
            data: Dados brutos da API
            include_pressure_levels: Se foram incluídos níveis de pressão
            
        Returns:
            Dict com dados processados
        """
        try:
            processed = {}
            
            if "hourly" not in data:
                raise ValidationError("Dados horárias não encontrados na resposta")
            
            hourly_data = data["hourly"]
            
            # Processar variáveis de superfície (21)
            surface_vars = [
                "temperature_2m", "relative_humidity_2m", "dewpoint_2m",
                "apparent_temperature", "precipitation_probability", "precipitation",
                "rain", "showers", "pressure_msl", "surface_pressure",
                "cloudcover", "cloudcover_low", "cloudcover_mid", "cloudcover_high",
                "windspeed_10m", "winddirection_10m", "windgusts_10m",
                "cape", "lifted_index", "vapour_pressure_deficit",
                "soil_temperature_0cm"
            ]
            
            for var in surface_vars:
                if var in hourly_data:
                    processed[var] = [float(v) if v is not None else 0.0 for v in hourly_data[var]]
                else:
                    # Valor padrão se variável não disponível
                    length = len(hourly_data.get("time", []))
                    processed[var] = [0.0] * length
            
            # Processar níveis de pressão se incluídos (125 variáveis)
            if include_pressure_levels:
                pressure_levels = [1000, 850, 700, 500, 300]
                pressure_vars = ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height"]
                
                for level in pressure_levels:
                    level_key = f"pressure_level_{level}"
                    if level_key in data:
                        level_data = data[level_key]
                        for var in pressure_vars:
                            feature_name = f"{var}_{level}hPa"
                            if var in level_data:
                                processed[feature_name] = [
                                    float(v) if v is not None else 0.0 
                                    for v in level_data[var]
                                ]
                            else:
                                length = len(hourly_data.get("time", []))
                                processed[feature_name] = [0.0] * length
            
            # Calcular features sinóticas derivadas (10)
            processed.update(self._calculate_synoptic_features(processed))
            
            return processed
            
        except Exception as e:
            logger.error(f"Erro no processamento de dados atmosféricos: {e}")
            raise ValidationError(f"Falha no processamento: {str(e)}")

    def _process_surface_response(self, data: Dict[str, Any]) -> List[Dict[str, float]]:
        """
        Processa resposta da Historical Weather API
        
        Args:
            data: Dados brutos da API
            
        Returns:
            List de dicionários com dados diários
        """
        try:
            if "daily" not in data:
                raise ValidationError("Dados diários não encontrados na resposta")
            
            daily_data = data["daily"]
            times = daily_data.get("time", [])
            
            processed_data = []
            
            # 25 variáveis de superfície
            surface_features = [
                "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
                "apparent_temperature_mean", "apparent_temperature_max", "apparent_temperature_min",
                "relativehumidity_2m_mean", "relativehumidity_2m_max",
                "relativehumidity_2m_min", "dewpoint_2m_mean",
                "precipitation_sum", "rain_sum", "showers_sum",
                "windspeed_10m_mean", "windspeed_10m_max",
                "winddirection_10m_dominant", "windgusts_10m_max",
                "pressure_msl_mean", "surface_pressure_mean", "pressure_msl_min",
                "cloudcover_mean", "cloudcover_low_mean", "cloudcover_high_mean",
                "shortwave_radiation_sum", "weathercode_mode"
            ]
            
            for i, time_str in enumerate(times):
                record = {"date": time_str}
                
                for feature in surface_features:
                    if feature in daily_data and i < len(daily_data[feature]):
                        value = daily_data[feature][i]
                        record[feature] = float(value) if value is not None else 0.0
                    else:
                        record[feature] = 0.0
                
                processed_data.append(record)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Erro no processamento de dados de superfície: {e}")
            raise ValidationError(f"Falha no processamento: {str(e)}")

    def _process_current_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa resposta da Current Weather API
        
        Args:
            data: Dados brutos da API
            
        Returns:
            Dict com dados atuais processados
        """
        try:
            processed = {
                "timestamp": datetime.now().isoformat(),
                "location": {
                    "latitude": data.get("latitude"),
                    "longitude": data.get("longitude"),
                    "elevation": data.get("elevation")
                }
            }
            
            # Dados atuais
            if "current" in data:
                current = data["current"]
                processed["current_conditions"] = {
                    "temperature": current.get("temperature_2m"),
                    "humidity": current.get("relative_humidity_2m"),
                    "precipitation": current.get("precipitation"),
                    "pressure": current.get("pressure_msl"),
                    "wind_speed": current.get("windspeed_10m"),
                    "wind_direction": current.get("winddirection_10m"),
                    "weather_code": current.get("weather_code")
                }
            
            # Dados horárias das últimas 24h
            if "hourly" in data:
                hourly = data["hourly"]
                processed["last_24h_trends"] = {
                    "times": hourly.get("time", [])[-24:],
                    "temperature": hourly.get("temperature_2m", [])[-24:],
                    "pressure": hourly.get("pressure_msl", [])[-24:],
                    "precipitation": hourly.get("precipitation", [])[-24:]
                }
            
            # Análise sinótica de níveis de pressão
            synoptic_analysis = {}
            
            # 850hPa (frentes frias)
            if "pressure_level_850" in data:
                level_850 = data["pressure_level_850"]
                if "temperature" in level_850:
                    temp_850 = level_850["temperature"][-6:]  # Últimas 6h
                    if len(temp_850) >= 2:
                        gradient = temp_850[-1] - temp_850[0]
                        synoptic_analysis["850hPa"] = {
                            "temperature_current": temp_850[-1],
                            "temperature_gradient_6h": gradient,
                            "frontal_activity": self._classify_frontal_activity(gradient)
                        }
            
            # 500hPa (vórtices)
            if "pressure_level_500" in data:
                level_500 = data["pressure_level_500"]
                if "wind_speed" in level_500:
                    wind_500 = level_500["wind_speed"][-12:]  # Últimas 12h
                    if wind_500:
                        wind_max = max(wind_500)
                        synoptic_analysis["500hPa"] = {
                            "wind_speed_max": wind_max,
                            "upper_level_activity": self._classify_upper_level_activity(wind_max)
                        }
            
            processed["synoptic_analysis"] = synoptic_analysis
            
            return processed
            
        except Exception as e:
            logger.error(f"Erro no processamento de dados atuais: {e}")
            raise ValidationError(f"Falha no processamento: {str(e)}")

    def _calculate_synoptic_features(self, data: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        Calcula features sinóticas derivadas (10 features)
        
        Args:
            data: Dados atmosféricos base
            
        Returns:
            Dict com features derivadas
        """
        try:
            derived = {}
            data_length = len(data.get("temperature_2m", []))
            
            # Wind shear features
            if "wind_speed_850hPa" in data and "wind_speed_500hPa" in data:
                derived["wind_shear_850_500"] = [
                    data["wind_speed_850hPa"][i] - data["wind_speed_500hPa"][i]
                    for i in range(data_length)
                ]
            else:
                derived["wind_shear_850_500"] = [0.0] * data_length
            
            if "wind_speed_1000hPa" in data and "wind_speed_850hPa" in data:
                derived["wind_shear_1000_850"] = [
                    data["wind_speed_1000hPa"][i] - data["wind_speed_850hPa"][i]
                    for i in range(data_length)
                ]
            else:
                derived["wind_shear_1000_850"] = [0.0] * data_length
            
            # Temperature gradients
            if "temperature_850hPa" in data and "temperature_500hPa" in data:
                derived["temp_gradient_850_500"] = [
                    data["temperature_850hPa"][i] - data["temperature_500hPa"][i]
                    for i in range(data_length)
                ]
            else:
                derived["temp_gradient_850_500"] = [0.0] * data_length
            
            if "temperature_2m" in data and "temperature_850hPa" in data:
                derived["temp_gradient_surface_850"] = [
                    data["temperature_2m"][i] - data["temperature_850hPa"][i]
                    for i in range(data_length)
                ]
            else:
                derived["temp_gradient_surface_850"] = [0.0] * data_length
            
            # Frontal analysis (850hPa)
            if "temperature_850hPa" in data:
                temp_850 = data["temperature_850hPa"]
                frontal_strength = [0.0]  # Primeiro valor
                temp_advection = [0.0]
                
                for i in range(1, len(temp_850)):
                    # Frontal strength (gradient temporal)
                    frontal_strength.append(temp_850[i] - temp_850[i-1])
                    
                    # Temperature advection (média móvel vs valor atual)
                    if i >= 3:
                        moving_avg = np.mean(temp_850[i-3:i])
                        temp_advection.append(moving_avg - temp_850[i])
                    else:
                        temp_advection.append(0.0)
                
                derived["frontal_strength_850"] = frontal_strength
                derived["temperature_advection_850"] = temp_advection
            else:
                derived["frontal_strength_850"] = [0.0] * data_length
                derived["temperature_advection_850"] = [0.0] * data_length
            
            # Vortex analysis (500hPa)
            if "wind_direction_500hPa" in data:
                wind_dir_500 = data["wind_direction_500hPa"]
                vorticity = [0.0]  # Primeiro valor
                
                for i in range(1, len(wind_dir_500)):
                    # Approximação de vorticidade (mudança de direção)
                    dir_change = wind_dir_500[i] - wind_dir_500[i-1]
                    # Normalizar para [-180, 180]
                    if dir_change > 180:
                        dir_change -= 360
                    elif dir_change < -180:
                        dir_change += 360
                    vorticity.append(dir_change)
                
                derived["vorticity_500"] = vorticity
            else:
                derived["vorticity_500"] = [0.0] * data_length
            
            if "wind_speed_500hPa" in data:
                wind_speed_500 = data["wind_speed_500hPa"]
                divergence = [0.0]  # Primeiro valor
                
                for i in range(1, len(wind_speed_500)):
                    # Approximação de divergência (mudança de velocidade)
                    divergence.append(wind_speed_500[i] - wind_speed_500[i-1])
                
                derived["divergence_500"] = divergence
            else:
                derived["divergence_500"] = [0.0] * data_length
            
            # Atmospheric instability
            if "cape" in data and "lifted_index" in data:
                cape_values = data["cape"]
                lifted_index = data["lifted_index"]
                instability = []
                
                for i in range(data_length):
                    cape_val = cape_values[i]
                    li_val = abs(lifted_index[i]) + 1e-6  # Evitar divisão por zero
                    instability.append(cape_val / li_val)
                
                derived["atmospheric_instability"] = instability
            else:
                derived["atmospheric_instability"] = [0.0] * data_length
            
            # Moisture flux
            if "relative_humidity_850hPa" in data and "wind_speed_850hPa" in data:
                humidity_850 = data["relative_humidity_850hPa"]
                wind_850 = data["wind_speed_850hPa"]
                moisture_flux = [
                    humidity_850[i] * wind_850[i] / 100.0
                    for i in range(data_length)
                ]
                derived["moisture_flux"] = moisture_flux
            else:
                derived["moisture_flux"] = [0.0] * data_length
            
            return derived
            
        except Exception as e:
            logger.warning(f"Erro ao calcular features sinóticas: {e}")
            return {}

    def _classify_frontal_activity(self, gradient: float) -> str:
        """Classifica atividade frontal baseada no gradiente de temperatura"""
        if gradient < -3:
            return "cold_front_approaching"
        elif gradient > 3:
            return "warm_front_approaching"
        else:
            return "stable"

    def _classify_upper_level_activity(self, wind_max: float) -> str:
        """Classifica atividade de níveis superiores baseada no vento máximo"""
        if wind_max > 60:
            return "strong_vortex"
        elif wind_max > 40:
            return "moderate_vortex"
        else:
            return "weak_pattern"

    def _check_cache(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """Verifica cache interno"""
        try:
            if key in self._data_cache:
                data, cache_time = self._data_cache[key]
                ttl_seconds = ttl or self._cache_ttl
                
                if (datetime.now() - cache_time).total_seconds() < ttl_seconds:
                    return data
                else:
                    del self._data_cache[key]
            
            return None
            
        except Exception:
            return None

    def _save_cache(self, key: str, data: Any, ttl: Optional[int] = None):
        """Salva dados no cache interno"""
        try:
            self._data_cache[key] = (data, datetime.now())
            
            # Limpar cache antigo se muito grande
            if len(self._data_cache) > 50:
                self._cleanup_cache()
                
        except Exception as e:
            logger.warning(f"Erro ao salvar cache: {e}")

    def _cleanup_cache(self):
        """Remove entradas antigas do cache"""
        try:
            current_time = datetime.now()
            keys_to_remove = []
            
            for key, (_, cache_time) in self._data_cache.items():
                if (current_time - cache_time).total_seconds() > self._cache_ttl:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._data_cache[key]
                
            logger.debug(f"Cache limpo: {len(keys_to_remove)} entradas removidas")
            
        except Exception as e:
            logger.warning(f"Erro na limpeza do cache: {e}")


# Factory function para facilitar injeção de dependência
def create_openmeteo_data_processor() -> OpenMeteoDataProcessor:
    """
    Factory function para criar OpenMeteoDataProcessor
    
    Returns:
        OpenMeteoDataProcessor: Instância configurada
    """
    return OpenMeteoDataProcessor() 