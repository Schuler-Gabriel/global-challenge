"""
Cache Service Implementation - Infrastructure Layer

Implementação de cache específico para dados de APIs externas.
Utiliza cache em memória com TTL e fallback para cache distribuído.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
import hashlib

from ..domain.entities import WeatherCondition, RiverLevel
from ..domain.services import ApiCacheService


class ExternalApiCacheService(ApiCacheService):
    """
    Implementação de cache para APIs externas
    
    Combina cache em memória para performance rápida com
    cache persistente opcional para recuperação após restart.
    """
    
    def __init__(self, default_ttl_seconds: int = 300):
        """
        Inicializa o serviço de cache
        
        Args:
            default_ttl_seconds: TTL padrão em segundos (5 minutos)
        """
        self.default_ttl = default_ttl_seconds
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Cache em memória
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # Lock para thread safety
        self._lock = asyncio.Lock()
        
        # Estatísticas de cache
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0,
            'total_size': 0
        }
    
    async def get_cached_weather(self, city: str) -> Optional[WeatherCondition]:
        """
        Obtém dados meteorológicos cacheados
        
        Args:
            city: Nome da cidade
            
        Returns:
            Optional[WeatherCondition]: Dados cacheados ou None
        """
        cache_key = self._create_weather_key(city)
        
        async with self._lock:
            cached_data = await self._get_from_memory(cache_key)
            
            if cached_data:
                try:
                    # Deserializar WeatherCondition
                    weather = self._deserialize_weather(cached_data['data'])
                    
                    self._stats['hits'] += 1
                    self.logger.debug(f"Cache hit para dados meteorológicos: {city}")
                    
                    return weather
                
                except Exception as e:
                    self.logger.warning(f"Erro ao deserializar dados meteorológicos do cache: {e}")
                    # Remover dados corrompidos
                    await self._delete_from_memory(cache_key)
            
            self._stats['misses'] += 1
            self.logger.debug(f"Cache miss para dados meteorológicos: {city}")
            
            return None
    
    async def cache_weather(
        self, 
        city: str, 
        weather: WeatherCondition, 
        ttl_seconds: int = None
    ) -> bool:
        """
        Armazena dados meteorológicos no cache
        
        Args:
            city: Nome da cidade
            weather: Dados meteorológicos
            ttl_seconds: Tempo de vida no cache
            
        Returns:
            bool: True se armazenou com sucesso
        """
        try:
            cache_key = self._create_weather_key(city)
            ttl = ttl_seconds or self.default_ttl
            
            # Serializar dados
            serialized_data = self._serialize_weather(weather)
            
            async with self._lock:
                success = await self._set_in_memory(cache_key, serialized_data, ttl)
                
                if success:
                    self._stats['sets'] += 1
                    self.logger.debug(f"Dados meteorológicos cacheados para {city} (TTL: {ttl}s)")
                
                return success
        
        except Exception as e:
            self.logger.error(f"Erro ao cachear dados meteorológicos para {city}: {e}")
            return False
    
    async def get_cached_river_level(self, station: str) -> Optional[RiverLevel]:
        """
        Obtém nível do rio cacheado
        
        Args:
            station: Nome da estação
            
        Returns:
            Optional[RiverLevel]: Dados cacheados ou None
        """
        cache_key = self._create_river_key(station)
        
        async with self._lock:
            cached_data = await self._get_from_memory(cache_key)
            
            if cached_data:
                try:
                    # Deserializar RiverLevel
                    river_level = self._deserialize_river_level(cached_data['data'])
                    
                    self._stats['hits'] += 1
                    self.logger.debug(f"Cache hit para nível do rio: {station}")
                    
                    return river_level
                
                except Exception as e:
                    self.logger.warning(f"Erro ao deserializar nível do rio do cache: {e}")
                    await self._delete_from_memory(cache_key)
            
            self._stats['misses'] += 1
            self.logger.debug(f"Cache miss para nível do rio: {station}")
            
            return None
    
    async def cache_river_level(
        self, 
        station: str, 
        level: RiverLevel, 
        ttl_seconds: int = None
    ) -> bool:
        """
        Armazena nível do rio no cache
        
        Args:
            station: Nome da estação
            level: Dados do nível
            ttl_seconds: Tempo de vida no cache
            
        Returns:
            bool: True se armazenou com sucesso
        """
        try:
            cache_key = self._create_river_key(station)
            ttl = ttl_seconds or self.default_ttl
            
            # Serializar dados
            serialized_data = self._serialize_river_level(level)
            
            async with self._lock:
                success = await self._set_in_memory(cache_key, serialized_data, ttl)
                
                if success:
                    self._stats['sets'] += 1
                    self.logger.debug(f"Nível do rio cacheado para {station} (TTL: {ttl}s)")
                
                return success
        
        except Exception as e:
            self.logger.error(f"Erro ao cachear nível do rio para {station}: {e}")
            return False
    
    async def invalidate_weather_cache(self, city: str) -> bool:
        """
        Invalida cache de dados meteorológicos
        
        Args:
            city: Nome da cidade
            
        Returns:
            bool: True se invalidou com sucesso
        """
        cache_key = self._create_weather_key(city)
        
        async with self._lock:
            success = await self._delete_from_memory(cache_key)
            
            if success:
                self.logger.debug(f"Cache de dados meteorológicos invalidado para {city}")
            
            return success
    
    async def invalidate_river_cache(self, station: str) -> bool:
        """
        Invalida cache de nível do rio
        
        Args:
            station: Nome da estação
            
        Returns:
            bool: True se invalidou com sucesso
        """
        cache_key = self._create_river_key(station)
        
        async with self._lock:
            success = await self._delete_from_memory(cache_key)
            
            if success:
                self.logger.debug(f"Cache de nível do rio invalidado para {station}")
            
            return success
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtém estatísticas do cache
        
        Returns:
            Dict: Estatísticas detalhadas
        """
        async with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0.0
            
            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'sets': self._stats['sets'],
                'evictions': self._stats['evictions'],
                'total_size': self._stats['total_size'],
                'memory_cache_size': len(self._memory_cache),
                'timestamp': datetime.now().isoformat()
            }
    
    async def clear_cache(self) -> bool:
        """
        Limpa todo o cache
        
        Returns:
            bool: True se limpou com sucesso
        """
        try:
            async with self._lock:
                cleared_items = len(self._memory_cache)
                self._memory_cache.clear()
                
                # Reset das estatísticas
                self._stats['total_size'] = 0
                
                self.logger.info(f"Cache limpo: {cleared_items} itens removidos")
                
                return True
        
        except Exception as e:
            self.logger.error(f"Erro ao limpar cache: {e}")
            return False
    
    async def cleanup_expired(self) -> int:
        """
        Remove itens expirados do cache
        
        Returns:
            int: Número de itens removidos
        """
        removed_count = 0
        current_time = datetime.now()
        
        async with self._lock:
            expired_keys = []
            
            for key, cached_item in self._memory_cache.items():
                expires_at = cached_item['expires_at']
                if expires_at and current_time > expires_at:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._memory_cache[key]
                removed_count += 1
                self._stats['evictions'] += 1
            
            # Atualizar tamanho total
            self._stats['total_size'] = len(self._memory_cache)
            
            if removed_count > 0:
                self.logger.debug(f"Limpeza de cache: {removed_count} itens expirados removidos")
        
        return removed_count
    
    # Métodos privados para cache em memória
    
    async def _get_from_memory(self, key: str) -> Optional[Dict[str, Any]]:
        """Obtém item do cache em memória"""
        cached_item = self._memory_cache.get(key)
        
        if not cached_item:
            return None
        
        # Verificar expiração
        expires_at = cached_item.get('expires_at')
        if expires_at and datetime.now() > expires_at:
            del self._memory_cache[key]
            self._stats['evictions'] += 1
            return None
        
        return cached_item
    
    async def _set_in_memory(self, key: str, data: Dict[str, Any], ttl_seconds: int) -> bool:
        """Armazena item no cache em memória"""
        try:
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            
            cached_item = {
                'data': data,
                'created_at': datetime.now(),
                'expires_at': expires_at,
                'ttl_seconds': ttl_seconds
            }
            
            self._memory_cache[key] = cached_item
            self._stats['total_size'] = len(self._memory_cache)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Erro ao armazenar no cache em memória: {e}")
            return False
    
    async def _delete_from_memory(self, key: str) -> bool:
        """Remove item do cache em memória"""
        if key in self._memory_cache:
            del self._memory_cache[key]
            self._stats['total_size'] = len(self._memory_cache)
            return True
        
        return False
    
    # Métodos utilitários
    
    def _create_weather_key(self, city: str) -> str:
        """Cria chave de cache para dados meteorológicos"""
        normalized_city = city.lower().strip().replace(' ', '_')
        return f"weather:{normalized_city}"
    
    def _create_river_key(self, station: str) -> str:
        """Cria chave de cache para nível do rio"""
        normalized_station = station.lower().strip().replace(' ', '_')
        return f"river:{normalized_station}"
    
    def _serialize_weather(self, weather: WeatherCondition) -> Dict[str, Any]:
        """Serializa WeatherCondition para cache"""
        return {
            'timestamp': weather.timestamp.isoformat(),
            'temperature': weather.temperature,
            'humidity': weather.humidity,
            'pressure': weather.pressure,
            'wind_speed': weather.wind_speed,
            'wind_direction': weather.wind_direction,
            'description': weather.description,
            'precipitation_current': weather.precipitation_current,
            'precipitation_forecast_1h': weather.precipitation_forecast_1h,
            'precipitation_forecast_6h': weather.precipitation_forecast_6h,
            'precipitation_forecast_24h': weather.precipitation_forecast_24h,
            'station_id': weather.station_id,
            'data_source': weather.data_source,
            'forecast_confidence': weather.forecast_confidence
        }
    
    def _deserialize_weather(self, data: Dict[str, Any]) -> WeatherCondition:
        """Deserializa WeatherCondition do cache"""
        # Converter timestamp de string para datetime
        timestamp_str = data['timestamp']
        timestamp = datetime.fromisoformat(timestamp_str)
        
        return WeatherCondition(
            timestamp=timestamp,
            temperature=data['temperature'],
            humidity=data['humidity'],
            pressure=data['pressure'],
            wind_speed=data['wind_speed'],
            wind_direction=data['wind_direction'],
            description=data['description'],
            precipitation_current=data.get('precipitation_current'),
            precipitation_forecast_1h=data.get('precipitation_forecast_1h'),
            precipitation_forecast_6h=data.get('precipitation_forecast_6h'),
            precipitation_forecast_24h=data.get('precipitation_forecast_24h'),
            station_id=data.get('station_id'),
            data_source=data.get('data_source', 'cache'),
            forecast_confidence=data.get('forecast_confidence')
        )
    
    def _serialize_river_level(self, level: RiverLevel) -> Dict[str, Any]:
        """Serializa RiverLevel para cache"""
        return {
            'timestamp': level.timestamp.isoformat(),
            'level_meters': level.level_meters,
            'station_name': level.station_name,
            'station_id': level.station_id,
            'source_timestamp': level.source_timestamp.isoformat() if level.source_timestamp else None,
            'data_quality': level.data_quality,
            'measurement_uncertainty': level.measurement_uncertainty
        }
    
    def _deserialize_river_level(self, data: Dict[str, Any]) -> RiverLevel:
        """Deserializa RiverLevel do cache"""
        # Converter timestamps de string para datetime
        timestamp = datetime.fromisoformat(data['timestamp'])
        source_timestamp = None
        if data.get('source_timestamp'):
            source_timestamp = datetime.fromisoformat(data['source_timestamp'])
        
        return RiverLevel(
            timestamp=timestamp,
            level_meters=data['level_meters'],
            station_name=data['station_name'],
            station_id=data.get('station_id'),
            source_timestamp=source_timestamp,
            data_quality=data.get('data_quality'),
            measurement_uncertainty=data.get('measurement_uncertainty')
        ) 