"""
Application Use Cases - External APIs Feature

Este módulo implementa os casos de uso para integração com APIs externas,
orquestrando operações entre diferentes serviços e garantindo resiliência.

Use Cases:
- GetCurrentConditionsUseCase: Consolida dados atuais de múltiplas APIs
- GetWeatherDataUseCase: Obtém dados meteorológicos
- GetRiverDataUseCase: Obtém dados do nível do rio
- HealthCheckUseCase: Monitora saúde das APIs
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from ..domain.entities import (
    WeatherCondition, RiverLevel, ExternalApiResponse,
    ApiHealthStatus, ApiStatus
)
from ..domain.services import (
    WeatherApiService, RiverApiService, CircuitBreakerService,
    ExternalApiAggregatorService, ApiCacheService
)
from ..domain.services import ExternalApiError


class GetCurrentConditionsUseCase:
    """
    Use case para obter condições atuais consolidadas
    
    Combina dados meteorológicos e nível do rio para
    fornecer visão completa das condições atuais.
    """
    
    def __init__(
        self,
        weather_service: WeatherApiService,
        river_service: RiverApiService,
        circuit_breaker: CircuitBreakerService,
        cache_service: ApiCacheService = None
    ):
        """
        Inicializa o use case
        
        Args:
            weather_service: Serviço de dados meteorológicos
            river_service: Serviço de dados do rio
            circuit_breaker: Serviço de circuit breaker
            cache_service: Serviço de cache (opcional)
        """
        self.weather_service = weather_service
        self.river_service = river_service
        self.circuit_breaker = circuit_breaker
        self.cache_service = cache_service
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def execute(self, city: str = "Porto Alegre", use_cache: bool = True) -> Dict[str, Any]:
        """
        Executa obtenção de condições atuais
        
        Args:
            city: Nome da cidade
            use_cache: Se deve usar cache
            
        Returns:
            Dict: Condições consolidadas
        """
        try:
            self.logger.info(f"Obtendo condições atuais para {city}")
            
            # Verificar cache primeiro
            cache_key = f"current_conditions:{city}"
            if use_cache and self.cache_service:
                cached_data = await self.cache_service.get_cached_response(cache_key)
                if cached_data:
                    self.logger.info("Retornando dados do cache")
                    return cached_data
            
            # Obter dados em paralelo com circuit breaker
            weather_task = self.circuit_breaker.execute(
                "weather", 
                self.weather_service.get_current_weather,
                city
            )
            
            river_task = self.circuit_breaker.execute(
                "river",
                self.river_service.get_current_level,
                city
            )
            
            # Aguardar ambas as operações
            weather_response = await weather_task
            river_response = await river_task
            
            # Processar resultados
            result = {
                "timestamp": datetime.now().isoformat(),
                "city": city,
                "weather": self._process_weather_response(weather_response),
                "river": self._process_river_response(river_response),
                "status": {
                    "weather_api": weather_response.status.value,
                    "river_api": river_response.status.value,
                    "overall": self._determine_overall_status(weather_response, river_response)
                }
            }
            
            # Adicionar análise de risco
            result["risk_assessment"] = self._calculate_risk_assessment(
                result["weather"], result["river"]
            )
            
            # Armazenar no cache
            if use_cache and self.cache_service:
                await self.cache_service.cache_response(cache_key, result, ttl_seconds=300)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro ao obter condições atuais: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "city": city,
                "error": str(e),
                "status": "error"
            }
    
    def _process_weather_response(self, response: ExternalApiResponse) -> Dict[str, Any]:
        """Processa resposta meteorológica"""
        if response.is_success() and response.data:
            weather = response.data
            if isinstance(weather, WeatherCondition):
                return weather.to_dict()
            else:
                return {"data": weather, "status": "success"}
        else:
            return {
                "status": "error",
                "error": response.error_message,
                "response_time_ms": response.response_time_ms
            }
    
    def _process_river_response(self, response: ExternalApiResponse) -> Dict[str, Any]:
        """Processa resposta do nível do rio"""
        if response.is_success() and response.data:
            river = response.data
            if isinstance(river, RiverLevel):
                return river.to_dict()
            else:
                return {"data": river, "status": "success"}
        else:
            return {
                "status": "error", 
                "error": response.error_message,
                "response_time_ms": response.response_time_ms
            }
    
    def _determine_overall_status(self, weather_resp: ExternalApiResponse, river_resp: ExternalApiResponse) -> str:
        """Determina status geral baseado nas respostas"""
        if weather_resp.is_success() and river_resp.is_success():
            return "healthy"
        elif weather_resp.is_success() or river_resp.is_success():
            return "partial"
        else:
            return "degraded"
    
    def _calculate_risk_assessment(self, weather_data: Dict[str, Any], river_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula avaliação de risco baseada nos dados"""
        risk = {
            "level": "low",
            "factors": [],
            "score": 0.0,
            "recommendations": []
        }
        
        # Analisar dados meteorológicos
        if weather_data.get("status") == "success" or "precipitation" in weather_data:
            precipitation = weather_data.get("precipitation", 0.0)
            wind_speed = weather_data.get("wind_speed", 0.0)
            
            if precipitation > 10.0:
                risk["factors"].append("Heavy precipitation")
                risk["score"] += 30
            elif precipitation > 2.0:
                risk["factors"].append("Moderate precipitation")
                risk["score"] += 15
            
            if wind_speed > 20.0:
                risk["factors"].append("Strong winds")
                risk["score"] += 10
        
        # Analisar dados do rio
        if river_data.get("status") == "success" or "level_meters" in river_data:
            level = river_data.get("level_meters", 0.0)
            
            if level > 3.5:
                risk["factors"].append("Critical river level")
                risk["score"] += 50
            elif level > 3.0:
                risk["factors"].append("High river level")
                risk["score"] += 30
            elif level > 2.5:
                risk["factors"].append("Elevated river level")
                risk["score"] += 15
        
        # Determinar nível de risco
        if risk["score"] >= 50:
            risk["level"] = "critical"
            risk["recommendations"].append("Immediate action required")
        elif risk["score"] >= 30:
            risk["level"] = "high"
            risk["recommendations"].append("Close monitoring recommended")
        elif risk["score"] >= 15:
            risk["level"] = "moderate"
            risk["recommendations"].append("Continue monitoring")
        else:
            risk["recommendations"].append("Normal conditions")
        
        return risk


class GetWeatherDataUseCase:
    """
    Use case para obter dados meteorológicos
    
    Gerencia obtenção de dados atuais e previsões
    meteorológicas com fallback e cache.
    """
    
    def __init__(
        self,
        weather_service: WeatherApiService,
        circuit_breaker: CircuitBreakerService,
        cache_service: ApiCacheService = None
    ):
        """Inicializa o use case"""
        self.weather_service = weather_service
        self.circuit_breaker = circuit_breaker
        self.cache_service = cache_service
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def get_current_weather(self, city: str = "Porto Alegre", use_cache: bool = True) -> Optional[WeatherCondition]:
        """Obtém condições meteorológicas atuais"""
        try:
            # Verificar cache
            cache_key = f"weather_current:{city}"
            if use_cache and self.cache_service:
                cached = await self.cache_service.get_cached_response(cache_key)
                if cached:
                    return cached
            
            # Executar com circuit breaker
            response = await self.circuit_breaker.execute(
                "weather",
                self.weather_service.get_current_weather,
                city
            )
            
            if response.is_success():
                weather = response.data
                
                # Armazenar no cache
                if use_cache and self.cache_service:
                    await self.cache_service.cache_response(cache_key, weather, ttl_seconds=300)
                
                return weather
            else:
                self.logger.warning(f"Erro ao obter dados meteorológicos: {response.error_message}")
                return None
                
        except Exception as e:
            self.logger.error(f"Erro no use case de dados meteorológicos: {e}")
            return None
    
    async def get_weather_forecast(self, city: str = "Porto Alegre", hours: int = 24, use_cache: bool = True) -> List[WeatherCondition]:
        """Obtém previsão meteorológica"""
        try:
            # Verificar cache
            cache_key = f"weather_forecast:{city}:{hours}"
            if use_cache and self.cache_service:
                cached = await self.cache_service.get_cached_response(cache_key)
                if cached:
                    return cached
            
            # Executar com circuit breaker
            response = await self.circuit_breaker.execute(
                "weather",
                self.weather_service.get_weather_forecast,
                city,
                hours
            )
            
            if response.is_success():
                forecast = response.data or []
                
                # Armazenar no cache
                if use_cache and self.cache_service:
                    await self.cache_service.cache_response(cache_key, forecast, ttl_seconds=600)
                
                return forecast
            else:
                self.logger.warning(f"Erro ao obter previsão meteorológica: {response.error_message}")
                return []
                
        except Exception as e:
            self.logger.error(f"Erro no use case de previsão meteorológica: {e}")
            return []


class GetRiverDataUseCase:
    """
    Use case para obter dados do nível do rio
    
    Gerencia obtenção de dados atuais e histórico
    do nível do rio com fallback e cache.
    """
    
    def __init__(
        self,
        river_service: RiverApiService,
        circuit_breaker: CircuitBreakerService,
        cache_service: ApiCacheService = None
    ):
        """Inicializa o use case"""
        self.river_service = river_service
        self.circuit_breaker = circuit_breaker
        self.cache_service = cache_service
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def get_current_level(self, station: str = "Porto Alegre", use_cache: bool = True) -> Optional[RiverLevel]:
        """Obtém nível atual do rio"""
        try:
            # Verificar cache
            cache_key = f"river_current:{station}"
            if use_cache and self.cache_service:
                cached = await self.cache_service.get_cached_response(cache_key)
                if cached:
                    return cached
            
            # Executar com circuit breaker
            response = await self.circuit_breaker.execute(
                "river",
                self.river_service.get_current_level,
                station
            )
            
            if response.is_success():
                level = response.data
                
                # Armazenar no cache
                if use_cache and self.cache_service:
                    await self.cache_service.cache_response(cache_key, level, ttl_seconds=180)
                
                return level
            else:
                self.logger.warning(f"Erro ao obter nível do rio: {response.error_message}")
                return None
                
        except Exception as e:
            self.logger.error(f"Erro no use case de nível do rio: {e}")
            return None
    
    async def get_level_history(self, station: str = "Porto Alegre", hours: int = 24, use_cache: bool = True) -> List[RiverLevel]:
        """Obtém histórico de níveis do rio"""
        try:
            # Verificar cache
            cache_key = f"river_history:{station}:{hours}"
            if use_cache and self.cache_service:
                cached = await self.cache_service.get_cached_response(cache_key)
                if cached:
                    return cached
            
            # Executar com circuit breaker
            response = await self.circuit_breaker.execute(
                "river",
                self.river_service.get_level_history,
                station,
                hours
            )
            
            if response.is_success():
                history = response.data or []
                
                # Armazenar no cache
                if use_cache and self.cache_service:
                    await self.cache_service.cache_response(cache_key, history, ttl_seconds=300)
                
                return history
            else:
                self.logger.warning(f"Erro ao obter histórico do rio: {response.error_message}")
                return []
                
        except Exception as e:
            self.logger.error(f"Erro no use case de histórico do rio: {e}")
            return []


class HealthCheckUseCase:
    """
    Use case para monitoramento de saúde das APIs
    
    Verifica disponibilidade e performance de todas
    as APIs externas integradas.
    """
    
    def __init__(
        self,
        weather_service: WeatherApiService,
        river_service: RiverApiService,
        circuit_breaker: CircuitBreakerService
    ):
        """Inicializa o use case"""
        self.weather_service = weather_service
        self.river_service = river_service
        self.circuit_breaker = circuit_breaker
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def execute(self) -> Dict[str, Any]:
        """
        Executa verificação de saúde de todas as APIs
        
        Returns:
            Dict: Status de saúde consolidado
        """
        try:
            self.logger.info("Iniciando health check das APIs externas")
            
            # Verificar saúde de cada API
            weather_health = await self.weather_service.health_check()
            river_health = await self.river_service.health_check()
            
            # Obter métricas do circuit breaker
            weather_metrics = await self.circuit_breaker.get_metrics("weather")
            river_metrics = await self.circuit_breaker.get_metrics("river")
            
            # Consolidar resultados
            result = {
                "timestamp": datetime.now().isoformat(),
                "overall_status": self._determine_overall_health(weather_health, river_health),
                "apis": {
                    "weather": {
                        "health": weather_health.to_dict(),
                        "circuit_breaker": weather_metrics
                    },
                    "river": {
                        "health": river_health.to_dict(),
                        "circuit_breaker": river_metrics
                    }
                },
                "summary": {
                    "healthy_apis": sum(1 for h in [weather_health, river_health] if h.is_healthy),
                    "total_apis": 2,
                    "health_percentage": self._calculate_health_percentage(weather_health, river_health)
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro no health check: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "error",
                "error": str(e)
            }
    
    def _determine_overall_health(self, weather_health: ApiHealthStatus, river_health: ApiHealthStatus) -> str:
        """Determina saúde geral do sistema"""
        if weather_health.is_healthy and river_health.is_healthy:
            return "healthy"
        elif weather_health.is_healthy or river_health.is_healthy:
            return "degraded"
        else:
            return "unhealthy"
    
    def _calculate_health_percentage(self, weather_health: ApiHealthStatus, river_health: ApiHealthStatus) -> float:
        """Calcula porcentagem de saúde"""
        healthy_count = sum(1 for h in [weather_health, river_health] if h.is_healthy)
        return (healthy_count / 2) * 100.0 