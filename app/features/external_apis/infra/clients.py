"""
API Clients - Infrastructure Layer

Implementações concretas dos clients para APIs externas:
- CptecWeatherClient: Client para API CPTEC/INPE
- GuaibaRiverClient: Client para API do nível do Rio Guaíba

Inclui tratamento de erros, timeouts, parsing de dados e logging.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import time

# HTTP client - usar aiohttp se disponível, senão httpx
try:
    import aiohttp
    HTTP_CLIENT = "aiohttp"
except ImportError:
    try:
        import httpx
        HTTP_CLIENT = "httpx"
    except ImportError:
        HTTP_CLIENT = "none"

from ..domain.entities import (
    WeatherCondition, RiverLevel, ExternalApiResponse, 
    ApiHealthStatus, ApiStatus, WeatherSeverity, RiverStatus
)
from ..domain.services import WeatherApiService, RiverApiService, ApiServiceConfig
from ..domain.services import (
    ExternalApiError, ApiTimeoutError, ApiUnavailableError,
    DataValidationError
)


class CptecWeatherClient(WeatherApiService):
    """
    Client para API CPTEC/INPE
    
    Implementa integração com o Centro de Previsão de Tempo e Estudos Climáticos
    para obtenção de dados meteorológicos em tempo real.
    """
    
    def __init__(self, base_url: str = None, timeout: float = None):
        """
        Inicializa o client CPTEC
        
        Args:
            base_url: URL base da API (opcional)
            timeout: Timeout para requests (opcional)
        """
        self.base_url = base_url or ApiServiceConfig.CPTEC_API_URL
        self.timeout = timeout or ApiServiceConfig.DEFAULT_TIMEOUT
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Estatísticas
        self._call_count = 0
        self._error_count = 0
        self._last_success = None
        self._last_error = None
    
    async def get_current_weather(self, city: str = "Porto Alegre") -> Optional[WeatherCondition]:
        """Obtém condições meteorológicas atuais do CPTEC"""
        try:
            self.logger.info(f"Buscando condições meteorológicas para {city}")
            
            # Construir URL com parâmetros
            url = f"{self.base_url}?city={city},RS"
            
            # Fazer requisição
            response_data = await self._make_request(url)
            
            if response_data and "data" in response_data:
                # Parse dos dados CPTEC
                weather = self._parse_weather_data(response_data["data"], city)
                self._last_success = datetime.now()
                return weather
            else:
                self.logger.warning("Resposta da API CPTEC vazia ou inválida")
                return None
                
        except Exception as e:
            self.logger.error(f"Erro ao obter dados meteorológicos: {e}")
            self._last_error = datetime.now()
            self._error_count += 1
            return None
    
    async def get_weather_forecast(self, city: str = "Porto Alegre", hours: int = 24) -> List[WeatherCondition]:
        """Obtém previsão meteorológica do CPTEC"""
        try:
            self.logger.info(f"Buscando previsão meteorológica para {city} ({hours}h)")
            
            # CPTEC API geralmente retorna previsão para alguns dias
            url = f"{self.base_url}?city={city},RS&forecast=true"
            
            response_data = await self._make_request(url)
            
            forecasts = []
            if response_data and "forecast" in response_data:
                # Parse dos dados de previsão
                for item in response_data["forecast"][:hours]:
                    weather = self._parse_weather_data(item, city)
                    if weather:
                        forecasts.append(weather)
            
            self._last_success = datetime.now()
            return forecasts
            
        except Exception as e:
            self.logger.error(f"Erro ao obter previsão meteorológica: {e}")
            self._last_error = datetime.now()
            self._error_count += 1
            return []
    
    async def health_check(self) -> ApiHealthStatus:
        """Verifica saúde da API CPTEC"""
        start_time = time.time()
        
        try:
            # Fazer requisição simples para testar disponibilidade
            url = f"{self.base_url}?city=Porto Alegre,RS"
            await self._make_request(url, timeout=5.0)
            
            response_time = (time.time() - start_time) * 1000  # ms
            
            # Calcular taxa de sucesso
            total_calls = max(self._call_count, 1)
            success_rate = (total_calls - self._error_count) / total_calls
            
            return ApiHealthStatus(
                api_name="CPTEC",
                is_healthy=True,
                last_check=datetime.now(),
                success_rate=success_rate,
                avg_response_time_ms=response_time,
                error_count_last_hour=self._error_count,
                circuit_state="CLOSED"
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return ApiHealthStatus(
                api_name="CPTEC",
                is_healthy=False,
                last_check=datetime.now(),
                success_rate=0.0,
                avg_response_time_ms=response_time,
                error_count_last_hour=self._error_count + 1,
                circuit_state="OPEN",
                recent_errors=[str(e)]
            )
    
    async def _make_request(self, url: str, timeout: float = None) -> Dict[str, Any]:
        """Faz requisição HTTP para a API"""
        timeout = timeout or self.timeout
        self._call_count += 1
        
        if HTTP_CLIENT == "none":
            # Fallback: simular resposta para testes
            return self._get_mock_response()
        
        try:
            if HTTP_CLIENT == "aiohttp":
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                    async with session.get(url, headers=ApiServiceConfig.DEFAULT_HEADERS) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            raise ApiUnavailableError(f"HTTP {response.status}")
            
            elif HTTP_CLIENT == "httpx":
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(url, headers=ApiServiceConfig.DEFAULT_HEADERS)
                    if response.status_code == 200:
                        return response.json()
                    else:
                        raise ApiUnavailableError(f"HTTP {response.status_code}")
                        
        except asyncio.TimeoutError:
            raise ApiTimeoutError(f"Timeout na requisição para {url}")
        except Exception as e:
            raise ExternalApiError(f"Erro na requisição: {e}")
    
    def _parse_weather_data(self, data: Dict[str, Any], city: str) -> Optional[WeatherCondition]:
        """Parse dos dados meteorológicos do CPTEC"""
        try:
            # Adaptar campos da API CPTEC para nossa estrutura
            return WeatherCondition(
                timestamp=datetime.now(),  # CPTEC pode não ter timestamp exato
                city=city,
                temperature=float(data.get("temperature", 25.0)),
                humidity=float(data.get("humidity", 60.0)),
                pressure=float(data.get("pressure", 1013.0)),
                wind_speed=float(data.get("wind_speed", 5.0)),
                wind_direction=float(data.get("wind_direction", 180.0)),
                precipitation=float(data.get("precipitation", 0.0)),
                cloud_cover=data.get("cloud_cover"),
                visibility=data.get("visibility"),
                uv_index=data.get("uv_index"),
                source_api="cptec",
                description=data.get("description")
            )
            
        except (ValueError, KeyError) as e:
            self.logger.error(f"Erro no parse dos dados CPTEC: {e}")
            return None
    
    def _get_mock_response(self) -> Dict[str, Any]:
        """Resposta mock para testes sem dependências HTTP"""
        return {
            "data": {
                "temperature": 23.5,
                "humidity": 65.0,
                "pressure": 1015.2,
                "wind_speed": 8.5,
                "wind_direction": 220.0,
                "precipitation": 0.2,
                "cloud_cover": 40.0,
                "description": "Parcialmente nublado"
            },
            "forecast": [
                {
                    "temperature": 24.0,
                    "humidity": 70.0,
                    "pressure": 1014.8,
                    "wind_speed": 9.0,
                    "wind_direction": 210.0,
                    "precipitation": 1.5
                }
            ]
        }


class GuaibaRiverClient(RiverApiService):
    """
    Client para API do nível do Rio Guaíba
    
    Implementa integração com o sistema de monitoramento do nível
    do Rio Guaíba em Porto Alegre.
    """
    
    def __init__(self, base_url: str = None, timeout: float = None):
        """
        Inicializa o client do Rio Guaíba
        
        Args:
            base_url: URL base da API (opcional)
            timeout: Timeout para requests (opcional)
        """
        self.base_url = base_url or ApiServiceConfig.GUAIBA_API_URL
        self.timeout = timeout or ApiServiceConfig.DEFAULT_TIMEOUT
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Estatísticas
        self._call_count = 0
        self._error_count = 0
        self._last_success = None
        self._last_error = None
    
    async def get_current_level(self, station: str = "Porto Alegre") -> Optional[RiverLevel]:
        """Obtém nível atual do Rio Guaíba"""
        try:
            self.logger.info(f"Buscando nível atual do rio para {station}")
            
            # Fazer requisição para API do Guaíba
            response_data = await self._make_request(self.base_url)
            
            if response_data and isinstance(response_data, list) and len(response_data) > 0:
                # Pegar o registro mais recente
                latest_data = response_data[0]
                level = self._parse_river_data(latest_data, station)
                self._last_success = datetime.now()
                return level
            else:
                self.logger.warning("Resposta da API do Guaíba vazia ou inválida")
                return None
                
        except Exception as e:
            self.logger.error(f"Erro ao obter nível do rio: {e}")
            self._last_error = datetime.now()
            self._error_count += 1
            return None
    
    async def get_level_history(self, station: str = "Porto Alegre", hours: int = 24) -> List[RiverLevel]:
        """Obtém histórico de níveis do Rio Guaíba"""
        try:
            self.logger.info(f"Buscando histórico do rio para {station} ({hours}h)")
            
            response_data = await self._make_request(self.base_url)
            
            levels = []
            if response_data and isinstance(response_data, list):
                # Filtrar dados das últimas horas
                cutoff_time = datetime.now() - timedelta(hours=hours)
                
                for item in response_data:
                    level = self._parse_river_data(item, station)
                    if level and level.timestamp >= cutoff_time:
                        levels.append(level)
            
            # Ordenar por timestamp
            levels.sort(key=lambda x: x.timestamp)
            self._last_success = datetime.now()
            return levels
            
        except Exception as e:
            self.logger.error(f"Erro ao obter histórico do rio: {e}")
            self._last_error = datetime.now()
            self._error_count += 1
            return []
    
    async def health_check(self) -> ApiHealthStatus:
        """Verifica saúde da API do Rio Guaíba"""
        start_time = time.time()
        
        try:
            await self._make_request(self.base_url, timeout=5.0)
            
            response_time = (time.time() - start_time) * 1000  # ms
            
            # Calcular taxa de sucesso
            total_calls = max(self._call_count, 1)
            success_rate = (total_calls - self._error_count) / total_calls
            
            return ApiHealthStatus(
                api_name="GuaibaRiver",
                is_healthy=True,
                last_check=datetime.now(),
                success_rate=success_rate,
                avg_response_time_ms=response_time,
                error_count_last_hour=self._error_count,
                circuit_state="CLOSED"
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return ApiHealthStatus(
                api_name="GuaibaRiver",
                is_healthy=False,
                last_check=datetime.now(),
                success_rate=0.0,
                avg_response_time_ms=response_time,
                error_count_last_hour=self._error_count + 1,
                circuit_state="OPEN",
                recent_errors=[str(e)]
            )
    
    async def _make_request(self, url: str, timeout: float = None) -> List[Dict[str, Any]]:
        """Faz requisição HTTP para a API do Guaíba"""
        timeout = timeout or self.timeout
        self._call_count += 1
        
        if HTTP_CLIENT == "none":
            # Fallback: simular resposta para testes
            return self._get_mock_response()
        
        try:
            if HTTP_CLIENT == "aiohttp":
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                    async with session.get(url, headers=ApiServiceConfig.DEFAULT_HEADERS) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            raise ApiUnavailableError(f"HTTP {response.status}")
            
            elif HTTP_CLIENT == "httpx":
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(url, headers=ApiServiceConfig.DEFAULT_HEADERS)
                    if response.status_code == 200:
                        return response.json()
                    else:
                        raise ApiUnavailableError(f"HTTP {response.status_code}")
                        
        except asyncio.TimeoutError:
            raise ApiTimeoutError(f"Timeout na requisição para {url}")
        except Exception as e:
            raise ExternalApiError(f"Erro na requisição: {e}")
    
    def _parse_river_data(self, data: Dict[str, Any], station: str) -> Optional[RiverLevel]:
        """Parse dos dados do nível do rio"""
        try:
            # Adaptar campos da API do Guaíba para nossa estrutura
            # Formato típico: {"time": "2024-01-01T12:00:00", "level": 2.85}
            
            timestamp_str = data.get("time") or data.get("timestamp") or data.get("date")
            level_value = data.get("level") or data.get("altura") or data.get("value")
            
            if timestamp_str and level_value is not None:
                # Parse do timestamp
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                else:
                    timestamp = datetime.now()
                
                return RiverLevel(
                    timestamp=timestamp,
                    level_meters=float(level_value),
                    station_name=station,
                    reference_level=0.0,  # Nível do mar
                    quality_code=data.get("quality"),
                    measurement_type="automatic",
                    source_api="nivelguaiba"
                )
            else:
                self.logger.warning(f"Dados insuficientes para criar RiverLevel: {data}")
                return None
            
        except (ValueError, KeyError) as e:
            self.logger.error(f"Erro no parse dos dados do rio: {e}")
            return None
    
    def _get_mock_response(self) -> List[Dict[str, Any]]:
        """Resposta mock para testes sem dependências HTTP"""
        now = datetime.now()
        return [
            {
                "time": now.isoformat(),
                "level": 2.85,
                "quality": "good"
            },
            {
                "time": (now - timedelta(hours=1)).isoformat(),
                "level": 2.83,
                "quality": "good"
            },
            {
                "time": (now - timedelta(hours=2)).isoformat(),
                "level": 2.81,
                "quality": "good"
            }
        ] 