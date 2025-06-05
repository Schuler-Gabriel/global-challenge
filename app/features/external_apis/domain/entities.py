"""
Domain Entities - External APIs Feature

Este módulo define as entidades de domínio para integração com APIs externas,
representando dados de nível do rio, condições meteorológicas e respostas de APIs.

Entidades:
- RiverLevel: Dados do nível do Rio Guaíba
- WeatherCondition: Condições meteorológicas atuais
- ExternalApiResponse: Resposta padronizada de APIs externas
- CircuitBreakerState: Estado do Circuit Breaker
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class RiverStatus(Enum):
    """Status do nível do rio baseado em thresholds"""
    NORMAL = "normal"           # < 2.5m
    ATTENTION = "attention"     # 2.5m - 3.0m
    ALERT = "alert"            # 3.0m - 3.5m
    EMERGENCY = "emergency"     # > 3.5m
    UNKNOWN = "unknown"


class WeatherSeverity(Enum):
    """Severidade das condições meteorológicas"""
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    EXTREME = "extreme"
    UNKNOWN = "unknown"


class ApiStatus(Enum):
    """Status de resposta das APIs"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CIRCUIT_OPEN = "circuit_open"
    UNAVAILABLE = "unavailable"


class RiskLevel(Enum):
    """Níveis de risco baseados no nível do rio"""
    BAIXO = "baixo"
    MODERADO = "moderado"
    ALTO = "alto"
    CRITICO = "critico"


class CircuitBreakerStatus(Enum):
    """Estados do Circuit Breaker"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RiverLevel:
    """
    Entidade representando dados do nível do Rio Guaíba
    
    Baseada na API: https://nivelguaiba.com.br/portoalegre.1day.json
    """
    timestamp: datetime
    level_meters: float  # Nível em metros
    station_name: str
    reference_level: float = 0.0  # Nível de referência
    
    # Metadados opcionais
    quality_code: Optional[str] = None
    measurement_type: Optional[str] = None
    source_api: str = "nivelguaiba"
    
    def __post_init__(self):
        """Validação após inicialização"""
        self._validate_level()
    
    def _validate_level(self):
        """Valida o nível do rio"""
        if self.level_meters < -1.0 or self.level_meters > 10.0:
            raise ValueError(f"Nível do rio fora do range válido: {self.level_meters}")
    
    def get_status(self) -> RiverStatus:
        """Determina o status baseado no nível"""
        if self.level_meters < 2.5:
            return RiverStatus.NORMAL
        elif self.level_meters < 3.0:
            return RiverStatus.ATTENTION
        elif self.level_meters < 3.5:
            return RiverStatus.ALERT
        else:
            return RiverStatus.EMERGENCY
    
    def is_critical_level(self) -> bool:
        """Verifica se o nível é crítico (acima de 3.5m)"""
        return self.level_meters > 3.5
    
    def get_relative_level(self) -> float:
        """Calcula nível relativo ao reference_level"""
        return self.level_meters - self.reference_level
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level_meters': self.level_meters,
            'station_name': self.station_name,
            'reference_level': self.reference_level,
            'quality_code': self.quality_code,
            'measurement_type': self.measurement_type,
            'source_api': self.source_api,
            'status': self.get_status().value,
            'is_critical': self.is_critical_level(),
            'relative_level': self.get_relative_level()
        }


@dataclass
class WeatherCondition:
    """
    Entidade representando condições meteorológicas atuais
    
    Baseada na API CPTEC/INPE para Porto Alegre
    """
    timestamp: datetime
    temperature: float  # °C
    humidity: float  # %
    pressure: float  # hPa/mbar
    wind_speed: float  # m/s
    wind_direction: float  # graus (0-360)
    description: str  # Descrição textual
    
    # Dados de precipitação
    precipitation_current: Optional[float] = None  # mm/h atual
    precipitation_forecast_1h: Optional[float] = None  # mm previsto próxima 1h
    precipitation_forecast_6h: Optional[float] = None  # mm previsto próximas 6h
    precipitation_forecast_24h: Optional[float] = None  # mm previsto próximas 24h
    
    # Metadados
    station_id: Optional[str] = None
    data_source: str = "CPTEC"
    forecast_confidence: Optional[float] = None  # 0.0-1.0
    
    def __post_init__(self):
        """Validação após inicialização"""
        self._validate_conditions()
    
    def _validate_conditions(self):
        """Valida se as condições estão dentro de ranges válidos"""
        # Validações baseadas em climatologia de Porto Alegre
        if not (-10 <= self.temperature <= 50):
            raise ValueError(f"Temperatura fora do range válido: {self.temperature}°C")
        
        if not (0 <= self.humidity <= 100):
            raise ValueError(f"Umidade fora do range válido: {self.humidity}%")
        
        if not (900 <= self.pressure <= 1100):
            raise ValueError(f"Pressão fora do range válido: {self.pressure}hPa")
        
        if not (0 <= self.wind_speed <= 50):
            raise ValueError(f"Velocidade do vento fora do range válido: {self.wind_speed}m/s")
        
        if not (0 <= self.wind_direction <= 360):
            raise ValueError(f"Direção do vento fora do range válido: {self.wind_direction}°")
    
    def is_storm_conditions(self) -> bool:
        """
        Verifica se há condições de tempestade
        
        Returns:
            bool: True se condições indicam tempestade
        """
        return (
            (self.precipitation_current and self.precipitation_current > 10.0) or
            self.wind_speed > 15.0 or
            (self.precipitation_forecast_1h and self.precipitation_forecast_1h > 20.0)
        )
    
    def is_rain_expected(self) -> bool:
        """
        Verifica se chuva é esperada nas próximas horas
        
        Returns:
            bool: True se chuva é esperada
        """
        return (
            (self.precipitation_forecast_1h and self.precipitation_forecast_1h > 0.1) or
            (self.precipitation_forecast_6h and self.precipitation_forecast_6h > 1.0)
        )
    
    def get_comfort_index(self) -> str:
        """
        Calcula índice de conforto térmico simplificado
        
        Returns:
            str: Nível de conforto ("confortável", "desconfortável", "extremo")
        """
        # Índice simplificado baseado em temperatura e umidade
        if 18 <= self.temperature <= 26 and 40 <= self.humidity <= 70:
            return "confortável"
        elif 10 <= self.temperature <= 35 and 20 <= self.humidity <= 80:
            return "desconfortável"
        else:
            return "extremo"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte para dicionário
        
        Returns:
            Dict: Representação em dicionário
        """
        return {
            'timestamp': self.timestamp.isoformat(),
            'temperature': self.temperature,
            'humidity': self.humidity,
            'pressure': self.pressure,
            'wind_speed': self.wind_speed,
            'wind_direction': self.wind_direction,
            'description': self.description,
            'precipitation_current': self.precipitation_current,
            'precipitation_forecast_1h': self.precipitation_forecast_1h,
            'precipitation_forecast_6h': self.precipitation_forecast_6h,
            'precipitation_forecast_24h': self.precipitation_forecast_24h,
            'station_id': self.station_id,
            'data_source': self.data_source,
            'forecast_confidence': self.forecast_confidence,
            'is_storm_conditions': self.is_storm_conditions(),
            'is_rain_expected': self.is_rain_expected(),
            'comfort_index': self.get_comfort_index()
        }


@dataclass
class ExternalApiResponse:
    """
    Entidade representando resposta padronizada de APIs externas
    
    Encapsula dados, metadados e status de qualquer chamada para API externa.
    """
    success: bool
    data: Optional[Any]  # RiverLevel, WeatherCondition, etc.
    timestamp: datetime
    api_name: str
    
    # Metadados da requisição
    response_time_ms: float
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Cache information
    from_cache: bool = False
    cache_ttl_seconds: Optional[int] = None
    
    def __post_init__(self):
        """Validação após inicialização"""
        if not self.success and not self.error_message:
            raise ValueError("Response sem sucesso deve ter error_message")
    
    def is_fresh_data(self, max_age_seconds: int = 300) -> bool:
        """
        Verifica se os dados são recentes
        
        Args:
            max_age_seconds: Idade máxima em segundos (padrão: 5 minutos)
            
        Returns:
            bool: True se dados são recentes
        """
        age = (datetime.now() - self.timestamp).total_seconds()
        return age <= max_age_seconds
    
    def is_fast_response(self, threshold_ms: float = 1000.0) -> bool:
        """
        Verifica se a resposta foi rápida
        
        Args:
            threshold_ms: Limite em millisegundos
            
        Returns:
            bool: True se resposta foi rápida
        """
        return self.response_time_ms <= threshold_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte para dicionário
        
        Returns:
            Dict: Representação em dicionário
        """
        return {
            'success': self.success,
            'data': self.data.to_dict() if hasattr(self.data, 'to_dict') else self.data,
            'timestamp': self.timestamp.isoformat(),
            'api_name': self.api_name,
            'response_time_ms': self.response_time_ms,
            'status_code': self.status_code,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'from_cache': self.from_cache,
            'cache_ttl_seconds': self.cache_ttl_seconds,
            'is_fresh_data': self.is_fresh_data(),
            'is_fast_response': self.is_fast_response()
        }


@dataclass
class CircuitBreakerState:
    """
    Entidade representando o estado do Circuit Breaker
    
    Controla o acesso a APIs externas baseado em falhas e recuperação.
    """
    status: CircuitBreakerStatus
    failure_count: int
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]
    
    # Configurações
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    half_open_max_calls: int = 3
    
    # Métricas
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    
    def __post_init__(self):
        """Validação após inicialização"""
        if self.failure_count < 0:
            raise ValueError("failure_count não pode ser negativo")
    
    def should_allow_call(self) -> bool:
        """
        Determina se uma chamada deve ser permitida
        
        Returns:
            bool: True se chamada deve ser permitida
        """
        if self.status == CircuitBreakerStatus.CLOSED:
            return True
        elif self.status == CircuitBreakerStatus.OPEN:
            # Verificar se é hora de tentar recuperação
            if self.last_failure_time:
                time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
                return time_since_failure >= self.recovery_timeout_seconds
            return False
        elif self.status == CircuitBreakerStatus.HALF_OPEN:
            # Permitir número limitado de tentativas
            return self.total_calls < self.half_open_max_calls
        
        return False
    
    def get_failure_rate(self) -> float:
        """
        Calcula taxa de falha
        
        Returns:
            float: Taxa de falha (0.0-1.0)
        """
        if self.total_calls == 0:
            return 0.0
        return self.total_failures / self.total_calls
    
    def get_availability(self) -> float:
        """
        Calcula disponibilidade
        
        Returns:
            float: Disponibilidade (0.0-1.0)
        """
        return 1.0 - self.get_failure_rate()
    
    def time_until_retry(self) -> Optional[float]:
        """
        Calcula tempo até próxima tentativa
        
        Returns:
            Optional[float]: Segundos até próxima tentativa, ou None se pode tentar agora
        """
        if self.status != CircuitBreakerStatus.OPEN or not self.last_failure_time:
            return None
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        remaining = self.recovery_timeout_seconds - time_since_failure
        
        return max(0, remaining) if remaining > 0 else None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte para dicionário
        
        Returns:
            Dict: Representação em dicionário
        """
        return {
            'status': self.status.value,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'last_success_time': self.last_success_time.isoformat() if self.last_success_time else None,
            'failure_threshold': self.failure_threshold,
            'recovery_timeout_seconds': self.recovery_timeout_seconds,
            'half_open_max_calls': self.half_open_max_calls,
            'total_calls': self.total_calls,
            'total_failures': self.total_failures,
            'total_successes': self.total_successes,
            'failure_rate': self.get_failure_rate(),
            'availability': self.get_availability(),
            'should_allow_call': self.should_allow_call(),
            'time_until_retry': self.time_until_retry()
        }


@dataclass
class ApiHealthStatus:
    """
    Entidade representando status de saúde de uma API externa
    
    Usado para monitoramento e health checks das APIs.
    """
    api_name: str
    status: str  # "healthy", "degraded", "unhealthy"
    response_time_ms: Optional[float] = None
    last_check_time: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Métricas agregadas
    success_rate_24h: Optional[float] = None
    avg_response_time_24h: Optional[float] = None
    total_requests_24h: Optional[int] = None
    
    def is_healthy(self) -> bool:
        """Verifica se a API está saudável"""
        return self.status == "healthy"
    
    def is_fast_response(self, threshold_ms: float = 1000.0) -> bool:
        """Verifica se a resposta está dentro do threshold"""
        if not self.response_time_ms:
            return False
        return self.response_time_ms <= threshold_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return {
            'api_name': self.api_name,
            'status': self.status,
            'response_time_ms': self.response_time_ms,
            'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
            'error_message': self.error_message,
            'success_rate_24h': self.success_rate_24h,
            'avg_response_time_24h': self.avg_response_time_24h,
            'total_requests_24h': self.total_requests_24h,
            'is_healthy': self.is_healthy(),
            'is_fast_response': self.is_fast_response()
        } 