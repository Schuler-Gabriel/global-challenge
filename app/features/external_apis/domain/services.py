"""
Domain Services - External APIs Feature

Este módulo define as interfaces de serviços de domínio para integração
com APIs externas. Os serviços abstratos definem contratos que serão
implementados na camada de infraestrutura.

Services:
- WeatherApiService: Interface para dados meteorológicos
- RiverApiService: Interface para dados do rio
- CircuitBreakerService: Interface para circuit breaker
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any, List

from .entities import (
    RiverLevel, WeatherCondition, ExternalApiResponse, 
    ApiHealthStatus, ApiStatus
)


class WeatherApiService(ABC):
    """
    Interface abstrata para serviços de dados meteorológicos
    
    Define o contrato para obtenção de dados meteorológicos
    em tempo real de diferentes provedores (CPTEC, OpenWeather, etc.)
    """
    
    @abstractmethod
    async def get_current_weather(self, city: str = "Porto Alegre") -> Optional[WeatherCondition]:
        """
        Obtém condições meteorológicas atuais
        
        Args:
            city: Nome da cidade
            
        Returns:
            Optional[WeatherCondition]: Condições atuais ou None se erro
        """
        pass
    
    @abstractmethod
    async def get_weather_forecast(self, city: str = "Porto Alegre", hours: int = 24) -> List[WeatherCondition]:
        """
        Obtém previsão meteorológica
        
        Args:
            city: Nome da cidade
            hours: Número de horas de previsão
            
        Returns:
            List[WeatherCondition]: Lista de previsões
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> ApiHealthStatus:
        """
        Verifica saúde da API meteorológica
        
        Returns:
            ApiHealthStatus: Status de saúde da API
        """
        pass


class RiverApiService(ABC):
    """
    Interface abstrata para serviços de dados do rio
    
    Define o contrato para obtenção de dados do nível
    do Rio Guaíba em tempo real.
    """
    
    @abstractmethod
    async def get_current_level(self, station: str = "Porto Alegre") -> Optional[RiverLevel]:
        """
        Obtém nível atual do rio
        
        Args:
            station: Nome da estação de medição
            
        Returns:
            Optional[RiverLevel]: Nível atual ou None se erro
        """
        pass
    
    @abstractmethod
    async def get_level_history(self, station: str = "Porto Alegre", hours: int = 24) -> List[RiverLevel]:
        """
        Obtém histórico de níveis do rio
        
        Args:
            station: Nome da estação de medição
            hours: Número de horas de histórico
            
        Returns:
            List[RiverLevel]: Lista de níveis históricos
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> ApiHealthStatus:
        """
        Verifica saúde da API do rio
        
        Returns:
            ApiHealthStatus: Status de saúde da API
        """
        pass


class CircuitBreakerService(ABC):
    """
    Interface abstrata para circuit breaker
    
    Implementa padrão Circuit Breaker para proteção
    contra falhas em cascata de APIs externas.
    """
    
    @abstractmethod
    async def execute(self, api_name: str, operation: callable, *args, **kwargs) -> ExternalApiResponse:
        """
        Executa operação com proteção de circuit breaker
        
        Args:
            api_name: Nome da API
            operation: Função a ser executada
            *args: Argumentos posicionais
            **kwargs: Argumentos nomeados
            
        Returns:
            ExternalApiResponse: Resposta da operação
        """
        pass
    
    @abstractmethod
    async def get_circuit_status(self, api_name: str) -> str:
        """
        Obtém status atual do circuit breaker
        
        Args:
            api_name: Nome da API
            
        Returns:
            str: Status do circuit ("CLOSED", "OPEN", "HALF_OPEN")
        """
        pass
    
    @abstractmethod
    async def reset_circuit(self, api_name: str) -> bool:
        """
        Reseta circuit breaker para CLOSED
        
        Args:
            api_name: Nome da API
            
        Returns:
            bool: True se resetou com sucesso
        """
        pass
    
    @abstractmethod
    async def get_metrics(self, api_name: str) -> Dict[str, Any]:
        """
        Obtém métricas do circuit breaker
        
        Args:
            api_name: Nome da API
            
        Returns:
            Dict: Métricas de performance e falhas
        """
        pass


class ExternalApiAggregatorService(ABC):
    """
    Interface abstrata para agregação de dados de APIs externas
    
    Combina dados de múltiplas APIs para fornecer
    visão consolidada das condições atuais.
    """
    
    @abstractmethod
    async def get_current_conditions(self) -> Dict[str, Any]:
        """
        Obtém condições atuais consolidadas
        
        Combina dados meteorológicos e nível do rio
        
        Returns:
            Dict: Condições consolidadas
        """
        pass
    
    @abstractmethod
    async def get_risk_assessment(self) -> Dict[str, Any]:
        """
        Avalia risco atual baseado em todas as APIs
        
        Returns:
            Dict: Avaliação de risco consolidada
        """
        pass
    
    @abstractmethod
    async def get_apis_health(self) -> Dict[str, ApiHealthStatus]:
        """
        Obtém status de saúde de todas as APIs
        
        Returns:
            Dict: Status de cada API por nome
        """
        pass


class ApiRetryService(ABC):
    """
    Interface abstrata para lógica de retry
    
    Implementa estratégias de retry com backoff
    exponencial para APIs externas.
    """
    
    @abstractmethod
    async def execute_with_retry(
        self, 
        operation: callable, 
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        timeout: float = 10.0,
        *args, 
        **kwargs
    ) -> ExternalApiResponse:
        """
        Executa operação com retry automático
        
        Args:
            operation: Função a ser executada
            max_retries: Número máximo de tentativas
            backoff_factor: Fator de backoff exponencial
            timeout: Timeout por tentativa
            *args: Argumentos posicionais
            **kwargs: Argumentos nomeados
            
        Returns:
            ExternalApiResponse: Resposta da operação
        """
        pass
    
    @abstractmethod
    async def should_retry(self, response: ExternalApiResponse, attempt: int) -> bool:
        """
        Determina se deve tentar novamente
        
        Args:
            response: Resposta da tentativa anterior
            attempt: Número da tentativa atual
            
        Returns:
            bool: True se deve tentar novamente
        """
        pass


class ApiCacheService(ABC):
    """
    Interface abstrata para cache de APIs externas
    
    Implementa cache inteligente com TTL baseado
    no tipo de dado e freshness requirements.
    """
    
    @abstractmethod
    async def get_cached_response(self, cache_key: str) -> Optional[Any]:
        """
        Obtém resposta do cache
        
        Args:
            cache_key: Chave do cache
            
        Returns:
            Optional[Any]: Dados cacheados ou None
        """
        pass
    
    @abstractmethod
    async def cache_response(
        self, 
        cache_key: str, 
        data: Any, 
        ttl_seconds: int = 300
    ) -> bool:
        """
        Armazena resposta no cache
        
        Args:
            cache_key: Chave do cache
            data: Dados a serem cacheados
            ttl_seconds: Tempo de vida em segundos
            
        Returns:
            bool: True se armazenou com sucesso
        """
        pass
    
    @abstractmethod
    async def invalidate_cache(self, pattern: str = "*") -> int:
        """
        Invalida entradas do cache
        
        Args:
            pattern: Padrão de chaves a invalidar
            
        Returns:
            int: Número de entradas invalidadas
        """
        pass
    
    @abstractmethod
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtém estatísticas do cache
        
        Returns:
            Dict: Estatísticas de hit rate, size, etc.
        """
        pass


# Configurações padrão para services
class ApiServiceConfig:
    """Configurações padrão para serviços de API"""
    
    # Timeouts
    DEFAULT_TIMEOUT = 10.0  # segundos
    LONG_TIMEOUT = 30.0     # para operações pesadas
    
    # Retry
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BACKOFF_FACTOR = 2.0
    
    # Circuit Breaker
    FAILURE_THRESHOLD = 5        # falhas para abrir circuit
    SUCCESS_THRESHOLD = 3        # sucessos para fechar circuit
    TIMEOUT_THRESHOLD = 5000     # ms para considerar timeout
    CIRCUIT_OPEN_DURATION = 60   # segundos que circuit fica aberto
    
    # Cache TTL (segundos)
    WEATHER_CACHE_TTL = 300      # 5 minutos
    RIVER_CACHE_TTL = 180        # 3 minutos
    FORECAST_CACHE_TTL = 600     # 10 minutos
    HEALTH_CACHE_TTL = 60        # 1 minuto
    
    # URLs das APIs
    GUAIBA_API_URL = "https://nivelguaiba.com.br/portoalegre.1day.json"
    CPTEC_API_URL = "https://www.cptec.inpe.br/api/forecast-input"
    
    # Headers padrão
    DEFAULT_HEADERS = {
        "User-Agent": "Weather-Flood-Alert-System/1.0",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }


# Exceções específicas para serviços de API
class ExternalApiError(Exception):
    """Exceção base para erros de APIs externas"""
    pass


class ApiTimeoutError(ExternalApiError):
    """Exceção para timeouts de API"""
    pass


class ApiUnavailableError(ExternalApiError):
    """Exceção para APIs indisponíveis"""
    pass


class CircuitBreakerOpenError(ExternalApiError):
    """Exceção quando circuit breaker está aberto"""
    pass


class ApiRateLimitError(ExternalApiError):
    """Exceção para rate limiting"""
    pass


class ApiAuthError(ExternalApiError):
    """Exceção para erros de autenticação"""
    pass


class DataValidationError(ExternalApiError):
    """Exceção para dados inválidos retornados pela API"""
    pass


class MonitoringService(ABC):
    """
    Interface abstrata para monitoramento de APIs externas
    
    Coleta métricas e monitora performance das APIs
    """
    
    @abstractmethod
    async def record_api_call(self, api_name: str, response_time: float, success: bool) -> None:
        """
        Registra chamada de API
        
        Args:
            api_name: Nome da API
            response_time: Tempo de resposta em ms
            success: Se a chamada foi bem-sucedida
        """
        pass
    
    @abstractmethod
    async def get_api_metrics(self, api_name: str) -> Dict[str, Any]:
        """
        Obtém métricas de uma API
        
        Args:
            api_name: Nome da API
            
        Returns:
            Dict: Métricas da API
        """
        pass
    
    @abstractmethod
    async def get_all_metrics(self) -> Dict[str, Any]:
        """
        Obtém métricas de todas as APIs
        
        Returns:
            Dict: Métricas consolidadas
        """
        pass 