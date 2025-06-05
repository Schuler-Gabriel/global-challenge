"""
Domain Repositories - Forecast Feature

Este módulo define as interfaces abstratas (repositories) que representam
contratos para acesso a dados. As implementações concretas ficam na camada
de infraestrutura.

Repositories:
- WeatherDataRepository: Interface para dados meteorológicos
- ForecastRepository: Interface para previsões
- ModelRepository: Interface para modelos ML
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any, Protocol
from dataclasses import dataclass

from .entities import WeatherData, Forecast, ModelMetrics


@dataclass
class WeatherDataQuery:
    """Query object para busca de dados meteorológicos"""
    station_id: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: Optional[int] = None
    order_by: str = 'timestamp'
    order_direction: str = 'asc'  # 'asc' ou 'desc'


@dataclass
class ForecastQuery:
    """Query object para busca de previsões"""
    model_version: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_confidence: Optional[float] = None
    limit: Optional[int] = None
    order_by: str = 'timestamp'
    order_direction: str = 'desc'  # 'asc' ou 'desc'


class WeatherDataRepository(ABC):
    """
    Interface abstrata para acesso a dados meteorológicos históricos
    
    Esta interface define o contrato para operações de dados meteorológicos,
    permitindo diferentes implementações (arquivo, banco de dados, API, etc.)
    """
    
    @abstractmethod
    async def get_latest_data(self, count: int = 24) -> List[WeatherData]:
        """
        Busca os dados meteorológicos mais recentes
        
        Args:
            count: Número de registros mais recentes
            
        Returns:
            List[WeatherData]: Lista de dados meteorológicos
        """
        pass
    
    @abstractmethod
    async def get_data_by_period(self, start_date: datetime, end_date: datetime) -> List[WeatherData]:
        """
        Busca dados meteorológicos por período
        
        Args:
            start_date: Data inicial
            end_date: Data final
            
        Returns:
            List[WeatherData]: Lista de dados no período
        """
        pass
    
    @abstractmethod
    async def get_data_by_query(self, query: WeatherDataQuery) -> List[WeatherData]:
        """
        Busca dados meteorológicos usando query object
        
        Args:
            query: Critérios de busca
            
        Returns:
            List[WeatherData]: Lista de dados que atendem aos critérios
        """
        pass
    
    @abstractmethod
    async def save_data(self, weather_data: WeatherData) -> bool:
        """
        Salva um registro de dados meteorológicos
        
        Args:
            weather_data: Dados a serem salvos
            
        Returns:
            bool: True se salvou com sucesso
        """
        pass
    
    @abstractmethod
    async def save_batch_data(self, weather_data_list: List[WeatherData]) -> int:
        """
        Salva múltiplos registros de dados meteorológicos
        
        Args:
            weather_data_list: Lista de dados a serem salvos
            
        Returns:
            int: Número de registros salvos com sucesso
        """
        pass
    
    @abstractmethod
    async def count_records(self, query: Optional[WeatherDataQuery] = None) -> int:
        """
        Conta o número de registros que atendem aos critérios
        
        Args:
            query: Critérios de busca (opcional)
            
        Returns:
            int: Número de registros
        """
        pass
    
    @abstractmethod
    async def get_data_statistics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Calcula estatísticas dos dados meteorológicos
        
        Args:
            start_date: Data inicial
            end_date: Data final
            
        Returns:
            Dict: Estatísticas calculadas
        """
        pass


class ForecastRepository(ABC):
    """
    Interface abstrata para acesso a previsões meteorológicas
    
    Define operações para armazenar e recuperar previsões geradas
    pelo modelo LSTM.
    """
    
    @abstractmethod
    async def save_forecast(self, forecast: Forecast) -> bool:
        """
        Salva uma previsão
        
        Args:
            forecast: Previsão a ser salva
            
        Returns:
            bool: True se salvou com sucesso
        """
        pass
    
    @abstractmethod
    async def get_latest_forecast(self) -> Optional[Forecast]:
        """
        Busca a previsão mais recente
        
        Returns:
            Optional[Forecast]: Última previsão ou None
        """
        pass
    
    @abstractmethod
    async def get_forecasts_by_period(self, start_date: datetime, end_date: datetime) -> List[Forecast]:
        """
        Busca previsões por período
        
        Args:
            start_date: Data inicial
            end_date: Data final
            
        Returns:
            List[Forecast]: Lista de previsões no período
        """
        pass
    
    @abstractmethod
    async def get_forecasts_by_query(self, query: ForecastQuery) -> List[Forecast]:
        """
        Busca previsões usando query object
        
        Args:
            query: Critérios de busca
            
        Returns:
            List[Forecast]: Lista de previsões que atendem aos critérios
        """
        pass
    
    @abstractmethod
    async def get_forecast_by_id(self, forecast_id: str) -> Optional[Forecast]:
        """
        Busca previsão por ID
        
        Args:
            forecast_id: ID da previsão
            
        Returns:
            Optional[Forecast]: Previsão ou None se não encontrada
        """
        pass
    
    @abstractmethod
    async def delete_old_forecasts(self, cutoff_date: datetime) -> int:
        """
        Remove previsões antigas
        
        Args:
            cutoff_date: Data limite (anteriores serão removidas)
            
        Returns:
            int: Número de registros removidos
        """
        pass
    
    @abstractmethod
    async def get_forecast_accuracy_metrics(self, model_version: str) -> Dict[str, Any]:
        """
        Calcula métricas de accuracy das previsões vs dados reais
        
        Args:
            model_version: Versão do modelo
            
        Returns:
            Dict: Métricas de accuracy
        """
        pass


class ModelRepository(ABC):
    """
    Interface abstrata para acesso a modelos de machine learning
    
    Define operações para carregar, salvar e gerenciar modelos LSTM
    e suas métricas de performance.
    """
    
    @abstractmethod
    async def load_model(self, model_version: str) -> Any:
        """
        Carrega um modelo treinado
        
        Args:
            model_version: Versão/ID do modelo
            
        Returns:
            Any: Modelo carregado (TensorFlow model)
        """
        pass
    
    @abstractmethod
    async def save_model(self, model: Any, model_version: str, metadata: Dict[str, Any]) -> bool:
        """
        Salva um modelo treinado
        
        Args:
            model: Modelo a ser salvo
            model_version: Versão/ID do modelo
            metadata: Metadados do modelo
            
        Returns:
            bool: True se salvou com sucesso
        """
        pass
    
    @abstractmethod
    async def get_latest_model_version(self) -> Optional[str]:
        """
        Busca a versão mais recente do modelo
        
        Returns:
            Optional[str]: Versão mais recente ou None
        """
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[str]:
        """
        Lista todas as versões de modelos disponíveis
        
        Returns:
            List[str]: Lista de versões disponíveis
        """
        pass
    
    @abstractmethod
    async def save_model_metrics(self, metrics: ModelMetrics) -> bool:
        """
        Salva métricas de performance do modelo
        
        Args:
            metrics: Métricas a serem salvas
            
        Returns:
            bool: True se salvou com sucesso
        """
        pass
    
    @abstractmethod
    async def get_model_metrics(self, model_version: str) -> Optional[ModelMetrics]:
        """
        Busca métricas de um modelo específico
        
        Args:
            model_version: Versão do modelo
            
        Returns:
            Optional[ModelMetrics]: Métricas ou None se não encontradas
        """
        pass
    
    @abstractmethod
    async def get_all_model_metrics(self) -> List[ModelMetrics]:
        """
        Busca métricas de todos os modelos
        
        Returns:
            List[ModelMetrics]: Lista de todas as métricas
        """
        pass
    
    @abstractmethod
    async def delete_model(self, model_version: str) -> bool:
        """
        Remove um modelo e suas métricas
        
        Args:
            model_version: Versão do modelo a ser removido
            
        Returns:
            bool: True se removeu com sucesso
        """
        pass
    
    @abstractmethod
    async def get_model_metadata(self, model_version: str) -> Optional[Dict[str, Any]]:
        """
        Busca metadados de um modelo específico
        
        Args:
            model_version: Versão do modelo
            
        Returns:
            Optional[Dict]: Metadados ou None se não encontrados
        """
        pass


class CacheRepository(ABC):
    """
    Interface abstrata para operações de cache
    
    Define operações para cache de previsões e dados temporários
    para melhorar performance da aplicação.
    """
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """
        Armazena valor no cache
        
        Args:
            key: Chave do cache
            value: Valor a ser armazenado
            ttl_seconds: Tempo de vida em segundos (opcional)
            
        Returns:
            bool: True se armazenou com sucesso
        """
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """
        Busca valor no cache
        
        Args:
            key: Chave do cache
            
        Returns:
            Optional[Any]: Valor ou None se não encontrado
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Remove valor do cache
        
        Args:
            key: Chave do cache
            
        Returns:
            bool: True se removeu com sucesso
        """
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Verifica se chave existe no cache
        
        Args:
            key: Chave do cache
            
        Returns:
            bool: True se existe
        """
        pass
    
    @abstractmethod
    async def set_forecast_cache(self, forecast: Forecast, ttl_seconds: int = 3600) -> bool:
        """
        Armazena previsão no cache com TTL específico
        
        Args:
            forecast: Previsão a ser cacheada
            ttl_seconds: Tempo de vida (padrão: 1 hora)
            
        Returns:
            bool: True se armazenou com sucesso
        """
        pass
    
    @abstractmethod
    async def get_cached_forecast(self, cache_key: str) -> Optional[Forecast]:
        """
        Busca previsão cacheada
        
        Args:
            cache_key: Chave da previsão no cache
            
        Returns:
            Optional[Forecast]: Previsão ou None se não encontrada
        """
        pass


# Protocol para implementações que precisam de configuração
class ConfigurableRepository(Protocol):
    """Protocol para repositories que precisam de configuração"""
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configura o repository
        
        Args:
            config: Dicionário de configurações
        """
        ...


# Protocol para implementações que precisam de health check
class HealthCheckRepository(Protocol):
    """Protocol para repositories que implementam health check"""
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Verifica saúde da conexão/implementação
        
        Returns:
            Dict: Status da saúde
        """
        ...


# Exceções específicas para repositories
class RepositoryError(Exception):
    """Exceção base para erros de repository"""
    pass


class DataNotFoundError(RepositoryError):
    """Exceção para quando dados não são encontrados"""
    pass


class DataValidationError(RepositoryError):
    """Exceção para erros de validação de dados"""
    pass


class ConnectionError(RepositoryError):
    """Exceção para erros de conexão"""
    pass


class ModelNotFoundError(RepositoryError):
    """Exceção para quando modelo não é encontrado"""
    pass


class CacheError(RepositoryError):
    """Exceção para erros de cache"""
    pass


# Utility functions para repositories
def create_cache_key(*args: Any) -> str:
    """
    Cria chave de cache padronizada
    
    Args:
        *args: Argumentos para compor a chave
        
    Returns:
        str: Chave de cache
    """
    return ":".join(str(arg) for arg in args)


def validate_date_range(start_date: datetime, end_date: datetime) -> None:
    """
    Valida range de datas
    
    Args:
        start_date: Data inicial
        end_date: Data final
        
    Raises:
        ValueError: Se range inválido
    """
    if start_date >= end_date:
        raise ValueError("Data inicial deve ser anterior à data final")
    
    # Limitar range máximo (ex: 5 anos)
    max_range = end_date - start_date
    if max_range.days > 5 * 365:  # 5 anos
        raise ValueError("Range de datas muito amplo (máximo: 5 anos)")


def validate_limit(limit: Optional[int]) -> None:
    """
    Valida limite de resultados
    
    Args:
        limit: Limite de resultados
        
    Raises:
        ValueError: Se limite inválido
    """
    if limit is not None:
        if limit <= 0:
            raise ValueError("Limite deve ser positivo")
        if limit > 10000:  # Limite máximo para performance
            raise ValueError("Limite muito alto (máximo: 10.000)") 