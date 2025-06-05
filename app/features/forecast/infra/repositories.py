"""
Repository Implementations - Infrastructure Layer

Este módulo implementa as interfaces de repository definidas na camada de domínio.
Contém implementações concretas para acesso a dados meteorológicos, previsões,
modelos ML e cache.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from pathlib import Path
import uuid

# Redis client for cache implementation
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# SQLAlchemy for database implementations
try:
    import sqlalchemy
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import select, func
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from ..domain.entities import WeatherData, Forecast, ModelMetrics
from ..domain.repositories import (
    WeatherDataRepository, ForecastRepository, 
    ModelRepository, CacheRepository,
    WeatherDataQuery, ForecastQuery,
    RepositoryError, DataNotFoundError, CacheError,
    create_cache_key
)

# Utilitários do módulo
from .model_loader import ModelLoader


class FileWeatherDataRepository(WeatherDataRepository):
    """
    Implementação baseada em arquivo para WeatherDataRepository
    
    Esta implementação utiliza arquivos CSV para armazenar e recuperar
    dados meteorológicos históricos.
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        """
        Inicializa o repository
        
        Args:
            data_dir: Diretório onde os dados estão armazenados
        """
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Garantir que diretório existe
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Arquivo principal de dados
        self.data_file = self.data_dir / "weather_data.json"
        
        # Cache em memória (opcional, para performance)
        self._cache: List[WeatherData] = None
    
    async def get_latest_data(self, count: int = 24) -> List[WeatherData]:
        """Busca os dados meteorológicos mais recentes"""
        data = await self._load_data()
        
        # Ordenar por timestamp (mais recente primeiro)
        sorted_data = sorted(data, key=lambda x: x.timestamp, reverse=True)
        
        # Retornar os mais recentes
        result = sorted_data[:count]
        
        # Inverter para ordem cronológica
        return list(reversed(result))
    
    async def get_data_by_period(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[WeatherData]:
        """Busca dados meteorológicos por período"""
        data = await self._load_data()
        
        # Filtrar por período
        filtered = [
            item for item in data
            if start_date <= item.timestamp <= end_date
        ]
        
        # Ordenar por timestamp
        return sorted(filtered, key=lambda x: x.timestamp)
    
    async def get_data_by_query(self, query: WeatherDataQuery) -> List[WeatherData]:
        """Busca dados meteorológicos usando query object"""
        data = await self._load_data()
        
        # Aplicar filtros da query
        filtered = data
        
        if query.station_id:
            filtered = [item for item in filtered if item.station_id == query.station_id]
        
        if query.start_date:
            filtered = [item for item in filtered if item.timestamp >= query.start_date]
        
        if query.end_date:
            filtered = [item for item in filtered if item.timestamp <= query.end_date]
        
        # Ordenar conforme query
        reverse = query.order_direction.lower() == 'desc'
        filtered = sorted(filtered, key=lambda x: getattr(x, query.order_by), reverse=reverse)
        
        # Aplicar limit se definido
        if query.limit:
            filtered = filtered[:query.limit]
        
        return filtered
    
    async def save_data(self, weather_data: WeatherData) -> bool:
        """Salva um registro de dados meteorológicos"""
        # Carregar dados existentes
        data = await self._load_data()
        
        # Adicionar novo registro
        data.append(weather_data)
        
        # Salvar de volta
        return await self._save_data(data)
    
    async def save_batch_data(self, weather_data_list: List[WeatherData]) -> int:
        """Salva múltiplos registros de dados meteorológicos"""
        # Carregar dados existentes
        data = await self._load_data()
        
        # Adicionar novos registros
        data.extend(weather_data_list)
        
        # Salvar de volta
        success = await self._save_data(data)
        
        return len(weather_data_list) if success else 0
    
    async def count_records(self, query: Optional[WeatherDataQuery] = None) -> int:
        """Conta o número de registros que atendem aos critérios"""
        if query:
            # Se há query, precisamos buscar e contar
            data = await self.get_data_by_query(query)
            return len(data)
        else:
            # Se não há query, apenas contamos todos
            data = await self._load_data()
            return len(data)
    
    async def get_data_statistics(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Calcula estatísticas dos dados meteorológicos"""
        # Obter dados do período
        data = await self.get_data_by_period(start_date, end_date)
        
        if not data:
            return {
                "count": 0,
                "period_start": start_date.isoformat(),
                "period_end": end_date.isoformat(),
                "stats": {}
            }
        
        # Calcular estatísticas básicas
        result = {
            "count": len(data),
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "stats": {}
        }
        
        # Campos numéricos para calcular estatísticas
        numeric_fields = [
            "precipitation", "pressure", "temperature", 
            "humidity", "wind_speed", "dew_point"
        ]
        
        # Calcular estatísticas para cada campo
        for field in numeric_fields:
            values = [getattr(item, field) for item in data if hasattr(item, field) and getattr(item, field) is not None]
            
            if values:
                result["stats"][field] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "count": len(values)
                }
        
        return result
    
    async def _load_data(self) -> List[WeatherData]:
        """Carrega dados do arquivo"""
        # Se já temos em cache, retornar
        if self._cache is not None:
            return self._cache
        
        # Se arquivo não existe, retornar lista vazia
        if not self.data_file.exists():
            self.logger.info(f"Arquivo de dados não encontrado: {self.data_file}")
            return []
        
        try:
            # Carregar do arquivo
            with open(self.data_file, "r") as f:
                data_list = json.load(f)
            
            # Converter para objetos WeatherData
            result = []
            for item in data_list:
                # Converter timestamp de string para datetime
                if "timestamp" in item:
                    item["timestamp"] = datetime.fromisoformat(item["timestamp"])
                
                # Criar objeto WeatherData
                result.append(WeatherData(**item))
            
            # Armazenar em cache
            self._cache = result
            
            return result
        
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados: {e}")
            return []
    
    async def _save_data(self, data: List[WeatherData]) -> bool:
        """Salva dados para o arquivo"""
        try:
            # Converter para formato serializável
            data_list = []
            for item in data:
                # Converter para dict e tratar campos especiais
                item_dict = item.__dict__.copy()
                
                # Converter datetime para string
                if "timestamp" in item_dict:
                    item_dict["timestamp"] = item_dict["timestamp"].isoformat()
                
                data_list.append(item_dict)
            
            # Salvar para arquivo
            with open(self.data_file, "w") as f:
                json.dump(data_list, f, indent=2)
            
            # Atualizar cache
            self._cache = data
            
            return True
        
        except Exception as e:
            self.logger.error(f"Erro ao salvar dados: {e}")
            return False
    
    def invalidate_cache(self):
        """Invalida o cache em memória"""
        self._cache = None


class FileForecastRepository(ForecastRepository):
    """
    Implementação baseada em arquivo para ForecastRepository
    
    Esta implementação utiliza arquivos JSON para armazenar e recuperar
    previsões meteorológicas.
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        """
        Inicializa o repository
        
        Args:
            data_dir: Diretório onde os dados estão armazenados
        """
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Garantir que diretório existe
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Arquivo principal de previsões
        self.forecast_file = self.data_dir / "forecasts.json"
    
    async def get_latest_forecast(self) -> Optional[Forecast]:
        """Busca a previsão mais recente"""
        forecasts = await self._load_forecasts()
        
        if not forecasts:
            return None
        
        # Ordenar por timestamp (mais recente primeiro)
        sorted_forecasts = sorted(forecasts, key=lambda x: x.timestamp, reverse=True)
        
        # Retornar a mais recente
        return sorted_forecasts[0]
    
    async def get_forecast_by_id(self, forecast_id: str) -> Optional[Forecast]:
        """Busca previsão por ID"""
        forecasts = await self._load_forecasts()
        
        # Buscar por ID
        for forecast in forecasts:
            if forecast.id == forecast_id:
                return forecast
        
        return None
    
    async def get_forecasts_by_period(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Forecast]:
        """Busca previsões por período"""
        forecasts = await self._load_forecasts()
        
        # Filtrar por período
        filtered = [
            item for item in forecasts
            if start_date <= item.timestamp <= end_date
        ]
        
        # Ordenar por timestamp
        return sorted(filtered, key=lambda x: x.timestamp)
    
    async def get_forecasts_by_query(self, query: ForecastQuery) -> List[Forecast]:
        """Busca previsões usando query object"""
        forecasts = await self._load_forecasts()
        
        # Aplicar filtros da query
        filtered = forecasts
        
        if query.start_date:
            filtered = [item for item in filtered if item.timestamp >= query.start_date]
        
        if query.end_date:
            filtered = [item for item in filtered if item.timestamp <= query.end_date]
        
        if query.model_version:
            filtered = [item for item in filtered if item.model_version == query.model_version]
        
        if query.min_confidence:
            filtered = [item for item in filtered if item.confidence_score >= query.min_confidence]
        
        # Ordenar conforme query
        reverse = query.order_direction.lower() == 'desc'
        filtered = sorted(filtered, key=lambda x: getattr(x, query.order_by), reverse=reverse)
        
        # Aplicar limit se definido
        if query.limit:
            filtered = filtered[:query.limit]
        
        return filtered
    
    async def save_forecast(self, forecast: Forecast) -> bool:
        """Salva uma previsão"""
        # Carregar previsões existentes
        forecasts = await self._load_forecasts()
        
        # Verificar se o forecast já tem um ID
        if not hasattr(forecast, 'id') or forecast.id is None:
            # Adicionar ID se não tiver
            forecast_dict = forecast.__dict__.copy()
            forecast_dict['id'] = str(uuid.uuid4())
            # Recriar o objeto com o ID
            forecast = Forecast(**forecast_dict)
        
        # Verificar se já existe forecast com mesmo ID
        for i, existing in enumerate(forecasts):
            if existing.id == forecast.id:
                # Substituir existente
                forecasts[i] = forecast
                return await self._save_forecasts(forecasts)
        
        # Adicionar nova previsão
        forecasts.append(forecast)
        
        # Salvar de volta
        return await self._save_forecasts(forecasts)
    
    async def delete_forecast(self, forecast_id: str) -> bool:
        """Remove uma previsão por ID"""
        # Carregar previsões existentes
        forecasts = await self._load_forecasts()
        
        # Filtrar removendo a previsão com o ID
        filtered = [f for f in forecasts if f.id != forecast_id]
        
        # Se não mudou, não encontrou
        if len(filtered) == len(forecasts):
            return False
        
        # Salvar de volta
        return await self._save_forecasts(filtered)
    
    async def delete_old_forecasts(self, cutoff_date: datetime) -> int:
        """Remove previsões antigas"""
        # Carregar previsões existentes
        forecasts = await self._load_forecasts()
        
        # Filtrar mantendo apenas as mais recentes que o cutoff
        filtered = [f for f in forecasts if f.timestamp >= cutoff_date]
        
        # Calcular quantas foram removidas
        removed = len(forecasts) - len(filtered)
        
        if removed > 0:
            # Salvar de volta
            if await self._save_forecasts(filtered):
                return removed
        
        return 0
    
    async def get_forecast_accuracy_metrics(self, model_version: str) -> Dict[str, Any]:
        """Calcula métricas de accuracy das previsões vs dados reais"""
        # Em uma implementação real, isso requereria acesso a dados reais 
        # para comparar com previsões. Aqui retornamos métricas simuladas.
        
        forecasts = await self._load_forecasts()
        
        # Filtrar para o modelo específico
        model_forecasts = [f for f in forecasts if f.model_version == model_version]
        
        if not model_forecasts:
            return {
                "model_version": model_version,
                "forecast_count": 0,
                "message": "Nenhuma previsão encontrada para este modelo"
            }
        
        # Métricas simuladas - em implementação real, calcularíamos comparando com dados reais
        return {
            "model_version": model_version,
            "forecast_count": len(model_forecasts),
            "metrics": {
                "mae": 1.2,
                "rmse": 1.8,
                "correlation": 0.85,
                "hit_rate": 0.78
            },
            "evaluation_date": datetime.now().isoformat()
        }
    
    async def _load_forecasts(self) -> List[Forecast]:
        """Carrega previsões do arquivo"""
        # Se arquivo não existe, retornar lista vazia
        if not self.forecast_file.exists():
            self.logger.info(f"Arquivo de previsões não encontrado: {self.forecast_file}")
            return []
        
        try:
            # Carregar do arquivo
            with open(self.forecast_file, "r") as f:
                forecast_list = json.load(f)
            
            # Converter para objetos Forecast
            result = []
            for item in forecast_list:
                # Converter timestamp de string para datetime
                if "timestamp" in item:
                    item["timestamp"] = datetime.fromisoformat(item["timestamp"])
                
                # Criar objeto Forecast
                result.append(Forecast(**item))
            
            return result
        
        except Exception as e:
            self.logger.error(f"Erro ao carregar previsões: {e}")
            return []
    
    async def _save_forecasts(self, forecasts: List[Forecast]) -> bool:
        """Salva previsões para o arquivo"""
        try:
            # Converter para formato serializável
            forecast_list = []
            for item in forecasts:
                # Converter para dict e tratar campos especiais
                item_dict = item.__dict__.copy()
                
                # Converter datetime para string
                if "timestamp" in item_dict:
                    item_dict["timestamp"] = item_dict["timestamp"].isoformat()
                
                forecast_list.append(item_dict)
            
            # Salvar para arquivo
            with open(self.forecast_file, "w") as f:
                json.dump(forecast_list, f, indent=2)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Erro ao salvar previsões: {e}")
            return False


class FileModelRepository(ModelRepository):
    """
    Implementação baseada em arquivo para ModelRepository
    
    Esta implementação utiliza o sistema de arquivos para gerenciar modelos
    de aprendizado de máquina.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Inicializa o repository
        
        Args:
            models_dir: Diretório onde os modelos estão armazenados
        """
        self.models_dir = Path(models_dir)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Garantir que diretório existe
        os.makedirs(self.models_dir, exist_ok=True)
        
        # ModelLoader para carregamento de modelos
        self.model_loader = ModelLoader(str(self.models_dir))
    
    async def get_available_models(self) -> List[str]:
        """Lista todos os modelos disponíveis"""
        # Obter modelos disponíveis do ModelLoader
        models = self.model_loader.get_available_models()
        
        # Retornar apenas as versões
        return list(models.keys())
    
    async def get_latest_model_version(self) -> Optional[str]:
        """Obtém a versão mais recente disponível"""
        return self.model_loader.get_latest_model_version()
    
    async def get_model_info(self, model_version: str) -> Dict[str, Any]:
        """Obtém informações sobre um modelo específico"""
        # Verificar se modelo existe
        if not await self.model_exists(model_version):
            raise ValueError(f"Modelo não encontrado: {model_version}")
        
        # Obter metadados do ModelLoader
        return self.model_loader.get_model_metadata(model_version)
    
    async def load_model(self, model_version: str) -> Any:
        """Carrega um modelo específico para uso"""
        try:
            return self.model_loader.load_model(model_version)
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo {model_version}: {e}")
            raise ValueError(f"Erro ao carregar modelo: {e}")
    
    async def save_model(self, model_version: str, model: Any, metadata: Dict[str, Any]) -> bool:
        """
        Salva um modelo com seus metadados
        
        Esta é uma implementação simplificada - em um sistema real,
        esta função salvaria o modelo TensorFlow no formato SavedModel.
        """
        try:
            # Criar diretório para o modelo
            model_dir = self.models_dir / model_version
            os.makedirs(model_dir, exist_ok=True)
            
            # Salvar metadados
            self.model_loader.save_model_metadata(model_version, metadata)
            
            # Em um sistema real, salvaríamos o modelo aqui:
            # model.save(str(model_dir))
            
            self.logger.info(f"Modelo {model_version} salvo com sucesso")
            return True
        
        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo {model_version}: {e}")
            return False
    
    async def delete_model(self, model_version: str) -> bool:
        """Remove um modelo"""
        try:
            # Verificar se modelo existe
            model_dir = self.models_dir / model_version
            if not model_dir.exists():
                return False
            
            # Remover diretório recursivamente
            import shutil
            shutil.rmtree(model_dir)
            
            # Limpar cache do ModelLoader
            self.model_loader.clear_cache()
            
            return True
        
        except Exception as e:
            self.logger.error(f"Erro ao remover modelo {model_version}: {e}")
            return False
    
    async def model_exists(self, model_version: str) -> bool:
        """Verifica se um modelo existe"""
        model_dir = self.models_dir / model_version
        return model_dir.exists()
    
    async def get_model_metrics(self, model_version: str) -> Optional[ModelMetrics]:
        """Obtém métricas de um modelo específico"""
        # Verificar se modelo existe
        if not await self.model_exists(model_version):
            return None
        
        # Obter metadados
        metadata = self.model_loader.get_model_metadata(model_version)
        
        # Extrair métricas
        performance = metadata.get('performance', {})
        
        # Valores padrão
        mae = performance.get('mae', 0.0)
        rmse = performance.get('rmse', 0.0)
        accuracy = performance.get('accuracy', 0.0)
        
        # Criar objeto ModelMetrics
        metrics = ModelMetrics(
            model_version=model_version,
            training_date=datetime.fromisoformat(metadata.get('created_at', datetime.now().isoformat())),
            mae=mae,
            rmse=rmse,
            accuracy=accuracy,
            r2_score=performance.get('r2_score'),
            precision=performance.get('precision'),
            recall=performance.get('recall'),
            f1_score=performance.get('f1_score'),
            skill_score=performance.get('skill_score'),
            train_samples=metadata.get('train_samples'),
            validation_samples=metadata.get('validation_samples'),
            test_samples=metadata.get('test_samples')
        )
        
        return metrics

    async def get_model_metadata(self, model_version: str) -> Dict[str, Any]:
        """Obtém metadados de um modelo específico"""
        # Verificar se modelo existe
        if not await self.model_exists(model_version):
            raise ValueError(f"Modelo não encontrado: {model_version}")
        
        # Obter metadados do ModelLoader
        return self.model_loader.get_model_metadata(model_version)

    async def save_model_metrics(self, model_version: str, metrics: ModelMetrics) -> bool:
        """Salva métricas de um modelo específico"""
        try:
            # Verificar se modelo existe
            if not await self.model_exists(model_version):
                return False
            
            # Obter metadados existentes
            metadata = self.model_loader.get_model_metadata(model_version)
            
            # Atualizar métricas no metadata
            metadata['performance'] = {
                'mae': metrics.mae,
                'rmse': metrics.rmse,
                'accuracy': metrics.accuracy,
                'r2_score': metrics.r2_score,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'skill_score': metrics.skill_score
            }
            
            # Salvar metadados atualizados
            self.model_loader.save_model_metadata(model_version, metadata)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Erro ao salvar métricas do modelo {model_version}: {e}")
            return False

    async def get_all_model_metrics(self) -> List[ModelMetrics]:
        """Obtém métricas de todos os modelos disponíveis"""
        try:
            # Obter modelos disponíveis
            versions = await self.get_available_models()
            
            # Obter métricas para cada modelo
            result = []
            for version in versions:
                metrics = await self.get_model_metrics(version)
                if metrics:
                    result.append(metrics)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Erro ao obter métricas de todos os modelos: {e}")
            return []


class MemoryCacheRepository(CacheRepository):
    """
    Implementação em memória para CacheRepository
    
    Ideal para desenvolvimento e testes sem dependência de Redis.
    """
    
    def __init__(self):
        """Inicializa o repository"""
        self.cache = {}
        self.ttl = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Armazena valor no cache"""
        try:
            self.cache[key] = value
            
            # Definir TTL se especificado
            if ttl_seconds is not None:
                self.ttl[key] = datetime.now() + timedelta(seconds=ttl_seconds)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Erro ao armazenar em cache: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Busca valor no cache"""
        # Verificar se chave existe
        if key not in self.cache:
            return None
        
        # Verificar TTL
        if key in self.ttl:
            if datetime.now() > self.ttl[key]:
                # Expirado
                del self.cache[key]
                del self.ttl[key]
                return None
        
        # Retornar valor
        return self.cache[key]
    
    async def delete(self, key: str) -> bool:
        """Remove valor do cache"""
        # Verificar se chave existe
        if key not in self.cache:
            return False
        
        # Remover
        del self.cache[key]
        if key in self.ttl:
            del self.ttl[key]
        
        return True
    
    async def exists(self, key: str) -> bool:
        """Verifica se chave existe no cache"""
        # Verificar se chave existe e não expirou
        if key not in self.cache:
            return False
        
        # Verificar TTL
        if key in self.ttl:
            if datetime.now() > self.ttl[key]:
                # Expirado
                del self.cache[key]
                del self.ttl[key]
                return False
        
        return True
    
    async def set_forecast_cache(self, forecast: Forecast, ttl_seconds: int = 3600) -> bool:
        """Armazena previsão no cache"""
        # Criar chave baseada na versão do modelo
        key = create_cache_key("forecast", forecast.model_version)
        
        # Armazenar em cache
        return await self.set(key, forecast, ttl_seconds)
    
    async def get_cached_forecast(self, cache_key: str) -> Optional[Forecast]:
        """Busca previsão cacheada"""
        # Buscar do cache
        value = await self.get(cache_key)
        
        # Verificar se é um objeto Forecast
        if value is not None and isinstance(value, Forecast):
            return value
        
        return None
    
    def clear_all(self) -> int:
        """Limpa todo o cache e retorna número de chaves removidas"""
        count = len(self.cache)
        self.cache.clear()
        self.ttl.clear()
        return count


class RedisCacheRepository(CacheRepository):
    """
    Implementação Redis para CacheRepository
    
    Utiliza Redis para cache distribuído com persistência.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Inicializa o repository
        
        Args:
            redis_url: URL de conexão com Redis
        """
        self.redis_url = redis_url
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._redis_client = None
        
        # Verificar disponibilidade do Redis
        if not REDIS_AVAILABLE:
            self.logger.warning("Módulo Redis não disponível. Usando fallback em memória.")
            self._memory_cache = MemoryCacheRepository()
    
    @property
    async def redis(self) -> Any:
        """Obtém cliente Redis, inicializando se necessário"""
        if not REDIS_AVAILABLE:
            raise CacheError("Módulo Redis não disponível")
        
        if self._redis_client is None:
            try:
                self._redis_client = redis.Redis.from_url(self.redis_url)
                # Testar conexão
                await self._redis_client.ping()
            except Exception as e:
                self.logger.error(f"Erro ao conectar ao Redis: {e}")
                raise CacheError(f"Erro ao conectar ao Redis: {e}")
        
        return self._redis_client
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Armazena valor no cache"""
        # Fallback para cache em memória
        if not REDIS_AVAILABLE:
            return await self._memory_cache.set(key, value, ttl_seconds)
        
        try:
            # Serializar valor para JSON
            serialized = json.dumps(value, default=self._serialize_complex)
            
            # Obter cliente Redis
            client = await self.redis
            
            # Armazenar com TTL opcional
            if ttl_seconds is not None:
                await client.setex(key, ttl_seconds, serialized)
            else:
                await client.set(key, serialized)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Erro ao armazenar em cache Redis: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Busca valor no cache"""
        # Fallback para cache em memória
        if not REDIS_AVAILABLE:
            return await self._memory_cache.get(key)
        
        try:
            # Obter cliente Redis
            client = await self.redis
            
            # Buscar valor
            value = await client.get(key)
            
            # Verificar se encontrou
            if value is None:
                return None
            
            # Deserializar
            return json.loads(value.decode('utf-8'), object_hook=self._deserialize_complex)
        
        except Exception as e:
            self.logger.error(f"Erro ao buscar em cache Redis: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Remove valor do cache"""
        # Fallback para cache em memória
        if not REDIS_AVAILABLE:
            return await self._memory_cache.delete(key)
        
        try:
            # Obter cliente Redis
            client = await self.redis
            
            # Remover chave
            deleted = await client.delete(key)
            
            return deleted > 0
        
        except Exception as e:
            self.logger.error(f"Erro ao remover do cache Redis: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Verifica se chave existe no cache"""
        # Fallback para cache em memória
        if not REDIS_AVAILABLE:
            return await self._memory_cache.exists(key)
        
        try:
            # Obter cliente Redis
            client = await self.redis
            
            # Verificar se existe
            return await client.exists(key) > 0
        
        except Exception as e:
            self.logger.error(f"Erro ao verificar existência no cache Redis: {e}")
            return False
    
    async def set_forecast_cache(self, forecast: Forecast, ttl_seconds: int = 3600) -> bool:
        """Armazena previsão no cache"""
        # Criar chave baseada na versão do modelo
        key = create_cache_key("forecast", forecast.model_version)
        
        # Converter Forecast para dict para serialização
        forecast_dict = self._forecast_to_dict(forecast)
        
        # Armazenar em cache
        return await self.set(key, forecast_dict, ttl_seconds)
    
    async def get_cached_forecast(self, cache_key: str) -> Optional[Forecast]:
        """Busca previsão cacheada"""
        # Buscar do cache
        value = await self.get(cache_key)
        
        # Verificar se encontrou
        if value is None:
            return None
        
        # Converter de volta para objeto Forecast
        try:
            if isinstance(value, dict):
                # Converter timestamp de string para datetime se necessário
                if "timestamp" in value and isinstance(value["timestamp"], str):
                    value["timestamp"] = datetime.fromisoformat(value["timestamp"])
                
                return Forecast(**value)
            elif isinstance(value, Forecast):
                return value
            else:
                self.logger.warning(f"Tipo inesperado no cache: {type(value)}")
                return None
        
        except Exception as e:
            self.logger.error(f"Erro ao deserializar previsão do cache: {e}")
            return None
    
    def _forecast_to_dict(self, forecast: Forecast) -> Dict[str, Any]:
        """Converte objeto Forecast para dict serializável"""
        forecast_dict = forecast.__dict__.copy()
        
        # Converter datetime para string
        if "timestamp" in forecast_dict and isinstance(forecast_dict["timestamp"], datetime):
            forecast_dict["timestamp"] = forecast_dict["timestamp"].isoformat()
        
        return forecast_dict
    
    def _serialize_complex(self, obj):
        """Serializa objetos complexos para JSON"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            raise TypeError(f"Type {type(obj)} not serializable")
    
    def _deserialize_complex(self, dct):
        """Deserializa objetos complexos de JSON"""
        # Detectar datetime
        if isinstance(dct, dict) and "timestamp" in dct and isinstance(dct["timestamp"], str):
            try:
                dct["timestamp"] = datetime.fromisoformat(dct["timestamp"])
            except (ValueError, TypeError):
                pass
        
        return dct 