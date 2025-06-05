"""
Application Use Cases - Forecast Feature

Este módulo implementa os use cases da camada de aplicação para previsões
meteorológicas. Os use cases coordenam entre a camada de domínio e infraestrutura,
orquestrando operações complexas.

Use Cases:
- GenerateForecastUseCase: Gera nova previsão meteorológica
- GetModelMetricsUseCase: Recupera métricas do modelo ML
- RefreshModelUseCase: Atualiza o modelo para nova versão
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from ..domain.entities import WeatherData, Forecast, ModelMetrics
from ..domain.services import ForecastService, ModelValidationService
from ..domain.repositories import (
    WeatherDataRepository, ForecastRepository, 
    ModelRepository, CacheRepository,
    create_cache_key
)


class GenerateForecastUseCase:
    """
    Use case para geração de previsão meteorológica
    
    Responsabilidades:
    - Coordenar entre repositories e services
    - Recuperar dados históricos recentes
    - Processar dados para formato do modelo
    - Executar inferência do modelo
    - Validar e salvar previsão
    - Cachear resultados
    """
    
    def __init__(
        self, 
        weather_data_repository: WeatherDataRepository,
        forecast_repository: ForecastRepository,
        model_repository: ModelRepository,
        cache_repository: CacheRepository,
        forecast_service: ForecastService
    ):
        self.weather_data_repository = weather_data_repository
        self.forecast_repository = forecast_repository
        self.model_repository = model_repository
        self.cache_repository = cache_repository
        self.forecast_service = forecast_service
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def execute(
        self, 
        use_cache: bool = True, 
        model_version: Optional[str] = None
    ) -> Forecast:
        """
        Executa o use case para gerar uma previsão
        
        Args:
            use_cache: Se deve usar cache
            model_version: Versão específica do modelo (opcional)
            
        Returns:
            Forecast: Previsão meteorológica gerada
            
        Raises:
            ValueError: Se dados de entrada inválidos
            ModelNotFoundError: Se modelo não encontrado
        """
        self.logger.info(f"Gerando previsão meteorológica (cache={use_cache}, model={model_version})")
        
        # 1. Verificar cache se permitido
        if use_cache:
            cache_key = create_cache_key("forecast", model_version or "latest")
            cached_forecast = await self.cache_repository.get_cached_forecast(cache_key)
            if cached_forecast:
                self.logger.info(f"Previsão encontrada em cache: {cached_forecast.model_version}")
                return cached_forecast
        
        # 2. Determinar versão do modelo
        if not model_version:
            model_version = await self.model_repository.get_latest_model_version()
            if not model_version:
                raise ValueError("Nenhum modelo disponível para previsão")
        
        # 3. Carregar dados meteorológicos recentes
        weather_data = await self.weather_data_repository.get_latest_data(
            count=self.forecast_service.config.sequence_length
        )
        
        # 4. Validar dados de entrada
        self.forecast_service.validate_input_sequence(weather_data)
        
        # 5. Carregar modelo
        model = await self.model_repository.load_model(model_version)
        
        # 6. Preparar dados para modelo (aqui seria feito na implementação real)
        # Este é um placeholder - na implementação real, os dados seriam processados
        # por um componente da camada de infraestrutura
        
        # 7. Realizar inferência (mock para demonstração)
        # Na implementação real, isso usaria o modelo TensorFlow carregado
        start_time = datetime.now()
        
        # Placeholder para inferência do modelo
        # Na implementação real: resultado = model.predict(dados_processados)
        mock_precipitation = 0.5  # mm/h (placeholder)
        mock_confidence = 0.85    # 0.0-1.0 (placeholder)
        
        inference_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # 8. Criar objeto Forecast
        forecast = Forecast(
            timestamp=datetime.now(),
            precipitation_mm=mock_precipitation,
            confidence_score=mock_confidence,
            model_version=model_version,
            inference_time_ms=inference_time_ms,
            input_sequence_length=len(weather_data),
            forecast_horizon_hours=24,
            features_used=16
        )
        
        # 9. Validar qualidade da previsão
        if not self.forecast_service.validate_forecast_quality(forecast):
            self.logger.warning(f"Previsão gerada não atende critérios de qualidade")
            # Em produção, poderia usar um fallback ou gerar alerta
        
        # 10. Salvar previsão
        await self.forecast_repository.save_forecast(forecast)
        
        # 11. Cachear resultado
        if use_cache:
            ttl = 3600  # 1 hora
            await self.cache_repository.set_forecast_cache(forecast, ttl_seconds=ttl)
        
        self.logger.info(f"Previsão gerada com sucesso: {forecast.precipitation_mm}mm/h, confiança={forecast.confidence_score}")
        return forecast


class GetModelMetricsUseCase:
    """
    Use case para recuperação de métricas do modelo
    
    Responsabilidades:
    - Recuperar métricas do modelo
    - Validar métricas contra critérios
    - Cachear resultados frequentemente solicitados
    """
    
    def __init__(
        self,
        model_repository: ModelRepository,
        cache_repository: CacheRepository,
        model_validation_service: ModelValidationService
    ):
        self.model_repository = model_repository
        self.cache_repository = cache_repository
        self.model_validation_service = model_validation_service
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def execute(
        self, 
        model_version: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Executa o use case para obter métricas do modelo
        
        Args:
            model_version: Versão do modelo (opcional)
            use_cache: Se deve usar cache
            
        Returns:
            Dict: Métricas do modelo com validação
            
        Raises:
            ModelNotFoundError: Se modelo não encontrado
        """
        self.logger.info(f"Recuperando métricas do modelo: {model_version or 'latest'}")
        
        # 1. Verificar cache se permitido
        if use_cache:
            cache_key = create_cache_key("model_metrics", model_version or "latest")
            cached_metrics = await self.cache_repository.get(cache_key)
            if cached_metrics:
                self.logger.info(f"Métricas encontradas em cache: {model_version or 'latest'}")
                return cached_metrics
        
        # 2. Determinar versão do modelo
        if not model_version:
            model_version = await self.model_repository.get_latest_model_version()
            if not model_version:
                raise ValueError("Nenhum modelo disponível")
        
        # 3. Recuperar métricas do repositório
        metrics = await self.model_repository.get_model_metrics(model_version)
        if not metrics:
            raise ValueError(f"Métricas não encontradas para modelo: {model_version}")
        
        # 4. Validar métricas com o serviço de domínio
        validation_result = self.model_validation_service.validate_model_metrics(metrics)
        
        # 5. Gerar resposta completa
        result = {
            "metrics": metrics.to_dict(),
            "validation": validation_result,
            "timestamp": datetime.now().isoformat()
        }
        
        # 6. Cachear resultado
        if use_cache:
            ttl = 86400  # 24 horas (métricas mudam com menos frequência)
            await self.cache_repository.set(
                create_cache_key("model_metrics", model_version),
                result,
                ttl_seconds=ttl
            )
        
        return result


class RefreshModelUseCase:
    """
    Use case para atualização do modelo
    
    Responsabilidades:
    - Verificar disponibilidade de novos modelos
    - Validar métricas do novo modelo
    - Comparar com modelo atual
    - Atualizar modelo ativo
    """
    
    def __init__(
        self,
        model_repository: ModelRepository,
        cache_repository: CacheRepository,
        model_validation_service: ModelValidationService
    ):
        self.model_repository = model_repository
        self.cache_repository = cache_repository
        self.model_validation_service = model_validation_service
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def execute(
        self, 
        new_model_version: Optional[str] = None,
        force_update: bool = False
    ) -> Dict[str, Any]:
        """
        Executa o use case para atualizar o modelo
        
        Args:
            new_model_version: Versão específica do novo modelo (opcional)
            force_update: Se deve forçar atualização mesmo com performance inferior
            
        Returns:
            Dict: Resultado da operação com detalhes
            
        Raises:
            ModelNotFoundError: Se modelo não encontrado
            ValueError: Se modelo não atende critérios mínimos
        """
        self.logger.info(f"Iniciando atualização de modelo: {new_model_version or 'latest disponível'}")
        
        # 1. Obter versão atual do modelo
        current_version = await self.model_repository.get_latest_model_version()
        if not current_version:
            self.logger.warning("Nenhum modelo atual encontrado")
            current_metrics = None
        else:
            # 2. Obter métricas do modelo atual
            current_metrics = await self.model_repository.get_model_metrics(current_version)
        
        # 3. Determinar nova versão se não especificada
        if not new_model_version:
            # Obter todas as versões disponíveis
            available_versions = await self.model_repository.get_available_models()
            # Filtrar versões mais recentes que a atual
            if current_version:
                newer_versions = [v for v in available_versions if v != current_version]
                if not newer_versions:
                    return {
                        "success": False,
                        "message": "Nenhuma versão mais recente disponível",
                        "current_version": current_version
                    }
                # Usar a versão mais recente disponível
                new_model_version = newer_versions[0]
            else:
                # Se não há modelo atual, usar o primeiro disponível
                if not available_versions:
                    return {
                        "success": False,
                        "message": "Nenhum modelo disponível para atualização"
                    }
                new_model_version = available_versions[0]
        
        # 4. Obter métricas do novo modelo
        new_metrics = await self.model_repository.get_model_metrics(new_model_version)
        if not new_metrics:
            raise ValueError(f"Métricas não encontradas para novo modelo: {new_model_version}")
        
        # 5. Validar novo modelo
        validation_result = self.model_validation_service.validate_model_metrics(new_metrics)
        
        # 6. Verificar se atende critérios mínimos
        if not new_metrics.meets_all_criteria() and not force_update:
            return {
                "success": False,
                "message": "Novo modelo não atende aos critérios mínimos",
                "validation": validation_result,
                "current_version": current_version,
                "new_version": new_model_version
            }
        
        # 7. Comparar com modelo atual (se existir)
        if current_metrics:
            comparison = self.model_validation_service.compare_models(
                current_metrics, new_metrics
            )
            recommendation = self.model_validation_service.recommend_model_update(
                comparison
            )
            
            # 8. Decidir se deve atualizar
            should_update = recommendation["should_update"] or force_update
            if not should_update:
                return {
                    "success": False,
                    "message": "Novo modelo não apresenta melhoria significativa",
                    "comparison": comparison,
                    "recommendation": recommendation,
                    "current_version": current_version,
                    "new_version": new_model_version
                }
        else:
            # Se não há modelo atual, sempre atualizar
            should_update = True
            comparison = None
            recommendation = {"should_update": True, "confidence": "high"}
        
        # 9. Carregar novo modelo para teste
        try:
            await self.model_repository.load_model(new_model_version)
        except Exception as e:
            self.logger.error(f"Erro ao carregar novo modelo: {e}")
            return {
                "success": False,
                "message": f"Erro ao carregar novo modelo: {str(e)}",
                "current_version": current_version,
                "new_version": new_model_version
            }
        
        # 10. Atualizar modelo e limpar cache
        self.logger.info(f"Atualizando para modelo: {new_model_version}")
        
        # Limpar cache de previsões
        await self.cache_repository.delete(create_cache_key("forecast", "latest"))
        
        return {
            "success": True,
            "message": f"Modelo atualizado com sucesso para {new_model_version}",
            "comparison": comparison,
            "recommendation": recommendation,
            "previous_version": current_version,
            "new_version": new_model_version,
            "metrics": new_metrics.to_dict(),
            "validation": validation_result
        } 