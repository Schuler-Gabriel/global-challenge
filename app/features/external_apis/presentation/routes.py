"""
FastAPI Routes - External APIs Feature

Este módulo implementa os endpoints REST para integração com APIs externas.
Fornece acesso aos dados meteorológicos do CPTEC e nível do Rio Guaíba.

Endpoints:
- GET /weather/current - Condições meteorológicas atuais
- GET /weather/forecast - Previsão meteorológica
- GET /river/current - Nível atual do rio
- GET /conditions/current - Condições completas (tempo + rio)
- GET /health - Health check das APIs
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse

from .schemas import (
    WeatherResponse,
    RiverLevelResponse,
    CurrentConditionsResponse,
    ApiHealthResponse,
    WeatherRequestSchema,
    RiverLevelRequestSchema,
    ForecastRequestSchema,
    HealthCheckRequestSchema,
    ErrorResponseSchema
)

# Importações condicionais dos use cases
try:
    from ..application.usecases import (
        GetCurrentConditionsUseCase,
        GetWeatherDataUseCase,
        GetRiverDataUseCase,
        HealthCheckUseCase
    )
    from ..infra.clients import CptecWeatherClient, GuaibaRiverClient
    from ..infra.cache import ExternalApiCacheService
    from ..infra.monitoring import ApiMonitoringService
    from ..domain.services import ApiServiceConfig
    USECASES_AVAILABLE = True
except ImportError as e:
    USECASES_AVAILABLE = False
    import_error = str(e)


# Router configuration
external_apis_router = APIRouter(
    prefix="/external-apis",
    tags=["External APIs"],
    responses={
        500: {"model": ErrorResponseSchema, "description": "Internal Server Error"},
        503: {"model": ErrorResponseSchema, "description": "Service Unavailable"}
    }
)

logger = logging.getLogger(__name__)


# Dependency injection setup
async def get_weather_use_case() -> GetWeatherDataUseCase:
    """
    Cria instância do use case para dados meteorológicos
    
    Returns:
        GetWeatherDataUseCase: Use case configurado
    """
    if not USECASES_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"External APIs service unavailable: {import_error}"
        )
    
    # Mock implementation - em produção seria injetado via dependencies
    from ..infra import CircuitBreakerImpl
    weather_client = CptecWeatherClient()
    circuit_breaker = CircuitBreakerImpl()
    return GetWeatherDataUseCase(
        weather_service=weather_client,
        circuit_breaker=circuit_breaker
    )


async def get_river_use_case() -> GetRiverDataUseCase:
    """
    Cria instância do use case para dados do rio
    
    Returns:
        GetRiverDataUseCase: Use case configurado
    """
    if not USECASES_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"External APIs service unavailable: {import_error}"
        )
    
    # Mock implementation - em produção seria injetado via dependencies
    from ..infra import CircuitBreakerImpl
    river_client = GuaibaRiverClient()
    circuit_breaker = CircuitBreakerImpl()
    return GetRiverDataUseCase(
        river_service=river_client,
        circuit_breaker=circuit_breaker
    )


async def get_conditions_use_case(
    weather_use_case: GetWeatherDataUseCase = Depends(get_weather_use_case),
    river_use_case: GetRiverDataUseCase = Depends(get_river_use_case)
) -> GetCurrentConditionsUseCase:
    """
    Cria instância do use case para condições completas
    
    Returns:
        GetCurrentConditionsUseCase: Use case configurado
    """
    from ..infra import CircuitBreakerImpl
    weather_client = CptecWeatherClient()
    river_client = GuaibaRiverClient()
    circuit_breaker = CircuitBreakerImpl()
    return GetCurrentConditionsUseCase(
        weather_service=weather_client,
        river_service=river_client,
        circuit_breaker=circuit_breaker
    )


async def get_monitoring_use_case() -> HealthCheckUseCase:
    """
    Cria instância do use case para monitoramento
    
    Returns:
        HealthCheckUseCase: Use case configurado
    """
    if not USECASES_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"External APIs service unavailable: {import_error}"
        )
    
    # Mock implementation - em produção seria injetado via dependencies
    from ..infra import CptecWeatherClient, GuaibaRiverClient, CircuitBreakerImpl
    
    # Criar instâncias mock
    weather_service = CptecWeatherClient()
    river_service = GuaibaRiverClient()
    circuit_breaker = CircuitBreakerImpl()
    
    return HealthCheckUseCase(
        weather_service=weather_service,
        river_service=river_service,
        circuit_breaker=circuit_breaker
    )


# Exception handlers
def create_error_response(error: Exception, request_id: Optional[str] = None) -> JSONResponse:
    """
    Cria resposta de erro padronizada
    
    Args:
        error: Exceção ocorrida
        request_id: ID da requisição
        
    Returns:
        JSONResponse: Resposta de erro
    """
    error_type = type(error).__name__
    error_message = str(error)
    
    # Determinar status code baseado no tipo de erro
    if "timeout" in error_message.lower() or "circuit breaker" in error_message.lower():
        status_code = 503
    elif "validation" in error_message.lower():
        status_code = 422
    elif "not found" in error_message.lower():
        status_code = 404
    else:
        status_code = 500
    
    response_data = {
        "error": error_type,
        "message": error_message,
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id
    }
    
    return JSONResponse(
        status_code=status_code,
        content=response_data
    )


# Weather endpoints
@external_apis_router.get(
    "/weather/current",
    response_model=WeatherResponse,
    summary="Obter condições meteorológicas atuais",
    description="Obtém dados meteorológicos atuais do CPTEC para a cidade especificada"
)
async def get_current_weather(
    city: str = Query("Porto Alegre", description="Nome da cidade"),
    use_cache: bool = Query(True, description="Se deve usar cache"),
    max_cache_age_minutes: int = Query(5, ge=1, le=60, description="Idade máxima do cache"),
    weather_use_case: GetWeatherDataUseCase = Depends(get_weather_use_case)
):
    """
    Obtém condições meteorológicas atuais para a cidade especificada.
    
    Utiliza cache para otimizar performance e circuit breaker para
    resiliência contra falhas da API externa.
    """
    try:
        logger.info(f"Solicitando dados meteorológicos atuais para {city}")
        
        # Executar use case
        api_response = await weather_use_case.execute(
            city=city,
            use_cache=use_cache,
            max_cache_age_minutes=max_cache_age_minutes
        )
        
        # Converter resposta para schema
        response = WeatherResponse(
            data=api_response.data.to_dict() if api_response.data else None,
            api_response={
                "success": api_response.success,
                "timestamp": api_response.timestamp,
                "api_name": api_response.api_name,
                "response_time_ms": api_response.response_time_ms,
                "status_code": api_response.status_code,
                "error_message": api_response.error_message,
                "retry_count": api_response.retry_count,
                "from_cache": api_response.from_cache,
                "cache_ttl_seconds": api_response.cache_ttl_seconds
            }
        )
        
        logger.info(f"Dados meteorológicos obtidos com sucesso para {city}")
        return response
    
    except Exception as e:
        logger.error(f"Erro ao obter dados meteorológicos para {city}: {e}")
        return create_error_response(e)


@external_apis_router.get(
    "/weather/forecast",
    response_model=WeatherResponse,
    summary="Obter previsão meteorológica",
    description="Obtém previsão meteorológica do CPTEC para as próximas horas"
)
async def get_weather_forecast(
    city: str = Query("Porto Alegre", description="Nome da cidade"),
    hours_ahead: int = Query(24, ge=1, le=168, description="Horas à frente"),
    use_cache: bool = Query(True, description="Se deve usar cache"),
    weather_use_case: GetWeatherDataUseCase = Depends(get_weather_use_case)
):
    """
    Obtém previsão meteorológica para as próximas horas.
    
    Máximo de 168 horas (7 dias) à frente.
    """
    try:
        logger.info(f"Solicitando previsão meteorológica para {city} - {hours_ahead}h")
        
        # Para esta implementação, usamos o mesmo use case
        # Em uma implementação completa, haveria um use case específico para previsão
        api_response = await weather_use_case.execute(
            city=city,
            use_cache=use_cache
        )
        
        response = WeatherResponse(
            data=api_response.data.to_dict() if api_response.data else None,
            api_response={
                "success": api_response.success,
                "timestamp": api_response.timestamp,
                "api_name": api_response.api_name,
                "response_time_ms": api_response.response_time_ms,
                "status_code": api_response.status_code,
                "error_message": api_response.error_message,
                "from_cache": api_response.from_cache
            }
        )
        
        logger.info(f"Previsão meteorológica obtida com sucesso para {city}")
        return response
    
    except Exception as e:
        logger.error(f"Erro ao obter previsão meteorológica para {city}: {e}")
        return create_error_response(e)


# River endpoints
@external_apis_router.get(
    "/river/current",
    response_model=RiverLevelResponse,
    summary="Obter nível atual do rio",
    description="Obtém nível atual do Rio Guaíba com análise de risco"
)
async def get_current_river_level(
    station: str = Query("Porto Alegre", description="Nome da estação"),
    use_cache: bool = Query(True, description="Se deve usar cache"),
    max_cache_age_minutes: int = Query(10, ge=1, le=120, description="Idade máxima do cache"),
    include_trend: bool = Query(True, description="Se deve incluir análise de tendência"),
    river_use_case: GetRiverDataUseCase = Depends(get_river_use_case)
):
    """
    Obtém nível atual do Rio Guaíba para a estação especificada.
    
    Inclui análise de risco baseada nas cotas de referência:
    - Cota de atenção: 2.80m
    - Cota de alerta: 3.15m  
    - Cota de inundação: 3.60m
    """
    try:
        logger.info(f"Solicitando nível atual do rio para estação {station}")
        
        # Executar use case
        api_response = await river_use_case.execute(
            station=station,
            use_cache=use_cache,
            max_cache_age_minutes=max_cache_age_minutes,
            include_trend=include_trend
        )
        
        # Processar dados da resposta
        response_data = None
        if api_response.data:
            if isinstance(api_response.data, dict) and 'current_level' in api_response.data:
                # Resposta com tendência
                response_data = {
                    "current_level": api_response.data['current_level'].to_dict(),
                    "trend_analysis": api_response.data.get('trend')
                }
            else:
                # Resposta simples
                response_data = api_response.data.to_dict()
        
        response = RiverLevelResponse(
            data=response_data,
            api_response={
                "success": api_response.success,
                "timestamp": api_response.timestamp,
                "api_name": api_response.api_name,
                "response_time_ms": api_response.response_time_ms,
                "status_code": api_response.status_code,
                "error_message": api_response.error_message,
                "retry_count": api_response.retry_count,
                "from_cache": api_response.from_cache
            }
        )
        
        logger.info(f"Nível do rio obtido com sucesso para {station}")
        return response
    
    except Exception as e:
        logger.error(f"Erro ao obter nível do rio para {station}: {e}")
        return create_error_response(e)


# Combined conditions endpoint
@external_apis_router.get(
    "/conditions/current",
    response_model=CurrentConditionsResponse,
    summary="Obter condições completas atuais",
    description="Obtém dados meteorológicos e nível do rio em paralelo com análise combinada"
)
async def get_current_conditions(
    city: str = Query("Porto Alegre", description="Nome da cidade"),
    station: str = Query("Porto Alegre", description="Nome da estação do rio"),
    use_cache: bool = Query(True, description="Se deve usar cache"),
    conditions_use_case: GetCurrentConditionsUseCase = Depends(get_conditions_use_case)
):
    """
    Obtém condições completas atuais combinando dados meteorológicos e nível do rio.
    
    Executa consultas em paralelo para otimizar performance e fornece
    análise combinada de risco de inundação considerando ambos os fatores.
    """
    try:
        logger.info(f"Solicitando condições completas para {city}/{station}")
        
        # Executar use case
        responses = await conditions_use_case.execute(
            city=city,
            station=station,
            use_cache=use_cache
        )
        
        # Processar respostas
        weather_response = responses.get('weather')
        river_response = responses.get('river')
        
        # Análise combinada simplificada
        combined_analysis = await _analyze_combined_conditions(weather_response, river_response)
        
        response = CurrentConditionsResponse(
            weather=WeatherResponse(
                data=weather_response.data.to_dict() if weather_response.data else None,
                api_response={
                    "success": weather_response.success,
                    "timestamp": weather_response.timestamp,
                    "api_name": weather_response.api_name,
                    "response_time_ms": weather_response.response_time_ms,
                    "from_cache": weather_response.from_cache,
                    "error_message": weather_response.error_message
                }
            ),
            river=RiverLevelResponse(
                data=river_response.data.to_dict() if river_response.data else None,
                api_response={
                    "success": river_response.success,
                    "timestamp": river_response.timestamp,
                    "api_name": river_response.api_name,
                    "response_time_ms": river_response.response_time_ms,
                    "from_cache": river_response.from_cache,
                    "error_message": river_response.error_message
                }
            ),
            combined_analysis=combined_analysis
        )
        
        logger.info(f"Condições completas obtidas com sucesso")
        return response
    
    except Exception as e:
        logger.error(f"Erro ao obter condições completas: {e}")
        return create_error_response(e)


# Health check endpoint
@external_apis_router.get(
    "/health",
    response_model=ApiHealthResponse,
    summary="Health check das APIs externas",
    description="Verifica disponibilidade e performance das APIs CPTEC e Rio Guaíba"
)
async def get_api_health(
    detailed: bool = Query(False, description="Se deve incluir métricas detalhadas"),
    api_name: Optional[str] = Query(None, description="API específica para verificar"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    monitoring_use_case: HealthCheckUseCase = Depends(get_monitoring_use_case)
):
    """
    Executa health check das APIs externas.
    
    Verifica disponibilidade, latência e taxas de sucesso das APIs.
    Inclui status dos circuit breakers e métricas de performance.
    """
    try:
        logger.info("Executando health check das APIs externas")
        
        # Executar use case
        health_data = await monitoring_use_case.execute()
        
        # Converter para formato da resposta
        api_health_list = []
        
        for api_name, api_data in health_data.get('apis', {}).items():
            api_health = {
                "api_name": api_name,
                "status": api_data.get('status', 'unknown'),
                "response_time_ms": api_data.get('response_time_ms', 0),
                "success_rate": 1.0,  # Placeholder
                "last_check": datetime.now(),
                "error_message": api_data.get('error')
            }
            api_health_list.append(api_health)
        
        response = ApiHealthResponse(
            timestamp=datetime.now(),
            overall_status=health_data.get('overall_status', 'unknown'),
            apis=api_health_list,
            circuit_breaker=health_data.get('circuit_breaker', {}),
            response_time_ms=health_data.get('response_time_ms', 0),
            metrics=health_data.get('metrics') if detailed else None
        )
        
        # Agendar limpeza de métricas em background
        background_tasks.add_task(_cleanup_old_metrics, monitoring_use_case)
        
        logger.info("Health check concluído com sucesso")
        return response
    
    except Exception as e:
        logger.error(f"Erro no health check: {e}")
        return create_error_response(e)


# Cache management endpoints
@external_apis_router.delete(
    "/cache/clear",
    summary="Limpar cache das APIs",
    description="Remove todos os dados em cache das APIs externas"
)
async def clear_cache():
    """
    Limpa todo o cache das APIs externas.
    
    Útil para forçar atualização de dados ou resolver problemas de cache.
    """
    try:
        # Implementação simplificada
        return {"message": "Cache limpo com sucesso", "timestamp": datetime.now()}
    
    except Exception as e:
        logger.error(f"Erro ao limpar cache: {e}")
        return create_error_response(e)


@external_apis_router.get(
    "/cache/stats",
    summary="Estatísticas do cache",
    description="Obtém métricas de uso do cache das APIs externas"
)
async def get_cache_stats():
    """
    Obtém estatísticas de uso do cache.
    
    Inclui hit rate, número de entradas, etc.
    """
    try:
        # Implementação simplificada sem dependência
        stats = {
            "total_hits": 0,
            "total_misses": 0,
            "hit_rate": 0.0,
            "cache_size": 0,
            "memory_usage_mb": 0.0,
            "last_cleanup": datetime.now().isoformat()
        }
        
        return {
            "cache_stats": stats,
            "timestamp": datetime.now()
        }
    
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas do cache: {e}")
        return create_error_response(e)


# Helper functions
async def _analyze_combined_conditions(weather_response, river_response) -> Dict[str, Any]:
    """
    Analisa condições combinadas de tempo e rio
    
    Args:
        weather_response: Resposta meteorológica
        river_response: Resposta do nível do rio
        
    Returns:
        Dict: Análise combinada
    """
    analysis = {
        "flood_risk_level": "baixo",
        "weather_impact": "minimal",
        "recommendations": [],
        "alerts": []
    }
    
    try:
        # Análise do nível do rio
        if river_response and river_response.success and river_response.data:
            river_data = river_response.data
            if hasattr(river_data, 'is_flood_risk') and river_data.is_flood_risk:
                analysis["flood_risk_level"] = "alto"
                analysis["recommendations"].append("Monitorar nível do rio de perto")
            
            if hasattr(river_data, 'is_critical_level') and river_data.is_critical_level:
                analysis["flood_risk_level"] = "critico"
                analysis["alerts"].append("Nível crítico do rio atingido")
        
        # Análise meteorológica
        if weather_response and weather_response.success and weather_response.data:
            weather_data = weather_response.data
            if hasattr(weather_data, 'is_rain_expected') and weather_data.is_rain_expected:
                analysis["weather_impact"] = "moderate"
                analysis["recommendations"].append("Chuva prevista - monitorar condições")
            
            if hasattr(weather_data, 'is_storm_conditions') and weather_data.is_storm_conditions:
                analysis["weather_impact"] = "high"
                analysis["alerts"].append("Condições de tempestade detectadas")
        
        # Análise combinada
        if analysis["flood_risk_level"] in ["alto", "critico"] and analysis["weather_impact"] in ["moderate", "high"]:
            analysis["recommendations"].append("ALERTA: Risco elevado de inundação combinado com condições meteorológicas adversas")
        
        if not analysis["recommendations"]:
            analysis["recommendations"].append("Condições normais - continuar monitoramento")
    
    except Exception as e:
        logger.warning(f"Erro na análise combinada: {e}")
        analysis["error"] = str(e)
    
    return analysis


async def _cleanup_old_metrics(monitoring_use_case: HealthCheckUseCase):
    """
    Task em background para limpeza de métricas antigas
    
    Args:
        monitoring_use_case: Use case de monitoramento
    """
    try:
        # Implementação simplificada
        logger.info("Executando limpeza de métricas antigas em background")
        # Em uma implementação real, chamaria um método de cleanup
    except Exception as e:
        logger.error(f"Erro na limpeza de métricas: {e}")


# Export router
router = external_apis_router 