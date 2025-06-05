"""
API Routes - Forecast Feature

Este módulo define os endpoints da API para a feature de previsão meteorológica.
Utiliza o FastAPI para definir rotas, validação automática, documentação e
resposta de erros.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status

from ..application.usecases import (
    GenerateForecastUseCase,
    GetModelMetricsUseCase,
    RefreshModelUseCase
)
from ..domain.entities import WeatherData, Forecast, ModelMetrics, PrecipitationLevel
from ..dependencies import (
    get_generate_forecast_usecase,
    get_model_metrics_usecase,
    get_refresh_model_usecase
)
from .schemas import (
    WeatherDataRequest,
    WeatherDataResponse,
    ForecastRequest,
    ForecastResponse,
    ForecastHourlyResponse,
    ModelMetricsResponse,
    RefreshModelRequest,
    RefreshModelResponse
)


# Criar logger para este módulo
logger = logging.getLogger(__name__)

# Criar router para a feature
router = APIRouter(
    prefix="/forecast",
    tags=["forecast"],
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Recurso não encontrado"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Erro interno do servidor"}
    }
)


@router.post(
    "/predict",
    response_model=ForecastResponse,
    summary="Gera previsão meteorológica",
    description="Gera previsão de precipitação para as próximas 24 horas."
)
async def generate_forecast(
    request: ForecastRequest = Body(...),
    usecase: GenerateForecastUseCase = Depends(get_generate_forecast_usecase)
) -> ForecastResponse:
    """
    Endpoint para geração de previsão meteorológica
    
    - Utiliza o modelo LSTM treinado
    - Considera dados das últimas 24 horas
    - Retorna previsão para as próximas 24 horas
    - Suporta uso de cache para performance
    """
    try:
        # Executar use case
        forecast = await usecase.execute(
            use_cache=request.use_cache,
            model_version=request.model_version
        )
        
        # Converter para DTO de resposta
        return ForecastResponse(
            timestamp=forecast.timestamp,
            precipitation_mm=forecast.precipitation_mm,
            confidence_score=forecast.confidence_score,
            precipitation_level=forecast.get_precipitation_level().value,
            is_rain_expected=forecast.is_rain_expected(),
            model_version=forecast.model_version,
            inference_time_ms=forecast.inference_time_ms,
            forecast_horizon_hours=forecast.forecast_horizon_hours,
            input_sequence_length=forecast.input_sequence_length
        )
    
    except ValueError as e:
        logger.warning(f"Erro de validação na previsão: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Erro ao gerar previsão: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao gerar previsão meteorológica"
        )


@router.get(
    "/hourly",
    response_model=ForecastHourlyResponse,
    summary="Previsão horária para 24h",
    description="Gera previsão detalhada hora a hora para as próximas 24 horas."
)
async def get_hourly_forecast(
    usecase: GenerateForecastUseCase = Depends(get_generate_forecast_usecase)
) -> ForecastHourlyResponse:
    """
    Endpoint para previsão meteorológica detalhada por hora
    
    - Gera previsões para cada hora das próximas 24h
    - Inclui confiança variável por horizonte temporal
    - Fornece resumo das condições esperadas
    """
    try:
        # Gerar previsão base
        forecast = await usecase.execute(use_cache=True)
        
        # Na implementação real, usaríamos o ForecastModel para gerar previsões horárias
        # Aqui, simulamos com valores derivados da previsão principal
        
        # Simular previsões horárias (placeholder)
        hours = []
        base_precipitation = forecast.precipitation_mm
        current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        for hour in range(24):
            # Simular variação ao longo do tempo (apenas para demonstração)
            hour_factor = 1.0 - (abs(12 - hour) / 24)  # Pico no meio do período
            precipitation = base_precipitation * hour_factor
            
            # Confiança diminui com horizonte temporal
            confidence = max(0.5, forecast.confidence_score * (1.0 - (hour / 48)))
            
            # Determinar nível de precipitação
            level = "none"
            if precipitation >= 0.1:
                level = "light"
            if precipitation >= 2.0:
                level = "moderate"
            if precipitation >= 10.0:
                level = "heavy"
            if precipitation >= 50.0:
                level = "extreme"
            
            # Adicionar à lista de horas
            hour_time = current_time + timedelta(hours=hour)
            hours.append({
                "timestamp": hour_time.isoformat(),
                "hour": hour,
                "precipitation_mm": round(precipitation, 2),
                "confidence_score": round(confidence, 2),
                "precipitation_level": level,
                "is_rain_expected": precipitation >= 0.1
            })
        
        # Gerar sumário
        max_precipitation = max(hours, key=lambda x: x["precipitation_mm"])
        rain_hours = sum(1 for h in hours if h["precipitation_mm"] >= 0.1)
        
        summary = {
            "max_precipitation": max_precipitation,
            "rain_hours": rain_hours,
            "average_precipitation": round(sum(h["precipitation_mm"] for h in hours) / len(hours), 2),
            "rain_expected": rain_hours > 0
        }
        
        # Informações do modelo
        model_info = {
            "version": forecast.model_version,
            "inference_time_ms": forecast.inference_time_ms,
            "input_sequence_length": forecast.input_sequence_length or 24
        }
        
        return ForecastHourlyResponse(
            hours=hours,
            summary=summary,
            model_info=model_info,
            generated_at=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Erro ao gerar previsão horária: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao gerar previsão horária"
        )


@router.get(
    "/metrics",
    response_model=ModelMetricsResponse,
    summary="Métricas do modelo",
    description="Retorna métricas de performance do modelo atual."
)
async def get_model_metrics(
    model_version: Optional[str] = Query(None, description="Versão do modelo (opcional)"),
    usecase: GetModelMetricsUseCase = Depends(get_model_metrics_usecase)
) -> ModelMetricsResponse:
    """
    Endpoint para métricas do modelo
    
    - Retorna MAE, RMSE, Accuracy
    - Valida contra critérios de performance
    - Suporta especificação de versão
    """
    try:
        # Executar use case
        result = await usecase.execute(model_version=model_version)
        
        # Extrair métricas
        metrics = result["metrics"]
        
        # Converter para DTO de resposta
        return ModelMetricsResponse(**metrics)
    
    except ValueError as e:
        logger.warning(f"Erro ao recuperar métricas: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Erro ao obter métricas do modelo: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao obter métricas do modelo"
        )


@router.post(
    "/refresh-model",
    response_model=RefreshModelResponse,
    summary="Atualiza modelo",
    description="Atualiza o modelo de previsão para uma nova versão."
)
async def refresh_model(
    request: RefreshModelRequest = Body(...),
    usecase: RefreshModelUseCase = Depends(get_refresh_model_usecase)
) -> RefreshModelResponse:
    """
    Endpoint para atualização do modelo
    
    - Compara performance com modelo atual
    - Valida métricas contra critérios
    - Suporta atualização forçada
    """
    try:
        # Executar use case
        result = await usecase.execute(
            new_model_version=request.new_model_version,
            force_update=request.force_update
        )
        
        # Converter para DTO de resposta
        return RefreshModelResponse(**result)
    
    except ValueError as e:
        logger.warning(f"Erro de validação na atualização do modelo: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Erro ao atualizar modelo: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao atualizar modelo de previsão"
        ) 