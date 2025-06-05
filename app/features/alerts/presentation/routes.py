"""
API Routes - Alerts Feature

Este módulo define os endpoints da API para a feature de alertas de cheias.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Path, Body, status
from fastapi.responses import JSONResponse

from ..application.usecases import (
    GenerateFloodAlertUseCase,
    GetActiveAlertsUseCase,
    GetAlertHistoryUseCase,
    UpdateAlertStatusUseCase
)
from ..domain.services import FloodAlertService, AlertHistoryService
from .schemas import (
    GenerateAlertRequest,
    FloodAlertResponse,
    AlertListResponse,
    AlertHistoryAnalysisResponse,
    AlertStatusUpdateRequest,
    AlertSummaryResponse,
    ErrorResponse
)


# Criar logger para este módulo
logger = logging.getLogger(__name__)

# Criar router para a feature
router = APIRouter(
    prefix="/alerts",
    tags=["alerts"],
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Recurso não encontrado"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Erro interno do servidor"}
    }
)

# Instâncias dos use cases (em produção, seria injetado via dependencies)
flood_alert_service = FloodAlertService()
generate_alert_usecase = GenerateFloodAlertUseCase(flood_alert_service)
active_alerts_usecase = GetActiveAlertsUseCase()
history_usecase = GetAlertHistoryUseCase(AlertHistoryService())
update_usecase = UpdateAlertStatusUseCase(flood_alert_service)


@router.post(
    "/generate",
    response_model=FloodAlertResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Gerar Alerta de Cheia",
    description="Gera um novo alerta de cheia baseado nas condições atuais do rio e previsão meteorológica"
)
async def generate_flood_alert(
    request: GenerateAlertRequest = Body(..., description="Dados para geração do alerta")
) -> FloodAlertResponse:
    """
    Gera um novo alerta de cheia
    
    Este endpoint combina informações do nível do rio e previsão meteorológica
    para calcular o risco e gerar um alerta apropriado.
    """
    try:
        logger.info(f"Gerando alerta - Rio: {request.river_level_meters}m, Precipitação: {request.precipitation_forecast_mm}mm")
        
        # Executar use case
        alert = await generate_alert_usecase.execute(
            river_level_meters=request.river_level_meters,
            precipitation_forecast_mm=request.precipitation_forecast_mm,
            confidence_score=request.confidence_score,
            forecast_horizon_hours=request.forecast_horizon_hours,
            model_version=request.model_version
        )
        
        # Adicionar à lista de alertas ativos (para demonstração)
        active_alerts_usecase.add_alert(alert)
        
        # Converter para response schema
        alert_dict = alert.to_dict()
        response = FloodAlertResponse(**alert_dict)
        
        logger.info(f"Alerta gerado com sucesso: {alert.alert_level.value} - ID: {alert.alert_id}")
        return response
        
    except ValueError as e:
        logger.error(f"Erro de validação ao gerar alerta: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dados inválidos: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Erro interno ao gerar alerta: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno do servidor"
        )


@router.get(
    "/active",
    response_model=AlertListResponse,
    summary="Listar Alertas Ativos",
    description="Recupera todos os alertas ativos no sistema"
)
async def get_active_alerts() -> AlertListResponse:
    """
    Lista todos os alertas ativos
    
    Retorna alertas que ainda estão dentro do período de validade.
    """
    try:
        logger.info("Buscando alertas ativos")
        
        # Executar use case
        active_alerts = await active_alerts_usecase.execute()
        
        # Converter para response schemas
        alert_responses = []
        critical_count = 0
        
        for alert in active_alerts:
            alert_dict = alert.to_dict()
            alert_response = FloodAlertResponse(**alert_dict)
            alert_responses.append(alert_response)
            
            if alert.alert_level.value == "critico":
                critical_count += 1
        
        response = AlertListResponse(
            alerts=alert_responses,
            total_count=len(alert_responses),
            active_count=len(alert_responses),
            critical_count=critical_count
        )
        
        logger.info(f"Alertas ativos encontrados: {len(active_alerts)}")
        return response
        
    except Exception as e:
        logger.error(f"Erro ao buscar alertas ativos: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno do servidor"
        )


@router.get(
    "/history",
    response_model=AlertHistoryAnalysisResponse,
    summary="Análise de Histórico",
    description="Analisa padrões e tendências no histórico de alertas"
)
async def get_alert_history(
    days_back: int = Query(30, ge=1, le=365, description="Número de dias para analisar"),
    alert_level_filter: Optional[str] = Query(None, description="Filtrar por nível de alerta")
) -> AlertHistoryAnalysisResponse:
    """
    Analisa histórico de alertas
    
    Fornece insights sobre padrões, frequência e eficácia dos alertas.
    """
    try:
        logger.info(f"Analisando histórico de {days_back} dias")
        
        # Executar use case
        analysis = await history_usecase.execute(
            days_back=days_back,
            alert_level_filter=alert_level_filter
        )
        
        response = AlertHistoryAnalysisResponse(**analysis)
        
        logger.info(f"Análise de histórico concluída para {days_back} dias")
        return response
        
    except Exception as e:
        logger.error(f"Erro ao analisar histórico: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno do servidor"
        )


@router.get(
    "/summary",
    response_model=Dict[str, Any],
    summary="Resumo de Alertas",
    description="Fornece um resumo do status atual dos alertas"
)
async def get_alerts_summary() -> Dict[str, Any]:
    """
    Resumo do status atual dos alertas
    
    Fornece uma visão geral rápida da situação atual.
    """
    try:
        logger.info("Gerando resumo de alertas")
        
        # Buscar alertas ativos
        active_alerts = await active_alerts_usecase.execute()
        
        # Calcular estatísticas
        total_active = len(active_alerts)
        critical_alerts = [a for a in active_alerts if a.alert_level.value == "critico"]
        high_alerts = [a for a in active_alerts if a.alert_level.value == "alto"]
        
        # Determinar status geral
        if critical_alerts:
            overall_status = "CRÍTICO"
            highest_level = "critico"
        elif high_alerts:
            overall_status = "ALTO"
            highest_level = "alto"
        elif active_alerts:
            overall_status = "MODERADO"
            highest_level = "moderado"
        else:
            overall_status = "NORMAL"
            highest_level = "baixo"
        
        # Alerta mais recente
        latest_alert = None
        if active_alerts:
            latest_alert = max(active_alerts, key=lambda x: x.timestamp)
            latest_alert_dict = latest_alert.to_dict()
        else:
            latest_alert_dict = None
        
        summary = {
            "overall_status": overall_status,
            "active_alerts_count": total_active,
            "critical_alerts_count": len(critical_alerts),
            "high_alerts_count": len(high_alerts),
            "highest_alert_level": highest_level,
            "latest_alert": latest_alert_dict,
            "last_updated": datetime.now().isoformat(),
            "system_status": "operational"
        }
        
        logger.info(f"Resumo gerado: {overall_status} - {total_active} alertas ativos")
        return summary
        
    except Exception as e:
        logger.error(f"Erro ao gerar resumo: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno do servidor"
        )


@router.put(
    "/{alert_id}/update",
    response_model=Optional[FloodAlertResponse],
    summary="Atualizar Alerta",
    description="Atualiza um alerta existente baseado em novas condições"
)
async def update_alert_status(
    alert_id: str = Path(..., description="ID do alerta a ser atualizado"),
    request: AlertStatusUpdateRequest = Body(..., description="Novas condições")
) -> Optional[FloodAlertResponse]:
    """
    Atualiza status de um alerta
    
    Verifica se as novas condições requerem upgrade do alerta.
    """
    try:
        logger.info(f"Atualizando alerta {alert_id}")
        
        # Buscar alerta atual (em produção, seria do repository)
        active_alerts = await active_alerts_usecase.execute()
        current_alert = None
        
        for alert in active_alerts:
            if alert.alert_id == alert_id:
                current_alert = alert
                break
        
        if not current_alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alerta {alert_id} não encontrado"
            )
        
        # Executar use case de atualização
        updated_alert = await update_usecase.execute(
            current_alert=current_alert,
            new_river_level=request.new_river_level,
            new_precipitation=request.new_precipitation,
            new_confidence=request.new_confidence
        )
        
        if updated_alert:
            # Adicionar novo alerta à lista
            active_alerts_usecase.add_alert(updated_alert)
            
            # Converter para response
            alert_dict = updated_alert.to_dict()
            response = FloodAlertResponse(**alert_dict)
            
            logger.info(f"Alerta atualizado: {updated_alert.alert_level.value}")
            return response
        else:
            logger.info(f"Nenhuma atualização necessária para alerta {alert_id}")
            return None
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao atualizar alerta: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno do servidor"
        )


@router.get(
    "/health",
    summary="Health Check",
    description="Verifica o status do sistema de alertas"
)
async def alerts_health_check() -> Dict[str, Any]:
    """
    Health check do sistema de alertas
    """
    try:
        # Verificar componentes básicos
        active_alerts = await active_alerts_usecase.execute()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "active_alerts_count": len(active_alerts),
            "services": {
                "alert_generation": "operational",
                "risk_calculation": "operational",
                "alert_classification": "operational"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check falhou: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        } 