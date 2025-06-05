"""
Application Use Cases - Alerts Feature

Este módulo implementa os use cases da camada de aplicação para alertas
de cheias. Os use cases coordenam entre a camada de domínio e infraestrutura.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

from ..domain.entities import FloodAlert, RiverLevel, WeatherAlert, AlertHistory
from ..domain.services import FloodAlertService, AlertHistoryService, AlertConfiguration


class GenerateFloodAlertUseCase:
    """
    Use case para geração de alertas de cheia
    
    Coordena a obtenção de dados do rio e meteorológicos para gerar alertas.
    """
    
    def __init__(self, flood_alert_service: FloodAlertService):
        self.flood_alert_service = flood_alert_service
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def execute(
        self, 
        river_level_meters: float,
        precipitation_forecast_mm: float,
        confidence_score: float,
        forecast_horizon_hours: int = 24,
        model_version: str = "latest"
    ) -> FloodAlert:
        """
        Executa geração de alerta de cheia
        
        Args:
            river_level_meters: Nível atual do rio em metros
            precipitation_forecast_mm: Precipitação prevista em mm
            confidence_score: Confiança da previsão (0.0-1.0)
            forecast_horizon_hours: Horizonte de previsão
            model_version: Versão do modelo usado
            
        Returns:
            FloodAlert: Alerta gerado
        """
        self.logger.info(f"Gerando alerta - Rio: {river_level_meters}m, Precipitação: {precipitation_forecast_mm}mm")
        
        # Criar entidades de entrada
        river_level = RiverLevel(
            timestamp=datetime.now(),
            level_meters=river_level_meters,
            trend="stable",  # Em produção, seria calculado
            source="api"
        )
        
        weather_alert = WeatherAlert(
            timestamp=datetime.now(),
            precipitation_forecast_mm=precipitation_forecast_mm,
            confidence_score=confidence_score,
            forecast_horizon_hours=forecast_horizon_hours,
            model_version=model_version
        )
        
        # Gerar alerta usando o serviço de domínio
        alert = self.flood_alert_service.generate_flood_alert(river_level, weather_alert)
        
        self.logger.info(f"Alerta gerado: {alert.alert_level.value} - ID: {alert.alert_id}")
        return alert


class GetActiveAlertsUseCase:
    """
    Use case para recuperar alertas ativos
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # Em produção, teria repository injection
        self._active_alerts: List[FloodAlert] = []
    
    async def execute(self) -> List[FloodAlert]:
        """
        Recupera todos os alertas ativos
        
        Returns:
            List[FloodAlert]: Lista de alertas ativos
        """
        current_time = datetime.now()
        active_alerts = [
            alert for alert in self._active_alerts 
            if alert.is_active(current_time)
        ]
        
        self.logger.info(f"Alertas ativos encontrados: {len(active_alerts)}")
        return active_alerts
    
    def add_alert(self, alert: FloodAlert):
        """Adiciona alerta à lista (para demonstração)"""
        self._active_alerts.append(alert)


class GetAlertHistoryUseCase:
    """
    Use case para análise de histórico de alertas
    """
    
    def __init__(self, alert_history_service: AlertHistoryService):
        self.alert_history_service = alert_history_service
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def execute(
        self, 
        days_back: int = 30,
        alert_level_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analisa histórico de alertas
        
        Args:
            days_back: Número de dias para analisar
            alert_level_filter: Filtro por nível de alerta
            
        Returns:
            Dict: Análise do histórico
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Em produção, buscaria do repository
        # Por ora, retorna análise mock
        mock_alerts = []  # Seria buscado do repository
        
        history = self.alert_history_service.analyze_alert_patterns(
            mock_alerts, start_date, end_date
        )
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days_back
            },
            "summary": history.to_dict(),
            "insights": {
                "trend": "stable",  # Seria calculado
                "peak_hours": [14, 15, 16],  # Horários de pico
                "seasonal_pattern": "winter_high"  # Padrão sazonal
            }
        }


class UpdateAlertStatusUseCase:
    """
    Use case para atualizar status de alertas
    """
    
    def __init__(self, flood_alert_service: FloodAlertService):
        self.flood_alert_service = flood_alert_service
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def execute(
        self,
        current_alert: FloodAlert,
        new_river_level: float,
        new_precipitation: float,
        new_confidence: float
    ) -> Optional[FloodAlert]:
        """
        Atualiza alerta baseado em novas condições
        
        Args:
            current_alert: Alerta atual
            new_river_level: Novo nível do rio
            new_precipitation: Nova precipitação prevista
            new_confidence: Nova confiança
            
        Returns:
            Optional[FloodAlert]: Novo alerta se upgrade necessário
        """
        # Criar novas condições
        new_river = RiverLevel(
            timestamp=datetime.now(),
            level_meters=new_river_level,
            trend="stable",
            source="api"
        )
        
        new_weather = WeatherAlert(
            timestamp=datetime.now(),
            precipitation_forecast_mm=new_precipitation,
            confidence_score=new_confidence,
            forecast_horizon_hours=24
        )
        
        # Verificar se precisa upgrade
        should_upgrade = self.flood_alert_service.should_upgrade_alert(
            current_alert, (new_river, new_weather)
        )
        
        if should_upgrade:
            self.logger.info(f"Upgrade necessário para alerta {current_alert.alert_id}")
            return self.flood_alert_service.generate_flood_alert(new_river, new_weather)
        
        return None 