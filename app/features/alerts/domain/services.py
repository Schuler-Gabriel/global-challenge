"""
Domain Services - Alerts Feature

Este módulo contém os serviços de domínio que encapsulam lógica de negócio
complexa relacionada ao sistema de alertas de cheias.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging

from .entities import (
    FloodAlert, RiverLevel, WeatherAlert, AlertLevel, RiskLevel, 
    AlertAction, AlertHistory
)


logger = logging.getLogger(__name__)


@dataclass
class AlertConfiguration:
    """Configuração para o sistema de alertas"""
    # Thresholds de nível do rio (metros)
    river_normal_threshold: float = 1.5
    river_attention_threshold: float = 2.5
    river_alert_threshold: float = 3.5
    river_emergency_threshold: float = 4.5
    
    # Thresholds de precipitação (mm/h)
    precipitation_light_threshold: float = 1.0
    precipitation_moderate_threshold: float = 10.0
    precipitation_heavy_threshold: float = 50.0
    
    # Configurações de confiança
    min_confidence_threshold: float = 0.6
    high_confidence_threshold: float = 0.8
    
    # Configurações temporais
    alert_validity_hours: int = 12
    max_forecast_horizon_hours: int = 48
    
    # Pesos para cálculo de risco
    river_level_weight: float = 0.6
    precipitation_weight: float = 0.3
    confidence_weight: float = 0.1


class RiskCalculationService:
    """
    Serviço responsável pelo cálculo de risco de cheias
    
    Combina informações de nível do rio, previsão meteorológica e confiança
    para calcular um score de risco normalizado.
    """
    
    def __init__(self, config: AlertConfiguration):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def calculate_risk_score(
        self, 
        river_level: RiverLevel, 
        weather_alert: WeatherAlert
    ) -> float:
        """
        Calcula score de risco combinado (0.0 - 1.0)
        
        Args:
            river_level: Nível atual do rio
            weather_alert: Alerta meteorológico
            
        Returns:
            float: Score de risco normalizado
        """
        # Score baseado no nível do rio (0.0 - 1.0)
        river_score = self._calculate_river_risk_score(river_level)
        
        # Score baseado na precipitação prevista (0.0 - 1.0)
        precipitation_score = self._calculate_precipitation_risk_score(weather_alert)
        
        # Score baseado na confiança da previsão (0.0 - 1.0)
        confidence_score = weather_alert.confidence_score
        
        # Cálculo ponderado
        weighted_score = (
            river_score * self.config.river_level_weight +
            precipitation_score * self.config.precipitation_weight +
            confidence_score * self.config.confidence_weight
        )
        
        # Garantir que está no range 0.0 - 1.0
        return min(max(weighted_score, 0.0), 1.0)
    
    def _calculate_river_risk_score(self, river_level: RiverLevel) -> float:
        """Calcula score de risco baseado no nível do rio"""
        level = river_level.level_meters
        
        if level <= self.config.river_normal_threshold:
            return 0.0
        elif level <= self.config.river_attention_threshold:
            # Linear entre normal e atenção (0.0 - 0.3)
            return 0.3 * (level - self.config.river_normal_threshold) / \
                   (self.config.river_attention_threshold - self.config.river_normal_threshold)
        elif level <= self.config.river_alert_threshold:
            # Linear entre atenção e alerta (0.3 - 0.7)
            return 0.3 + 0.4 * (level - self.config.river_attention_threshold) / \
                   (self.config.river_alert_threshold - self.config.river_attention_threshold)
        elif level <= self.config.river_emergency_threshold:
            # Linear entre alerta e emergência (0.7 - 1.0)
            return 0.7 + 0.3 * (level - self.config.river_alert_threshold) / \
                   (self.config.river_emergency_threshold - self.config.river_alert_threshold)
        else:
            # Acima da emergência = risco máximo
            return 1.0
    
    def _calculate_precipitation_risk_score(self, weather_alert: WeatherAlert) -> float:
        """Calcula score de risco baseado na precipitação prevista"""
        precip = weather_alert.precipitation_forecast_mm
        
        if precip <= self.config.precipitation_light_threshold:
            return 0.0
        elif precip <= self.config.precipitation_moderate_threshold:
            # Linear entre leve e moderada (0.0 - 0.4)
            return 0.4 * (precip - self.config.precipitation_light_threshold) / \
                   (self.config.precipitation_moderate_threshold - self.config.precipitation_light_threshold)
        elif precip <= self.config.precipitation_heavy_threshold:
            # Linear entre moderada e forte (0.4 - 0.8)
            return 0.4 + 0.4 * (precip - self.config.precipitation_moderate_threshold) / \
                   (self.config.precipitation_heavy_threshold - self.config.precipitation_moderate_threshold)
        else:
            # Precipitação intensa = risco alto
            return 0.8
    
    def determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determina o nível de risco baseado no score"""
        if risk_score < 0.2:
            return RiskLevel.MINIMAL
        elif risk_score < 0.4:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MODERATE
        elif risk_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME


class AlertClassificationService:
    """
    Serviço responsável pela classificação de alertas
    
    Determina o nível de alerta e ações recomendadas baseado no risco calculado
    e nas condições atuais.
    """
    
    def __init__(self, config: AlertConfiguration):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def classify_alert(
        self, 
        river_level: RiverLevel, 
        weather_alert: WeatherAlert,
        risk_score: float
    ) -> Tuple[AlertLevel, AlertAction]:
        """
        Classifica o nível de alerta e ação recomendada
        
        Args:
            river_level: Nível atual do rio
            weather_alert: Alerta meteorológico
            risk_score: Score de risco calculado
            
        Returns:
            Tuple[AlertLevel, AlertAction]: Nível e ação recomendada
        """
        # Matriz de decisão baseada em nível do rio e precipitação
        river_meters = river_level.level_meters
        precipitation_mm = weather_alert.precipitation_forecast_mm
        confidence = weather_alert.confidence_score
        
        # Regras críticas (override outras condições)
        if river_meters > self.config.river_emergency_threshold:
            return AlertLevel.CRITICAL, AlertAction.EMERGENCY
        
        # Regras de alta prioridade
        if (river_meters > self.config.river_alert_threshold and 
            precipitation_mm > self.config.precipitation_moderate_threshold and
            confidence > self.config.high_confidence_threshold):
            return AlertLevel.CRITICAL, AlertAction.EMERGENCY
        
        if river_meters > self.config.river_alert_threshold:
            return AlertLevel.HIGH, AlertAction.ALERT
        
        # Regras de média prioridade
        if (river_meters > self.config.river_attention_threshold and 
            precipitation_mm > self.config.precipitation_light_threshold):
            if confidence > self.config.high_confidence_threshold:
                return AlertLevel.HIGH, AlertAction.ALERT
            else:
                return AlertLevel.MODERATE, AlertAction.ATTENTION
        
        if river_meters > self.config.river_attention_threshold:
            return AlertLevel.MODERATE, AlertAction.ATTENTION
        
        # Regras baseadas apenas na precipitação
        if (precipitation_mm > self.config.precipitation_heavy_threshold and 
            confidence > self.config.high_confidence_threshold):
            return AlertLevel.MODERATE, AlertAction.ATTENTION
        
        # Condições normais
        return AlertLevel.LOW, AlertAction.MONITORING
    
    def generate_alert_message(
        self, 
        alert_level: AlertLevel,
        river_level: RiverLevel,
        weather_alert: WeatherAlert,
        risk_score: float
    ) -> str:
        """Gera mensagem descritiva do alerta"""
        river_category = river_level.get_level_category()
        precip_level = weather_alert.get_precipitation_level()
        
        base_messages = {
            AlertLevel.LOW: f"Situação normal. Rio em nível {river_category} ({river_level.level_meters:.2f}m). Precipitação {precip_level} prevista.",
            AlertLevel.MODERATE: f"Atenção. Rio em nível {river_category} ({river_level.level_meters:.2f}m). Precipitação {precip_level} prevista nas próximas {weather_alert.forecast_horizon_hours}h.",
            AlertLevel.HIGH: f"Alerta de cheia. Rio em nível {river_category} ({river_level.level_meters:.2f}m). Precipitação {precip_level} prevista. Risco elevado.",
            AlertLevel.CRITICAL: f"Emergência de cheia. Rio em nível crítico ({river_level.level_meters:.2f}m). Evacuação recomendada. Risco extremo."
        }
        
        message = base_messages[alert_level]
        message += f" Score de risco: {risk_score:.2f}. Confiança: {weather_alert.confidence_score:.0%}."
        
        return message


class FloodAlertService:
    """
    Serviço principal para geração de alertas de cheia
    
    Coordena os outros serviços para gerar alertas completos.
    """
    
    def __init__(self, config: Optional[AlertConfiguration] = None):
        self.config = config or AlertConfiguration()
        self.risk_service = RiskCalculationService(self.config)
        self.classification_service = AlertClassificationService(self.config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def generate_flood_alert(
        self, 
        river_level: RiverLevel, 
        weather_alert: WeatherAlert
    ) -> FloodAlert:
        """
        Gera um alerta de cheia completo
        
        Args:
            river_level: Nível atual do rio
            weather_alert: Alerta meteorológico
            
        Returns:
            FloodAlert: Alerta completo gerado
        """
        self.logger.info(f"Gerando alerta de cheia - Rio: {river_level.level_meters}m, Precipitação: {weather_alert.precipitation_forecast_mm}mm")
        
        # 1. Calcular score de risco
        risk_score = self.risk_service.calculate_risk_score(river_level, weather_alert)
        risk_level = self.risk_service.determine_risk_level(risk_score)
        
        # 2. Classificar alerta
        alert_level, recommended_action = self.classification_service.classify_alert(
            river_level, weather_alert, risk_score
        )
        
        # 3. Gerar mensagem
        message = self.classification_service.generate_alert_message(
            alert_level, river_level, weather_alert, risk_score
        )
        
        # 4. Determinar validade
        valid_until = datetime.now() + timedelta(hours=self.config.alert_validity_hours)
        
        # 5. Criar alerta
        alert = FloodAlert(
            timestamp=datetime.now(),
            alert_id=str(uuid.uuid4()),
            alert_level=alert_level,
            risk_level=risk_level,
            recommended_action=recommended_action,
            river_level=river_level,
            weather_alert=weather_alert,
            risk_score=risk_score,
            message=message,
            valid_until=valid_until
        )
        
        self.logger.info(f"Alerta gerado: {alert_level.value} - Score: {risk_score:.2f}")
        return alert
    
    def should_upgrade_alert(
        self, 
        current_alert: FloodAlert, 
        new_conditions: Tuple[RiverLevel, WeatherAlert]
    ) -> bool:
        """
        Determina se um alerta deve ser atualizado baseado em novas condições
        
        Args:
            current_alert: Alerta atual
            new_conditions: Novas condições (rio, clima)
            
        Returns:
            bool: True se deve atualizar o alerta
        """
        river_level, weather_alert = new_conditions
        
        # Calcular novo score de risco
        new_risk_score = self.risk_service.calculate_risk_score(river_level, weather_alert)
        
        # Classificar novo alerta
        new_alert_level, _ = self.classification_service.classify_alert(
            river_level, weather_alert, new_risk_score
        )
        
        # Atualizar se:
        # 1. Nível de alerta aumentou
        # 2. Score de risco aumentou significativamente (>20%)
        # 3. Alerta atual expirando em breve (<2h)
        
        level_upgrade = new_alert_level.value != current_alert.alert_level.value
        significant_risk_increase = new_risk_score > current_alert.risk_score * 1.2
        expiring_soon = (current_alert.valid_until - datetime.now()).total_seconds() < 7200  # 2h
        
        return level_upgrade or significant_risk_increase or expiring_soon


class AlertHistoryService:
    """
    Serviço para análise de histórico de alertas
    
    Fornece insights sobre padrões de alertas e eficácia do sistema.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def analyze_alert_patterns(
        self, 
        alerts: List[FloodAlert], 
        period_start: datetime, 
        period_end: datetime
    ) -> AlertHistory:
        """
        Analisa padrões em um período de alertas
        
        Args:
            alerts: Lista de alertas do período
            period_start: Início do período
            period_end: Fim do período
            
        Returns:
            AlertHistory: Análise do histórico
        """
        if not alerts:
            return AlertHistory(
                period_start=period_start,
                period_end=period_end,
                total_alerts=0,
                alerts_by_level={level: 0 for level in AlertLevel},
                average_risk_score=0.0,
                max_river_level=0.0,
                total_precipitation=0.0
            )
        
        # Contar alertas por nível
        alerts_by_level = {level: 0 for level in AlertLevel}
        for alert in alerts:
            alerts_by_level[alert.alert_level] += 1
        
        # Calcular estatísticas
        total_alerts = len(alerts)
        average_risk_score = sum(alert.risk_score for alert in alerts) / total_alerts
        max_river_level = max(alert.river_level.level_meters for alert in alerts)
        total_precipitation = sum(alert.weather_alert.precipitation_forecast_mm for alert in alerts)
        
        return AlertHistory(
            period_start=period_start,
            period_end=period_end,
            total_alerts=total_alerts,
            alerts_by_level=alerts_by_level,
            average_risk_score=average_risk_score,
            max_river_level=max_river_level,
            total_precipitation=total_precipitation
        )
    
    def calculate_alert_effectiveness(
        self, 
        alerts: List[FloodAlert],
        actual_flood_events: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calcula métricas de eficácia dos alertas
        
        Args:
            alerts: Lista de alertas emitidos
            actual_flood_events: Lista de eventos reais de cheia
            
        Returns:
            Dict: Métricas de eficácia
        """
        if not alerts or not actual_flood_events:
            return {
                "true_positive_rate": 0.0,
                "false_positive_rate": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }
        
        # Simplificação: considera que alertas HIGH/CRITICAL são positivos
        # e eventos reais de cheia validam os alertas
        high_alerts = [a for a in alerts if a.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]]
        
        # Em uma implementação real, seria necessário correlacionar
        # alertas com eventos reais por timestamp e localização
        
        # Placeholder para demonstração
        true_positives = min(len(high_alerts), len(actual_flood_events))
        false_positives = max(0, len(high_alerts) - len(actual_flood_events))
        false_negatives = max(0, len(actual_flood_events) - len(high_alerts))
        
        precision = true_positives / len(high_alerts) if high_alerts else 0.0
        recall = true_positives / len(actual_flood_events) if actual_flood_events else 0.0
        
        return {
            "true_positive_rate": recall,
            "false_positive_rate": false_positives / len(alerts) if alerts else 0.0,
            "precision": precision,
            "recall": recall
        } 