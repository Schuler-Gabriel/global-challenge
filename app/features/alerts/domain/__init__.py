"""
Alerts Domain Layer

Este módulo exporta todas as entidades, serviços e interfaces de repository
da camada de domínio da feature Alerts.

A Domain Layer contém:
- Entities: Objetos de negócio puros (FloodAlert, RiverLevel, WeatherAlert, etc.)
- Services: Lógica de negócio complexa (FloodAlertService, RiskCalculationService, etc.)
"""

# Entities
from .entities import (
    FloodAlert,
    RiverLevel,
    WeatherAlert,
    AlertHistory,
    AlertLevel,
    RiskLevel,
    AlertAction
)

# Services
from .services import (
    FloodAlertService,
    RiskCalculationService,
    AlertClassificationService,
    AlertHistoryService,
    AlertConfiguration
)

__all__ = [
    # Entities
    "FloodAlert",
    "RiverLevel", 
    "WeatherAlert",
    "AlertHistory",
    
    # Enums
    "AlertLevel",
    "RiskLevel",
    "AlertAction",
    
    # Services
    "FloodAlertService",
    "RiskCalculationService",
    "AlertClassificationService", 
    "AlertHistoryService",
    "AlertConfiguration"
]

