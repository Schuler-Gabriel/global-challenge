"""
Monitoring Service Implementation - Infrastructure Layer

Implementação de monitoramento para APIs externas com coleta de métricas,
health checks e alertas de disponibilidade.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field

from ..domain.entities import ApiStatus
from ..domain.services import MonitoringService


@dataclass
class ApiMetric:
    """Métrica individual de uma chamada de API"""
    timestamp: datetime
    api_name: str
    operation: str
    success: bool
    response_time_ms: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class ApiHealthCheck:
    """Resultado de health check de uma API"""
    timestamp: datetime
    api_name: str
    status: ApiStatus
    response_time_ms: float
    error_message: Optional[str] = None


@dataclass
class ApiAggregatedMetrics:
    """Métricas agregadas para uma API"""
    api_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    avg_response_time_ms: float = 0.0
    min_response_time_ms: float = float('inf')
    max_response_time_ms: float = 0.0
    success_rate: float = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    current_streak: int = 0  # Sequência atual (positivo=sucessos, negativo=falhas)
    
    # Métricas por período
    calls_last_hour: int = 0
    calls_last_day: int = 0
    errors_last_hour: int = 0
    errors_last_day: int = 0


class ApiMonitoringService(MonitoringService):
    """
    Implementação do serviço de monitoramento para APIs externas
    
    Coleta e agrega métricas de performance, disponibilidade e saúde
    das APIs externas utilizadas pelo sistema.
    """
    
    def __init__(self, max_metrics_history: int = 10000):
        """
        Inicializa o serviço de monitoramento
        
        Args:
            max_metrics_history: Número máximo de métricas mantidas em memória
        """
        self.max_metrics_history = max_metrics_history
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Armazenamento de métricas
        self._metrics: deque = deque(maxlen=max_metrics_history)
        self._health_checks: Dict[str, List[ApiHealthCheck]] = defaultdict(list)
        
        # Métricas agregadas por API
        self._aggregated_metrics: Dict[str, ApiAggregatedMetrics] = {}
        
        # Lock para thread safety
        self._lock = asyncio.Lock()
        
        # Configurações de alertas
        self._alert_thresholds = {
            'max_response_time_ms': 5000,
            'min_success_rate': 0.95,
            'max_consecutive_failures': 5
        }
        
        # Status atual das APIs
        self._current_status: Dict[str, ApiStatus] = {}
    
    async def record_api_call(
        self, 
        api_name: str, 
        operation: str, 
        success: bool, 
        response_time_ms: float,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Registra chamada à API para métricas
        
        Args:
            api_name: Nome da API
            operation: Nome da operação
            success: Se foi bem-sucedida
            response_time_ms: Tempo de resposta em ms
            error_type: Tipo do erro (se houver)
            error_message: Mensagem de erro (se houver)
        """
        timestamp = datetime.now()
        
        metric = ApiMetric(
            timestamp=timestamp,
            api_name=api_name,
            operation=operation,
            success=success,
            response_time_ms=response_time_ms,
            error_type=error_type,
            error_message=error_message
        )
        
        async with self._lock:
            # Adicionar métrica
            self._metrics.append(metric)
            
            # Atualizar métricas agregadas
            await self._update_aggregated_metrics(metric)
            
            # Verificar alertas
            await self._check_alerts(api_name)
            
            # Log da métrica
            if success:
                self.logger.debug(f"API {api_name}.{operation} - SUCCESS ({response_time_ms:.1f}ms)")
            else:
                self.logger.warning(f"API {api_name}.{operation} - FAILED ({response_time_ms:.1f}ms): {error_message}")
    
    async def get_api_metrics(self, api_name: str, hours_back: int = 24) -> Dict[str, Any]:
        """
        Obtém métricas de uma API
        
        Args:
            api_name: Nome da API
            hours_back: Horas para trás para análise
            
        Returns:
            Dict: Métricas de performance e disponibilidade
        """
        async with self._lock:
            # Métricas agregadas
            aggregated = self._aggregated_metrics.get(api_name, ApiAggregatedMetrics(api_name))
            
            # Filtrar métricas por período
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            period_metrics = [
                m for m in self._metrics 
                if m.api_name == api_name and m.timestamp >= cutoff_time
            ]
            
            # Calcular métricas do período
            period_stats = self._calculate_period_stats(period_metrics)
            
            # Status atual
            current_status = self._current_status.get(api_name, ApiStatus.UNKNOWN)
            
            # Health checks recentes
            recent_health_checks = self._health_checks.get(api_name, [])[-10:]  # Últimos 10
            
            return {
                'api_name': api_name,
                'current_status': current_status.value,
                'aggregated_metrics': {
                    'total_calls': aggregated.total_calls,
                    'successful_calls': aggregated.successful_calls,
                    'failed_calls': aggregated.failed_calls,
                    'success_rate': aggregated.success_rate,
                    'avg_response_time_ms': aggregated.avg_response_time_ms,
                    'min_response_time_ms': aggregated.min_response_time_ms if aggregated.min_response_time_ms != float('inf') else 0,
                    'max_response_time_ms': aggregated.max_response_time_ms,
                    'current_streak': aggregated.current_streak,
                    'last_success': aggregated.last_success.isoformat() if aggregated.last_success else None,
                    'last_failure': aggregated.last_failure.isoformat() if aggregated.last_failure else None
                },
                'period_metrics': period_stats,
                'recent_health_checks': [
                    {
                        'timestamp': hc.timestamp.isoformat(),
                        'status': hc.status.value,
                        'response_time_ms': hc.response_time_ms,
                        'error_message': hc.error_message
                    }
                    for hc in recent_health_checks
                ],
                'alert_status': await self._get_alert_status(api_name),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_system_health(self) -> Dict[str, Any]:
        """
        Obtém saúde geral do sistema de APIs externas
        
        Returns:
            Dict: Status de saúde de todas as APIs
        """
        async with self._lock:
            all_apis = set(self._aggregated_metrics.keys()) | set(self._current_status.keys())
            
            api_statuses = []
            overall_healthy = True
            
            for api_name in all_apis:
                api_metrics = await self.get_api_metrics(api_name, hours_back=1)
                status = self._current_status.get(api_name, ApiStatus.UNKNOWN)
                
                api_status = {
                    'api_name': api_name,
                    'status': status.value,
                    'success_rate': api_metrics['aggregated_metrics']['success_rate'],
                    'avg_response_time_ms': api_metrics['aggregated_metrics']['avg_response_time_ms'],
                    'current_streak': api_metrics['aggregated_metrics']['current_streak']
                }
                
                api_statuses.append(api_status)
                
                # Verificar se afeta saúde geral
                if status not in [ApiStatus.ONLINE, ApiStatus.UNKNOWN]:
                    overall_healthy = False
            
            # Estatísticas gerais
            total_metrics = len(self._metrics)
            successful_metrics = sum(1 for m in self._metrics if m.success)
            overall_success_rate = successful_metrics / total_metrics if total_metrics > 0 else 0.0
            
            return {
                'overall_healthy': overall_healthy,
                'overall_success_rate': overall_success_rate,
                'total_apis': len(all_apis),
                'online_apis': sum(1 for status in self._current_status.values() if status == ApiStatus.ONLINE),
                'total_metrics_collected': total_metrics,
                'apis': api_statuses,
                'timestamp': datetime.now().isoformat()
            }
    
    async def check_api_health(self, api_name: str) -> ApiStatus:
        """
        Verifica saúde específica de uma API
        
        Args:
            api_name: Nome da API
            
        Returns:
            ApiStatus: Status da API
        """
        async with self._lock:
            # Obter métricas recentes (última hora)
            recent_metrics = [
                m for m in self._metrics 
                if m.api_name == api_name and m.timestamp >= datetime.now() - timedelta(hours=1)
            ]
            
            if not recent_metrics:
                return ApiStatus.UNKNOWN
            
            # Calcular métricas de saúde
            total_calls = len(recent_metrics)
            successful_calls = sum(1 for m in recent_metrics if m.success)
            success_rate = successful_calls / total_calls
            
            avg_response_time = sum(m.response_time_ms for m in recent_metrics) / total_calls
            
            # Determinar status baseado em critérios
            if success_rate >= 0.95 and avg_response_time <= 2000:
                status = ApiStatus.ONLINE
            elif success_rate >= 0.80 and avg_response_time <= 5000:
                status = ApiStatus.DEGRADED
            else:
                status = ApiStatus.OFFLINE
            
            # Atualizar status atual
            self._current_status[api_name] = status
            
            return status
    
    async def record_health_check(
        self, 
        api_name: str, 
        status: ApiStatus, 
        response_time_ms: float,
        error_message: Optional[str] = None
    ) -> None:
        """
        Registra resultado de health check
        
        Args:
            api_name: Nome da API
            status: Status verificado
            response_time_ms: Tempo de resposta do health check
            error_message: Mensagem de erro (se houver)
        """
        health_check = ApiHealthCheck(
            timestamp=datetime.now(),
            api_name=api_name,
            status=status,
            response_time_ms=response_time_ms,
            error_message=error_message
        )
        
        async with self._lock:
            # Adicionar health check
            self._health_checks[api_name].append(health_check)
            
            # Manter apenas os últimos 100 health checks por API
            if len(self._health_checks[api_name]) > 100:
                self._health_checks[api_name] = self._health_checks[api_name][-100:]
            
            # Atualizar status atual
            self._current_status[api_name] = status
            
            self.logger.debug(f"Health check {api_name}: {status.value} ({response_time_ms:.1f}ms)")
    
    async def get_alerts(self, api_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Obtém alertas ativos
        
        Args:
            api_name: Nome da API (opcional, todas se None)
            
        Returns:
            List[Dict]: Lista de alertas ativos
        """
        alerts = []
        
        async with self._lock:
            apis_to_check = [api_name] if api_name else self._aggregated_metrics.keys()
            
            for api in apis_to_check:
                api_alerts = await self._get_alert_status(api)
                if api_alerts['active_alerts']:
                    alerts.extend(api_alerts['active_alerts'])
        
        return alerts
    
    async def cleanup_old_metrics(self, days_to_keep: int = 7) -> int:
        """
        Remove métricas antigas
        
        Args:
            days_to_keep: Dias de métricas para manter
            
        Returns:
            int: Número de métricas removidas
        """
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)
        
        async with self._lock:
            original_size = len(self._metrics)
            
            # Filtrar métricas recentes
            self._metrics = deque(
                (m for m in self._metrics if m.timestamp >= cutoff_time),
                maxlen=self.max_metrics_history
            )
            
            # Limpar health checks antigos
            for api_name in self._health_checks:
                self._health_checks[api_name] = [
                    hc for hc in self._health_checks[api_name] 
                    if hc.timestamp >= cutoff_time
                ]
            
            removed_count = original_size - len(self._metrics)
            
            if removed_count > 0:
                self.logger.info(f"Limpeza de métricas: {removed_count} métricas antigas removidas")
            
            return removed_count
    
    # Métodos privados
    
    async def _update_aggregated_metrics(self, metric: ApiMetric) -> None:
        """Atualiza métricas agregadas com nova métrica"""
        api_name = metric.api_name
        
        if api_name not in self._aggregated_metrics:
            self._aggregated_metrics[api_name] = ApiAggregatedMetrics(api_name)
        
        agg = self._aggregated_metrics[api_name]
        
        # Atualizar contadores
        agg.total_calls += 1
        
        if metric.success:
            agg.successful_calls += 1
            agg.last_success = metric.timestamp
            
            # Atualizar streak
            if agg.current_streak < 0:
                agg.current_streak = 1
            else:
                agg.current_streak += 1
        else:
            agg.failed_calls += 1
            agg.last_failure = metric.timestamp
            
            # Atualizar streak
            if agg.current_streak > 0:
                agg.current_streak = -1
            else:
                agg.current_streak -= 1
        
        # Atualizar taxa de sucesso
        agg.success_rate = agg.successful_calls / agg.total_calls
        
        # Atualizar tempos de resposta
        response_time = metric.response_time_ms
        agg.avg_response_time_ms = (
            (agg.avg_response_time_ms * (agg.total_calls - 1) + response_time) 
            / agg.total_calls
        )
        agg.min_response_time_ms = min(agg.min_response_time_ms, response_time)
        agg.max_response_time_ms = max(agg.max_response_time_ms, response_time)
        
        # Atualizar contadores por período
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        # Recontar métricas por período (simplificado)
        recent_metrics = [
            m for m in self._metrics 
            if m.api_name == api_name
        ]
        
        agg.calls_last_hour = sum(1 for m in recent_metrics if m.timestamp >= hour_ago)
        agg.calls_last_day = sum(1 for m in recent_metrics if m.timestamp >= day_ago)
        agg.errors_last_hour = sum(1 for m in recent_metrics if m.timestamp >= hour_ago and not m.success)
        agg.errors_last_day = sum(1 for m in recent_metrics if m.timestamp >= day_ago and not m.success)
    
    def _calculate_period_stats(self, metrics: List[ApiMetric]) -> Dict[str, Any]:
        """Calcula estatísticas para um período de métricas"""
        if not metrics:
            return {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'success_rate': 0.0,
                'avg_response_time_ms': 0.0,
                'min_response_time_ms': 0.0,
                'max_response_time_ms': 0.0
            }
        
        total_calls = len(metrics)
        successful_calls = sum(1 for m in metrics if m.success)
        failed_calls = total_calls - successful_calls
        success_rate = successful_calls / total_calls
        
        response_times = [m.response_time_ms for m in metrics]
        avg_response_time = sum(response_times) / len(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        
        return {
            'total_calls': total_calls,
            'successful_calls': successful_calls,
            'failed_calls': failed_calls,
            'success_rate': success_rate,
            'avg_response_time_ms': avg_response_time,
            'min_response_time_ms': min_response_time,
            'max_response_time_ms': max_response_time
        }
    
    async def _check_alerts(self, api_name: str) -> None:
        """Verifica e gera alertas para uma API"""
        if api_name not in self._aggregated_metrics:
            return
        
        agg = self._aggregated_metrics[api_name]
        
        # Alerta de taxa de sucesso baixa
        if agg.success_rate < self._alert_thresholds['min_success_rate'] and agg.total_calls >= 10:
            self.logger.warning(f"ALERT: {api_name} success rate ({agg.success_rate:.2%}) below threshold")
        
        # Alerta de tempo de resposta alto
        if agg.avg_response_time_ms > self._alert_thresholds['max_response_time_ms']:
            self.logger.warning(f"ALERT: {api_name} response time ({agg.avg_response_time_ms:.1f}ms) above threshold")
        
        # Alerta de falhas consecutivas
        if agg.current_streak <= -self._alert_thresholds['max_consecutive_failures']:
            self.logger.error(f"ALERT: {api_name} has {abs(agg.current_streak)} consecutive failures")
    
    async def _get_alert_status(self, api_name: str) -> Dict[str, Any]:
        """Obtém status de alertas para uma API"""
        if api_name not in self._aggregated_metrics:
            return {'active_alerts': [], 'alert_count': 0}
        
        agg = self._aggregated_metrics[api_name]
        alerts = []
        
        # Verificar alertas ativos
        if agg.success_rate < self._alert_thresholds['min_success_rate'] and agg.total_calls >= 10:
            alerts.append({
                'type': 'low_success_rate',
                'severity': 'warning',
                'message': f"Success rate ({agg.success_rate:.2%}) below threshold ({self._alert_thresholds['min_success_rate']:.2%})",
                'value': agg.success_rate,
                'threshold': self._alert_thresholds['min_success_rate']
            })
        
        if agg.avg_response_time_ms > self._alert_thresholds['max_response_time_ms']:
            alerts.append({
                'type': 'high_response_time',
                'severity': 'warning',
                'message': f"Response time ({agg.avg_response_time_ms:.1f}ms) above threshold ({self._alert_thresholds['max_response_time_ms']}ms)",
                'value': agg.avg_response_time_ms,
                'threshold': self._alert_thresholds['max_response_time_ms']
            })
        
        if agg.current_streak <= -self._alert_thresholds['max_consecutive_failures']:
            alerts.append({
                'type': 'consecutive_failures',
                'severity': 'critical',
                'message': f"{abs(agg.current_streak)} consecutive failures",
                'value': abs(agg.current_streak),
                'threshold': self._alert_thresholds['max_consecutive_failures']
            })
        
        return {
            'active_alerts': alerts,
            'alert_count': len(alerts)
        }
    
    async def get_all_metrics(self) -> Dict[str, Any]:
        """
        Obtém métricas de todas as APIs
        
        Returns:
            Dict: Métricas consolidadas de todas as APIs
        """
        async with self._lock:
            all_apis = list(self._aggregated_metrics.keys())
            api_metrics = {}
            
            for api_name in all_apis:
                api_metrics[api_name] = await self.get_api_metrics(api_name, hours_back=24)
            
            # Métricas globais
            total_metrics = len(self._metrics)
            successful_metrics = sum(1 for m in self._metrics if m.success)
            
            return {
                'total_apis': len(all_apis),
                'total_metrics_collected': total_metrics,
                'overall_success_rate': successful_metrics / total_metrics if total_metrics > 0 else 0.0,
                'apis': api_metrics,
                'timestamp': datetime.now().isoformat()
            } 