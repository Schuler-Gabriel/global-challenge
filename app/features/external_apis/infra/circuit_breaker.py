"""
Circuit Breaker Implementation - Infrastructure Layer

Implementação do padrão Circuit Breaker para proteção contra
falhas em cascata de APIs externas.

Estados:
- CLOSED: Funcionando normalmente
- OPEN: Circuit aberto, bloqueando chamadas
- HALF_OPEN: Teste de recuperação

Features:
- Threshold configurável de falhas
- Timeout configurável
- Recovery automático
- Métricas de monitoramento
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from enum import Enum

from ..domain.entities import ExternalApiResponse, ApiStatus
from ..domain.services import CircuitBreakerService, ApiServiceConfig
from ..domain.services import CircuitBreakerOpenError


class CircuitState(Enum):
    """Estados do Circuit Breaker"""
    CLOSED = "CLOSED"       # Funcionando normalmente
    OPEN = "OPEN"           # Circuit aberto
    HALF_OPEN = "HALF_OPEN" # Testando recuperação


class CircuitBreakerStats:
    """Estatísticas do Circuit Breaker para uma API"""
    
    def __init__(self):
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.total_requests = 0
        self.total_failures = 0
        self.avg_response_time = 0.0
        self.state_changes = 0
        self.created_at = datetime.now()
    
    def record_success(self, response_time_ms: float):
        """Registra uma chamada bem-sucedida"""
        self.success_count += 1
        self.total_requests += 1
        self.last_success_time = datetime.now()
        
        # Atualizar média de tempo de resposta
        self.avg_response_time = (
            (self.avg_response_time * (self.total_requests - 1) + response_time_ms) 
            / self.total_requests
        )
        
        # Reset do contador de falhas consecutivas
        self.failure_count = 0
    
    def record_failure(self):
        """Registra uma falha"""
        self.failure_count += 1
        self.total_failures += 1
        self.total_requests += 1
        self.last_failure_time = datetime.now()
    
    def reset_failures(self):
        """Reset do contador de falhas"""
        self.failure_count = 0
    
    def get_failure_rate(self) -> float:
        """Calcula taxa de falhas"""
        if self.total_requests == 0:
            return 0.0
        return self.total_failures / self.total_requests
    
    def get_success_rate(self) -> float:
        """Calcula taxa de sucesso"""
        return 1.0 - self.get_failure_rate()


class CircuitBreakerImpl(CircuitBreakerService):
    """
    Implementação concreta do Circuit Breaker
    
    Gerencia estado de múltiplas APIs com configurações
    independentes e métricas de monitoramento.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o Circuit Breaker
        
        Args:
            config: Configurações customizadas (opcional)
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Configurações padrão
        self.config = {
            "failure_threshold": ApiServiceConfig.FAILURE_THRESHOLD,
            "success_threshold": ApiServiceConfig.SUCCESS_THRESHOLD,
            "timeout_threshold_ms": ApiServiceConfig.TIMEOUT_THRESHOLD,
            "open_duration_seconds": ApiServiceConfig.CIRCUIT_OPEN_DURATION,
        }
        
        # Aplicar configurações customizadas
        if config:
            self.config.update(config)
        
        # Estado dos circuits por API
        self._circuits: Dict[str, CircuitState] = {}
        self._stats: Dict[str, CircuitBreakerStats] = {}
        self._open_since: Dict[str, datetime] = {}
        
        # Lock para operações thread-safe
        self._lock = asyncio.Lock()
    
    async def execute(self, api_name: str, operation: Callable, *args, **kwargs) -> ExternalApiResponse:
        """
        Executa operação com proteção de circuit breaker
        
        Args:
            api_name: Nome da API
            operation: Função a ser executada
            *args: Argumentos posicionais
            **kwargs: Argumentos nomeados
            
        Returns:
            ExternalApiResponse: Resposta da operação
        """
        async with self._lock:
            # Verificar estado do circuit
            circuit_state = await self._get_circuit_state(api_name)
            
            # Se circuit está aberto, verificar se pode testar recuperação
            if circuit_state == CircuitState.OPEN:
                if not await self._should_attempt_reset(api_name):
                    # Circuit ainda deve permanecer aberto
                    return ExternalApiResponse(
                        api_name=api_name,
                        status=ApiStatus.CIRCUIT_OPEN,
                        timestamp=datetime.now(),
                        response_time_ms=0.0,
                        error_message="Circuit breaker está aberto",
                        error_code="CIRCUIT_OPEN"
                    )
                else:
                    # Tentar meio-aberto
                    self._circuits[api_name] = CircuitState.HALF_OPEN
                    self.logger.info(f"Circuit breaker {api_name} mudou para HALF_OPEN")
        
        # Executar operação
        start_time = time.time()
        
        try:
            # Chamar operação
            result = await operation(*args, **kwargs)
            
            # Calcular tempo de resposta
            response_time_ms = (time.time() - start_time) * 1000
            
            # Registrar sucesso
            async with self._lock:
                await self._record_success(api_name, response_time_ms)
            
            # Retornar resposta de sucesso
            return ExternalApiResponse(
                api_name=api_name,
                status=ApiStatus.SUCCESS,
                timestamp=datetime.now(),
                response_time_ms=response_time_ms,
                data=result
            )
            
        except Exception as e:
            # Calcular tempo até falha
            response_time_ms = (time.time() - start_time) * 1000
            
            # Registrar falha
            async with self._lock:
                await self._record_failure(api_name, e)
            
            # Determinar status baseado no tipo de erro
            status = self._classify_error(e, response_time_ms)
            
            return ExternalApiResponse(
                api_name=api_name,
                status=status,
                timestamp=datetime.now(),
                response_time_ms=response_time_ms,
                error_message=str(e),
                error_code=type(e).__name__
            )
    
    async def get_circuit_status(self, api_name: str) -> str:
        """Obtém status atual do circuit breaker"""
        async with self._lock:
            state = await self._get_circuit_state(api_name)
            return state.value
    
    async def reset_circuit(self, api_name: str) -> bool:
        """Reseta circuit breaker para CLOSED"""
        try:
            async with self._lock:
                self._circuits[api_name] = CircuitState.CLOSED
                
                # Reset das estatísticas de falha
                if api_name in self._stats:
                    self._stats[api_name].reset_failures()
                
                # Remover timestamp de abertura
                if api_name in self._open_since:
                    del self._open_since[api_name]
                
                self.logger.info(f"Circuit breaker {api_name} resetado para CLOSED")
                return True
                
        except Exception as e:
            self.logger.error(f"Erro ao resetar circuit breaker {api_name}: {e}")
            return False
    
    async def get_metrics(self, api_name: str) -> Dict[str, Any]:
        """Obtém métricas do circuit breaker"""
        async with self._lock:
            if api_name not in self._stats:
                return {
                    "api_name": api_name,
                    "circuit_state": "CLOSED",
                    "message": "Nenhuma métrica disponível"
                }
            
            stats = self._stats[api_name]
            state = await self._get_circuit_state(api_name)
            
            return {
                "api_name": api_name,
                "circuit_state": state.value,
                "failure_count": stats.failure_count,
                "success_count": stats.success_count,
                "total_requests": stats.total_requests,
                "total_failures": stats.total_failures,
                "failure_rate": stats.get_failure_rate(),
                "success_rate": stats.get_success_rate(),
                "avg_response_time_ms": stats.avg_response_time,
                "last_failure": stats.last_failure_time.isoformat() if stats.last_failure_time else None,
                "last_success": stats.last_success_time.isoformat() if stats.last_success_time else None,
                "state_changes": stats.state_changes,
                "created_at": stats.created_at.isoformat(),
                "open_since": self._open_since.get(api_name, {}).isoformat() if api_name in self._open_since else None,
                "thresholds": {
                    "failure_threshold": self.config["failure_threshold"],
                    "success_threshold": self.config["success_threshold"],
                    "timeout_threshold_ms": self.config["timeout_threshold_ms"],
                    "open_duration_seconds": self.config["open_duration_seconds"]
                }
            }
    
    async def _get_circuit_state(self, api_name: str) -> CircuitState:
        """Obtém estado atual do circuit para uma API"""
        if api_name not in self._circuits:
            # Primeira vez - inicializar como CLOSED
            self._circuits[api_name] = CircuitState.CLOSED
            self._stats[api_name] = CircuitBreakerStats()
        
        return self._circuits[api_name]
    
    async def _record_success(self, api_name: str, response_time_ms: float):
        """Registra sucesso e atualiza estado do circuit"""
        # Inicializar se necessário
        if api_name not in self._stats:
            self._stats[api_name] = CircuitBreakerStats()
        
        # Registrar sucesso
        self._stats[api_name].record_success(response_time_ms)
        
        # Verificar se deve fechar circuit
        current_state = await self._get_circuit_state(api_name)
        
        if current_state == CircuitState.HALF_OPEN:
            # Se estava em HALF_OPEN e teve sucesso, pode fechar
            success_count = self._stats[api_name].success_count
            if success_count >= self.config["success_threshold"]:
                self._circuits[api_name] = CircuitState.CLOSED
                self._stats[api_name].state_changes += 1
                
                # Remover timestamp de abertura
                if api_name in self._open_since:
                    del self._open_since[api_name]
                
                self.logger.info(f"Circuit breaker {api_name} fechado após {success_count} sucessos")
    
    async def _record_failure(self, api_name: str, error: Exception):
        """Registra falha e atualiza estado do circuit"""
        # Inicializar se necessário
        if api_name not in self._stats:
            self._stats[api_name] = CircuitBreakerStats()
        
        # Registrar falha
        self._stats[api_name].record_failure()
        
        # Verificar se deve abrir circuit
        current_state = await self._get_circuit_state(api_name)
        
        if current_state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]:
            failure_count = self._stats[api_name].failure_count
            
            if failure_count >= self.config["failure_threshold"]:
                # Abrir circuit
                self._circuits[api_name] = CircuitState.OPEN
                self._open_since[api_name] = datetime.now()
                self._stats[api_name].state_changes += 1
                
                self.logger.warning(
                    f"Circuit breaker {api_name} aberto após {failure_count} falhas consecutivas"
                )
    
    async def _should_attempt_reset(self, api_name: str) -> bool:
        """Verifica se deve tentar resetar um circuit aberto"""
        if api_name not in self._open_since:
            return False
        
        open_since = self._open_since[api_name]
        open_duration = datetime.now() - open_since
        
        return open_duration.total_seconds() >= self.config["open_duration_seconds"]
    
    def _classify_error(self, error: Exception, response_time_ms: float) -> ApiStatus:
        """Classifica tipo de erro para status apropriado"""
        error_type = type(error).__name__
        
        # Timeout
        if "timeout" in error_type.lower() or response_time_ms > self.config["timeout_threshold_ms"]:
            return ApiStatus.TIMEOUT
        
        # Erro de conexão/rede
        if any(term in error_type.lower() for term in ["connection", "network", "dns"]):
            return ApiStatus.UNAVAILABLE
        
        # Erro genérico
        return ApiStatus.ERROR
    
    async def get_all_circuits_status(self) -> Dict[str, Any]:
        """Obtém status de todos os circuits"""
        async with self._lock:
            status = {}
            
            for api_name in self._circuits.keys():
                status[api_name] = await self.get_metrics(api_name)
            
            # Estatísticas globais
            total_apis = len(self._circuits)
            healthy_apis = sum(1 for state in self._circuits.values() if state == CircuitState.CLOSED)
            
            status["_summary"] = {
                "total_apis": total_apis,
                "healthy_apis": healthy_apis,
                "unhealthy_apis": total_apis - healthy_apis,
                "health_rate": healthy_apis / total_apis if total_apis > 0 else 1.0,
                "last_updated": datetime.now().isoformat()
            }
            
            return status
    
    async def force_open_circuit(self, api_name: str, reason: str = "Manual") -> bool:
        """Força abertura de um circuit (para testes/manutenção)"""
        try:
            async with self._lock:
                self._circuits[api_name] = CircuitState.OPEN
                self._open_since[api_name] = datetime.now()
                
                if api_name in self._stats:
                    self._stats[api_name].state_changes += 1
                
                self.logger.warning(f"Circuit breaker {api_name} forçado para OPEN. Razão: {reason}")
                return True
                
        except Exception as e:
            self.logger.error(f"Erro ao forçar abertura do circuit {api_name}: {e}")
            return False 