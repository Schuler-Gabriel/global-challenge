"""
Dependency injection module for FastAPI.

This module provides all dependencies used throughout the application,
including configuration, logging, caching, and external services.
"""

from typing import Generator, Optional, Dict, Any
from functools import lru_cache
import httpx
import logging
from fastapi import Depends, Request, HTTPException, status
from contextlib import asynccontextmanager

from .config import Settings, get_settings
from .exceptions import (
    CacheConnectionError,
    APIConnectionError,
    ConfigurationError
)


# Configuration Dependencies
def get_current_settings() -> Settings:
    """
    Get current application settings.
    
    Returns:
        Settings: Current application settings
    """
    return get_settings()


# Logging Dependencies
@lru_cache()
def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get logger instance with proper configuration.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


def get_request_logger(request: Request) -> logging.Logger:
    """
    Get logger with request context.
    
    Args:
        request: FastAPI request object
        
    Returns:
        logging.Logger: Logger with request context
    """
    logger = get_logger("request")
    
    # Add request ID for tracing
    request_id = getattr(request.state, "request_id", "unknown")
    logger = logging.LoggerAdapter(logger, {"request_id": request_id})
    
    return logger


# HTTP Client Dependencies
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_http_client():
    """
    Get HTTP client for external API calls.
    
    Yields:
        httpx.AsyncClient: Configured HTTP client
    """
    settings = get_settings()
    
    timeout = httpx.Timeout(
        connect=5.0,
        read=float(settings.api_timeout),
        write=5.0,
        pool=10.0
    )
    
    limits = httpx.Limits(
        max_keepalive_connections=20,
        max_connections=100,
        keepalive_expiry=30.0
    )
    
    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        follow_redirects=True,
        headers={
            "User-Agent": f"{settings.app_name}/{settings.app_version}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    ) as client:
        yield client


# Cache Dependencies
class CacheManager:
    """
    Cache manager for handling different cache backends.
    
    Supports in-memory caching and Redis (when available).
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._memory_cache: Dict[str, Any] = {}
        self._redis_client = None
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize cache backend."""
        if self.settings.redis_url:
            try:
                import redis
                self._redis_client = redis.from_url(
                    self.settings.redis_url,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    health_check_interval=30
                )
                # Test connection
                self._redis_client.ping()
            except Exception as e:
                logger = get_logger("cache")
                logger.warning(f"Redis connection failed, falling back to memory cache: {e}")
                self._redis_client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if self._redis_client:
                import json
                value = self._redis_client.get(key)
                return json.loads(value) if value else None
            else:
                return self._memory_cache.get(key)
        except Exception as e:
            logger = get_logger("cache")
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            ttl = ttl or self.settings.cache_ttl_seconds
            
            if self._redis_client:
                import json
                return self._redis_client.setex(key, ttl, json.dumps(value))
            else:
                # Simple memory cache without TTL (for simplicity)
                if len(self._memory_cache) >= self.settings.cache_max_size:
                    # Remove oldest item (simple FIFO)
                    oldest_key = next(iter(self._memory_cache))
                    del self._memory_cache[oldest_key]
                
                self._memory_cache[key] = value
                return True
        except Exception as e:
            logger = get_logger("cache")
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            if self._redis_client:
                return bool(self._redis_client.delete(key))
            else:
                return self._memory_cache.pop(key, None) is not None
        except Exception as e:
            logger = get_logger("cache")
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache."""
        try:
            if self._redis_client:
                return self._redis_client.flushdb()
            else:
                self._memory_cache.clear()
                return True
        except Exception as e:
            logger = get_logger("cache")
            logger.error(f"Cache clear error: {e}")
            return False
    
    def health_check(self) -> bool:
        """Check cache health."""
        try:
            if self._redis_client:
                self._redis_client.ping()
                return True
            else:
                return True  # Memory cache is always available
        except Exception:
            return False


@lru_cache()
def get_cache_manager() -> CacheManager:
    """
    Get cache manager instance.
    
    Returns:
        CacheManager: Configured cache manager
    """
    settings = get_settings()
    return CacheManager(settings)


async def get_cache() -> CacheManager:
    """
    FastAPI dependency for cache manager.
    
    Returns:
        CacheManager: Cache manager instance
    """
    return get_cache_manager()


# Request Context Dependencies
def get_request_id() -> str:
    """
    Generate unique request ID for tracing.
    
    Returns:
        str: Unique request ID
    """
    import uuid
    return str(uuid.uuid4())


async def add_request_id(request: Request) -> None:
    """
    Add request ID to request state.
    
    Args:
        request: FastAPI request object
    """
    request_id = request.headers.get("X-Request-ID") or get_request_id()
    request.state.request_id = request_id


# Rate Limiting Dependencies  
class RateLimiter:
    """
    Simple rate limiter for API endpoints.
    """
    
    def __init__(self, cache_manager: CacheManager, settings: Settings):
        self.cache = cache_manager
        self.settings = settings
    
    async def check_rate_limit(self, identifier: str, limit: int, window: int = 60) -> bool:
        """
        Check if identifier is within rate limits.
        
        Args:
            identifier: Unique identifier (IP, user ID, etc.)
            limit: Maximum requests allowed
            window: Time window in seconds
            
        Returns:
            bool: True if within limits, False otherwise
        """
        import time
        
        key = f"rate_limit:{identifier}"
        current_time = int(time.time())
        
        # Get current request count
        data = await self.cache.get(key) or {"count": 0, "window_start": current_time}
        
        # Reset if window expired
        if current_time - data["window_start"] >= window:
            data = {"count": 0, "window_start": current_time}
        
        # Check limit
        if data["count"] >= limit:
            return False
        
        # Increment count
        data["count"] += 1
        await self.cache.set(key, data, ttl=window)
        
        return True


async def get_rate_limiter(
    cache: CacheManager = Depends(get_cache)
) -> RateLimiter:
    """
    Get rate limiter instance.
    
    Args:
        cache: Cache manager dependency
        
    Returns:
        RateLimiter: Rate limiter instance
    """
    settings = get_settings()
    return RateLimiter(cache, settings)


async def check_rate_limit(
    request: Request,
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
) -> None:
    """
    Check rate limit for current request.
    
    Args:
        request: FastAPI request object
        rate_limiter: Rate limiter dependency
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    client_ip = request.client.host if request.client else "unknown"
    settings = get_settings()
    
    if not await rate_limiter.check_rate_limit(
        identifier=client_ip,
        limit=settings.rate_limit_per_minute,
        window=60
    ):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )


# Health Check Dependencies
class HealthChecker:
    """
    Health checker for various application components.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    async def check_cache_health(self, cache: CacheManager) -> Dict[str, Any]:
        """Check cache health."""
        try:
            is_healthy = cache.health_check()
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "cache_type": "redis" if cache._redis_client else "memory"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def check_external_apis_health(self, http_client: httpx.AsyncClient) -> Dict[str, Any]:
        """Check external APIs health."""
        results = {}
        
        # Check Guaíba API
        try:
            response = await http_client.get(
                self.settings.guaiba_api_url,
                timeout=5.0
            )
            results["guaiba_api"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            results["guaiba_api"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check CPTEC API
        try:
            response = await http_client.get(
                self.settings.cptec_api_url,
                timeout=5.0
            )
            results["cptec_api"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            results["cptec_api"] = {
                "status": "error",
                "error": str(e)
            }
        
        return results
    
    async def check_model_health(self) -> Dict[str, Any]:
        """Check ML model health."""
        import os
        
        model_path = self.settings.model_full_path
        
        return {
            "status": "healthy" if os.path.exists(model_path) else "unhealthy",
            "model_path": model_path,
            "model_version": self.settings.model_version
        }


async def get_health_checker() -> HealthChecker:
    """
    Get health checker instance.
    
    Returns:
        HealthChecker: Health checker instance
    """
    settings = get_settings()
    return HealthChecker(settings)


# Authentication Dependencies (placeholder for future use)
async def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """
    Get current authenticated user (placeholder).
    
    Args:
        request: FastAPI request object
        
    Returns:
        Optional[Dict]: User information or None
    """
    # TODO: Implement authentication logic
    return None


def require_auth(current_user: Optional[Dict] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Require authentication (placeholder).
    
    Args:
        current_user: Current user dependency
        
    Returns:
        Dict: User information
        
    Raises:
        HTTPException: If not authenticated
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return current_user


# Utility Dependencies
def get_client_ip(request: Request) -> str:
    """
    Get client IP address from request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        str: Client IP address
    """
    # Check for forwarded headers
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    return request.client.host if request.client else "unknown"


def get_user_agent(request: Request) -> str:
    """
    Get user agent from request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        str: User agent string
    """
    return request.headers.get("User-Agent", "unknown") 