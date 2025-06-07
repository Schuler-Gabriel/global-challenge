"""
Main FastAPI application for the Flood Alert System.

This module creates and configures the FastAPI application with all necessary
middleware, exception handlers, and route registration.
"""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.dependencies import add_request_id, get_cache, get_health_checker
from app.core.exceptions import (
    ERROR_STATUS_MAPPING,
    BaseApplicationException,
    create_http_exception_from_app_exception,
    get_error_details,
)
from app.core.logging import (
    get_request_context_logger,
    get_structured_logger,
    setup_logging,
)

# Initialize settings and logging
settings = get_settings()
setup_logging(settings)
logger = get_structured_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info(
        "Starting Flood Alert System API",
        extra={
            "version": settings.app_version,
            "environment": settings.environment,
            "debug": settings.debug,
        },
    )

    # Initialize cache
    try:
        cache = await get_cache()
        cache_health = cache.health_check()
        logger.info("Cache initialized", extra={"cache_healthy": cache_health})
    except Exception as e:
        logger.error("Failed to initialize cache", extra={"error": str(e)})

    # TODO: Initialize ML model
    # TODO: Validate external API connections

    yield

    # Shutdown
    logger.info("Shutting down Flood Alert System API")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Sistema inteligente de previsão meteorológica e alertas de cheias para Porto Alegre",
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    openapi_url="/openapi.json" if not settings.is_production else None,
)


# Middleware Configuration
@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Add request ID to all requests for tracing."""
    await add_request_id(request)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests with timing and context."""
    start_time = time.time()

    # Get request context logger
    request_logger = get_request_context_logger(
        "app.request",
        request_id=getattr(request.state, "request_id", "unknown"),
        client_ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("User-Agent", "unknown"),
    )

    # Log request start
    request_logger.info(
        f"Request started: {request.method} {request.url}",
        extra={
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
        },
    )

    try:
        response = await call_next(request)

        # Calculate response time
        process_time = time.time() - start_time

        # Log successful response
        request_logger.info(
            f"Request completed: {request.method} {request.url}",
            extra={
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "process_time": process_time,
            },
        )

        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)

        return response

    except Exception as e:
        # Calculate response time for failed requests
        process_time = time.time() - start_time

        # Log error
        request_logger.error(
            f"Request failed: {request.method} {request.url}",
            extra={
                "method": request.method,
                "url": str(request.url),
                "process_time": process_time,
                "error": str(e),
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )

        raise


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to responses."""
    response = await call_next(request)

    if settings.is_production:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

    return response


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time"],
)

# Trusted Host Middleware (for production)
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[
            "*.yourdomain.com",
            "yourdomain.com",
        ],  # TODO: Configure actual domains
    )


# Exception Handlers
@app.exception_handler(BaseApplicationException)
async def application_exception_handler(
    request: Request, exc: BaseApplicationException
):
    """Handle application-specific exceptions."""
    request_logger = get_request_context_logger(
        "app.error", request_id=getattr(request.state, "request_id", "unknown")
    )

    # Determine status code
    status_code = ERROR_STATUS_MAPPING.get(
        type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR
    )

    # Log the error
    request_logger.error(
        f"Application error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "error_type": type(exc).__name__,
            "context": exc.context,
            "status_code": status_code,
        },
    )

    # Create HTTP exception
    http_exc = create_http_exception_from_app_exception(exc, status_code)

    return JSONResponse(status_code=http_exc.status_code, content=http_exc.detail)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging."""
    request_logger = get_request_context_logger(
        "app.error", request_id=getattr(request.state, "request_id", "unknown")
    )

    request_logger.warning(
        f"HTTP error: {exc.detail}",
        extra={"status_code": exc.status_code, "detail": exc.detail},
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": "HTTP_ERROR",
            "message": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    request_logger = get_request_context_logger(
        "app.error", request_id=getattr(request.state, "request_id", "unknown")
    )

    # Get error details
    error_details = get_error_details(exc)

    request_logger.error(
        f"Unexpected error: {str(exc)}", extra=error_details, exc_info=True
    )

    # Return generic error in production, detailed in development
    if settings.is_production:
        detail = {
            "error_code": "INTERNAL_ERROR",
            "message": "An internal error occurred. Please try again later.",
        }
    else:
        detail = {
            "error_code": "INTERNAL_ERROR",
            "message": str(exc),
            "type": type(exc).__name__,
            "context": error_details,
        }

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=detail
    )


# Health Check Endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Basic health check endpoint.

    Returns:
        Dict: Health status
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.app_version,
        "environment": settings.environment,
    }


@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check():
    """
    Detailed health check including dependencies.

    Returns:
        Dict: Detailed health status
    """
    from app.core.dependencies import get_http_client

    health_checker = await get_health_checker()
    cache = await get_cache()

    # Check cache health
    cache_health = await health_checker.check_cache_health(cache)

    # Check external APIs health
    async with get_http_client() as http_client:
        apis_health = await health_checker.check_external_apis_health(http_client)

    # Check model health
    model_health = await health_checker.check_model_health()

    # Overall health status
    all_healthy = (
        cache_health.get("status") == "healthy"
        and all(api.get("status") == "healthy" for api in apis_health.values())
        and model_health.get("status") == "healthy"
    )

    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": time.time(),
        "version": settings.app_version,
        "environment": settings.environment,
        "components": {
            "cache": cache_health,
            "external_apis": apis_health,
            "ml_model": model_health,
        },
    }


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.

    Returns:
        Dict: API information
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Sistema inteligente de previsão meteorológica e alertas de cheias para Porto Alegre",
        "docs_url": "/docs" if not settings.is_production else None,
        "health_url": "/health",
        "api_prefix": settings.api_v1_prefix,
    }


# API Router Registration
# TODO: Add feature routers when implemented
# from app.features.forecast.presentation.routes import router as forecast_router
# from app.features.alerts.presentation.routes import router as alerts_router

# app.include_router(forecast_router, prefix=f"{settings.api_v1_prefix}/forecast", tags=["Forecast"])
# app.include_router(alerts_router, prefix=f"{settings.api_v1_prefix}/alerts", tags=["Alerts"])


# Custom OpenAPI Schema
def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=settings.app_name,
        version=settings.app_version,
        description="""
        # Sistema de Alertas de Cheias - Rio Guaíba
        
        API para previsão meteorológica e geração de alertas de cheias em Porto Alegre.
        
        ## Funcionalidades
        
        - **Previsão Meteorológica**: Modelos LSTM para previsão de chuva 24h
        - **Alertas Inteligentes**: Sistema automatizado baseado em matriz de risco
        - **APIs Externas**: Integração com CPTEC e Nível do Guaíba
        - **Cache Inteligente**: Sistema de cache para alta performance
        
        ## Endpoints Principais
        
        - `/api/v1/forecast/predict`: Previsão meteorológica
        - `/api/v1/alerts/current`: Alerta atual
        - `/health`: Status da aplicação
        
        ## Autenticação
        
        Atualmente a API é aberta. Autenticação será implementada nas próximas versões.
        """,
        routes=app.routes,
    )

    # Add custom metadata
    openapi_schema["info"]["contact"] = {
        "name": "Sistema de Alertas de Cheias",
        "email": "suporte@alertas-cheias.com.br",
    }

    openapi_schema["info"]["license"] = {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }

    # Add server information
    openapi_schema["servers"] = [
        {
            "url": settings.api_base_url,
            "description": f"API Server - {settings.environment.title()}",
        }
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload and settings.is_development,
        log_level=settings.log_level.lower(),
        access_log=not settings.is_production,
        use_colors=settings.is_development,
    )
