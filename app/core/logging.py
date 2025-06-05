"""
Structured logging configuration for the Flood Alert System.

This module provides JSON-based structured logging with proper formatting,
rotation, and context management for better monitoring and debugging.
"""

import logging
import logging.handlers
import sys
import json
import traceback
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path

from .config import Settings, get_settings


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Converts log records to JSON format with consistent structure
    including timestamp, level, message, and context information.
    """
    
    def __init__(self):
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            str: JSON-formatted log entry
        """
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add process/thread info
        log_data.update({
            "process_id": record.process,
            "thread_id": record.thread,
            "thread_name": record.threadName,
        })
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from LoggerAdapter or custom fields
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id
        
        # Add custom fields from extra parameter
        for key, value in record.__dict__.items():
            if key not in [
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "lineno", "funcName", "created",
                "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "message", "exc_info", "exc_text",
                "stack_info", "getMessage"
            ] and not key.startswith("_"):
                log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """
    Enhanced text formatter for development and debugging.
    
    Provides colored output and detailed formatting for better readability
    during development.
    """
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
    }
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stderr.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as colored text.
        
        Args:
            record: Log record to format
            
        Returns:
            str: Formatted log entry
        """
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        
        # Get color for level
        color = self.COLORS.get(record.levelname, '') if self.use_colors else ''
        reset = self.COLORS['RESET'] if self.use_colors else ''
        
        # Build log entry
        parts = [
            f"{timestamp}",
            f"{color}{record.levelname:8}{reset}",
            f"{record.name}",
            f"{record.funcName}:{record.lineno}",
            f"{record.getMessage()}"
        ]
        
        # Add request ID if present
        if hasattr(record, "request_id"):
            parts.insert(-1, f"[req:{record.request_id[:8]}]")
        
        log_entry = " | ".join(parts)
        
        # Add exception info if present
        if record.exc_info:
            log_entry += "\n" + self.formatException(record.exc_info)
        
        return log_entry


class StructuredLogger:
    """
    Structured logger setup and management.
    
    Provides centralized logging configuration with support for different
    formatters, handlers, and log levels based on environment.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._configured = False
    
    def configure_logging(self) -> None:
        """
        Configure application logging based on settings.
        
        Sets up formatters, handlers, and loggers according to the
        current environment and configuration.
        """
        if self._configured:
            return
        
        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Set root log level
        root_logger.setLevel(getattr(logging, self.settings.log_level))
        
        # Choose formatter based on settings
        if self.settings.log_format == "JSON":
            formatter = JSONFormatter()
        else:
            formatter = TextFormatter(use_colors=self.settings.is_development)
        
        # Configure console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, self.settings.log_level))
        root_logger.addHandler(console_handler)
        
        # Configure file handler if specified
        if self.settings.log_file:
            self._configure_file_logging(formatter)
        
        # Configure specific loggers
        self._configure_application_loggers()
        
        # Suppress noisy third-party loggers
        self._configure_third_party_loggers()
        
        self._configured = True
    
    def _configure_file_logging(self, formatter: logging.Formatter) -> None:
        """
        Configure file-based logging with rotation.
        
        Args:
            formatter: Log formatter to use
        """
        log_file_path = Path(self.settings.log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse rotation size (e.g., "10 MB" -> 10 * 1024 * 1024)
        size_str = self.settings.log_rotation_size.upper()
        if "MB" in size_str:
            max_bytes = int(size_str.replace("MB", "").strip()) * 1024 * 1024
        elif "KB" in size_str:
            max_bytes = int(size_str.replace("KB", "").strip()) * 1024
        elif "GB" in size_str:
            max_bytes = int(size_str.replace("GB", "").strip()) * 1024 * 1024 * 1024
        else:
            max_bytes = 10 * 1024 * 1024  # Default 10 MB
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file_path),
            maxBytes=max_bytes,
            backupCount=self.settings.log_retention_days,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, self.settings.log_level))
        
        logging.getLogger().addHandler(file_handler)
    
    def _configure_application_loggers(self) -> None:
        """Configure application-specific loggers."""
        # Main application logger
        app_logger = logging.getLogger("app")
        app_logger.setLevel(getattr(logging, self.settings.log_level))
        
        # Feature-specific loggers
        forecast_logger = logging.getLogger("app.forecast")
        forecast_logger.setLevel(getattr(logging, self.settings.log_level))
        
        alerts_logger = logging.getLogger("app.alerts")
        alerts_logger.setLevel(getattr(logging, self.settings.log_level))
        
        # Infrastructure loggers
        cache_logger = logging.getLogger("cache")
        cache_logger.setLevel(logging.WARNING if self.settings.is_production else logging.INFO)
        
        api_logger = logging.getLogger("external_api")
        api_logger.setLevel(logging.INFO)
        
        model_logger = logging.getLogger("ml_model")
        model_logger.setLevel(logging.INFO)
    
    def _configure_third_party_loggers(self) -> None:
        """Configure third-party library loggers to reduce noise."""
        # HTTP clients
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        
        # FastAPI/Uvicorn
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("fastapi").setLevel(logging.INFO)
        
        # TensorFlow
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("tf").setLevel(logging.ERROR)
        
        # Other libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)


class RequestContextLogger(logging.LoggerAdapter):
    """
    Logger adapter that adds request context to log entries.
    
    Automatically includes request ID, user information, and other
    contextual data in all log entries.
    """
    
    def __init__(self, logger: logging.Logger, request_context: Dict[str, Any]):
        super().__init__(logger, request_context)
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Add context to log message.
        
        Args:
            msg: Log message
            kwargs: Keyword arguments
            
        Returns:
            tuple: Processed message and kwargs
        """
        # Add context to extra fields
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        
        kwargs["extra"].update(self.extra)
        
        return msg, kwargs


# Global logger setup
_logger_instance: Optional[StructuredLogger] = None


def setup_logging(settings: Optional[Settings] = None) -> None:
    """
    Setup application logging.
    
    Args:
        settings: Application settings (optional, will use global if not provided)
    """
    global _logger_instance
    
    if settings is None:
        settings = get_settings()
    
    _logger_instance = StructuredLogger(settings)
    _logger_instance.configure_logging()


def get_structured_logger(name: str) -> logging.Logger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Configured logger
    """
    if _logger_instance is None:
        setup_logging()
    
    return logging.getLogger(name)


def get_request_context_logger(
    name: str,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    **extra_context: Any
) -> RequestContextLogger:
    """
    Get logger with request context.
    
    Args:
        name: Logger name
        request_id: Request ID for tracing
        user_id: User ID for audit trail
        correlation_id: Correlation ID for distributed tracing
        **extra_context: Additional context fields
        
    Returns:
        RequestContextLogger: Logger with context
    """
    logger = get_structured_logger(name)
    
    context = {}
    if request_id:
        context["request_id"] = request_id
    if user_id:
        context["user_id"] = user_id
    if correlation_id:
        context["correlation_id"] = correlation_id
    
    context.update(extra_context)
    
    return RequestContextLogger(logger, context)


# Utility functions for common logging patterns
def log_api_call(
    logger: logging.Logger,
    api_name: str,
    url: str,
    method: str = "GET",
    status_code: Optional[int] = None,
    response_time: Optional[float] = None,
    error: Optional[str] = None
) -> None:
    """
    Log external API call with structured data.
    
    Args:
        logger: Logger instance
        api_name: Name of the API
        url: API URL
        method: HTTP method
        status_code: Response status code
        response_time: Response time in seconds
        error: Error message if any
    """
    log_data = {
        "api_name": api_name,
        "url": url,
        "method": method,
        "status_code": status_code,
        "response_time": response_time,
    }
    
    if error:
        log_data["error"] = error
        logger.error(f"API call failed: {api_name}", extra=log_data)
    else:
        logger.info(f"API call completed: {api_name}", extra=log_data)


def log_model_prediction(
    logger: logging.Logger,
    model_name: str,
    input_features: int,
    prediction_time: float,
    confidence: Optional[float] = None,
    error: Optional[str] = None
) -> None:
    """
    Log ML model prediction with performance metrics.
    
    Args:
        logger: Logger instance
        model_name: Name of the model
        input_features: Number of input features
        prediction_time: Time taken for prediction
        confidence: Prediction confidence score
        error: Error message if any
    """
    log_data = {
        "model_name": model_name,
        "input_features": input_features,
        "prediction_time": prediction_time,
        "confidence": confidence,
    }
    
    if error:
        log_data["error"] = error
        logger.error(f"Model prediction failed: {model_name}", extra=log_data)
    else:
        logger.info(f"Model prediction completed: {model_name}", extra=log_data)


def log_cache_operation(
    logger: logging.Logger,
    operation: str,
    key: str,
    hit: bool = False,
    ttl: Optional[int] = None,
    error: Optional[str] = None
) -> None:
    """
    Log cache operation with performance data.
    
    Args:
        logger: Logger instance
        operation: Cache operation (get, set, delete, etc.)
        key: Cache key
        hit: Whether it was a cache hit
        ttl: Time to live for set operations
        error: Error message if any
    """
    log_data = {
        "operation": operation,
        "key": key,
        "cache_hit": hit,
        "ttl": ttl,
    }
    
    if error:
        log_data["error"] = error
        logger.warning(f"Cache operation failed: {operation}", extra=log_data)
    else:
        logger.debug(f"Cache operation: {operation}", extra=log_data)


# Initialize logging on import in development
settings = get_settings()
if settings.is_development:
    setup_logging(settings) 