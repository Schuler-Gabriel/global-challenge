"""
Custom exceptions for the Flood Alert System.

This module defines all custom exceptions used throughout the application,
providing clear error handling and appropriate HTTP status codes.
"""

from typing import Any, Dict, Optional, Union
import traceback
from fastapi import HTTPException, status


class BaseApplicationException(Exception):
    """
    Base exception class for all application exceptions.
    
    Provides common functionality for all custom exceptions including
    error codes, messages, and context information.
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "GENERIC_ERROR",
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.original_exception = original_exception
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and API responses."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "type": self.__class__.__name__
        }


# Configuration Exceptions
class ConfigurationError(BaseApplicationException):
    """Raised when there are configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        context = {"config_key": config_key} if config_key else {}
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            context=context
        )


class EnvironmentVariableError(ConfigurationError):
    """Raised when required environment variables are missing."""
    
    def __init__(self, variable_name: str):
        super().__init__(
            message=f"Required environment variable '{variable_name}' is not set",
            config_key=variable_name
        )


# Model and ML Exceptions
class ModelError(BaseApplicationException):
    """Base class for ML model-related errors."""
    pass

# Alias for compatibility
ModelException = ModelError


class ModelNotFoundError(ModelError):
    """Raised when ML model files are not found."""
    
    def __init__(self, model_path: str):
        super().__init__(
            message=f"ML model not found at path: {model_path}",
            error_code="MODEL_NOT_FOUND",
            context={"model_path": model_path}
        )


class ModelLoadError(ModelError):
    """Raised when ML model fails to load."""
    
    def __init__(self, model_path: str, original_exception: Optional[Exception] = None):
        super().__init__(
            message=f"Failed to load ML model from: {model_path}",
            error_code="MODEL_LOAD_ERROR",
            context={"model_path": model_path},
            original_exception=original_exception
        )


class ModelPredictionError(ModelError):
    """Raised when ML model prediction fails."""
    
    def __init__(self, message: str = "Model prediction failed", context: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="MODEL_PREDICTION_ERROR",
            context=context or {}
        )


class InvalidInputDataError(ModelError):
    """Raised when input data for model is invalid."""
    
    def __init__(self, message: str, input_data: Optional[Any] = None):
        context = {"input_data": str(input_data)} if input_data else {}
        super().__init__(
            message=message,
            error_code="INVALID_INPUT_DATA",
            context=context
        )


# External API Exceptions
class ExternalAPIError(BaseApplicationException):
    """Base class for external API-related errors."""
    pass

# Alias for compatibility
ExternalApiException = ExternalAPIError

class APITimeoutError(ExternalAPIError):
    """Raised when external API calls timeout."""
    
    def __init__(self, api_name: str, timeout_seconds: int):
        super().__init__(
            message=f"API '{api_name}' request timed out after {timeout_seconds} seconds",
            error_code="API_TIMEOUT",
            context={"api_name": api_name, "timeout": timeout_seconds}
        )


class APIConnectionError(ExternalAPIError):
    """Raised when connection to external API fails."""
    
    def __init__(self, api_name: str, url: str, original_exception: Optional[Exception] = None):
        super().__init__(
            message=f"Failed to connect to API '{api_name}' at {url}",
            error_code="API_CONNECTION_ERROR",
            context={"api_name": api_name, "url": url},
            original_exception=original_exception
        )


class APIResponseError(ExternalAPIError):
    """Raised when external API returns invalid response."""
    
    def __init__(self, api_name: str, status_code: int, response_text: str = ""):
        super().__init__(
            message=f"API '{api_name}' returned error status {status_code}",
            error_code="API_RESPONSE_ERROR",
            context={
                "api_name": api_name,
                "status_code": status_code,
                "response": response_text[:500]  # Limit response text
            }
        )


class APIDataParsingError(ExternalAPIError):
    """Raised when external API response cannot be parsed."""
    
    def __init__(self, api_name: str, message: str = "Failed to parse API response"):
        super().__init__(
            message=f"Failed to parse response from API '{api_name}': {message}",
            error_code="API_PARSING_ERROR",
            context={"api_name": api_name}
        )


# Data Processing Exceptions
class DataProcessingError(BaseApplicationException):
    """Base class for data processing errors."""
    pass


class DataValidationError(DataProcessingError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, value: Optional[Any] = None):
        context = {}
        if field_name:
            context["field_name"] = field_name
        if value is not None:
            context["value"] = str(value)
        
        super().__init__(
            message=message,
            error_code="DATA_VALIDATION_ERROR",
            context=context
        )

# Alias for compatibility
DataValidationException = DataValidationError


class DataNotFoundError(DataProcessingError):
    """Raised when required data is not found."""
    
    def __init__(self, data_type: str, identifier: Optional[str] = None):
        message = f"{data_type} not found"
        if identifier:
            message += f" (ID: {identifier})"
        
        super().__init__(
            message=message,
            error_code="DATA_NOT_FOUND",
            context={"data_type": data_type, "identifier": identifier}
        )


class DataFileError(DataProcessingError):
    """Raised when there are issues with data files."""
    
    def __init__(self, file_path: str, operation: str, original_exception: Optional[Exception] = None):
        super().__init__(
            message=f"Failed to {operation} data file: {file_path}",
            error_code="DATA_FILE_ERROR",
            context={"file_path": file_path, "operation": operation},
            original_exception=original_exception
        )


# Alert System Exceptions
class AlertSystemError(BaseApplicationException):
    """Base class for alert system errors."""
    pass


class AlertGenerationError(AlertSystemError):
    """Raised when alert generation fails."""
    
    def __init__(self, message: str = "Failed to generate alert", context: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="ALERT_GENERATION_ERROR",
            context=context or {}
        )


class InvalidAlertLevelError(AlertSystemError):
    """Raised when invalid alert level is specified."""
    
    def __init__(self, level: str, valid_levels: Optional[list] = None):
        context = {"invalid_level": level}
        if valid_levels:
            context["valid_levels"] = valid_levels
        
        super().__init__(
            message=f"Invalid alert level: {level}",
            error_code="INVALID_ALERT_LEVEL",
            context=context
        )


# Cache and Storage Exceptions
class CacheError(BaseApplicationException):
    """Base class for cache-related errors."""
    pass


class CacheConnectionError(CacheError):
    """Raised when cache connection fails."""
    
    def __init__(self, cache_type: str = "cache", original_exception: Optional[Exception] = None):
        super().__init__(
            message=f"Failed to connect to {cache_type}",
            error_code="CACHE_CONNECTION_ERROR",
            context={"cache_type": cache_type},
            original_exception=original_exception
        )


class CacheOperationError(CacheError):
    """Raised when cache operations fail."""
    
    def __init__(self, operation: str, key: Optional[str] = None):
        context = {"operation": operation}
        if key:
            context["key"] = key
        
        super().__init__(
            message=f"Cache {operation} operation failed",
            error_code="CACHE_OPERATION_ERROR",
            context=context
        )


# HTTP Exceptions for FastAPI
class HTTPExceptionWithLogging(HTTPException):
    """
    Extended HTTPException that includes logging context.
    
    This exception automatically logs errors when raised,
    providing better debugging and monitoring capabilities.
    """
    
    def __init__(
        self,
        status_code: int,
        detail: Union[str, Dict[str, Any]],
        headers: Optional[Dict[str, str]] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code
        self.context = context or {}


# Utility functions for exception handling
def create_http_exception_from_app_exception(
    app_exception: BaseApplicationException,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
) -> HTTPExceptionWithLogging:
    """
    Convert application exception to HTTP exception.
    
    Args:
        app_exception: Application exception to convert
        status_code: HTTP status code to use
        
    Returns:
        HTTPExceptionWithLogging: HTTP exception for FastAPI
    """
    detail = {
        "error_code": app_exception.error_code,
        "message": app_exception.message,
        "context": app_exception.context
    }
    
    return HTTPExceptionWithLogging(
        status_code=status_code,
        detail=detail,
        error_code=app_exception.error_code,
        context=app_exception.context
    )


def get_error_details(exception: Exception) -> Dict[str, Any]:
    """
    Extract error details from any exception.
    
    Args:
        exception: Exception to analyze
        
    Returns:
        Dict with error details including traceback
    """
    error_details = {
        "type": type(exception).__name__,
        "message": str(exception),
        "traceback": traceback.format_exc()
    }
    
    if isinstance(exception, BaseApplicationException):
        error_details.update({
            "error_code": exception.error_code,
            "context": exception.context
        })
        
        if exception.original_exception:
            error_details["original_exception"] = {
                "type": type(exception.original_exception).__name__,
                "message": str(exception.original_exception)
            }
    
    return error_details


# Common HTTP status code mappings
ERROR_STATUS_MAPPING = {
    DataNotFoundError: status.HTTP_404_NOT_FOUND,
    DataValidationError: status.HTTP_422_UNPROCESSABLE_ENTITY,
    InvalidInputDataError: status.HTTP_400_BAD_REQUEST,
    InvalidAlertLevelError: status.HTTP_400_BAD_REQUEST,
    ModelNotFoundError: status.HTTP_503_SERVICE_UNAVAILABLE,
    ModelLoadError: status.HTTP_503_SERVICE_UNAVAILABLE,
    APITimeoutError: status.HTTP_503_SERVICE_UNAVAILABLE,
    APIConnectionError: status.HTTP_503_SERVICE_UNAVAILABLE,
    CacheConnectionError: status.HTTP_503_SERVICE_UNAVAILABLE,
    ConfigurationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
} 