"""
Core configuration module using Pydantic Settings.

This module centralizes all application configuration including:
- Environment variables
- API settings
- Database configurations
- Logging levels
- External API configurations
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings using Pydantic Settings."""

    # Application Settings
    app_name: str = "Sistema de Alertas de Cheias - Rio Guaíba"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True

    # API Settings
    api_v1_prefix: str = "/api/v1"
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ]

    # Security Settings
    secret_key: str = "your-secret-key-change-in-production"
    access_token_expire_minutes: int = 30
    algorithm: str = "HS256"

    # Rate Limiting
    rate_limit_per_minute: int = 1000
    rate_limit_burst: int = 100

    # External APIs Configuration
    guaiba_api_url: str = "https://nivelguaiba.com.br/portoalegre.1day.json"
    cptec_api_url: str = (
        "https://www.cptec.inpe.br/api/forecast-input?city=Porto%20Alegre%2C%20RS"
    )

    # API Client Settings
    api_timeout: int = 10
    max_retries: int = 3
    backoff_factor: float = 2.0

    # Cache Settings
    cache_ttl_seconds: int = 300  # 5 minutes
    cache_max_size: int = 1000

    # Redis Settings (optional)
    redis_url: Optional[str] = None
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    # Database Settings (optional for future use)
    database_url: Optional[str] = None
    db_echo: bool = False

    # ML Model Settings
    model_path: str = "data/modelos_treinados"
    model_version: str = "v1.0"
    sequence_length: int = 24
    features_count: int = 8

    # Model Parameters
    lstm_units: List[int] = [128, 64, 32]
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100

    # Model Performance Thresholds
    min_accuracy: float = 0.75
    max_mae: float = 2.0
    max_rmse: float = 3.0
    max_inference_time_ms: int = 100

    # Alert Thresholds
    critical_river_level: float = 3.60
    high_river_level: float = 3.15
    moderate_river_level: float = 2.80
    high_rain_threshold: float = 50.0
    moderate_rain_threshold: float = 20.0

    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "JSON"  # JSON or TEXT
    log_file: Optional[str] = None
    log_rotation_size: str = "10 MB"
    log_retention_days: int = 30

    # Monitoring Settings
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_timeout: int = 5

    # Data Processing
    data_raw_path: str = "data/raw"
    data_processed_path: str = "data/processed"
    historical_data_path: str = "data/raw/dados_historicos"

    # Background Tasks
    forecast_update_interval_minutes: int = 60
    model_retrain_interval_hours: int = 24
    cache_cleanup_interval_minutes: int = 30

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment values."""
        allowed = ["development", "testing", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level values."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of: {allowed}")
        return v.upper()

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v):
        """Validate log format values."""
        allowed = ["JSON", "TEXT"]
        if v.upper() not in allowed:
            raise ValueError(f"Log format must be one of: {allowed}")
        return v.upper()

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == "testing"

    @property
    def model_full_path(self) -> str:
        """Get full path to the ML model."""
        return os.path.join(self.model_path, f"model_{self.model_version}")

    @property
    def api_base_url(self) -> str:
        """Get the base API URL."""
        protocol = "https" if self.is_production else "http"
        return f"{protocol}://{self.host}:{self.port}{self.api_v1_prefix}"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are loaded once and reused.
    This is important for performance and consistency.

    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Global settings instance
settings = get_settings()
