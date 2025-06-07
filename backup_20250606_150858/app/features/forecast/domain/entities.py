"""
Domain Entities - Forecast Feature

Este módulo define as entidades de domínio para previsão meteorológica,
representando os conceitos centrais do negócio sem dependências externas.

Entidades:
- WeatherData: Dados meteorológicos históricos
- Forecast: Resultado de previsão meteorológica
- ModelMetrics: Métricas de performance do modelo
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class WeatherCondition(Enum):
    """Condições meteorológicas categorizadas"""

    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    STORMY = "stormy"
    UNKNOWN = "unknown"


class PrecipitationLevel(Enum):
    """Níveis de precipitação baseados na documentação"""

    NONE = "none"  # < 0.1 mm/h
    LIGHT = "light"  # 0.1 - 2.0 mm/h
    MODERATE = "moderate"  # 2.0 - 10.0 mm/h
    HEAVY = "heavy"  # 10.0 - 50.0 mm/h
    EXTREME = "extreme"  # > 50.0 mm/h


@dataclass
class WeatherData:
    """
    Entidade representando dados meteorológicos históricos

    Baseada nas variáveis do INMET (2000-2025):
    - 16+ features meteorológicas
    - Dados horários de Porto Alegre
    - Validação de ranges baseada na documentação
    """

    timestamp: datetime
    precipitation: float  # mm/h
    pressure: float  # mB
    temperature: float  # °C
    dew_point: float  # °C
    humidity: float  # %
    wind_speed: float  # m/s
    wind_direction: float  # graus
    radiation: Optional[float] = None  # Kj/m²

    # Valores máximos/mínimos (para feature engineering)
    pressure_max: Optional[float] = None
    pressure_min: Optional[float] = None
    temperature_max: Optional[float] = None
    temperature_min: Optional[float] = None
    humidity_max: Optional[float] = None
    humidity_min: Optional[float] = None
    dew_point_max: Optional[float] = None
    dew_point_min: Optional[float] = None

    # Metadados
    station_id: Optional[str] = None
    quality_flag: Optional[str] = None

    def __post_init__(self):
        """Validação dos dados após inicialização"""
        self._validate_ranges()

    def _validate_ranges(self):
        """
        Valida se os valores estão dentro dos ranges esperados
        baseados na documentação do projeto
        """
        # Ranges válidos baseados na documentação
        valid_ranges = {
            "precipitation": (0, 200),  # mm/h
            "temperature": (-10, 50),  # °C
            "humidity": (0, 100),  # %
            "pressure": (900, 1100),  # mB
            "wind_speed": (0, 50),  # m/s
        }

        # Validar precipitação
        if not (0 <= self.precipitation <= valid_ranges["precipitation"][1]):
            raise ValueError(f"Precipitação fora do range válido: {self.precipitation}")

        # Validar temperatura
        if not (
            valid_ranges["temperature"][0]
            <= self.temperature
            <= valid_ranges["temperature"][1]
        ):
            raise ValueError(f"Temperatura fora do range válido: {self.temperature}")

        # Validar umidade
        if not (0 <= self.humidity <= 100):
            raise ValueError(f"Umidade fora do range válido: {self.humidity}")

        # Validar pressão
        if not (
            valid_ranges["pressure"][0] <= self.pressure <= valid_ranges["pressure"][1]
        ):
            raise ValueError(f"Pressão fora do range válido: {self.pressure}")

        # Validar velocidade do vento
        if not (0 <= self.wind_speed <= valid_ranges["wind_speed"][1]):
            raise ValueError(
                f"Velocidade do vento fora do range válido: {self.wind_speed}"
            )

        # Validar direção do vento
        if not (0 <= self.wind_direction <= 360):
            raise ValueError(
                f"Direção do vento fora do range válido: {self.wind_direction}"
            )

    def get_precipitation_level(self) -> PrecipitationLevel:
        """
        Classifica o nível de precipitação

        Returns:
            PrecipitationLevel: Nível categorizado
        """
        if self.precipitation < 0.1:
            return PrecipitationLevel.NONE
        elif self.precipitation < 2.0:
            return PrecipitationLevel.LIGHT
        elif self.precipitation < 10.0:
            return PrecipitationLevel.MODERATE
        elif self.precipitation < 50.0:
            return PrecipitationLevel.HEAVY
        else:
            return PrecipitationLevel.EXTREME

    def get_weather_condition(self) -> WeatherCondition:
        """
        Determina condição meteorológica geral

        Returns:
            WeatherCondition: Condição categorizada
        """
        if self.precipitation > 10.0:
            return WeatherCondition.STORMY
        elif self.precipitation > 0.1:
            return WeatherCondition.RAINY
        elif self.humidity > 80:
            return WeatherCondition.CLOUDY
        else:
            return WeatherCondition.CLEAR

    def is_extreme_weather(self) -> bool:
        """
        Verifica se representa condições meteorológicas extremas

        Returns:
            bool: True se condições são extremas
        """
        return (
            self.precipitation > 50.0
            or self.temperature > 40.0
            or self.temperature < 0.0
            or self.wind_speed > 30.0
            or self.humidity < 10.0
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Converte para dicionário

        Returns:
            Dict: Representação em dicionário
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "precipitation": self.precipitation,
            "pressure": self.pressure,
            "temperature": self.temperature,
            "dew_point": self.dew_point,
            "humidity": self.humidity,
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
            "radiation": self.radiation,
            "pressure_max": self.pressure_max,
            "pressure_min": self.pressure_min,
            "temperature_max": self.temperature_max,
            "temperature_min": self.temperature_min,
            "humidity_max": self.humidity_max,
            "humidity_min": self.humidity_min,
            "dew_point_max": self.dew_point_max,
            "dew_point_min": self.dew_point_min,
            "station_id": self.station_id,
            "quality_flag": self.quality_flag,
            "precipitation_level": self.get_precipitation_level().value,
            "weather_condition": self.get_weather_condition().value,
            "is_extreme": self.is_extreme_weather(),
        }


@dataclass
class Forecast:
    """
    Entidade representando uma previsão meteorológica

    Resultado da inferência do modelo LSTM para previsão de precipitação
    24h à frente baseada em 24h de histórico.
    """

    timestamp: datetime  # Momento da previsão
    precipitation_mm: float  # Precipitação prevista (mm/h)
    confidence_score: float  # Score de confiança (0.0 - 1.0)
    model_version: str  # Versão do modelo utilizado
    inference_time_ms: float  # Tempo de inferência em ms

    # Metadados opcionais
    input_sequence_length: Optional[int] = None
    forecast_horizon_hours: Optional[int] = None
    features_used: Optional[int] = None

    def __post_init__(self):
        """Validação após inicialização"""
        self._validate_forecast()

    def _validate_forecast(self):
        """Valida os valores da previsão"""
        if self.precipitation_mm < 0:
            raise ValueError("Precipitação não pode ser negativa")

        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError("Score de confiança deve estar entre 0.0 e 1.0")

        if self.inference_time_ms < 0:
            raise ValueError("Tempo de inferência não pode ser negativo")

    def get_precipitation_level(self) -> PrecipitationLevel:
        """
        Classifica o nível de precipitação previsto

        Returns:
            PrecipitationLevel: Nível categorizado
        """
        if self.precipitation_mm < 0.1:
            return PrecipitationLevel.NONE
        elif self.precipitation_mm < 2.0:
            return PrecipitationLevel.LIGHT
        elif self.precipitation_mm < 10.0:
            return PrecipitationLevel.MODERATE
        elif self.precipitation_mm < 50.0:
            return PrecipitationLevel.HEAVY
        else:
            return PrecipitationLevel.EXTREME

    def is_rain_expected(self) -> bool:
        """
        Verifica se chuva é esperada

        Returns:
            bool: True se precipitação > 0.1 mm/h
        """
        return self.precipitation_mm >= 0.1

    def is_high_confidence(self) -> bool:
        """
        Verifica se a previsão tem alta confiança

        Returns:
            bool: True se confidence_score > 0.8
        """
        return self.confidence_score > 0.8

    def meets_performance_criteria(self) -> bool:
        """
        Verifica se atende aos critérios de performance
        baseados na documentação (tempo de inferência < 100ms)

        Returns:
            bool: True se atende aos critérios
        """
        return self.inference_time_ms < 100.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Converte para dicionário

        Returns:
            Dict: Representação em dicionário
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "precipitation_mm": self.precipitation_mm,
            "confidence_score": self.confidence_score,
            "model_version": self.model_version,
            "inference_time_ms": self.inference_time_ms,
            "input_sequence_length": self.input_sequence_length,
            "forecast_horizon_hours": self.forecast_horizon_hours,
            "features_used": self.features_used,
            "precipitation_level": self.get_precipitation_level().value,
            "is_rain_expected": self.is_rain_expected(),
            "is_high_confidence": self.is_high_confidence(),
            "meets_performance_criteria": self.meets_performance_criteria(),
        }


@dataclass
class ModelMetrics:
    """
    Entidade representando métricas de performance do modelo LSTM

    Baseada nos critérios de sucesso da documentação:
    - MAE < 2.0 mm/h para precipitação
    - RMSE < 3.0 mm/h para precipitação
    - Accuracy > 75% em classificação de eventos
    """

    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    accuracy: float  # Accuracy para classificação de eventos
    model_version: str  # Versão do modelo
    training_date: datetime  # Data do treinamento

    # Métricas adicionais opcionais
    r2_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

    # Métricas específicas para meteorologia
    skill_score: Optional[float] = None  # Skill Score para eventos de chuva
    bias: Optional[float] = None  # Bias do modelo

    # Informações do dataset
    train_samples: Optional[int] = None
    validation_samples: Optional[int] = None
    test_samples: Optional[int] = None

    def __post_init__(self):
        """Validação após inicialização"""
        self._validate_metrics()

    def _validate_metrics(self):
        """Valida as métricas"""
        if self.mae < 0:
            raise ValueError("MAE não pode ser negativo")

        if self.rmse < 0:
            raise ValueError("RMSE não pode ser negativo")

        if not (0.0 <= self.accuracy <= 1.0):
            raise ValueError("Accuracy deve estar entre 0.0 e 1.0")

    def meets_mae_criteria(self) -> bool:
        """
        Verifica se MAE atende ao critério (< 2.0 mm/h)

        Returns:
            bool: True se atende ao critério
        """
        return self.mae < 2.0

    def meets_rmse_criteria(self) -> bool:
        """
        Verifica se RMSE atende ao critério (< 3.0 mm/h)

        Returns:
            bool: True se atende ao critério
        """
        return self.rmse < 3.0

    def meets_accuracy_criteria(self) -> bool:
        """
        Verifica se accuracy atende ao critério (> 75%)

        Returns:
            bool: True se atende ao critério
        """
        return self.accuracy > 0.75

    def meets_all_criteria(self) -> bool:
        """
        Verifica se todas as métricas atendem aos critérios

        Returns:
            bool: True se todas as métricas atendem
        """
        return (
            self.meets_mae_criteria()
            and self.meets_rmse_criteria()
            and self.meets_accuracy_criteria()
        )

    def get_performance_grade(self) -> str:
        """
        Retorna nota de performance baseada nas métricas

        Returns:
            str: Nota (A, B, C, D, F)
        """
        criteria_met = sum(
            [
                self.meets_mae_criteria(),
                self.meets_rmse_criteria(),
                self.meets_accuracy_criteria(),
            ]
        )

        if criteria_met == 3:
            return "A"  # Excelente
        elif criteria_met == 2:
            return "B"  # Bom
        elif criteria_met == 1:
            return "C"  # Regular
        else:
            return "F"  # Insuficiente

    def to_dict(self) -> Dict[str, Any]:
        """
        Converte para dicionário

        Returns:
            Dict: Representação em dicionário
        """
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "accuracy": self.accuracy,
            "model_version": self.model_version,
            "training_date": self.training_date.isoformat(),
            "r2_score": self.r2_score,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "skill_score": self.skill_score,
            "bias": self.bias,
            "train_samples": self.train_samples,
            "validation_samples": self.validation_samples,
            "test_samples": self.test_samples,
            "meets_mae_criteria": self.meets_mae_criteria(),
            "meets_rmse_criteria": self.meets_rmse_criteria(),
            "meets_accuracy_criteria": self.meets_accuracy_criteria(),
            "meets_all_criteria": self.meets_all_criteria(),
            "performance_grade": self.get_performance_grade(),
        }
