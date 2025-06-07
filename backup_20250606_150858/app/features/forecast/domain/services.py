"""
Domain Services - Forecast Feature

Este módulo contém os serviços de domínio que encapsulam lógica de negócio
complexa relacionada à previsão meteorológica.

Services:
- ForecastService: Lógica de negócio para previsões
- WeatherAnalysisService: Análise de dados meteorológicos
- ModelValidationService: Validação de modelos ML
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .entities import (
    Forecast,
    ModelMetrics,
    PrecipitationLevel,
    WeatherCondition,
    WeatherData,
)

logger = logging.getLogger(__name__)


@dataclass
class ForecastConfiguration:
    """Configuração para geração de previsões"""

    sequence_length: int = 24  # Horas de histórico
    forecast_horizon: int = 24  # Horas de previsão à frente
    confidence_threshold: float = 0.7  # Threshold mínimo de confiança
    max_inference_time_ms: float = 100.0  # Tempo máximo de inferência
    features_count: int = 16  # Número de features do modelo


class ForecastService:
    """
    Serviço de domínio para lógica de negócio de previsões meteorológicas

    Responsabilidades:
    - Validar dados de entrada para previsão
    - Aplicar regras de negócio específicas
    - Validar qualidade das previsões
    - Gerar alertas baseados em previsões
    """

    def __init__(self, config: ForecastConfiguration):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def validate_input_sequence(self, weather_data: List[WeatherData]) -> bool:
        """
        Valida sequência de entrada para previsão

        Args:
            weather_data: Lista de dados meteorológicos

        Returns:
            bool: True se válida

        Raises:
            ValueError: Se dados inválidos
        """
        if not weather_data:
            raise ValueError("Sequência de dados não pode estar vazia")

        if len(weather_data) < self.config.sequence_length:
            raise ValueError(
                f"Sequência deve ter pelo menos {self.config.sequence_length} pontos. "
                f"Recebido: {len(weather_data)}"
            )

        # Verificar ordem cronológica
        timestamps = [data.timestamp for data in weather_data]
        if timestamps != sorted(timestamps):
            raise ValueError("Dados devem estar em ordem cronológica")

        # Verificar continuidade temporal (gaps < 2 horas)
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i - 1]
            if gap > timedelta(hours=2):
                self.logger.warning(
                    f"Gap temporal detectado: {gap} entre {timestamps[i-1]} e {timestamps[i]}"
                )

        # Verificar qualidade dos dados
        extreme_count = sum(1 for data in weather_data if data.is_extreme_weather())
        if extreme_count > len(weather_data) * 0.3:  # Mais de 30% extremos
            self.logger.warning(
                f"Alta proporção de condições extremas: {extreme_count}/{len(weather_data)}"
            )

        return True

    def validate_forecast_quality(self, forecast: Forecast) -> bool:
        """
        Valida qualidade da previsão gerada

        Args:
            forecast: Previsão a ser validada

        Returns:
            bool: True se atende aos critérios de qualidade
        """
        # Critério 1: Tempo de inferência
        if not forecast.meets_performance_criteria():
            self.logger.warning(
                f"Previsão não atende critério de performance: "
                f"{forecast.inference_time_ms}ms > {self.config.max_inference_time_ms}ms"
            )
            return False

        # Critério 2: Confiança mínima
        if forecast.confidence_score < self.config.confidence_threshold:
            self.logger.warning(
                f"Previsão com baixa confiança: "
                f"{forecast.confidence_score} < {self.config.confidence_threshold}"
            )
            return False

        # Critério 3: Valores plausíveis
        if forecast.precipitation_mm > 200:  # Limite extremo
            self.logger.warning(
                f"Previsão de precipitação muito alta: {forecast.precipitation_mm}mm/h"
            )
            return False

        return True

    def should_generate_alert(
        self, forecast: Forecast, river_level: Optional[float] = None
    ) -> bool:
        """
        Determina se deve gerar alerta baseado na previsão

        Args:
            forecast: Previsão meteorológica
            river_level: Nível atual do rio (opcional)

        Returns:
            bool: True se deve gerar alerta
        """
        # Precipitação moderada ou superior
        if forecast.get_precipitation_level() in [
            PrecipitationLevel.MODERATE,
            PrecipitationLevel.HEAVY,
            PrecipitationLevel.EXTREME,
        ]:
            return True

        # Se temos nível do rio, considerar conjunto
        if river_level is not None:
            # Precipitação leve + nível alto do rio
            if (
                forecast.get_precipitation_level() == PrecipitationLevel.LIGHT
                and river_level > 2.5
            ):
                return True

        return False

    def calculate_risk_score(
        self, forecast: Forecast, river_level: Optional[float] = None
    ) -> float:
        """
        Calcula score de risco baseado na previsão

        Args:
            forecast: Previsão meteorológica
            river_level: Nível atual do rio (opcional)

        Returns:
            float: Score de risco (0.0 - 1.0)
        """
        risk_score = 0.0

        # Componente: Precipitação
        if forecast.precipitation_mm > 50:
            risk_score += 0.5  # Risco alto
        elif forecast.precipitation_mm > 10:
            risk_score += 0.3  # Risco moderado
        elif forecast.precipitation_mm > 2:
            risk_score += 0.1  # Risco baixo

        # Componente: Confiança (inverso)
        risk_score += (1.0 - forecast.confidence_score) * 0.2

        # Componente: Nível do rio
        if river_level is not None:
            if river_level > 3.6:
                risk_score += 0.3  # Crítico
            elif river_level > 3.15:
                risk_score += 0.2  # Alto
            elif river_level > 2.8:
                risk_score += 0.1  # Moderado

        return min(risk_score, 1.0)  # Máximo 1.0

    def get_forecast_summary(self, forecast: Forecast) -> Dict[str, Any]:
        """
        Gera sumário da previsão para tomada de decisão

        Args:
            forecast: Previsão meteorológica

        Returns:
            Dict: Sumário com informações chave
        """
        return {
            "timestamp": forecast.timestamp.isoformat(),
            "precipitation_level": forecast.get_precipitation_level().value,
            "precipitation_mm": forecast.precipitation_mm,
            "confidence": forecast.confidence_score,
            "rain_expected": forecast.is_rain_expected(),
            "high_confidence": forecast.is_high_confidence(),
            "performance_ok": forecast.meets_performance_criteria(),
            "quality_ok": self.validate_forecast_quality(forecast),
            "should_alert": self.should_generate_alert(forecast),
            "risk_score": self.calculate_risk_score(forecast),
        }


class WeatherAnalysisService:
    """
    Serviço para análise avançada de dados meteorológicos

    Responsabilidades:
    - Detectar padrões nos dados
    - Identificar anomalias
    - Calcular estatísticas meteorológicas
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def detect_patterns(self, weather_data: List[WeatherData]) -> Dict[str, Any]:
        """
        Detecta padrões meteorológicos nos dados

        Args:
            weather_data: Lista de dados meteorológicos

        Returns:
            Dict: Padrões detectados
        """
        if not weather_data:
            return {}

        patterns = {
            "total_precipitation": sum(data.precipitation for data in weather_data),
            "avg_temperature": sum(data.temperature for data in weather_data)
            / len(weather_data),
            "avg_humidity": sum(data.humidity for data in weather_data)
            / len(weather_data),
            "avg_wind_speed": sum(data.wind_speed for data in weather_data)
            / len(weather_data),
            "extreme_events": sum(
                1 for data in weather_data if data.is_extreme_weather()
            ),
            "rain_hours": sum(1 for data in weather_data if data.precipitation >= 0.1),
            "dominant_condition": self._get_dominant_condition(weather_data),
        }

        return patterns

    def _get_dominant_condition(self, weather_data: List[WeatherData]) -> str:
        """Determina condição meteorológica dominante"""
        conditions = [data.get_weather_condition() for data in weather_data]
        condition_counts = {}

        for condition in conditions:
            condition_counts[condition] = condition_counts.get(condition, 0) + 1

        if condition_counts:
            dominant = max(condition_counts, key=condition_counts.get)
            return dominant.value

        return WeatherCondition.UNKNOWN.value

    def detect_anomalies(self, weather_data: List[WeatherData]) -> List[Dict[str, Any]]:
        """
        Detecta anomalias nos dados meteorológicos

        Args:
            weather_data: Lista de dados meteorológicos

        Returns:
            List: Lista de anomalias detectadas
        """
        anomalies = []

        for i, data in enumerate(weather_data):
            anomaly_info = {"index": i, "timestamp": data.timestamp, "anomalies": []}

            # Detectar condições extremas
            if data.is_extreme_weather():
                anomaly_info["anomalies"].append("extreme_weather")

            # Detectar mudanças bruscas (se há dados anteriores)
            if i > 0:
                prev_data = weather_data[i - 1]

                # Mudança brusca de temperatura (>10°C em 1h)
                temp_change = abs(data.temperature - prev_data.temperature)
                if temp_change > 10:
                    anomaly_info["anomalies"].append(f"temp_change_{temp_change:.1f}C")

                # Mudança brusca de pressão (>20mB em 1h)
                pressure_change = abs(data.pressure - prev_data.pressure)
                if pressure_change > 20:
                    anomaly_info["anomalies"].append(
                        f"pressure_change_{pressure_change:.1f}mB"
                    )

            if anomaly_info["anomalies"]:
                anomalies.append(anomaly_info)

        return anomalies

    def calculate_meteorological_indices(
        self, weather_data: List[WeatherData]
    ) -> Dict[str, float]:
        """
        Calcula índices meteorológicos específicos

        Args:
            weather_data: Lista de dados meteorológicos

        Returns:
            Dict: Índices calculados
        """
        if not weather_data:
            return {}

        # Heat Index simplificado
        avg_temp = sum(data.temperature for data in weather_data) / len(weather_data)
        avg_humidity = sum(data.humidity for data in weather_data) / len(weather_data)
        heat_index = avg_temp + (avg_humidity / 100) * 5  # Simplificado

        # Wind Chill simplificado
        avg_wind = sum(data.wind_speed for data in weather_data) / len(weather_data)
        wind_chill = avg_temp - (avg_wind * 0.5)  # Simplificado

        # Precipitation Rate
        total_precip = sum(data.precipitation for data in weather_data)
        precip_rate = total_precip / len(weather_data) if weather_data else 0

        return {
            "heat_index": heat_index,
            "wind_chill": wind_chill,
            "precipitation_rate": precip_rate,
            "pressure_trend": self._calculate_pressure_trend(weather_data),
            "humidity_comfort": self._calculate_humidity_comfort(avg_humidity),
        }

    def _calculate_pressure_trend(self, weather_data: List[WeatherData]) -> str:
        """Calcula tendência da pressão atmosférica"""
        if len(weather_data) < 3:
            return "insufficient_data"

        first_half = weather_data[: len(weather_data) // 2]
        second_half = weather_data[len(weather_data) // 2 :]

        avg_first = sum(data.pressure for data in first_half) / len(first_half)
        avg_second = sum(data.pressure for data in second_half) / len(second_half)

        diff = avg_second - avg_first

        if diff > 2:
            return "rising"
        elif diff < -2:
            return "falling"
        else:
            return "stable"

    def _calculate_humidity_comfort(self, humidity: float) -> str:
        """Calcula conforto baseado na umidade"""
        if humidity < 30:
            return "dry"
        elif humidity < 60:
            return "comfortable"
        elif humidity < 80:
            return "humid"
        else:
            return "very_humid"


class ModelValidationService:
    """
    Serviço para validação de modelos de machine learning

    Responsabilidades:
    - Validar métricas de modelos
    - Comparar versões de modelos
    - Determinar qualidade de modelos
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def validate_model_metrics(self, metrics: ModelMetrics) -> Dict[str, Any]:
        """
        Valida métricas do modelo contra critérios estabelecidos

        Args:
            metrics: Métricas do modelo

        Returns:
            Dict: Resultado da validação
        """
        validation_result = {
            "model_version": metrics.model_version,
            "training_date": metrics.training_date.isoformat(),
            "performance_grade": metrics.get_performance_grade(),
            "meets_all_criteria": metrics.meets_all_criteria(),
            "individual_criteria": {
                "mae": {
                    "value": metrics.mae,
                    "threshold": 2.0,
                    "meets": metrics.meets_mae_criteria(),
                },
                "rmse": {
                    "value": metrics.rmse,
                    "threshold": 3.0,
                    "meets": metrics.meets_rmse_criteria(),
                },
                "accuracy": {
                    "value": metrics.accuracy,
                    "threshold": 0.75,
                    "meets": metrics.meets_accuracy_criteria(),
                },
            },
        }

        return validation_result

    def compare_models(
        self, current_metrics: ModelMetrics, new_metrics: ModelMetrics
    ) -> Dict[str, Any]:
        """
        Compara métricas entre dois modelos

        Args:
            current_metrics: Métricas do modelo atual
            new_metrics: Métricas do novo modelo

        Returns:
            Dict: Comparação detalhada
        """
        comparison = {
            "current_model": current_metrics.model_version,
            "new_model": new_metrics.model_version,
            "improvements": {},
            "overall_better": False,
        }

        # Comparar MAE (menor é melhor)
        mae_improvement = current_metrics.mae - new_metrics.mae
        comparison["improvements"]["mae"] = {
            "current": current_metrics.mae,
            "new": new_metrics.mae,
            "improvement": mae_improvement,
            "better": mae_improvement > 0,
        }

        # Comparar RMSE (menor é melhor)
        rmse_improvement = current_metrics.rmse - new_metrics.rmse
        comparison["improvements"]["rmse"] = {
            "current": current_metrics.rmse,
            "new": new_metrics.rmse,
            "improvement": rmse_improvement,
            "better": rmse_improvement > 0,
        }

        # Comparar Accuracy (maior é melhor)
        accuracy_improvement = new_metrics.accuracy - current_metrics.accuracy
        comparison["improvements"]["accuracy"] = {
            "current": current_metrics.accuracy,
            "new": new_metrics.accuracy,
            "improvement": accuracy_improvement,
            "better": accuracy_improvement > 0,
        }

        # Determinar se o novo modelo é melhor no geral
        improvements_count = sum(
            1 for metric in comparison["improvements"].values() if metric["better"]
        )

        comparison["overall_better"] = improvements_count >= 2
        comparison["improvements_count"] = improvements_count

        return comparison

    def recommend_model_update(
        self, comparison: Dict[str, Any], threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Recomenda se deve atualizar o modelo baseado na comparação

        Args:
            comparison: Resultado da comparação entre modelos
            threshold: Threshold mínimo de melhoria

        Returns:
            Dict: Recomendação de atualização
        """
        recommendation = {"should_update": False, "confidence": "low", "reasons": []}

        improvements = comparison["improvements"]

        # Verificar melhorias significativas
        significant_improvements = 0

        if (
            improvements["mae"]["better"]
            and abs(improvements["mae"]["improvement"]) > threshold
        ):
            significant_improvements += 1
            recommendation["reasons"].append(
                f"MAE improvement: {improvements['mae']['improvement']:.3f}"
            )

        if (
            improvements["rmse"]["better"]
            and abs(improvements["rmse"]["improvement"]) > threshold
        ):
            significant_improvements += 1
            recommendation["reasons"].append(
                f"RMSE improvement: {improvements['rmse']['improvement']:.3f}"
            )

        if (
            improvements["accuracy"]["better"]
            and abs(improvements["accuracy"]["improvement"]) > threshold
        ):
            significant_improvements += 1
            recommendation["reasons"].append(
                f"Accuracy improvement: {improvements['accuracy']['improvement']:.3f}"
            )

        # Decisão de atualização
        if significant_improvements >= 2:
            recommendation["should_update"] = True
            recommendation["confidence"] = "high"
        elif significant_improvements == 1:
            recommendation["should_update"] = True
            recommendation["confidence"] = "medium"

        return recommendation
