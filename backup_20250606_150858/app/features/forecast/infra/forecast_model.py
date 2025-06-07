"""
Forecast Model Infrastructure - LSTM Weather Prediction

Este módulo implementa o wrapper para o modelo LSTM treinado, integrando
preprocessing, inferência e postprocessing para previsões meteorológicas.

Baseado na documentação do projeto:
- Modelo LSTM com precisão > 75% para previsão de chuva 24h
- Features: 16+ variáveis meteorológicas do INMET
- Sequence length: 24 horas de histórico
- Forecast horizon: 24 horas à frente
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from app.core.exceptions import ModelLoadError, PredictionError, ValidationError
from app.features.forecast.domain.entities import Forecast, ModelMetrics, WeatherData

logger = logging.getLogger(__name__)


class WeatherLSTMModel:
    """
    Wrapper para modelo LSTM de previsão meteorológica

    Integra preprocessing, inferência e postprocessing para previsões
    baseadas nos dados históricos do INMET (2000-2025).
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Inicializa o wrapper do modelo LSTM

        Args:
            model_path: Caminho para o modelo treinado (opcional)
        """
        self.model_path = model_path or Path("data/modelos_treinados")
        self.model: Optional[tf.keras.Model] = None
        self.feature_scaler: Optional[StandardScaler] = None
        self.target_scaler: Optional[StandardScaler] = None
        self.metadata: Dict[str, Any] = {}
        self.feature_columns: List[str] = []
        self.target_column: str = ""
        self.sequence_length: int = 24
        self.forecast_horizon: int = 24

        logger.info(f"WeatherLSTMModel inicializado com path: {self.model_path}")

    def load_model(self, model_name: str = "lstm_weather_model") -> bool:
        """
        Carrega modelo LSTM treinado e seus artefatos

        Args:
            model_name: Nome do modelo a ser carregado

        Returns:
            bool: True se carregamento foi bem-sucedido

        Raises:
            ModelLoadError: Se não conseguir carregar o modelo
        """
        try:
            logger.info(f"Carregando modelo: {model_name}")

            # Carregar modelo TensorFlow
            model_dir = self.model_path / model_name
            if not model_dir.exists():
                raise ModelLoadError(f"Modelo não encontrado: {model_dir}")

            self.model = tf.keras.models.load_model(model_dir)
            logger.info(
                f"✓ Modelo TensorFlow carregado: {self.model.count_params():,} parâmetros"
            )

            # Carregar metadados
            metadata_path = self.model_path / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)

                # Extrair configurações dos metadados
                model_config = self.metadata.get("model_config", {})
                self.feature_columns = model_config.get("feature_columns", [])
                self.target_column = model_config.get(
                    "target_column", "precipitacao_mm"
                )
                self.sequence_length = model_config.get("sequence_length", 24)
                self.forecast_horizon = model_config.get("forecast_horizon", 24)

                logger.info(
                    f"✓ Metadados carregados: {len(self.feature_columns)} features"
                )
            else:
                logger.warning("Metadados não encontrados, usando configurações padrão")
                self._set_default_config()

            # Carregar scalers
            self._load_scalers()

            logger.info("✓ Modelo carregado com sucesso")
            return True

        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise ModelLoadError(f"Falha ao carregar modelo {model_name}: {str(e)}")

    def _load_scalers(self):
        """Carrega scalers de normalização"""
        try:
            # Tentar carregar scalers salvos
            feature_scaler_path = self.model_path / "feature_scaler.joblib"
            target_scaler_path = self.model_path / "target_scaler.joblib"

            if feature_scaler_path.exists():
                self.feature_scaler = joblib.load(feature_scaler_path)
                logger.info("✓ Feature scaler carregado")
            else:
                logger.warning("Feature scaler não encontrado, criando novo")
                self.feature_scaler = StandardScaler()

            if target_scaler_path.exists():
                self.target_scaler = joblib.load(target_scaler_path)
                logger.info("✓ Target scaler carregado")
            else:
                logger.warning("Target scaler não encontrado, criando novo")
                self.target_scaler = StandardScaler()

        except Exception as e:
            logger.warning(f"Erro ao carregar scalers: {e}, criando novos")
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()

    def _set_default_config(self):
        """Define configurações padrão baseadas na documentação"""
        self.feature_columns = [
            "precipitacao_mm",
            "pressao_mb",
            "temperatura_c",
            "ponto_orvalho_c",
            "umidade_relativa",
            "velocidade_vento_ms",
            "direcao_vento_gr",
            "radiacao_kjm2",
            "pressao_max_mb",
            "pressao_min_mb",
            "temperatura_max_c",
            "temperatura_min_c",
            "umidade_max",
            "umidade_min",
            "ponto_orvalho_max_c",
            "ponto_orvalho_min_c",
        ]
        self.target_column = "precipitacao_mm"
        self.sequence_length = 24
        self.forecast_horizon = 24

    def preprocess_data(self, weather_data: List[WeatherData]) -> np.ndarray:
        """
        Preprocessa dados meteorológicos para inferência

        Args:
            weather_data: Lista de dados meteorológicos históricos

        Returns:
            np.ndarray: Dados preprocessados para o modelo

        Raises:
            ValidationError: Se dados são inválidos
        """
        try:
            if len(weather_data) < self.sequence_length:
                raise ValidationError(
                    f"Dados insuficientes: {len(weather_data)} < {self.sequence_length}"
                )

            # Converter para DataFrame
            data_dict = []
            for wd in weather_data[
                -self.sequence_length :
            ]:  # Últimas sequence_length horas
                data_dict.append(
                    {
                        "precipitacao_mm": wd.precipitation,
                        "pressao_mb": wd.pressure,
                        "temperatura_c": wd.temperature,
                        "ponto_orvalho_c": wd.dew_point,
                        "umidade_relativa": wd.humidity,
                        "velocidade_vento_ms": wd.wind_speed,
                        "direcao_vento_gr": wd.wind_direction,
                        "radiacao_kjm2": wd.radiation or 0.0,
                        "pressao_max_mb": wd.pressure_max or wd.pressure,
                        "pressao_min_mb": wd.pressure_min or wd.pressure,
                        "temperatura_max_c": wd.temperature_max or wd.temperature,
                        "temperatura_min_c": wd.temperature_min or wd.temperature,
                        "umidade_max": wd.humidity_max or wd.humidity,
                        "umidade_min": wd.humidity_min or wd.humidity,
                        "ponto_orvalho_max_c": wd.dew_point_max or wd.dew_point,
                        "ponto_orvalho_min_c": wd.dew_point_min or wd.dew_point,
                    }
                )

            df = pd.DataFrame(data_dict)

            # Validar colunas necessárias
            missing_cols = set(self.feature_columns) - set(df.columns)
            if missing_cols:
                logger.warning(f"Colunas faltantes: {missing_cols}, preenchendo com 0")
                for col in missing_cols:
                    df[col] = 0.0

            # Extrair features na ordem correta
            features = df[self.feature_columns].values

            # Normalizar se scaler estiver disponível
            if self.feature_scaler and hasattr(self.feature_scaler, "mean_"):
                features = self.feature_scaler.transform(features)

            # Reshape para formato do LSTM: (1, sequence_length, features)
            features = features.reshape(
                1, self.sequence_length, len(self.feature_columns)
            )

            logger.debug(f"Dados preprocessados: shape={features.shape}")
            return features

        except Exception as e:
            logger.error(f"Erro no preprocessing: {e}")
            raise ValidationError(f"Falha no preprocessing: {str(e)}")

    def predict(self, weather_data: List[WeatherData]) -> Forecast:
        """
        Gera previsão meteorológica usando o modelo LSTM

        Args:
            weather_data: Dados meteorológicos históricos (últimas 24h)

        Returns:
            Forecast: Previsão de precipitação para 24h à frente

        Raises:
            PredictionError: Se não conseguir gerar previsão
        """
        try:
            if self.model is None:
                raise PredictionError("Modelo não carregado")

            # Preprocessar dados
            X = self.preprocess_data(weather_data)

            # Inferência
            logger.debug("Executando inferência...")
            start_time = datetime.now()

            prediction = self.model.predict(X, verbose=0)

            inference_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.debug(f"Inferência concluída em {inference_time:.1f}ms")

            # Postprocessar resultado
            predicted_value = float(prediction[0, 0])

            # Desnormalizar se scaler estiver disponível
            if self.target_scaler and hasattr(self.target_scaler, "mean_"):
                predicted_value = self.target_scaler.inverse_transform(
                    [[predicted_value]]
                )[0, 0]

            # Garantir que precipitação não seja negativa
            predicted_value = max(0.0, predicted_value)

            # Criar objeto Forecast
            forecast_time = weather_data[-1].timestamp + timedelta(
                hours=self.forecast_horizon
            )

            forecast = Forecast(
                timestamp=forecast_time,
                precipitation_mm=predicted_value,
                confidence_score=self._calculate_confidence(predicted_value),
                model_version=self.metadata.get("training_date", "unknown"),
                inference_time_ms=inference_time,
            )

            logger.info(
                f"Previsão gerada: {predicted_value:.2f}mm em {inference_time:.1f}ms"
            )
            return forecast

        except Exception as e:
            logger.error(f"Erro na previsão: {e}")
            raise PredictionError(f"Falha na previsão: {str(e)}")

    def _calculate_confidence(self, predicted_value: float) -> float:
        """
        Calcula score de confiança baseado no valor previsto

        Args:
            predicted_value: Valor de precipitação previsto

        Returns:
            float: Score de confiança (0.0 a 1.0)
        """
        # Lógica simples baseada em ranges típicos de precipitação
        if predicted_value < 0.1:  # Sem chuva
            return 0.9
        elif predicted_value < 2.0:  # Chuva leve
            return 0.8
        elif predicted_value < 10.0:  # Chuva moderada
            return 0.7
        elif predicted_value < 50.0:  # Chuva forte
            return 0.6
        else:  # Chuva muito forte
            return 0.5

    def get_model_metrics(self) -> ModelMetrics:
        """
        Retorna métricas do modelo carregado

        Returns:
            ModelMetrics: Métricas de performance do modelo
        """
        if not self.metadata:
            return ModelMetrics(
                mae=0.0,
                rmse=0.0,
                accuracy=0.0,
                model_version="unknown",
                training_date=datetime.now(),
            )

        model_metrics = self.metadata.get("model_metrics", {})

        return ModelMetrics(
            mae=model_metrics.get("test_mae", 0.0),
            rmse=model_metrics.get("test_rmse", 0.0),
            accuracy=model_metrics.get("test_accuracy", 0.0),
            model_version=self.metadata.get("training_date", "unknown"),
            training_date=datetime.fromisoformat(
                self.metadata.get("training_date", datetime.now().isoformat())
            ),
        )

    def validate_model_performance(self) -> bool:
        """
        Valida se o modelo atende aos critérios de performance

        Returns:
            bool: True se modelo atende aos critérios
        """
        metrics = self.get_model_metrics()

        # Critérios baseados na documentação
        mae_ok = metrics.mae < 2.0  # MAE < 2.0 mm/h
        rmse_ok = metrics.rmse < 3.0  # RMSE < 3.0 mm/h
        accuracy_ok = metrics.accuracy > 0.75  # Accuracy > 75%

        logger.info(f"Validação do modelo:")
        logger.info(f"  MAE: {metrics.mae:.4f} < 2.0: {'✓' if mae_ok else '✗'}")
        logger.info(f"  RMSE: {metrics.rmse:.4f} < 3.0: {'✓' if rmse_ok else '✗'}")
        logger.info(
            f"  Accuracy: {metrics.accuracy:.4f} > 0.75: {'✓' if accuracy_ok else '✗'}"
        )

        return mae_ok and rmse_ok and accuracy_ok

    def is_loaded(self) -> bool:
        """Verifica se o modelo está carregado"""
        return self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo carregado

        Returns:
            Dict: Informações do modelo
        """
        if not self.is_loaded():
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "model_path": str(self.model_path),
            "parameters": self.model.count_params(),
            "sequence_length": self.sequence_length,
            "forecast_horizon": self.forecast_horizon,
            "features_count": len(self.feature_columns),
            "target_column": self.target_column,
            "tensorflow_version": tf.__version__,
            "metadata": self.metadata,
        }
