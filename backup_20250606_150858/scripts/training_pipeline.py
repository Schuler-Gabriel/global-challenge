#!/usr/bin/env python3
"""
Pipeline de Treinamento e Validação Avançado - Fase 3.2
Sistema de Alertas de Cheias - Rio Guaíba

Este módulo implementa um pipeline completo de treinamento e validação temporal
para modelos LSTM de previsão meteorológica, incluindo:

- Cross-validation temporal (walk-forward validation)
- Otimização de hiperparâmetros com grid search
- Métricas específicas para meteorologia
- Preservação de ordem cronológica
- Target: Accuracy > 75% em previsão de chuva 24h

Baseado na documentação do projeto - Fase 3.2:
- Pipeline de treinamento completo
- Validation split temporal (não aleatório)
- Walk-forward validation
- Métricas meteorológicas (MAE, RMSE, Skill Score)
- Otimização sistemática de hiperparâmetros

Uso:
    python scripts/training_pipeline.py --mode full-pipeline
    python scripts/training_pipeline.py --mode temporal-cv
    python scripts/training_pipeline.py --mode hyperopt --max-trials 20
"""

import argparse
import json
import logging
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# TensorFlow/Keras
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    roc_auc_score,
)

# Scikit-learn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

# Configurações
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
np.random.seed(42)
tf.random.set_seed(42)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training_pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
MODELS_PATH = DATA_PATH / "modelos_treinados"
CONFIGS_PATH = PROJECT_ROOT / "configs"

# Criar diretórios se não existem
MODELS_PATH.mkdir(parents=True, exist_ok=True)
(MODELS_PATH / "temporal_validation").mkdir(exist_ok=True)
(MODELS_PATH / "hyperopt").mkdir(exist_ok=True)


# ========================================
# CONFIGURAÇÕES ESPECÍFICAS FASE 3.2
# ========================================

# Configurações de hiperparâmetros para grid search
HYPERPARAMETER_GRID = {
    "learning_rate": [0.001, 0.0001, 0.00001],
    "batch_size": [16, 32, 64, 128],
    "sequence_length": [12, 24, 48, 72],  # horas
    "lstm_units": [[64], [128], [64, 32], [128, 64], [256, 128, 64]],
    "dropout_rate": [0.1, 0.2, 0.3],
}

# Configurações de validação temporal
TEMPORAL_VALIDATION_CONFIG = {
    "min_train_months": 12,  # Mínimo 12 meses para treino
    "validation_months": 3,  # 3 meses para validação
    "step_months": 1,  # Avançar 1 mês por fold
    "max_folds": 10,  # Máximo 10 folds
}

# Thresholds para classificação de eventos de chuva
RAIN_THRESHOLDS = {
    "light": 0.1,  # mm/h - chuva leve
    "moderate": 2.5,  # mm/h - chuva moderada
    "heavy": 10.0,  # mm/h - chuva forte
    "very_heavy": 50.0,  # mm/h - chuva muito forte
}

# Features meteorológicas específicas
METEOROLOGICAL_FEATURES = [
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


# ========================================
# CLASSES PRINCIPAIS
# ========================================


class TemporalDataSplitter:
    """
    Implementa divisão temporal para séries temporais meteorológicas
    Preserva ordem cronológica e evita data leakage
    """

    def __init__(self, config: Dict[str, Any]):
        self.min_train_months = config["min_train_months"]
        self.validation_months = config["validation_months"]
        self.step_months = config["step_months"]
        self.max_folds = config["max_folds"]

    def create_temporal_splits(
        self, data: pd.DataFrame
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Cria splits temporais para walk-forward validation

        Args:
            data: DataFrame com coluna 'timestamp' ordenada

        Yields:
            Tuple com (train_data, validation_data)
        """
        if "timestamp" not in data.columns:
            raise ValueError("Data must have 'timestamp' column")

        # Garantir que dados estão ordenados
        data = data.sort_values("timestamp")

        # Converter timestamp para datetime se necessário
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])

        start_date = data["timestamp"].min()
        end_date = data["timestamp"].max()

        logger.info(f"Período total dos dados: {start_date} até {end_date}")

        current_date = start_date + pd.DateOffset(months=self.min_train_months)
        fold_count = 0

        while (
            current_date + pd.DateOffset(months=self.validation_months) <= end_date
            and fold_count < self.max_folds
        ):
            # Definir períodos
            train_end = current_date
            val_start = current_date
            val_end = current_date + pd.DateOffset(months=self.validation_months)

            # Criar máscaras
            train_mask = data["timestamp"] < train_end
            val_mask = (data["timestamp"] >= val_start) & (data["timestamp"] < val_end)

            train_split = data[train_mask].copy()
            val_split = data[val_mask].copy()

            if len(train_split) > 0 and len(val_split) > 0:
                logger.info(
                    f"Fold {fold_count + 1}: Train até {train_end.strftime('%Y-%m-%d')}, "
                    f"Val {val_start.strftime('%Y-%m-%d')} - {val_end.strftime('%Y-%m-%d')}"
                )
                logger.info(
                    f"  Train samples: {len(train_split)}, Val samples: {len(val_split)}"
                )

                yield train_split, val_split
                fold_count += 1

            # Avançar para próximo fold
            current_date += pd.DateOffset(months=self.step_months)

        logger.info(f"Total de folds gerados: {fold_count}")


class MeteorologicalMetrics:
    """
    Implementa métricas específicas para avaliação meteorológica
    """

    @staticmethod
    def calculate_mae_by_intensity(
        y_true: np.ndarray, y_pred: np.ndarray, thresholds: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calcula MAE estratificado por intensidade de chuva
        """
        results = {}

        for intensity, threshold in thresholds.items():
            if intensity == "light":
                mask = (y_true >= 0) & (y_true < thresholds["moderate"])
            elif intensity == "moderate":
                mask = (y_true >= thresholds["moderate"]) & (
                    y_true < thresholds["heavy"]
                )
            elif intensity == "heavy":
                mask = (y_true >= thresholds["heavy"]) & (
                    y_true < thresholds["very_heavy"]
                )
            elif intensity == "very_heavy":
                mask = y_true >= thresholds["very_heavy"]
            else:
                continue

            if np.sum(mask) > 0:
                mae = mean_absolute_error(y_true[mask], y_pred[mask])
                results[f"mae_{intensity}"] = mae
                results[f"count_{intensity}"] = np.sum(mask)

        return results

    @staticmethod
    def calculate_skill_score(
        y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.1
    ) -> float:
        """
        Calcula Skill Score (Equitable Threat Score) para eventos de chuva
        """
        # Converter para classificação binária
        obs = (y_true >= threshold).astype(int)
        pred = (y_pred >= threshold).astype(int)

        # Matriz de confusão
        tn, fp, fn, tp = confusion_matrix(obs, pred).ravel()

        # Calcular skill score
        hits = tp
        misses = fn
        false_alarms = fp
        correct_negatives = tn

        total = hits + misses + false_alarms + correct_negatives
        hits_random = (hits + misses) * (hits + false_alarms) / total

        if (hits + misses + false_alarms - hits_random) == 0:
            return 0.0

        skill_score = (hits - hits_random) / (
            hits + misses + false_alarms - hits_random
        )
        return skill_score

    @staticmethod
    def calculate_precipitation_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcula conjunto completo de métricas para precipitação
        """
        metrics = {}

        # Métricas básicas de regressão
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics["mse"] = mean_squared_error(y_true, y_pred)

        # Métricas por intensidade
        intensity_metrics = MeteorologicalMetrics.calculate_mae_by_intensity(
            y_true, y_pred, RAIN_THRESHOLDS
        )
        metrics.update(intensity_metrics)

        # Skill scores para diferentes thresholds
        for name, threshold in RAIN_THRESHOLDS.items():
            if name != "very_heavy":  # Evitar threshold muito alto
                skill = MeteorologicalMetrics.calculate_skill_score(
                    y_true, y_pred, threshold
                )
                metrics[f"skill_score_{name}"] = skill

        # Métricas de classificação para eventos de chuva (>= 0.1mm/h)
        rain_threshold = RAIN_THRESHOLDS["light"]
        y_true_binary = (y_true >= rain_threshold).astype(int)
        y_pred_binary = (y_pred >= rain_threshold).astype(int)

        if len(np.unique(y_true_binary)) > 1:  # Evitar erro se só há uma classe
            metrics["accuracy"] = accuracy_score(y_true_binary, y_pred_binary)
            metrics["f1_score"] = f1_score(y_true_binary, y_pred_binary)

            # Calcular AUC se possível
            try:
                metrics["auc"] = roc_auc_score(y_true_binary, y_pred)
            except ValueError:
                metrics["auc"] = 0.0

        return metrics


class LSTMModelBuilder:
    """
    Construtor de modelos LSTM com diferentes arquiteturas
    """

    @staticmethod
    def build_model(
        sequence_length: int,
        features_count: int,
        lstm_units: List[int],
        dropout_rate: float,
        learning_rate: float,
    ) -> tf.keras.Model:
        """
        Constrói modelo LSTM com configuração específica
        """
        model = Sequential()

        # Primeira camada LSTM
        if len(lstm_units) == 1:
            model.add(
                LSTM(lstm_units[0], input_shape=(sequence_length, features_count))
            )
        else:
            model.add(
                LSTM(
                    lstm_units[0],
                    return_sequences=True,
                    input_shape=(sequence_length, features_count),
                )
            )

        model.add(Dropout(dropout_rate))

        # Camadas intermediárias
        for i in range(1, len(lstm_units) - 1):
            model.add(LSTM(lstm_units[i], return_sequences=True))
            model.add(Dropout(dropout_rate))

        # Última camada LSTM (se há mais de uma)
        if len(lstm_units) > 1:
            model.add(LSTM(lstm_units[-1]))
            model.add(Dropout(dropout_rate))

        # Camada de saída
        model.add(Dense(1, activation="relu"))  # ReLU para precipitação (não negativa)

        # Compilar modelo
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model


class TrainingPipeline:
    """
    Pipeline principal de treinamento e validação
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.data_splitter = TemporalDataSplitter(TEMPORAL_VALIDATION_CONFIG)
        self.metrics_calculator = MeteorologicalMetrics()
        self.results = defaultdict(list)

    def load_data(self) -> pd.DataFrame:
        """
        Carrega dados processados unificados
        """
        logger.info("Carregando dados para pipeline de treinamento...")

        # Tentar carregar dados unificados primeiro
        unified_file = PROCESSED_DATA_PATH / "inmet_unified_data.parquet"
        if unified_file.exists():
            data = pd.read_parquet(unified_file)
            logger.info(f"Dados unificados carregados: {data.shape}")
            return data

        # Fallback para dados separados
        train_file = PROCESSED_DATA_PATH / "train_data.parquet"
        val_file = PROCESSED_DATA_PATH / "validation_data.parquet"
        test_file = PROCESSED_DATA_PATH / "test_data.parquet"

        if all(f.exists() for f in [train_file, val_file, test_file]):
            train_data = pd.read_parquet(train_file)
            val_data = pd.read_parquet(val_file)
            test_data = pd.read_parquet(test_file)

            # Concatenar dados para validação temporal
            data = pd.concat([train_data, val_data, test_data], ignore_index=True)
            logger.info(f"Dados concatenados carregados: {data.shape}")
            return data

        raise FileNotFoundError("Nenhum arquivo de dados processados encontrado")

    def prepare_sequences(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        sequence_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara sequências temporais para treinamento
        """
        # Verificar colunas disponíveis
        available_features = [col for col in feature_cols if col in data.columns]
        if len(available_features) != len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            logger.warning(f"Features faltando: {missing}")
            feature_cols = available_features

        if target_col not in data.columns:
            # Procurar coluna de precipitação
            precip_cols = [col for col in data.columns if "precipitacao" in col.lower()]
            if precip_cols:
                target_col = precip_cols[0]
                logger.info(f"Usando {target_col} como target")
            else:
                raise ValueError(f"Coluna target '{target_col}' não encontrada")

        # Ordenar por timestamp
        if "timestamp" in data.columns:
            data = data.sort_values("timestamp")

        # Extrair features e target
        features = data[feature_cols].values
        target = data[target_col].values

        X, y = [], []

        for i in range(len(data) - sequence_length):
            X.append(features[i : (i + sequence_length)])
            y.append(target[i + sequence_length])

        return np.array(X), np.array(y)

    def run_temporal_cross_validation(self, max_folds: int = 5) -> Dict[str, Any]:
        """
        Executa validação cruzada temporal
        """
        logger.info("Iniciando validação cruzada temporal...")

        # Carregar dados
        data = self.load_data()

        # Configuração do modelo base
        model_config = {
            "sequence_length": 24,
            "lstm_units": [128, 64],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
        }

        fold_results = []

        # Executar folds temporais
        for fold_idx, (train_data, val_data) in enumerate(
            self.data_splitter.create_temporal_splits(data)
        ):
            if fold_idx >= max_folds:
                break

            logger.info(f"Processando fold {fold_idx + 1}...")

            try:
                # Preparar sequências
                X_train, y_train = self.prepare_sequences(
                    train_data,
                    METEOROLOGICAL_FEATURES,
                    "precipitacao_mm",
                    model_config["sequence_length"],
                )
                X_val, y_val = self.prepare_sequences(
                    val_data,
                    METEOROLOGICAL_FEATURES,
                    "precipitacao_mm",
                    model_config["sequence_length"],
                )

                if len(X_train) == 0 or len(X_val) == 0:
                    logger.warning(
                        f"Fold {fold_idx + 1} sem dados suficientes, pulando..."
                    )
                    continue

                # Normalizar dados
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()

                X_train_scaled = scaler_X.fit_transform(
                    X_train.reshape(-1, X_train.shape[-1])
                ).reshape(X_train.shape)
                X_val_scaled = scaler_X.transform(
                    X_val.reshape(-1, X_val.shape[-1])
                ).reshape(X_val.shape)

                y_train_scaled = scaler_y.fit_transform(
                    y_train.reshape(-1, 1)
                ).flatten()
                y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

                # Construir modelo
                model = LSTMModelBuilder.build_model(
                    model_config["sequence_length"],
                    len(METEOROLOGICAL_FEATURES),
                    model_config["lstm_units"],
                    model_config["dropout_rate"],
                    model_config["learning_rate"],
                )

                # Treinar modelo
                callbacks = [
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=5),
                ]

                history = model.fit(
                    X_train_scaled,
                    y_train_scaled,
                    validation_data=(X_val_scaled, y_val_scaled),
                    epochs=model_config["epochs"],
                    batch_size=model_config["batch_size"],
                    callbacks=callbacks,
                    verbose=0,
                )

                # Fazer previsões
                y_pred_scaled = model.predict(X_val_scaled, verbose=0)
                y_pred = scaler_y.inverse_transform(
                    y_pred_scaled.reshape(-1, 1)
                ).flatten()

                # Calcular métricas
                metrics = self.metrics_calculator.calculate_precipitation_metrics(
                    y_val, y_pred
                )
                metrics["fold"] = fold_idx + 1
                metrics["train_samples"] = len(X_train)
                metrics["val_samples"] = len(X_val)

                fold_results.append(metrics)

                logger.info(
                    f"Fold {fold_idx + 1} - MAE: {metrics['mae']:.3f}, "
                    f"RMSE: {metrics['rmse']:.3f}, Accuracy: {metrics.get('accuracy', 0):.3f}"
                )

            except Exception as e:
                logger.error(f"Erro no fold {fold_idx + 1}: {str(e)}")
                continue

        # Calcular estatísticas finais
        if fold_results:
            final_results = self._calculate_cv_statistics(fold_results)

            # Salvar resultados
            results_file = (
                MODELS_PATH
                / "temporal_validation"
                / f'cv_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            with open(results_file, "w") as f:
                json.dump(final_results, f, indent=2, default=str)

            logger.info(
                f"Validação cruzada temporal concluída. Resultados salvos em {results_file}"
            )
            return final_results
        else:
            logger.error("Nenhum fold foi executado com sucesso")
            return {}

    def _calculate_cv_statistics(
        self, fold_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calcula estatísticas da validação cruzada
        """
        stats = {}

        # Métricas numéricas para calcular média e std
        numeric_metrics = ["mae", "rmse", "mse", "accuracy", "f1_score", "auc"]

        for metric in numeric_metrics:
            values = [
                fold[metric]
                for fold in fold_results
                if metric in fold and fold[metric] is not None
            ]
            if values:
                stats[f"{metric}_mean"] = np.mean(values)
                stats[f"{metric}_std"] = np.std(values)
                stats[f"{metric}_min"] = np.min(values)
                stats[f"{metric}_max"] = np.max(values)

        # Critérios de sucesso
        accuracy_mean = stats.get("accuracy_mean", 0)
        mae_mean = stats.get("mae_mean", float("inf"))

        stats["meets_accuracy_target"] = accuracy_mean >= 0.75  # 75%
        stats["meets_mae_target"] = mae_mean <= 2.0  # MAE < 2.0 mm/h
        stats["overall_success"] = (
            stats["meets_accuracy_target"] and stats["meets_mae_target"]
        )

        stats["fold_results"] = fold_results
        stats["n_folds"] = len(fold_results)
        stats["timestamp"] = datetime.now().isoformat()

        return stats

    def run_hyperparameter_optimization(self, max_trials: int = 20) -> Dict[str, Any]:
        """
        Executa otimização de hiperparâmetros com grid search
        """
        logger.info(
            f"Iniciando otimização de hiperparâmetros com {max_trials} trials..."
        )

        # Carregar dados
        data = self.load_data()

        # Dividir dados para otimização (usar só parte dos dados para rapidez)
        if len(data) > 50000:
            data = data.sample(n=50000, random_state=42).sort_values("timestamp")
            logger.info(f"Usando amostra de {len(data)} registros para otimização")

        # Criar combinações de hiperparâmetros
        param_combinations = self._generate_param_combinations(max_trials)

        best_score = float("inf")
        best_params = None
        trial_results = []

        for trial_idx, params in enumerate(param_combinations):
            logger.info(f"Trial {trial_idx + 1}/{len(param_combinations)}: {params}")

            try:
                # Executar treinamento com parâmetros
                result = self._train_with_params(data, params)
                result["trial"] = trial_idx + 1
                result["params"] = params

                trial_results.append(result)

                # Atualizar melhor resultado
                if result["mae"] < best_score:
                    best_score = result["mae"]
                    best_params = params
                    logger.info(f"Novo melhor resultado: MAE = {best_score:.3f}")

            except Exception as e:
                logger.error(f"Erro no trial {trial_idx + 1}: {str(e)}")
                continue

        # Compilar resultados finais
        optimization_results = {
            "best_params": best_params,
            "best_mae": best_score,
            "total_trials": len(trial_results),
            "trial_results": trial_results,
            "timestamp": datetime.now().isoformat(),
        }

        # Salvar resultados
        results_file = (
            MODELS_PATH
            / "hyperopt"
            / f'hyperopt_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(results_file, "w") as f:
            json.dump(optimization_results, f, indent=2, default=str)

        logger.info(f"Otimização concluída. Melhores parâmetros: {best_params}")
        logger.info(f"Resultados salvos em {results_file}")

        return optimization_results

    def _generate_param_combinations(
        self, max_combinations: int
    ) -> List[Dict[str, Any]]:
        """
        Gera combinações de hiperparâmetros para grid search
        """
        from itertools import product

        # Reduzir grid para otimização eficiente
        reduced_grid = {
            "learning_rate": [0.001, 0.0001],
            "batch_size": [32, 64],
            "sequence_length": [24, 48],
            "lstm_units": [[128], [128, 64], [256, 128]],
            "dropout_rate": [0.2, 0.3],
        }

        # Gerar todas as combinações
        keys = list(reduced_grid.keys())
        values = list(reduced_grid.values())

        combinations = []
        for combination in product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)

        # Limitar número de combinações
        if len(combinations) > max_combinations:
            np.random.shuffle(combinations)
            combinations = combinations[:max_combinations]

        logger.info(f"Geradas {len(combinations)} combinações de parâmetros")
        return combinations

    def _train_with_params(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Treina modelo com parâmetros específicos
        """
        # Split temporal simples para otimização
        split_point = int(len(data) * 0.8)
        train_data = data.iloc[:split_point]
        val_data = data.iloc[split_point:]

        # Preparar sequências
        X_train, y_train = self.prepare_sequences(
            train_data,
            METEOROLOGICAL_FEATURES,
            "precipitacao_mm",
            params["sequence_length"],
        )
        X_val, y_val = self.prepare_sequences(
            val_data,
            METEOROLOGICAL_FEATURES,
            "precipitacao_mm",
            params["sequence_length"],
        )

        # Normalizar
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        ).reshape(X_train.shape)
        X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(
            X_val.shape
        )

        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

        # Construir e treinar modelo
        model = LSTMModelBuilder.build_model(
            params["sequence_length"],
            len(METEOROLOGICAL_FEATURES),
            params["lstm_units"],
            params["dropout_rate"],
            params["learning_rate"],
        )

        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=3),
        ]

        history = model.fit(
            X_train_scaled,
            y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=30,  # Reduzido para otimização
            batch_size=params["batch_size"],
            callbacks=callbacks,
            verbose=0,
        )

        # Avaliar
        y_pred_scaled = model.predict(X_val_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        # Calcular métricas
        metrics = self.metrics_calculator.calculate_precipitation_metrics(y_val, y_pred)

        return metrics


# ========================================
# FUNÇÃO PRINCIPAL
# ========================================


def main():
    """
    Função principal do pipeline de treinamento
    """
    parser = argparse.ArgumentParser(
        description="Pipeline de Treinamento e Validação - Fase 3.2"
    )
    parser.add_argument(
        "--mode",
        choices=["temporal-cv", "hyperopt", "full-pipeline"],
        default="temporal-cv",
        help="Modo de execução",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=5,
        help="Número máximo de folds para validação temporal",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=20,
        help="Número máximo de trials para otimização",
    )

    args = parser.parse_args()

    logger.info(f"Iniciando pipeline de treinamento - Modo: {args.mode}")
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"GPU Available: {len(tf.config.list_physical_devices('GPU'))} devices")

    pipeline = TrainingPipeline()

    try:
        if args.mode == "temporal-cv":
            results = pipeline.run_temporal_cross_validation(max_folds=args.max_folds)

            if results:
                logger.info("=== RESULTADOS DA VALIDAÇÃO CRUZADA TEMPORAL ===")
                logger.info(
                    f"Accuracy média: {results.get('accuracy_mean', 0):.3f} ± {results.get('accuracy_std', 0):.3f}"
                )
                logger.info(
                    f"MAE médio: {results.get('mae_mean', 0):.3f} ± {results.get('mae_std', 0):.3f}"
                )
                logger.info(
                    f"RMSE médio: {results.get('rmse_mean', 0):.3f} ± {results.get('rmse_std', 0):.3f}"
                )
                logger.info(
                    f"Accuracy >= 75%: {'✅' if results.get('meets_accuracy_target', False) else '❌'}"
                )
                logger.info(
                    f"MAE <= 2.0: {'✅' if results.get('meets_mae_target', False) else '❌'}"
                )

        elif args.mode == "hyperopt":
            results = pipeline.run_hyperparameter_optimization(
                max_trials=args.max_trials
            )

            if results:
                logger.info("=== RESULTADOS DA OTIMIZAÇÃO DE HIPERPARÂMETROS ===")
                logger.info(f"Melhor MAE: {results.get('best_mae', 0):.3f}")
                logger.info(f"Melhores parâmetros: {results.get('best_params', {})}")
                logger.info(f"Total de trials: {results.get('total_trials', 0)}")

        elif args.mode == "full-pipeline":
            logger.info("Executando pipeline completo...")

            # 1. Validação cruzada temporal
            logger.info("Etapa 1: Validação cruzada temporal")
            cv_results = pipeline.run_temporal_cross_validation(
                max_folds=args.max_folds
            )

            # 2. Otimização de hiperparâmetros
            logger.info("Etapa 2: Otimização de hiperparâmetros")
            hyperopt_results = pipeline.run_hyperparameter_optimization(
                max_trials=args.max_trials
            )

            # 3. Compilar resultados finais
            final_results = {
                "temporal_cv": cv_results,
                "hyperparameter_optimization": hyperopt_results,
                "timestamp": datetime.now().isoformat(),
            }

            # Salvar resultados completos
            results_file = (
                MODELS_PATH
                / f'full_pipeline_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            with open(results_file, "w") as f:
                json.dump(final_results, f, indent=2, default=str)

            logger.info(
                f"Pipeline completo finalizado. Resultados salvos em {results_file}"
            )

    except Exception as e:
        logger.error(f"Erro durante execução: {str(e)}")
        raise

    logger.info("Pipeline de treinamento concluído!")


if __name__ == "__main__":
    main()
