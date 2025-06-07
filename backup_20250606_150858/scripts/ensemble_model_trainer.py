#!/usr/bin/env python3
"""
Treinamento de Modelo Ensemble para Previsão de 4 Dias
Projeto: Sistema de Alertas de Cheias - Rio Guaíba

Este script implementa a estratégia de ensemble multi-escala:
1. Modelo Meteorológico Básico (LSTM + Attention)
2. Modelo Sinótico-Dinâmico (Transformer + CNN)
3. Modelo de Teleconexões (Graph Neural Network)
4. Meta-Modelo Ensemble
"""

import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleForecastTrainer:
    """Treinador de modelo ensemble para previsão de 4 dias"""

    def __init__(self):
        self.models_dir = Path("models/ensemble")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = Path("results/training")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Configurações do modelo
        self.sequence_length = 24  # 24 horas de histórico
        self.forecast_horizon = 96  # 96 horas (4 dias)
        self.ensemble_size = 3

        # Scalers para diferentes tipos de features
        self.scalers = {
            "meteorological": StandardScaler(),
            "synoptic": StandardScaler(),
            "teleconnections": MinMaxScaler(),
        }

    def prepare_training_data(self, data_file: str = None) -> Dict[str, np.ndarray]:
        """
        Prepara dados de treinamento simulados ou carrega dados reais
        """
        logger.info("Preparando dados de treinamento...")

        if data_file and Path(data_file).exists():
            # Carrega dados reais se disponível
            return self._load_real_data(data_file)
        else:
            # Simula dados de treinamento
            return self._simulate_training_data()

    def _simulate_training_data(self) -> Dict[str, np.ndarray]:
        """Simula dados de treinamento realistas"""
        np.random.seed(42)

        # Simula 2 anos de dados horários
        n_hours = 24 * 365 * 2
        dates = pd.date_range(start="2022-01-01", periods=n_hours, freq="H")

        # Features meteorológicas básicas
        meteorological_features = self._generate_meteorological_features(n_hours, dates)

        # Features sinóticas
        synoptic_features = self._generate_synoptic_features(n_hours, dates)

        # Features de teleconexões
        teleconnection_features = self._generate_teleconnection_features(n_hours, dates)

        # Target: nível do rio (simplificado)
        river_level = self._generate_river_level_target(
            meteorological_features, synoptic_features
        )

        return {
            "meteorological": meteorological_features,
            "synoptic": synoptic_features,
            "teleconnections": teleconnection_features,
            "target": river_level,
            "dates": dates,
        }

    def _generate_meteorological_features(
        self, n_hours: int, dates: pd.DatetimeIndex
    ) -> np.ndarray:
        """Gera features meteorológicas realistas"""
        # Sazonalidade e tendências
        day_of_year = dates.dayofyear
        hour_of_day = dates.hour

        # Temperatura com sazonalidade
        temp_seasonal = 20 + 10 * np.sin(2 * np.pi * day_of_year / 365.25)
        temp_daily = 5 * np.sin(2 * np.pi * hour_of_day / 24)
        temperature = temp_seasonal + temp_daily + np.random.normal(0, 2, n_hours)

        # Precipitação com eventos extremos
        precipitation = np.random.exponential(0.5, n_hours)
        # Adiciona eventos de chuva intensa ocasionais
        extreme_events = np.random.choice(
            n_hours, size=int(n_hours * 0.02), replace=False
        )
        precipitation[extreme_events] *= np.random.exponential(10, len(extreme_events))

        # Pressão atmosférica
        pressure = (
            1013
            + 10 * np.sin(2 * np.pi * day_of_year / 365.25)
            + np.random.normal(0, 5, n_hours)
        )

        # Vento
        wind_speed = (
            5
            + 3 * np.sin(2 * np.pi * day_of_year / 365.25)
            + np.random.exponential(2, n_hours)
        )
        wind_direction = np.random.uniform(0, 360, n_hours)

        # Umidade
        humidity = (
            60
            + 20 * np.sin(2 * np.pi * day_of_year / 365.25)
            + np.random.normal(0, 10, n_hours)
        )
        humidity = np.clip(humidity, 0, 100)

        return np.column_stack(
            [temperature, precipitation, pressure, wind_speed, wind_direction, humidity]
        )

    def _generate_synoptic_features(
        self, n_hours: int, dates: pd.DatetimeIndex
    ) -> np.ndarray:
        """Gera features sinóticas (frentes, sistemas de pressão)"""
        # Índices de frentes frias (mais frequentes no inverno)
        day_of_year = dates.dayofyear
        winter_intensity = 0.5 + 0.5 * np.cos(
            2 * np.pi * (day_of_year - 172) / 365.25
        )  # Pico no inverno

        front_intensity = np.random.exponential(winter_intensity, n_hours)
        front_speed = 20 + np.random.normal(0, 10, n_hours)

        # Sistemas de alta/baixa pressão
        pressure_system_type = np.random.choice(
            [-1, 0, 1], n_hours, p=[0.3, 0.4, 0.3]
        )  # -1: baixa, 0: neutro, 1: alta
        system_intensity = np.random.exponential(1, n_hours)

        # Cisalhamento do vento
        wind_shear = np.random.exponential(5, n_hours)

        # CAPE (energia convectiva)
        cape = np.random.exponential(500, n_hours)
        cape[dates.month.isin([12, 1, 2, 6, 7, 8])] *= 2  # Maior no verão

        return np.column_stack(
            [
                front_intensity,
                front_speed,
                pressure_system_type,
                system_intensity,
                wind_shear,
                cape,
            ]
        )

    def _generate_teleconnection_features(
        self, n_hours: int, dates: pd.DatetimeIndex
    ) -> np.ndarray:
        """Gera features de teleconexões (El Niño, oscilações)"""
        # El Niño/La Niña (varia lentamente)
        enso_index = np.sin(
            2 * np.pi * dates.dayofyear / (365.25 * 3)
        ) + np.random.normal(0, 0.1, n_hours)

        # Oscilação do Atlântico Sul
        sao_index = np.cos(
            2 * np.pi * dates.dayofyear / (365.25 * 2)
        ) + np.random.normal(0, 0.2, n_hours)

        # Temperatura da superfície do mar
        sst_anomaly = 0.5 * enso_index + np.random.normal(0, 0.3, n_hours)

        # Índices de dipolo
        dipole_index = np.random.normal(0, 0.5, n_hours)

        return np.column_stack([enso_index, sao_index, sst_anomaly, dipole_index])

    def _generate_river_level_target(
        self, met_features: np.ndarray, syn_features: np.ndarray
    ) -> np.ndarray:
        """Gera nível do rio baseado nas features meteorológicas e sinóticas"""
        n_hours = len(met_features)

        # Nível base do rio
        base_level = 2.0

        # Contribuição da precipitação (com delay)
        precipitation = met_features[:, 1]
        runoff = np.convolve(
            precipitation, np.exp(-np.arange(24) / 6), mode="same"
        )  # Decay exponencial

        # Contribuição de frentes frias
        front_contribution = syn_features[:, 0] * 0.5

        # Efeito de maré e sazonalidade
        seasonal_effect = 0.3 * np.sin(2 * np.pi * np.arange(n_hours) / (24 * 365.25))
        tidal_effect = 0.1 * np.sin(
            2 * np.pi * np.arange(n_hours) / 12.42
        )  # Maré semi-diurna

        # Ruído
        noise = np.random.normal(0, 0.1, n_hours)

        river_level = (
            base_level
            + runoff * 0.01
            + front_contribution
            + seasonal_effect
            + tidal_effect
            + noise
        )

        return np.maximum(river_level, 0.5)  # Nível mínimo

    def create_sequences(
        self, data: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Cria sequências de entrada e saída para o modelo"""
        logger.info("Criando sequências de treinamento...")

        n_samples = (
            len(data["target"]) - self.sequence_length - self.forecast_horizon + 1
        )

        # Preparar X (entradas)
        X = {}
        for feature_type in ["meteorological", "synoptic", "teleconnections"]:
            features = data[feature_type]
            # Normalizar
            features_scaled = self.scalers[feature_type].fit_transform(features)

            # Criar sequências
            X[feature_type] = np.array(
                [
                    features_scaled[i : i + self.sequence_length]
                    for i in range(n_samples)
                ]
            )

        # Preparar y (saídas) - múltiplos horizontes
        y = np.array(
            [
                data["target"][
                    i
                    + self.sequence_length : i
                    + self.sequence_length
                    + self.forecast_horizon
                ]
                for i in range(n_samples)
            ]
        )

        return X, y

    def build_meteorological_model(self, input_shapes: Dict[str, Tuple]) -> keras.Model:
        """
        Modelo 1: LSTM + Attention para dados meteorológicos básicos
        Foco: 1-2 dias
        """
        logger.info("Construindo modelo meteorológico (LSTM + Attention)...")

        # Input para features meteorológicas
        met_input = keras.Input(
            shape=input_shapes["meteorological"], name="meteorological_input"
        )

        # LSTM layers com dropout
        lstm1 = layers.LSTM(128, return_sequences=True, dropout=0.2)(met_input)
        lstm2 = layers.LSTM(64, return_sequences=True, dropout=0.2)(lstm1)

        # Attention mechanism
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(lstm2, lstm2)
        attention = layers.LayerNormalization()(attention + lstm2)

        # Global pooling
        pooled = layers.GlobalAveragePooling1D()(attention)

        # Dense layers
        dense1 = layers.Dense(128, activation="relu")(pooled)
        dense1 = layers.Dropout(0.3)(dense1)
        dense2 = layers.Dense(64, activation="relu")(dense1)

        # Output para múltiplos horizontes
        output = layers.Dense(
            self.forecast_horizon, activation="linear", name="meteorological_output"
        )(dense2)

        model = keras.Model(
            inputs=met_input, outputs=output, name="meteorological_model"
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"],
        )

        return model

    def build_synoptic_model(self, input_shapes: Dict[str, Tuple]) -> keras.Model:
        """
        Modelo 2: Transformer + CNN para análise sinótica
        Foco: 2-4 dias
        """
        logger.info("Construindo modelo sinótico (Transformer + CNN)...")

        # Inputs
        met_input = keras.Input(shape=input_shapes["meteorological"], name="met_input")
        syn_input = keras.Input(shape=input_shapes["synoptic"], name="syn_input")

        # CNN para features meteorológicas (padrões locais)
        conv1 = layers.Conv1D(64, 3, activation="relu", padding="same")(met_input)
        conv2 = layers.Conv1D(32, 3, activation="relu", padding="same")(conv1)
        pool = layers.MaxPooling1D(2)(conv2)

        # Transformer para features sinóticas (padrões de longo prazo)
        # Positional encoding
        positions = tf.range(start=0, limit=input_shapes["synoptic"][0], delta=1)
        positions = tf.cast(positions, tf.float32)
        pos_encoding = layers.Dense(input_shapes["synoptic"][1])(
            positions[:, tf.newaxis]
        )
        syn_with_pos = syn_input + pos_encoding

        # Multi-head attention
        attention = layers.MultiHeadAttention(num_heads=8, key_dim=64)(
            syn_with_pos, syn_with_pos
        )
        attention = layers.LayerNormalization()(attention + syn_with_pos)

        # Feed forward
        ff = layers.Dense(128, activation="relu")(attention)
        ff = layers.Dense(input_shapes["synoptic"][1])(ff)
        transformer_out = layers.LayerNormalization()(ff + attention)

        # Combine CNN and Transformer outputs
        cnn_flat = layers.GlobalAveragePooling1D()(pool)
        transformer_flat = layers.GlobalAveragePooling1D()(transformer_out)
        combined = layers.Concatenate()([cnn_flat, transformer_flat])

        # Dense layers
        dense1 = layers.Dense(256, activation="relu")(combined)
        dense1 = layers.Dropout(0.3)(dense1)
        dense2 = layers.Dense(128, activation="relu")(dense1)
        dense2 = layers.Dropout(0.2)(dense2)

        # Output
        output = layers.Dense(
            self.forecast_horizon, activation="linear", name="synoptic_output"
        )(dense2)

        model = keras.Model(
            inputs=[met_input, syn_input], outputs=output, name="synoptic_model"
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss="mse",
            metrics=["mae"],
        )

        return model

    def build_teleconnection_model(self, input_shapes: Dict[str, Tuple]) -> keras.Model:
        """
        Modelo 3: Graph Neural Network para teleconexões
        Foco: 3-7 dias
        """
        logger.info("Construindo modelo de teleconexões (GNN simplificado)...")

        # Simplificação: usar Dense layers para simular GNN
        tel_input = keras.Input(
            shape=input_shapes["teleconnections"], name="teleconnection_input"
        )

        # "Graph" processing (simulado com Dense layers)
        # Na implementação real, usaria bibliotecas como DGL ou PyTorch Geometric

        # Temporal processing
        lstm = layers.LSTM(32, return_sequences=True)(tel_input)

        # "Node" processing
        node1 = layers.Dense(64, activation="tanh")(lstm)
        node2 = layers.Dense(32, activation="tanh")(node1)

        # Global features
        global_pool = layers.GlobalAveragePooling1D()(node2)

        # Dense layers
        dense1 = layers.Dense(64, activation="relu")(global_pool)
        dense1 = layers.Dropout(0.2)(dense1)
        dense2 = layers.Dense(32, activation="relu")(dense1)

        # Output (pesos menores para horizontes longos)
        output = layers.Dense(
            self.forecast_horizon, activation="linear", name="teleconnection_output"
        )(dense2)

        model = keras.Model(
            inputs=tel_input, outputs=output, name="teleconnection_model"
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss="mse",
            metrics=["mae"],
        )

        return model

    def build_meta_ensemble_model(self, n_base_models: int = 3) -> keras.Model:
        """
        Meta-modelo que combina as previsões dos modelos base
        """
        logger.info("Construindo meta-modelo ensemble...")

        # Inputs das previsões dos modelos base
        model_inputs = []
        for i in range(n_base_models):
            model_inputs.append(
                keras.Input(
                    shape=(self.forecast_horizon,), name=f"model_{i}_prediction"
                )
            )

        # Concatena as previsões
        combined = layers.Concatenate()(model_inputs)

        # Rede para aprender pesos adaptativos
        dense1 = layers.Dense(128, activation="relu")(combined)
        dense1 = layers.Dropout(0.2)(dense1)
        dense2 = layers.Dense(64, activation="relu")(dense1)

        # Pesos para cada modelo (softmax para garantir soma = 1)
        weights = layers.Dense(
            n_base_models, activation="softmax", name="model_weights"
        )(dense2)

        # Previsão final ponderada
        weighted_predictions = []
        for i in range(n_base_models):
            weight = layers.Lambda(lambda x: x[:, i : i + 1])(weights)
            weighted_pred = layers.Multiply()([model_inputs[i], weight])
            weighted_predictions.append(weighted_pred)

        final_prediction = layers.Add()(weighted_predictions)

        model = keras.Model(
            inputs=model_inputs,
            outputs=[final_prediction, weights],
            name="meta_ensemble",
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={"add": "mse", "model_weights": "categorical_crossentropy"},
            loss_weights={"add": 1.0, "model_weights": 0.1},
            metrics={"add": ["mae"], "model_weights": ["accuracy"]},
        )

        return model

    def train_ensemble(self, data: Dict[str, np.ndarray]) -> Dict[str, keras.Model]:
        """Treina o ensemble completo"""
        logger.info("Iniciando treinamento do ensemble...")

        # Preparar dados
        X, y = self.create_sequences(data)

        # Split temporal
        split_point = int(len(y) * 0.8)
        X_train = {k: v[:split_point] for k, v in X.items()}
        X_val = {k: v[split_point:] for k, v in X.items()}
        y_train = y[:split_point]
        y_val = y[split_point:]

        input_shapes = {k: v.shape[1:] for k, v in X.items()}

        # Treinar modelos base
        models = {}
        base_predictions = {"train": [], "val": []}

        # Modelo 1: Meteorológico
        logger.info("Treinando modelo meteorológico...")
        met_model = self.build_meteorological_model(input_shapes)
        met_history = met_model.fit(
            X_train["meteorological"],
            y_train,
            validation_data=(X_val["meteorological"], y_val),
            epochs=50,
            batch_size=32,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ],
        )
        models["meteorological"] = met_model
        base_predictions["train"].append(
            met_model.predict(X_train["meteorological"], verbose=0)
        )
        base_predictions["val"].append(
            met_model.predict(X_val["meteorological"], verbose=0)
        )

        # Modelo 2: Sinótico
        logger.info("Treinando modelo sinótico...")
        syn_model = self.build_synoptic_model(input_shapes)
        syn_history = syn_model.fit(
            [X_train["meteorological"], X_train["synoptic"]],
            y_train,
            validation_data=([X_val["meteorological"], X_val["synoptic"]], y_val),
            epochs=50,
            batch_size=32,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ],
        )
        models["synoptic"] = syn_model
        base_predictions["train"].append(
            syn_model.predict(
                [X_train["meteorological"], X_train["synoptic"]], verbose=0
            )
        )
        base_predictions["val"].append(
            syn_model.predict([X_val["meteorological"], X_val["synoptic"]], verbose=0)
        )

        # Modelo 3: Teleconexões
        logger.info("Treinando modelo de teleconexões...")
        tel_model = self.build_teleconnection_model(input_shapes)
        tel_history = tel_model.fit(
            X_train["teleconnections"],
            y_train,
            validation_data=(X_val["teleconnections"], y_val),
            epochs=50,
            batch_size=32,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ],
        )
        models["teleconnections"] = tel_model
        base_predictions["train"].append(
            tel_model.predict(X_train["teleconnections"], verbose=0)
        )
        base_predictions["val"].append(
            tel_model.predict(X_val["teleconnections"], verbose=0)
        )

        # Meta-modelo
        logger.info("Treinando meta-modelo ensemble...")
        meta_model = self.build_meta_ensemble_model()

        # Preparar dados para meta-modelo (dummy weights para treino)
        dummy_weights = np.ones((len(y_train), 3)) / 3  # Pesos uniformes iniciais

        meta_history = meta_model.fit(
            base_predictions["train"],
            [y_train, dummy_weights],
            validation_data=(
                base_predictions["val"],
                [y_val, np.ones((len(y_val), 3)) / 3],
            ),
            epochs=30,
            batch_size=32,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ],
        )
        models["meta_ensemble"] = meta_model

        # Salvar modelos
        self._save_models(models)

        # Avaliar ensemble
        self._evaluate_ensemble(models, X_val, y_val, base_predictions["val"])

        return models

    def _save_models(self, models: Dict[str, keras.Model]) -> None:
        """Salva os modelos treinados"""
        logger.info("Salvando modelos...")

        for name, model in models.items():
            model_path = self.models_dir / f"{name}_model.h5"
            model.save(model_path)

        # Salvar scalers
        for name, scaler in self.scalers.items():
            scaler_path = self.models_dir / f"{name}_scaler.joblib"
            joblib.dump(scaler, scaler_path)

    def _evaluate_ensemble(
        self,
        models: Dict[str, keras.Model],
        X_val: Dict,
        y_val: np.ndarray,
        base_preds: List[np.ndarray],
    ) -> None:
        """Avalia o desempenho do ensemble"""
        logger.info("Avaliando ensemble...")

        # Previsão do meta-modelo
        ensemble_pred, weights = models["meta_ensemble"].predict(base_preds, verbose=0)

        # Métricas por horizonte de tempo
        horizons = [24, 48, 72, 96]  # horas
        results = {}

        for horizon in horizons:
            if horizon <= self.forecast_horizon:
                idx = horizon - 1

                # Métricas individuais
                for i, (name, _) in enumerate(
                    [(k, v) for k, v in models.items() if k != "meta_ensemble"]
                ):
                    pred = base_preds[i][:, idx]
                    true = y_val[:, idx]

                    results[f"{name}_{horizon}h"] = {
                        "mae": mean_absolute_error(true, pred),
                        "rmse": np.sqrt(mean_squared_error(true, pred)),
                        "r2": r2_score(true, pred),
                    }

                # Métricas do ensemble
                ensemble_pred_h = ensemble_pred[:, idx]
                true_h = y_val[:, idx]

                results[f"ensemble_{horizon}h"] = {
                    "mae": mean_absolute_error(true_h, ensemble_pred_h),
                    "rmse": np.sqrt(mean_squared_error(true_h, ensemble_pred_h)),
                    "r2": r2_score(true_h, ensemble_pred_h),
                }

        # Salvar resultados
        results_file = (
            self.results_dir
            / f"ensemble_evaluation_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        )
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Imprimir resumo
        print("\n" + "=" * 60)
        print("📊 AVALIAÇÃO DO ENSEMBLE")
        print("=" * 60)

        for horizon in horizons:
            if f"ensemble_{horizon}h" in results:
                print(f"\n⏰ {horizon} horas:")
                ensemble_metrics = results[f"ensemble_{horizon}h"]
                print(f"  📏 MAE: {ensemble_metrics['mae']:.3f}")
                print(f"  📐 RMSE: {ensemble_metrics['rmse']:.3f}")
                print(f"  📈 R²: {ensemble_metrics['r2']:.3f}")

        # Análise dos pesos
        avg_weights = np.mean(weights, axis=0)
        print(f"\n🏋️ Pesos médios do ensemble:")
        model_names = ["Meteorológico", "Sinótico", "Teleconexões"]
        for i, (name, weight) in enumerate(zip(model_names, avg_weights)):
            print(f"  {name}: {weight:.1%}")

        print("=" * 60)


def main():
    """Demonstração do treinamento do ensemble"""

    print("🚀 INICIANDO TREINAMENTO DO ENSEMBLE")
    print("=" * 50)

    trainer = EnsembleForecastTrainer()

    # Preparar dados
    print("📊 Preparando dados de treinamento...")
    training_data = trainer.prepare_training_data()

    print(f"✅ Dados preparados:")
    print(f"  📅 Período: {len(training_data['dates'])} horas")
    print(f"  🌡️ Features meteorológicas: {training_data['meteorological'].shape[1]}")
    print(f"  🌀 Features sinóticas: {training_data['synoptic'].shape[1]}")
    print(f"  🌍 Features teleconexões: {training_data['teleconnections'].shape[1]}")

    # Treinar ensemble
    print("\n🧠 Treinando modelos...")
    models = trainer.train_ensemble(training_data)

    print(f"\n✅ Treinamento concluído!")
    print(f"📁 Modelos salvos em: {trainer.models_dir}")
    print(f"📈 Resultados salvos em: {trainer.results_dir}")

    print("\n🎯 RECOMENDAÇÕES PARA PRODUÇÃO:")
    print("1. 📡 Integrar APIs reais (NOAA, ECMWF)")
    print("2. 🔄 Implementar retreinamento automático")
    print("3. 📊 Adicionar monitoramento de drift")
    print("4. ⚡ Otimizar para inferência em tempo real")
    print("5. 🎛️ Implementar ajuste fino por região")


if __name__ == "__main__":
    main()
