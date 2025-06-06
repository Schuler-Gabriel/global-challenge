#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid LSTM Model Trainer - Phase 3.1
Sistema de Alertas de Cheias - Rio Guaíba

Implementa treinamento completo do modelo híbrido usando TensorFlow:
- Component 1: LSTM atmosférico (149 features)
- Component 2: LSTM superfície (25 features)  
- Component 3: Ensemble stacking

Target: 82-87% accuracy com <1.5mm MAE, <2.5mm RMSE
"""

import os
import sys
import warnings
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, optimizers
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    print(f"✓ TensorFlow {tf.__version__} carregado com sucesso")
except ImportError as e:
    print(f"❌ Erro ao importar TensorFlow: {e}")
    sys.exit(1)

# Configurações
np.random.seed(42)
tf.random.set_seed(42)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hybrid_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
MODELS_PATH = DATA_PATH / "modelos_treinados"
MODELS_PATH.mkdir(parents=True, exist_ok=True)


class HybridLSTMTrainer:
    """
    Trainer para modelo híbrido LSTM - Phase 3.1
    
    Arquitetura:
    - Atmospheric LSTM: 149 features → 96 units → 48 units → 1 output
    - Surface LSTM: 25 features → 64 units → 32 units → 1 output
    - Ensemble: Weighted average (0.7/0.3) + Stacking RandomForest
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.atmospheric_model = None
        self.surface_model = None
        self.stacking_model = None
        self.scalers = {}
        
        logger.info("HybridLSTMTrainer inicializado para Phase 3.1")
        
    def _default_config(self) -> Dict[str, Any]:
        """Configuração padrão otimizada para meteorologia"""
        return {
            # Atmospheric LSTM
            'atmospheric': {
                'sequence_length': 72,  # 3 dias
                'features_count': 149,
                'lstm_units': [96, 48],
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'patience': 15
            },
            # Surface LSTM  
            'surface': {
                'sequence_length': 48,  # 2 dias
                'features_count': 25,
                'lstm_units': [64, 32],
                'dropout_rate': 0.15,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'patience': 15
            },
            # Ensemble
            'ensemble': {
                'atmospheric_weight': 0.7,
                'surface_weight': 0.3,
                'stacking_params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                }
            },
            # Training
            'validation_split': 0.2,
            'test_split': 0.15
        }
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega dados processados para treinamento"""
        logger.info("Carregando dados processados...")
        
        # Dados atmosféricos
        atm_file = PROCESSED_DATA_PATH / "openmeteo_historical_forecast_consolidated.parquet"
        if atm_file.exists():
            atmospheric_data = pd.read_parquet(atm_file)
            logger.info(f"✓ Dados atmosféricos: {len(atmospheric_data)} registros")
        else:
            logger.warning("Dados atmosféricos não encontrados, simulando...")
            atmospheric_data = self._simulate_atmospheric_data()
        
        # Dados de superfície (simular por agora)
        surface_data = self._simulate_surface_data(len(atmospheric_data))
        
        return atmospheric_data, surface_data
    
    def _simulate_atmospheric_data(self) -> pd.DataFrame:
        """Simula dados atmosféricos realistas"""
        logger.info("Simulando dados atmosféricos...")
        
        # 2 anos de dados horários
        n_hours = 24 * 365 * 2
        dates = pd.date_range(start='2022-01-01', periods=n_hours, freq='H')
        
        # Features atmosféricas simuladas (149 features)
        np.random.seed(42)
        data = {}
        
        # Surface features (21)
        surface_features = [
            'temperature_2m', 'relative_humidity_2m', 'dewpoint_2m',
            'precipitation', 'rain', 'pressure_msl', 'surface_pressure',
            'cloudcover', 'windspeed_10m', 'winddirection_10m',
            'windgusts_10m', 'cape', 'lifted_index'
        ]
        
        for feature in surface_features:
            if 'temperature' in feature:
                # Temperatura com sazonalidade
                seasonal = 20 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
                daily = 5 * np.sin(2 * np.pi * dates.hour / 24)
                data[feature] = seasonal + daily + np.random.normal(0, 2, n_hours)
            elif 'precipitation' in feature or 'rain' in feature:
                # Precipitação com eventos extremos
                precip = np.random.exponential(0.3, n_hours)
                extreme_idx = np.random.choice(n_hours, size=int(n_hours * 0.02), replace=False)
                precip[extreme_idx] *= np.random.exponential(20, len(extreme_idx))
                data[feature] = precip
            elif 'pressure' in feature:
                # Pressão atmosférica
                data[feature] = 1013 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365.25) + np.random.normal(0, 5, n_hours)
            elif 'wind' in feature:
                if 'direction' in feature:
                    data[feature] = np.random.uniform(0, 360, n_hours)
                else:
                    data[feature] = 5 + 3 * np.sin(2 * np.pi * dates.dayofyear / 365.25) + np.random.exponential(2, n_hours)
            elif 'humidity' in feature:
                data[feature] = np.clip(60 + 20 * np.sin(2 * np.pi * dates.dayofyear / 365.25) + np.random.normal(0, 10, n_hours), 0, 100)
            else:
                data[feature] = np.random.normal(0, 1, n_hours)
        
        # Pressure level features (125 = 5 levels × 25 vars)
        pressure_levels = ['1000hPa', '850hPa', '700hPa', '500hPa', '300hPa']
        pressure_vars = ['temperature', 'relative_humidity', 'wind_speed', 'wind_direction', 'geopotential_height']
        
        for level in pressure_levels:
            for var in pressure_vars:
                feature_name = f"{var}_{level}"
                if 'temperature' in var:
                    # Temperatura diminui com altitude
                    alt_factor = {'1000hPa': 0, '850hPa': -10, '700hPa': -20, '500hPa': -35, '300hPa': -55}[level]
                    base_temp = data['temperature_2m'] + alt_factor
                    data[feature_name] = base_temp + np.random.normal(0, 3, n_hours)
                elif 'humidity' in var:
                    data[feature_name] = np.clip(data['relative_humidity_2m'] + np.random.normal(0, 15, n_hours), 0, 100)
                elif 'wind' in var:
                    if 'direction' in var:
                        data[feature_name] = np.random.uniform(0, 360, n_hours)
                    else:
                        # Vento aumenta com altitude
                        alt_factor = {'1000hPa': 1.0, '850hPa': 1.5, '700hPa': 2.0, '500hPa': 3.0, '300hPa': 4.0}[level]
                        data[feature_name] = data['windspeed_10m'] * alt_factor + np.random.exponential(5, n_hours)
                else:
                    data[feature_name] = np.random.normal(0, 1, n_hours)
        
        # Synoptic derived features (10)
        synoptic_features = [
            'wind_shear_850_500', 'wind_shear_1000_850', 'temp_gradient_850_500',
            'temp_gradient_surface_850', 'frontal_strength_850', 'temperature_advection_850',
            'vorticity_500', 'divergence_500', 'atmospheric_instability', 'moisture_flux'
        ]
        
        for feature in synoptic_features:
            data[feature] = np.random.normal(0, 1, n_hours)
        
        # Target: precipitação futura (24h ahead)
        target_precip = data['precipitation'].copy()
        # Adicionar dependência das features atmosféricas
        for i in range(24, len(target_precip)):
            # Correlação com temperatura e umidade
            temp_effect = (data['temperature_2m'][i-24:i].mean() - 15) / 20
            humidity_effect = data['relative_humidity_2m'][i-24:i].mean() / 100
            pressure_effect = (1013 - data['pressure_msl'][i-24:i].mean()) / 20
            
            target_precip[i] = max(0, temp_effect + humidity_effect + pressure_effect + np.random.exponential(0.5))
        
        data['target_precipitation'] = target_precip
        data['timestamp'] = dates
        
        df = pd.DataFrame(data)
        logger.info(f"✓ Dados atmosféricos simulados: {len(df)} registros, {len(df.columns)} features")
        
        return df
    
    def _simulate_surface_data(self, n_records: int) -> pd.DataFrame:
        """Simula dados de superfície (25 features)"""
        logger.info("Simulando dados de superfície...")
        
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=n_records, freq='H')
        
        # 25 surface features
        surface_features = [
            'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
            'apparent_temperature_mean', 'apparent_temperature_max', 'apparent_temperature_min',
            'relativehumidity_2m_mean', 'relativehumidity_2m_max', 'relativehumidity_2m_min',
            'dewpoint_2m_mean', 'precipitation_sum', 'rain_sum', 'showers_sum',
            'windspeed_10m_mean', 'windspeed_10m_max', 'winddirection_10m_dominant',
            'windgusts_10m_max', 'pressure_msl_mean', 'surface_pressure_mean',
            'pressure_msl_min', 'cloudcover_mean', 'cloudcover_low_mean',
            'cloudcover_high_mean', 'shortwave_radiation_sum', 'weathercode_mode'
        ]
        
        data = {}
        for feature in surface_features:
            if 'temperature' in feature:
                base_temp = 20 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
                if 'max' in feature:
                    data[feature] = base_temp + 5 + np.random.normal(0, 2, n_records)
                elif 'min' in feature:
                    data[feature] = base_temp - 5 + np.random.normal(0, 2, n_records)
                else:
                    data[feature] = base_temp + np.random.normal(0, 2, n_records)
            elif 'precipitation' in feature or 'rain' in feature or 'showers' in feature:
                data[feature] = np.random.exponential(0.2, n_records)
            elif 'humidity' in feature:
                base_hum = 60 + 20 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
                data[feature] = np.clip(base_hum + np.random.normal(0, 10, n_records), 0, 100)
            elif 'pressure' in feature:
                data[feature] = 1013 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365.25) + np.random.normal(0, 5, n_records)
            elif 'wind' in feature:
                if 'direction' in feature:
                    data[feature] = np.random.uniform(0, 360, n_records)
                else:
                    data[feature] = 5 + np.random.exponential(3, n_records)
            else:
                data[feature] = np.random.normal(0, 1, n_records)
        
        data['timestamp'] = dates
        df = pd.DataFrame(data)
        logger.info(f"✓ Dados de superfície simulados: {len(df)} registros, {len(df.columns)} features")
        
        return df
    
    def prepare_sequences(self, data: pd.DataFrame, feature_cols: List[str], 
                         target_col: str, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara sequências para LSTM"""
        logger.info(f"Preparando sequências de {sequence_length} timesteps...")
        
        # Normalizar features
        scaler_name = f"scaler_{len(feature_cols)}features"
        if scaler_name not in self.scalers:
            self.scalers[scaler_name] = StandardScaler()
            
        features_scaled = self.scalers[scaler_name].fit_transform(data[feature_cols])
        target = data[target_col].values
        
        # Criar sequências
        X, y = [], []
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            y.append(target[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"✓ Sequências criadas: X={X.shape}, y={y.shape}")
        return X, y
    
    def build_atmospheric_lstm(self) -> tf.keras.Model:
        """Constrói LSTM atmosférico (149 features)"""
        config = self.config['atmospheric']
        
        model = tf.keras.Sequential([
            layers.Input(shape=(config['sequence_length'], config['features_count'])),
            
            # Primeira camada LSTM
            layers.LSTM(config['lstm_units'][0], return_sequences=True, dropout=config['dropout_rate']),
            layers.BatchNormalization(),
            
            # Segunda camada LSTM
            layers.LSTM(config['lstm_units'][1], dropout=config['dropout_rate']),
            layers.BatchNormalization(),
            
            # Camadas densas
            layers.Dense(32, activation='relu'),
            layers.Dropout(config['dropout_rate']),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=config['learning_rate']),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        logger.info(f"✓ Atmospheric LSTM criado: {model.count_params():,} parâmetros")
        return model
    
    def build_surface_lstm(self) -> tf.keras.Model:
        """Constrói LSTM de superfície (25 features)"""
        config = self.config['surface']
        
        model = tf.keras.Sequential([
            layers.Input(shape=(config['sequence_length'], config['features_count'])),
            
            # Primeira camada LSTM
            layers.LSTM(config['lstm_units'][0], return_sequences=True, dropout=config['dropout_rate']),
            layers.BatchNormalization(),
            
            # Segunda camada LSTM  
            layers.LSTM(config['lstm_units'][1], dropout=config['dropout_rate']),
            layers.BatchNormalization(),
            
            # Camadas densas
            layers.Dense(24, activation='relu'),
            layers.Dropout(config['dropout_rate']),
            layers.Dense(12, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=config['learning_rate']),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        logger.info(f"✓ Surface LSTM criado: {model.count_params():,} parâmetros")
        return model
    
    def train_atmospheric_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                               X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.Model:
        """Treina modelo atmosférico"""
        logger.info("Treinando modelo atmosférico...")
        
        model = self.build_atmospheric_lstm()
        config = self.config['atmospheric']
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=config['patience'], restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=config['patience']//2, min_lr=1e-7),
            callbacks.ModelCheckpoint(
                str(MODELS_PATH / 'atmospheric_lstm_best.keras'),
                monitor='val_loss', save_best_only=True
            )
        ]
        
        # Treinamento
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Avaliação
        train_loss = model.evaluate(X_train, y_train, verbose=0)
        val_loss = model.evaluate(X_val, y_val, verbose=0)
        
        logger.info(f"✓ Atmospheric LSTM treinado:")
        logger.info(f"  Train Loss: {train_loss[0]:.4f}, MAE: {train_loss[1]:.4f}")
        logger.info(f"  Val Loss: {val_loss[0]:.4f}, MAE: {val_loss[1]:.4f}")
        
        return model
    
    def train_surface_model(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.Model:
        """Treina modelo de superfície"""
        logger.info("Treinando modelo de superfície...")
        
        model = self.build_surface_lstm()
        config = self.config['surface']
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=config['patience'], restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=config['patience']//2, min_lr=1e-7),
            callbacks.ModelCheckpoint(
                str(MODELS_PATH / 'surface_lstm_best.keras'),
                monitor='val_loss', save_best_only=True
            )
        ]
        
        # Treinamento
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Avaliação
        train_loss = model.evaluate(X_train, y_train, verbose=0)
        val_loss = model.evaluate(X_val, y_val, verbose=0)
        
        logger.info(f"✓ Surface LSTM treinado:")
        logger.info(f"  Train Loss: {train_loss[0]:.4f}, MAE: {train_loss[1]:.4f}")
        logger.info(f"  Val Loss: {val_loss[0]:.4f}, MAE: {val_loss[1]:.4f}")
        
        return model
    
    def train_stacking_model(self, atm_preds: np.ndarray, surf_preds: np.ndarray, 
                            y_true: np.ndarray) -> RandomForestRegressor:
        """Treina meta-modelo de stacking"""
        logger.info("Treinando stacking model...")
        
        # Meta-features
        X_meta = np.column_stack([atm_preds, surf_preds])
        
        # Modelo RandomForest
        config = self.config['ensemble']['stacking_params']
        model = RandomForestRegressor(**config)
        model.fit(X_meta, y_true)
        
        # Avaliação
        meta_preds = model.predict(X_meta)
        mae = mean_absolute_error(y_true, meta_preds)
        rmse = np.sqrt(mean_squared_error(y_true, meta_preds))
        r2 = r2_score(y_true, meta_preds)
        
        logger.info(f"✓ Stacking model treinado:")
        logger.info(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return model
    
    def train_hybrid_model(self):
        """Pipeline completo de treinamento híbrido"""
        logger.info("=== INICIANDO TREINAMENTO HÍBRIDO PHASE 3.1 ===")
        
        # 1. Carregar dados
        atmospheric_data, surface_data = self.load_data()
        
        # 2. Preparar features atmosféricas
        atm_features = [col for col in atmospheric_data.columns 
                       if col not in ['timestamp', 'target_precipitation']][:149]
        X_atm, y_atm = self.prepare_sequences(
            atmospheric_data, atm_features, 'target_precipitation',
            self.config['atmospheric']['sequence_length']
        )
        
        # 3. Preparar features de superfície
        surf_features = [col for col in surface_data.columns 
                        if col != 'timestamp'][:25]
        # Alinhar com target atmosférico
        aligned_surface = surface_data.iloc[self.config['atmospheric']['sequence_length']:].reset_index(drop=True)
        target_aligned = atmospheric_data['target_precipitation'].iloc[self.config['atmospheric']['sequence_length']:].values
        
        X_surf, y_surf = self.prepare_sequences(
            aligned_surface, surf_features, 'target_precipitation',
            self.config['surface']['sequence_length']
        )
        
        # Ajustar tamanhos
        min_len = min(len(X_atm), len(X_surf))
        X_atm = X_atm[:min_len]
        X_surf = X_surf[:min_len]
        y_target = y_atm[:min_len]
        
        # 4. Split dados
        val_split = self.config['validation_split']
        test_split = self.config['test_split']
        
        train_size = int(len(X_atm) * (1 - val_split - test_split))
        val_size = int(len(X_atm) * val_split)
        
        # Atmospheric splits
        X_atm_train = X_atm[:train_size]
        X_atm_val = X_atm[train_size:train_size+val_size]
        X_atm_test = X_atm[train_size+val_size:]
        
        # Surface splits
        X_surf_train = X_surf[:train_size]
        X_surf_val = X_surf[train_size:train_size+val_size]
        X_surf_test = X_surf[train_size+val_size:]
        
        # Target splits
        y_train = y_target[:train_size]
        y_val = y_target[train_size:train_size+val_size]
        y_test = y_target[train_size+val_size:]
        
        logger.info(f"Splits: Train={train_size}, Val={val_size}, Test={len(y_test)}")
        
        # 5. Treinar modelos
        self.atmospheric_model = self.train_atmospheric_model(
            X_atm_train, y_train, X_atm_val, y_val
        )
        
        self.surface_model = self.train_surface_model(
            X_surf_train, y_train, X_surf_val, y_val
        )
        
        # 6. Predições para stacking
        logger.info("Gerando predições para stacking...")
        atm_val_preds = self.atmospheric_model.predict(X_atm_val, verbose=0).flatten()
        surf_val_preds = self.surface_model.predict(X_surf_val, verbose=0).flatten()
        
        # 7. Treinar stacking
        self.stacking_model = self.train_stacking_model(
            atm_val_preds, surf_val_preds, y_val
        )
        
        # 8. Avaliação final no test set
        logger.info("=== AVALIAÇÃO FINAL ===")
        self.evaluate_hybrid_model(X_atm_test, X_surf_test, y_test)
        
        # 9. Salvar modelos
        self.save_models()
        
        logger.info("=== TREINAMENTO HÍBRIDO CONCLUÍDO ===")
    
    def evaluate_hybrid_model(self, X_atm_test: np.ndarray, X_surf_test: np.ndarray, y_test: np.ndarray):
        """Avaliação completa do modelo híbrido"""
        
        # Predições individuais
        atm_preds = self.atmospheric_model.predict(X_atm_test, verbose=0).flatten()
        surf_preds = self.surface_model.predict(X_surf_test, verbose=0).flatten()
        
        # Weighted average
        weights = self.config['ensemble']
        weighted_preds = (weights['atmospheric_weight'] * atm_preds + 
                         weights['surface_weight'] * surf_preds)
        
        # Stacking
        X_meta = np.column_stack([atm_preds, surf_preds])
        stacking_preds = self.stacking_model.predict(X_meta)
        
        # Métricas
        results = {}
        for name, preds in [
            ('Atmospheric', atm_preds),
            ('Surface', surf_preds), 
            ('Weighted', weighted_preds),
            ('Stacking', stacking_preds)
        ]:
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            
            results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
            
            logger.info(f"{name:12} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Check targets Phase 3.1
        stacking_mae = results['Stacking']['MAE']
        stacking_rmse = results['Stacking']['RMSE']
        
        target_mae = 1.5  # < 1.5mm/h
        target_rmse = 2.5  # < 2.5mm/h
        
        if stacking_mae < target_mae and stacking_rmse < target_rmse:
            logger.info(f"🎯 TARGETS ATINGIDOS! MAE: {stacking_mae:.3f} < {target_mae}, RMSE: {stacking_rmse:.3f} < {target_rmse}")
        else:
            logger.warning(f"⚠️  Targets não atingidos. MAE: {stacking_mae:.3f}, RMSE: {stacking_rmse:.3f}")
        
        return results
    
    def save_models(self):
        """Salva todos os componentes do modelo híbrido"""
        logger.info("Salvando modelos...")
        
        # Modelos TensorFlow
        self.atmospheric_model.save(MODELS_PATH / 'atmospheric_lstm')
        self.surface_model.save(MODELS_PATH / 'surface_lstm')
        
        # Stacking model
        joblib.dump(self.stacking_model, MODELS_PATH / 'stacking_model.joblib')
        
        # Scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, MODELS_PATH / f'{name}.joblib')
        
        # Configuração
        with open(MODELS_PATH / 'hybrid_config.json', 'w') as f:
            import json
            json.dump(self.config, f, indent=2)
        
        # Metadados
        metadata = {
            'model_version': 'hybrid_v1.0_phase3.1',
            'creation_date': datetime.now().isoformat(),
            'target_accuracy': '82-87%',
            'target_mae': '<1.5mm/h',
            'target_rmse': '<2.5mm/h',
            'tensorflow_version': tf.__version__,
            'components': {
                'atmospheric_lstm': 'atmospheric_lstm/',
                'surface_lstm': 'surface_lstm/',
                'stacking_model': 'stacking_model.joblib'
            }
        }
        
        with open(MODELS_PATH / 'hybrid_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Modelos salvos em: {MODELS_PATH}")


def main():
    """Função principal"""
    print("🌊 Sistema de Alertas de Cheias - Hybrid LSTM Trainer Phase 3.1")
    print("=" * 70)
    
    try:
        # Inicializar trainer
        trainer = HybridLSTMTrainer()
        
        # Treinar modelo híbrido
        trainer.train_hybrid_model()
        
        print("\n🎉 Treinamento concluído com sucesso!")
        print(f"📁 Modelos salvos em: {MODELS_PATH}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Treinamento interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro durante o treinamento: {e}")
        logger.error(f"Erro: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 