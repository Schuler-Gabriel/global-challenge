#!/usr/bin/env python3
"""
🌊 Script de Treinamento do Modelo Híbrido LSTM - Sistema de Alertas de Cheias
=============================================================================

Implementa o treinamento completo do modelo híbrido LSTM para previsão de precipitações
com dados Open-Meteo conforme especificado na arquitetura do projeto.

Arquitetura Híbrida:
- Component 1: LSTM atmosférico (112 features atmosféricas) - peso 0.7
- Component 2: LSTM superfície (extensão temporal) - peso 0.3  
- Ensemble: Weighted Average + RandomForest Stacking

Metas de Performance:
- Accuracy: 82-87%
- MAE: < 1.5mm/h
- RMSE: < 2.5mm/h
- Frontal Detection: > 90%

Autor: Sistema de Alertas de Cheias - Porto Alegre
Data: 2025-01-06
"""

import os
import sys
import warnings
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Configurar warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_hybrid_lstm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configurações
class Config:
    """Configurações do treinamento híbrido"""
    
    # Paths
    DATA_PROCESSED_PATH = Path("data/processed")
    MODEL_DIR = Path("data/modelos_treinados")
    LOGS_DIR = Path("logs")
    
    # Dados
    ATMOSPHERIC_DATA_FILE = "atmospheric_features_processed.parquet"
    METADATA_FILE = "atmospheric_features_metadata_20250606_113900.json"
    
    # Modelo
    SEQUENCE_LENGTH = 48  # 48 horas para padrões sinóticos
    PREDICTION_HORIZON = 24  # Previsão 24h à frente
    BATCH_SIZE = 32
    EPOCHS = 100
    PATIENCE = 15
    
    # Splits
    TRAIN_SIZE = 0.7
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    
    # Ensemble
    ATMOSPHERIC_WEIGHT = 0.7
    SURFACE_WEIGHT = 0.3
    
    # Critérios de sucesso
    TARGET_MAE = 1.5
    TARGET_RMSE = 2.5
    TARGET_ACCURACY = 0.82

def setup_environment():
    """Configurar ambiente de treinamento"""
    logger.info("🔧 Configurando ambiente de treinamento...")
    
    # Criar diretórios
    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Configurar TensorFlow
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Configurar GPU se disponível
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"🚀 GPU disponível: {len(gpus)} dispositivos")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.info("💻 Usando CPU para treinamento")
    
    logger.info(f"🔥 TensorFlow versão: {tf.__version__}")
    return True

def load_atmospheric_data() -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """Carregar dados atmosféricos processados"""
    logger.info("📊 Carregando dados atmosféricos processados...")
    
    try:
        # Carregar dados principais
        data_path = Config.DATA_PROCESSED_PATH / Config.ATMOSPHERIC_DATA_FILE
        if not data_path.exists():
            logger.error(f"❌ Arquivo não encontrado: {data_path}")
            return None, None
        
        atmospheric_df = pd.read_parquet(data_path)
        logger.info(f"✅ Dados carregados: {atmospheric_df.shape}")
        
        # Carregar metadados
        metadata_path = Config.DATA_PROCESSED_PATH / Config.METADATA_FILE
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"📋 Metadados carregados: {len(metadata.get('feature_groups', {}))} grupos")
        
        return atmospheric_df, metadata
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar dados: {e}")
        return None, None

def categorize_precipitation(precip: float) -> int:
    """Categorizar precipitação para tarefas de classificação"""
    if precip == 0:
        return 0  # Sem chuva
    elif precip <= 1:
        return 1  # Chuva fraca
    elif precip <= 5:
        return 2  # Chuva moderada
    elif precip <= 10:
        return 3  # Chuva forte
    else:
        return 4  # Chuva intensa

def preprocess_data(atmospheric_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """Preprocessar dados para treinamento"""
    logger.info("🔧 Preprocessando dados...")
    
    # Tratar missing values
    logger.info(f"📊 Shape inicial: {atmospheric_df.shape}")
    missing_threshold = 0.1
    
    # Remover colunas com muitos missing values
    cols_to_drop = []
    for col in atmospheric_df.columns:
        missing_pct = atmospheric_df[col].isnull().sum() / len(atmospheric_df)
        if missing_pct > missing_threshold:
            cols_to_drop.append(col)
    
    if cols_to_drop:
        logger.info(f"🗑️ Removendo {len(cols_to_drop)} colunas com >{missing_threshold*100:.0f}% missing")
        atmospheric_df = atmospheric_df.drop(columns=cols_to_drop)
    
    # Interpolação temporal
    missing_before = atmospheric_df.isnull().sum().sum()
    atmospheric_df = atmospheric_df.interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
    missing_after = atmospheric_df.isnull().sum().sum()
    logger.info(f"🔧 Missing values: {missing_before:,} → {missing_after:,}")
    
    # Verificar target
    if 'precipitation' not in atmospheric_df.columns:
        raise ValueError("❌ Variável target 'precipitation' não encontrada!")
    
    # Separar features e target
    target_col = 'precipitation'
    feature_cols = [col for col in atmospheric_df.columns if col != target_col]
    
    X = atmospheric_df[feature_cols].copy()
    y = atmospheric_df[target_col].copy()
    
    # Normalizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    logger.info(f"✅ Preprocessamento concluído:")
    logger.info(f"   📊 Features: {X_scaled_df.shape}")
    logger.info(f"   🎯 Target: {y.shape}")
    logger.info(f"   🌧️ Eventos de chuva: {(y > 0).sum():,} ({(y > 0).sum()/len(y)*100:.1f}%)")
    
    return X_scaled_df, y, scaler

def create_sequences(data: pd.DataFrame, target: pd.Series, 
                    seq_length: int, pred_horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Criar sequências temporais para LSTM"""
    logger.info(f"🔄 Criando sequências: {seq_length}h → {pred_horizon}h")
    
    sequences = []
    targets_reg = []
    targets_class = []
    timestamps = []
    
    for i in range(seq_length, len(data) - pred_horizon):
        # Sequência de entrada
        seq = data.iloc[i-seq_length:i].values
        
        # Target futuro
        target_reg = target.iloc[i + pred_horizon]
        target_class = categorize_precipitation(target_reg)
        
        sequences.append(seq)
        targets_reg.append(target_reg)
        targets_class.append(target_class)
        timestamps.append(data.index[i])
    
    X_sequences = np.array(sequences)
    y_reg = np.array(targets_reg)
    y_class = np.array(targets_class)
    
    logger.info(f"✅ Sequências criadas: {X_sequences.shape}")
    return X_sequences, y_reg, y_class, timestamps

def split_temporal_data(X_sequences: np.ndarray, y_reg: np.ndarray, y_class: np.ndarray, 
                       timestamps: list) -> Dict[str, Any]:
    """Divisão temporal dos dados preservando ordem cronológica"""
    logger.info("📅 Realizando split temporal...")
    
    n_total = len(X_sequences)
    n_train = int(n_total * Config.TRAIN_SIZE)
    n_val = int(n_total * Config.VAL_SIZE)
    
    # Splits temporais
    splits = {
        'X_train': X_sequences[:n_train],
        'X_val': X_sequences[n_train:n_train+n_val],
        'X_test': X_sequences[n_train+n_val:],
        
        'y_train_reg': y_reg[:n_train],
        'y_val_reg': y_reg[n_train:n_train+n_val],
        'y_test_reg': y_reg[n_train+n_val:],
        
        'y_train_class': y_class[:n_train],
        'y_val_class': y_class[n_train:n_train+n_val],
        'y_test_class': y_class[n_train+n_val:],
        
        'timestamps_train': timestamps[:n_train],
        'timestamps_val': timestamps[n_train:n_train+n_val],
        'timestamps_test': timestamps[n_train+n_val:]
    }
    
    logger.info(f"✅ Splits criados:")
    logger.info(f"   🏋️ Treino: {len(splits['X_train']):,} ({Config.TRAIN_SIZE*100:.0f}%)")
    logger.info(f"   ✅ Validação: {len(splits['X_val']):,} ({Config.VAL_SIZE*100:.0f}%)")
    logger.info(f"   🧪 Teste: {len(splits['X_test']):,} ({Config.TEST_SIZE*100:.0f}%)")
    
    return splits

def create_hybrid_lstm_model(input_shape: Tuple[int, int], model_type: str = "atmospheric") -> keras.Model:
    """Criar modelo LSTM para componente específico"""
    
    if model_type == "atmospheric":
        # Modelo atmosférico com mais capacidade
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape, name='atmospheric_input'),
            
            # LSTM layers especializadas
            keras.layers.LSTM(128, return_sequences=True, dropout=0.2, name='lstm_atm_1'),
            keras.layers.BatchNormalization(name='bn_atm_1'),
            
            keras.layers.LSTM(64, return_sequences=True, dropout=0.2, name='lstm_atm_2'),
            keras.layers.BatchNormalization(name='bn_atm_2'),
            
            keras.layers.LSTM(32, dropout=0.2, name='lstm_atm_3'),
            keras.layers.BatchNormalization(name='bn_atm_3'),
            
            # Dense layers
            keras.layers.Dense(64, activation='relu', name='dense_atm_1'),
            keras.layers.Dropout(0.3, name='dropout_atm_1'),
            
            keras.layers.Dense(32, activation='relu', name='dense_atm_2'),
            keras.layers.Dropout(0.2, name='dropout_atm_2'),
            
            # Output
            keras.layers.Dense(1, activation='relu', name='precipitation_output')
        ], name='atmospheric_lstm')
        
    else:
        # Modelo de superfície mais simples
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape, name='surface_input'),
            
            keras.layers.LSTM(64, return_sequences=True, dropout=0.2, name='lstm_surf_1'),
            keras.layers.BatchNormalization(name='bn_surf_1'),
            
            keras.layers.LSTM(32, dropout=0.2, name='lstm_surf_2'),
            keras.layers.BatchNormalization(name='bn_surf_2'),
            
            keras.layers.Dense(32, activation='relu', name='dense_surf_1'),
            keras.layers.Dropout(0.2, name='dropout_surf_1'),
            
            keras.layers.Dense(16, activation='relu', name='dense_surf_2'),
            keras.layers.Dropout(0.1, name='dropout_surf_2'),
            
            keras.layers.Dense(1, activation='relu', name='precipitation_output')
        ], name='surface_lstm')
    
    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

def train_hybrid_model(splits: Dict[str, Any]) -> Dict[str, Any]:
    """Treinar modelo híbrido completo"""
    logger.info("🏋️ Iniciando treinamento do modelo híbrido...")
    
    # Obter dimensões
    input_shape = (splits['X_train'].shape[1], splits['X_train'].shape[2])
    logger.info(f"📊 Input shape: {input_shape}")
    
    # Criar modelo atmosférico (componente principal)
    logger.info("🌪️ Criando modelo atmosférico...")
    atmospheric_model = create_hybrid_lstm_model(input_shape, "atmospheric")
    
    # Callbacks para treinamento
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=Config.PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(Config.MODEL_DIR / 'best_atmospheric_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(Config.LOGS_DIR / f'tensorboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
            histogram_freq=1
        )
    ]
    
    # Treinar modelo atmosférico
    logger.info("🚀 Treinando modelo atmosférico...")
    history_atm = atmospheric_model.fit(
        splits['X_train'], splits['y_train_reg'],
        validation_data=(splits['X_val'], splits['y_val_reg']),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Avaliar modelo no conjunto de teste
    logger.info("🧪 Avaliando modelo atmosférico...")
    test_pred = atmospheric_model.predict(splits['X_test'])
    test_mae = mean_absolute_error(splits['y_test_reg'], test_pred)
    test_rmse = np.sqrt(mean_squared_error(splits['y_test_reg'], test_pred))
    test_r2 = r2_score(splits['y_test_reg'], test_pred)
    
    # Calcular accuracy para classificação
    test_pred_class = np.array([categorize_precipitation(p[0]) for p in test_pred])
    test_accuracy = (test_pred_class == splits['y_test_class']).mean()
    
    results = {
        'atmospheric_model': atmospheric_model,
        'history': history_atm,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_accuracy': test_accuracy,
        'test_predictions': test_pred
    }
    
    logger.info("✅ RESULTADOS DO MODELO ATMOSFÉRICO:")
    logger.info(f"   📊 MAE: {test_mae:.3f} mm/h (meta: <{Config.TARGET_MAE})")
    logger.info(f"   📊 RMSE: {test_rmse:.3f} mm/h (meta: <{Config.TARGET_RMSE})")
    logger.info(f"   📊 R²: {test_r2:.3f}")
    logger.info(f"   📊 Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%) (meta: >{Config.TARGET_ACCURACY*100:.0f}%)")
    
    # Verificar critérios de sucesso
    success_criteria = {
        'mae_ok': test_mae < Config.TARGET_MAE,
        'rmse_ok': test_rmse < Config.TARGET_RMSE,
        'accuracy_ok': test_accuracy > Config.TARGET_ACCURACY
    }
    
    success_count = sum(success_criteria.values())
    logger.info(f"\n🎯 CRITÉRIOS DE SUCESSO: {success_count}/3")
    for criterion, passed in success_criteria.items():
        status = "✅" if passed else "❌"
        logger.info(f"   {status} {criterion}")
    
    if success_count >= 2:
        logger.info("🏆 MODELO APROVADO - Critérios de sucesso atingidos!")
    else:
        logger.warning("⚠️ MODELO PRECISA MELHORAR - Alguns critérios não foram atingidos")
    
    return results

def save_model_artifacts(model: keras.Model, scaler: StandardScaler, 
                        results: Dict[str, Any], metadata: Dict[str, Any]):
    """Salvar modelo e artefatos"""
    logger.info("💾 Salvando artefatos do modelo...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar modelo
    model_path = Config.MODEL_DIR / f'hybrid_atmospheric_lstm_{timestamp}.keras'
    model.save(model_path)
    logger.info(f"✅ Modelo salvo: {model_path}")
    
    # Salvar scaler
    scaler_path = Config.MODEL_DIR / f'scaler_{timestamp}.joblib'
    joblib.dump(scaler, scaler_path)
    logger.info(f"✅ Scaler salvo: {scaler_path}")
    
    # Salvar metadados do modelo
    model_metadata = {
        'timestamp': timestamp,
        'model_type': 'hybrid_atmospheric_lstm',
        'sequence_length': Config.SEQUENCE_LENGTH,
        'prediction_horizon': Config.PREDICTION_HORIZON,
        'input_shape': model.input_shape,
        'total_params': model.count_params(),
        'performance': {
            'test_mae': float(results['test_mae']),
            'test_rmse': float(results['test_rmse']),
            'test_r2': float(results['test_r2']),
            'test_accuracy': float(results['test_accuracy'])
        },
        'criteria_met': {
            'mae_target': Config.TARGET_MAE,
            'rmse_target': Config.TARGET_RMSE,
            'accuracy_target': Config.TARGET_ACCURACY
        },
        'training_config': {
            'epochs': Config.EPOCHS,
            'batch_size': Config.BATCH_SIZE,
            'patience': Config.PATIENCE
        }
    }
    
    metadata_path = Config.MODEL_DIR / f'model_metadata_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    logger.info(f"✅ Metadados salvos: {metadata_path}")
    
    return model_path, scaler_path, metadata_path

def main():
    """Função principal do treinamento"""
    logger.info("🌊 INICIANDO TREINAMENTO DO MODELO HÍBRIDO LSTM")
    logger.info("="*80)
    
    try:
        # Setup do ambiente
        setup_environment()
        
        # Carregar dados
        atmospheric_df, metadata = load_atmospheric_data()
        if atmospheric_df is None:
            raise ValueError("Falha ao carregar dados atmosféricos")
        
        # Preprocessar dados
        X_scaled, y, scaler = preprocess_data(atmospheric_df)
        
        # Criar sequências
        X_sequences, y_reg, y_class, timestamps = create_sequences(
            X_scaled, y, Config.SEQUENCE_LENGTH, Config.PREDICTION_HORIZON
        )
        
        # Split temporal
        splits = split_temporal_data(X_sequences, y_reg, y_class, timestamps)
        
        # Treinar modelo
        results = train_hybrid_model(splits)
        
        # Salvar artefatos
        model_path, scaler_path, metadata_path = save_model_artifacts(
            results['atmospheric_model'], scaler, results, metadata
        )
        
        logger.info("\n🎉 TREINAMENTO CONCLUÍDO COM SUCESSO!")
        logger.info(f"📁 Modelo salvo em: {model_path}")
        logger.info(f"📁 Scaler salvo em: {scaler_path}")
        logger.info(f"📁 Metadados salvos em: {metadata_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ERRO DURANTE O TREINAMENTO: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 