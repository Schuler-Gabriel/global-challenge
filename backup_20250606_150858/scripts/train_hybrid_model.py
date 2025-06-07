#!/usr/bin/env python3
"""
Hybrid LSTM Trainer - Phase 3.1
Sistema de Alertas de Cheias - Rio Guaíba
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

print("🔄 Carregando TensorFlow...")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

print(f"✅ TensorFlow {tf.__version__} carregado!")

# Configuração
np.random.seed(42)
tf.random.set_seed(42)

MODELS_PATH = Path("data/modelos_treinados")
MODELS_PATH.mkdir(parents=True, exist_ok=True)

def simulate_training_data():
    """Simula dados realistas para treinamento"""
    print("📊 Gerando dados de treinamento...")
    
    # 1.5 anos de dados horários
    n_hours = 24 * 365 + 24 * 180  # ~1.5 anos
    dates = pd.date_range('2022-01-01', periods=n_hours, freq='H')
    
    # Padrões sazonais básicos
    temp_base = 20 + 8 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    humidity_base = 70 + 15 * np.sin(2 * np.pi * (dates.dayofyear - 90) / 365.25)
    pressure_base = 1013 + 8 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    
    # Dados atmosféricos (149 features)
    atm_data = {}
    for i in range(149):
        if i < 30:  # Features de temperatura
            atm_data[f'atm_{i}'] = temp_base + np.random.normal(0, 2, n_hours)
        elif i < 60:  # Features de umidade  
            atm_data[f'atm_{i}'] = humidity_base + np.random.normal(0, 5, n_hours)
        elif i < 90:  # Features de pressão
            atm_data[f'atm_{i}'] = pressure_base + np.random.normal(0, 3, n_hours)
        else:  # Outras features
            atm_data[f'atm_{i}'] = np.random.normal(0, 1, n_hours)
    
    # Dados de superfície (25 features)  
    surf_data = {}
    for i in range(25):
        if i < 8:  # Temperatura
            surf_data[f'surf_{i}'] = temp_base + np.random.normal(0, 1, n_hours)
        elif i < 16:  # Umidade
            surf_data[f'surf_{i}'] = humidity_base + np.random.normal(0, 3, n_hours)
        else:  # Pressão e outros
            surf_data[f'surf_{i}'] = pressure_base + np.random.normal(0, 2, n_hours)
    
    # Target: precipitação com dependências realistas
    target = np.zeros(n_hours)
    for i in range(48, n_hours):
        # Efeito da temperatura (ótimo ~25°C)
        temp_effect = np.exp(-((temp_base[i] - 25) ** 2) / 100)
        
        # Efeito da umidade (maior com alta umidade)
        humidity_effect = humidity_base[i] / 100
        
        # Efeito da pressão (maior com baixa pressão)
        pressure_effect = (1015 - pressure_base[i]) / 10
        
        # Persistência da chuva
        recent_rain = np.mean(target[max(0, i-24):i])
        
        # Sazonalidade (mais chuva no verão)
        seasonal = 1 + 0.3 * np.sin(2 * np.pi * (dates[i].month - 3) / 12)
        
        combined = (temp_effect + humidity_effect + pressure_effect + recent_rain * 0.4) * seasonal
        target[i] = max(0, combined + np.random.exponential(0.2))
    
    print(f"✅ Dados simulados: {n_hours:,} registros")
    print(f"   Precipitação média: {target.mean():.3f}mm")
    print(f"   Precipitação máxima: {target.max():.1f}mm")
    
    return atm_data, surf_data, target, dates

def prepare_lstm_data(data_dict, target, seq_len):
    """Prepara dados para LSTM"""
    features = np.array([data_dict[key] for key in sorted(data_dict.keys())]).T
    
    # Normalizar
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Criar sequências
    X, y = [], []
    for i in range(seq_len, len(features_scaled)):
        X.append(features_scaled[i-seq_len:i])
        y.append(target[i])
    
    return np.array(X), np.array(y), scaler

def build_atmospheric_model():
    """Modelo LSTM atmosférico"""
    model = keras.Sequential([
        keras.Input(shape=(72, 149)),  # 3 dias, 149 features
        layers.LSTM(64, return_sequences=True, dropout=0.2),
        layers.LSTM(32, dropout=0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_surface_model():
    """Modelo LSTM de superfície"""  
    model = keras.Sequential([
        keras.Input(shape=(48, 25)),  # 2 dias, 25 features
        layers.LSTM(48, return_sequences=True, dropout=0.15),
        layers.LSTM(24, dropout=0.15),
        layers.Dense(12, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    print("🌊 Treinador Híbrido LSTM - Phase 3.1")
    print("="*50)
    
    # 1. Simular dados
    atm_data, surf_data, target, dates = simulate_training_data()
    
    # 2. Preparar dados atmosféricos
    print("\n🌍 Preparando dados atmosféricos...")
    X_atm, y_atm, scaler_atm = prepare_lstm_data(atm_data, target, 72)
    print(f"   Sequências atmosféricas: {X_atm.shape}")
    
    # 3. Preparar dados de superfície  
    print("🏔️ Preparando dados de superfície...")
    X_surf, y_surf, scaler_surf = prepare_lstm_data(surf_data, target, 48)
    print(f"   Sequências superfície: {X_surf.shape}")
    
    # 4. Alinhar dados (mesmo target)
    min_len = min(len(X_atm), len(X_surf))
    X_atm = X_atm[:min_len]
    X_surf = X_surf[:min_len] 
    y_target = y_atm[:min_len]
    
    print(f"📏 Dados alinhados: {min_len:,} amostras")
    
    # 5. Split dados
    train_size = int(0.7 * min_len)
    val_size = int(0.15 * min_len)
    
    X_atm_train = X_atm[:train_size]
    X_atm_val = X_atm[train_size:train_size+val_size] 
    X_atm_test = X_atm[train_size+val_size:]
    
    X_surf_train = X_surf[:train_size]
    X_surf_val = X_surf[train_size:train_size+val_size]
    X_surf_test = X_surf[train_size+val_size:]
    
    y_train = y_target[:train_size]
    y_val = y_target[train_size:train_size+val_size]
    y_test = y_target[train_size+val_size:]
    
    print(f"📊 Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    # 6. Treinar modelo atmosférico
    print("\n🚀 Treinando modelo atmosférico...")
    atm_model = build_atmospheric_model()
    print(f"   Parâmetros: {atm_model.count_params():,}")
    
    history_atm = atm_model.fit(
        X_atm_train, y_train,
        validation_data=(X_atm_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
    )
    
    # 7. Treinar modelo de superfície
    print("\n🚀 Treinando modelo de superfície...")
    surf_model = build_surface_model()
    print(f"   Parâmetros: {surf_model.count_params():,}")
    
    history_surf = surf_model.fit(
        X_surf_train, y_train,
        validation_data=(X_surf_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
    )
    
    # 8. Predições para ensemble
    print("\n🔗 Criando ensemble...")
    atm_val_preds = atm_model.predict(X_atm_val, verbose=0).flatten()
    surf_val_preds = surf_model.predict(X_surf_val, verbose=0).flatten()
    
    # 9. Treinar stacking
    X_meta = np.column_stack([atm_val_preds, surf_val_preds])
    stacking_model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
    stacking_model.fit(X_meta, y_val)
    
    # 10. Avaliação final
    print("\n📈 Avaliação no test set:")
    atm_test_preds = atm_model.predict(X_atm_test, verbose=0).flatten()
    surf_test_preds = surf_model.predict(X_surf_test, verbose=0).flatten()
    weighted_preds = 0.7 * atm_test_preds + 0.3 * surf_test_preds
    
    X_meta_test = np.column_stack([atm_test_preds, surf_test_preds])
    stacking_preds = stacking_model.predict(X_meta_test)
    
    # Métricas
    for name, preds in [
        ('Atmospheric', atm_test_preds),
        ('Surface', surf_test_preds),
        ('Weighted', weighted_preds),
        ('Stacking', stacking_preds)
    ]:
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"   {name:12} MAE: {mae:.4f}  RMSE: {rmse:.4f}")
    
    # 11. Salvar modelos
    print(f"\n💾 Salvando modelos em {MODELS_PATH}...")
    
    atm_model.save(MODELS_PATH / 'atmospheric_lstm')
    surf_model.save(MODELS_PATH / 'surface_lstm') 
    joblib.dump(stacking_model, MODELS_PATH / 'stacking_model.joblib')
    joblib.dump(scaler_atm, MODELS_PATH / 'atmospheric_scaler.joblib')
    joblib.dump(scaler_surf, MODELS_PATH / 'surface_scaler.joblib')
    
    # Metadados
    metadata = {
        'model_version': 'hybrid_v1.0_phase3.1',
        'creation_date': datetime.now().isoformat(),
        'tensorflow_version': tf.__version__,
        'atmospheric_params': int(atm_model.count_params()),
        'surface_params': int(surf_model.count_params()),
        'target_mae': float(mean_absolute_error(y_test, stacking_preds)),
        'target_rmse': float(np.sqrt(mean_squared_error(y_test, stacking_preds)))
    }
    
    with open(MODELS_PATH / 'hybrid_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Treinamento concluído!")
    print(f"🎯 MAE final: {metadata['target_mae']:.4f}")
    print(f"🎯 RMSE final: {metadata['target_rmse']:.4f}")
    print("\n🔗 Próximo: Phase 3.3 - Integração FastAPI")

if __name__ == "__main__":
    main() 