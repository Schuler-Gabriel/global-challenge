# %% [markdown]
# # Treinamento do Modelo LSTM - Previsão Meteorológica
#
# Este notebook implementa a arquitetura LSTM para previsão de chuva baseada nos dados históricos do INMET (2000-2025).
#
# ## Objetivos:
# - Arquitetura LSTM multivariada (16+ features)
# - Sequence length otimizado para dados horários
# - Previsão de precipitação para 24h à frente
# - Accuracy > 75% em classificação de eventos de chuva
# - MAE < 2.0 mm/h para precipitação

# %%
# Imports necessários
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# Scikit-learn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report

# Configurações
plt.style.use('seaborn-v0_8')
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# %%
# Configurações do modelo baseadas na documentação
MODEL_CONFIG = {
    'sequence_length': 24,  # 24 horas de histórico
    'forecast_horizon': 24,  # Previsão para 24h à frente
    'features_count': 16,    # Variáveis meteorológicas disponíveis
    'lstm_units': [128, 64, 32],  # Configuração de camadas
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'patience': 15
}

# Features principais dos dados INMET
FEATURE_COLUMNS = [
    'precipitacao_mm',
    'pressao_mb', 
    'temperatura_c',
    'ponto_orvalho_c',
    'umidade_relativa',
    'velocidade_vento_ms',
    'direcao_vento_gr',
    'radiacao_kjm2',
    'pressao_max_mb',
    'pressao_min_mb',
    'temperatura_max_c',
    'temperatura_min_c',
    'umidade_max',
    'umidade_min',
    'ponto_orvalho_max_c',
    'ponto_orvalho_min_c'
]

# Paths
DATA_PATH = Path('../data/processed')
MODEL_PATH = Path('../data/modelos_treinados')
MODEL_PATH.mkdir(exist_ok=True)

print(f"Configuração do modelo: {MODEL_CONFIG}")
print(f"Features disponíveis: {len(FEATURE_COLUMNS)}")

# %% [markdown]
# ## 1. Carregamento e Preparação dos Dados

# %%
# Carregar dados processados
print("Carregando dados de treinamento...")
train_data = pd.read_parquet(DATA_PATH / 'train_data.parquet')
val_data = pd.read_parquet(DATA_PATH / 'validation_data.parquet')
test_data = pd.read_parquet(DATA_PATH / 'test_data.parquet')

print(f"Train shape: {train_data.shape}")
print(f"Validation shape: {val_data.shape}")
print(f"Test shape: {test_data.shape}")

# Verificar colunas disponíveis
print(f"\nColunas disponíveis: {train_data.columns.tolist()}")

# Ajustar FEATURE_COLUMNS baseado nas colunas reais
available_features = [col for col in FEATURE_COLUMNS if col in train_data.columns]
if len(available_features) < len(FEATURE_COLUMNS):
    print(f"\nAjustando features disponíveis: {len(available_features)} de {len(FEATURE_COLUMNS)}")
    FEATURE_COLUMNS = available_features
    MODEL_CONFIG['features_count'] = len(FEATURE_COLUMNS)

print(f"\nFeatures finais para o modelo: {FEATURE_COLUMNS}")

# %%
def prepare_sequences(data, feature_cols, target_col, sequence_length, forecast_horizon):
    """
    Prepara sequências temporais para treinamento do LSTM
    
    Args:
        data: DataFrame com dados temporais
        feature_cols: Lista de colunas de features
        target_col: Nome da coluna target
        sequence_length: Comprimento da sequência de entrada
        forecast_horizon: Horizonte de previsão
    
    Returns:
        X: Sequências de features
        y: Targets correspondentes
    """
    # Ordenar por timestamp se disponível
    if 'timestamp' in data.columns:
        data = data.sort_values('timestamp')
    
    # Extrair features e target
    features = data[feature_cols].values
    target = data[target_col].values
    
    X, y = [], []
    
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        # Sequência de entrada (últimas sequence_length horas)
        X.append(features[i:(i + sequence_length)])
        
        # Target (precipitação daqui a forecast_horizon horas)
        target_idx = i + sequence_length + forecast_horizon - 1
        y.append(target[target_idx])
    
    return np.array(X), np.array(y)

# Determinar coluna target (precipitação)
target_column = None
for col in train_data.columns:
    if 'precipitacao' in col.lower() or 'chuva' in col.lower():
        target_column = col
        break

if target_column is None:
    # Usar primeira coluna de feature como target temporário
    target_column = FEATURE_COLUMNS[0]
    print(f"ATENÇÃO: Usando {target_column} como target temporário")

print(f"Target column: {target_column}")

# %%
# Preparar sequências para treinamento
print("Preparando sequências temporais...")

X_train, y_train = prepare_sequences(
    train_data, 
    FEATURE_COLUMNS, 
    target_column,
    MODEL_CONFIG['sequence_length'],
    MODEL_CONFIG['forecast_horizon']
)

X_val, y_val = prepare_sequences(
    val_data, 
    FEATURE_COLUMNS, 
    target_column,
    MODEL_CONFIG['sequence_length'],
    MODEL_CONFIG['forecast_horizon']
)

X_test, y_test = prepare_sequences(
    test_data, 
    FEATURE_COLUMNS, 
    target_column,
    MODEL_CONFIG['sequence_length'],
    MODEL_CONFIG['forecast_horizon']
)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Estatísticas do target
print(f"\nEstatísticas do target (precipitação):")
print(f"Train - Mean: {y_train.mean():.3f}, Std: {y_train.std():.3f}, Max: {y_train.max():.3f}")
print(f"Val - Mean: {y_val.mean():.3f}, Std: {y_val.std():.3f}, Max: {y_val.max():.3f}")
print(f"Test - Mean: {y_test.mean():.3f}, Std: {y_test.std():.3f}, Max: {y_test.max():.3f}")

# %% [markdown]
# ## 2. Normalização dos Dados

# %%
# Normalização das features
print("Normalizando features...")

# Reshape para normalização
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

# Normalizar features
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train_reshaped)
X_val_scaled = feature_scaler.transform(X_val_reshaped)
X_test_scaled = feature_scaler.transform(X_test_reshaped)

# Reshape de volta
X_train_scaled = X_train_scaled.reshape(X_train.shape)
X_val_scaled = X_val_scaled.reshape(X_val.shape)
X_test_scaled = X_test_scaled.reshape(X_test.shape)

# Normalizar target (precipitação)
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

print(f"Features normalizadas - Shape: {X_train_scaled.shape}")
print(f"Target normalizado - Mean: {y_train_scaled.mean():.3f}, Std: {y_train_scaled.std():.3f}")

# Salvar scalers
import joblib
joblib.dump(feature_scaler, MODEL_PATH / 'feature_scaler.pkl')
joblib.dump(target_scaler, MODEL_PATH / 'target_scaler.pkl')
print("Scalers salvos!")

# %% [markdown]
# ## 3. Arquitetura do Modelo LSTM

# %%
def create_lstm_model(sequence_length, features_count, lstm_units, dropout_rate):
    """
    Cria modelo LSTM para previsão de séries temporais
    
    Args:
        sequence_length: Comprimento da sequência de entrada
        features_count: Número de features
        lstm_units: Lista com número de unidades por camada LSTM
        dropout_rate: Taxa de dropout
    
    Returns:
        model: Modelo Keras compilado
    """
    model = Sequential()
    
    # Primeira camada LSTM
    model.add(LSTM(
        lstm_units[0],
        return_sequences=len(lstm_units) > 1,
        input_shape=(sequence_length, features_count),
        name='lstm_1'
    ))
    model.add(Dropout(dropout_rate, name='dropout_1'))
    
    # Camadas LSTM adicionais
    for i, units in enumerate(lstm_units[1:], 2):
        return_sequences = i < len(lstm_units)
        model.add(LSTM(
            units,
            return_sequences=return_sequences,
            name=f'lstm_{i}'
        ))
        model.add(Dropout(dropout_rate, name=f'dropout_{i}'))
    
    # Camada densa final
    model.add(Dense(50, activation='relu', name='dense_1'))
    model.add(Dropout(dropout_rate, name='dropout_final'))
    model.add(Dense(1, name='output'))
    
    return model

# Criar modelo principal
print("Criando modelo LSTM...")
model = create_lstm_model(
    MODEL_CONFIG['sequence_length'],
    MODEL_CONFIG['features_count'],
    MODEL_CONFIG['lstm_units'],
    MODEL_CONFIG['dropout_rate']
)

# Compilar modelo
optimizer = Adam(learning_rate=MODEL_CONFIG['learning_rate'])
model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae']
)

# Resumo do modelo
model.summary()

# Plotar arquitetura
plot_model(
    model, 
    to_file=MODEL_PATH / 'model_architecture.png',
    show_shapes=True,
    show_layer_names=True
)
print(f"\nArquitetura salva em: {MODEL_PATH / 'model_architecture.png'}")

# %% [markdown]
# ## 4. Configuração de Callbacks

# %%
# Configurar callbacks
callbacks = [
    # Early Stopping
    EarlyStopping(
        monitor='val_loss',
        patience=MODEL_CONFIG['patience'],
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce Learning Rate
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Model Checkpoint
    ModelCheckpoint(
        filepath=str(MODEL_PATH / 'best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    
    # TensorBoard
    TensorBoard(
        log_dir=str(MODEL_PATH / 'tensorboard_logs'),
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
]

print("Callbacks configurados:")
for callback in callbacks:
    print(f"  - {callback.__class__.__name__}")

# %% [markdown]
# ## 5. Treinamento do Modelo

# %%
# Treinamento
print("Iniciando treinamento...")
print(f"Epochs: {MODEL_CONFIG['epochs']}")
print(f"Batch size: {MODEL_CONFIG['batch_size']}")
print(f"Dados de treino: {X_train_scaled.shape[0]} samples")
print(f"Dados de validação: {X_val_scaled.shape[0]} samples")

history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=MODEL_CONFIG['epochs'],
    batch_size=MODEL_CONFIG['batch_size'],
    callbacks=callbacks,
    verbose=1
)

print("\nTreinamento concluído!")

# %% [markdown]
# ## 6. Visualização do Treinamento

# %%
# Plotar histórico de treinamento
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss
axes[0, 0].plot(history.history['loss'], label='Training Loss')
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
axes[0, 0].set_title('Model Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# MAE
axes[0, 1].plot(history.history['mae'], label='Training MAE')
axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
axes[0, 1].set_title('Model MAE')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Learning Rate (se disponível)
if 'lr' in history.history:
    axes[1, 0].plot(history.history['lr'])
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
else:
    axes[1, 0].text(0.5, 0.5, 'Learning Rate\nhistory not available', 
                    ha='center', va='center', transform=axes[1, 0].transAxes)

# Comparação final
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
final_train_mae = history.history['mae'][-1]
final_val_mae = history.history['val_mae'][-1]

metrics_text = f"""
Métricas Finais:
Train Loss: {final_train_loss:.4f}
Val Loss: {final_val_loss:.4f}
Train MAE: {final_train_mae:.4f}
Val MAE: {final_val_mae:.4f}

Épocas: {len(history.history['loss'])}
"""

axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                fontsize=12, verticalalignment='center')
axes[1, 1].set_title('Métricas Finais')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig(MODEL_PATH / 'training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Gráficos salvos em: {MODEL_PATH / 'training_history.png'}")

# %% [markdown]
# ## 7. Avaliação Inicial

# %%
# Carregar melhor modelo
best_model = tf.keras.models.load_model(MODEL_PATH / 'best_model.h5')

# Previsões
print("Gerando previsões...")
y_pred_train = best_model.predict(X_train_scaled)
y_pred_val = best_model.predict(X_val_scaled)
y_pred_test = best_model.predict(X_test_scaled)

# Desnormalizar previsões
y_pred_train_denorm = target_scaler.inverse_transform(y_pred_train).flatten()
y_pred_val_denorm = target_scaler.inverse_transform(y_pred_val).flatten()
y_pred_test_denorm = target_scaler.inverse_transform(y_pred_test).flatten()

y_train_denorm = target_scaler.inverse_transform(y_train_scaled.reshape(-1, 1)).flatten()
y_val_denorm = target_scaler.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
y_test_denorm = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

# Métricas de regressão
print("\n=== MÉTRICAS DE REGRESSÃO ===")
print("TRAIN:")
print(f"  MAE: {mean_absolute_error(y_train_denorm, y_pred_train_denorm):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_train_denorm, y_pred_train_denorm)):.4f}")

print("VALIDATION:")
val_mae = mean_absolute_error(y_val_denorm, y_pred_val_denorm)
val_rmse = np.sqrt(mean_squared_error(y_val_denorm, y_pred_val_denorm))
print(f"  MAE: {val_mae:.4f}")
print(f"  RMSE: {val_rmse:.4f}")

print("TEST:")
test_mae = mean_absolute_error(y_test_denorm, y_pred_test_denorm)
test_rmse = np.sqrt(mean_squared_error(y_test_denorm, y_pred_test_denorm))
print(f"  MAE: {test_mae:.4f}")
print(f"  RMSE: {test_rmse:.4f}")

# Verificar critérios de sucesso
print("\n=== CRITÉRIOS DE SUCESSO ===")
print(f"MAE < 2.0: {'✓' if test_mae < 2.0 else '✗'} (atual: {test_mae:.4f})")
print(f"RMSE < 3.0: {'✓' if test_rmse < 3.0 else '✗'} (atual: {test_rmse:.4f})")

# %% [markdown]
# ## 8. Salvar Modelo e Configurações

# %%
# Salvar modelo em formato SavedModel
model_save_path = MODEL_PATH / 'lstm_weather_model'
best_model.save(model_save_path)
print(f"Modelo salvo em: {model_save_path}")

# Salvar configurações e metadados
import json
from datetime import datetime

model_metadata = {
    'model_config': MODEL_CONFIG,
    'feature_columns': FEATURE_COLUMNS,
    'target_column': target_column,
    'training_date': datetime.now().isoformat(),
    'tensorflow_version': tf.__version__,
    'model_metrics': {
        'train_mae': float(mean_absolute_error(y_train_denorm, y_pred_train_denorm)),
        'val_mae': float(val_mae),
        'test_mae': float(test_mae),
        'train_rmse': float(np.sqrt(mean_squared_error(y_train_denorm, y_pred_train_denorm))),
        'val_rmse': float(val_rmse),
        'test_rmse': float(test_rmse),
        'epochs_trained': len(history.history['loss'])
    },
    'data_shapes': {
        'train_samples': int(X_train_scaled.shape[0]),
        'val_samples': int(X_val_scaled.shape[0]),
        'test_samples': int(X_test_scaled.shape[0]),
        'sequence_length': int(X_train_scaled.shape[1]),
        'features_count': int(X_train_scaled.shape[2])
    }
}

with open(MODEL_PATH / 'model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print(f"Metadados salvos em: {MODEL_PATH / 'model_metadata.json'}")

# Salvar histórico de treinamento
history_df = pd.DataFrame(history.history)
history_df.to_csv(MODEL_PATH / 'training_history.csv', index=False)
print(f"Histórico salvo em: {MODEL_PATH / 'training_history.csv'}")

print("\n=== MODELO LSTM TREINADO COM SUCESSO! ===")
print(f"Arquivos gerados em: {MODEL_PATH}")
print("\nPróximos passos:")
print("1. Implementar avaliação detalhada (notebooks/model_evaluation.ipynb)")
print("2. Integrar modelo na aplicação (app/features/forecast/infra/)")
print("3. Implementar classificação de eventos de chuva")
print("4. Testar diferentes arquiteturas se necessário")

# %%
