# %% [markdown]
# # Experimentos de Arquitetura LSTM - Otimização de Hiperparâmetros
#
# Este notebook implementa experimentos sistemáticos para encontrar a melhor arquitetura LSTM para previsão meteorológica.
#
# ## Objetivos:
# - Testar diferentes configurações de arquitetura LSTM
# - Grid search automatizado para hiperparâmetros
# - Comparação de performance entre arquiteturas
# - Análise de trade-offs entre complexidade e performance

# %%
# Imports necessários
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import itertools
import json
from datetime import datetime
import time
warnings.filterwarnings('ignore')

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configurações
plt.style.use('seaborn-v0_8')
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# %%
# Configurações base
BASE_CONFIG = {
    'sequence_length': 24,
    'forecast_horizon': 24,
    'batch_size': 32,
    'epochs': 50,  # Reduzido para experimentos
    'patience': 10,
    'validation_split': 0.2
}

# Paths
DATA_PATH = Path('../data/processed')
MODEL_PATH = Path('../data/modelos_treinados')
EXPERIMENTS_PATH = MODEL_PATH / 'experiments'
EXPERIMENTS_PATH.mkdir(exist_ok=True)

print(f"Configuração base: {BASE_CONFIG}")
print(f"Experimentos serão salvos em: {EXPERIMENTS_PATH}")

# %% [markdown]
# ## 1. Definição de Arquiteturas para Teste

# %%
# Definir diferentes arquiteturas para teste
ARCHITECTURES = {
    'simple_1_layer': {
        'lstm_units': [64],
        'dropout_rate': 0.1,
        'learning_rate': 0.001,
        'description': 'Arquitetura simples com 1 camada LSTM'
    },
    'simple_2_layers': {
        'lstm_units': [128, 64],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'description': 'Arquitetura com 2 camadas LSTM'
    },
    'simple_3_layers': {
        'lstm_units': [256, 128, 64],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'description': 'Arquitetura com 3 camadas LSTM'
    },
    'heavy_2_layers': {
        'lstm_units': [256, 128],
        'dropout_rate': 0.3,
        'learning_rate': 0.0001,
        'description': 'Arquitetura pesada com 2 camadas'
    },
    'light_3_layers': {
        'lstm_units': [64, 32, 16],
        'dropout_rate': 0.1,
        'learning_rate': 0.01,
        'description': 'Arquitetura leve com 3 camadas'
    },
    'production': {
        'lstm_units': [128, 64, 32],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'description': 'Arquitetura balanceada para produção'
    }
}

print("Arquiteturas definidas:")
for name, config in ARCHITECTURES.items():
    print(f"  {name}: {config['lstm_units']} - {config['description']}")

# %% [markdown]
# ## 2. Grid Search Parameters

# %%
# Parâmetros para grid search
GRID_SEARCH_PARAMS = {
    'sequence_lengths': [12, 24, 48],
    'learning_rates': [0.01, 0.001, 0.0001],
    'batch_sizes': [16, 32, 64],
    'dropout_rates': [0.1, 0.2, 0.3]
}

print("Parâmetros para Grid Search:")
for param, values in GRID_SEARCH_PARAMS.items():
    print(f"  {param}: {values}")

total_combinations = np.prod([len(v) for v in GRID_SEARCH_PARAMS.values()])
print(f"\nTotal de combinações possíveis: {total_combinations}")

# %% [markdown]
# ## 3. Funções Utilitárias

# %%
def create_lstm_model(sequence_length, features_count, lstm_units, dropout_rate, learning_rate):
    """
    Cria modelo LSTM com configuração específica
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
    
    # Compilar
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def prepare_sequences(data, feature_cols, target_col, sequence_length, forecast_horizon):
    """
    Prepara sequências temporais para treinamento
    """
    if 'timestamp' in data.columns:
        data = data.sort_values('timestamp')
    
    features = data[feature_cols].values
    target = data[target_col].values
    
    X, y = [], []
    
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(features[i:(i + sequence_length)])
        target_idx = i + sequence_length + forecast_horizon - 1
        y.append(target[target_idx])
    
    return np.array(X), np.array(y)

def evaluate_model(model, X_test, y_test, target_scaler):
    """
    Avalia modelo e retorna métricas
    """
    y_pred = model.predict(X_test, verbose=0)
    
    # Desnormalizar
    y_pred_denorm = target_scaler.inverse_transform(y_pred).flatten()
    y_test_denorm = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calcular métricas
    mae = mean_absolute_error(y_test_denorm, y_pred_denorm)
    rmse = np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mae_normalized': mean_absolute_error(y_test, y_pred.flatten()),
        'rmse_normalized': np.sqrt(mean_squared_error(y_test, y_pred.flatten()))
    }

# %% [markdown]
# ## 4. Carregamento de Dados

# %%
# Carregar dados processados
print("Carregando dados...")
train_data = pd.read_parquet(DATA_PATH / 'train_data.parquet')
val_data = pd.read_parquet(DATA_PATH / 'validation_data.parquet')
test_data = pd.read_parquet(DATA_PATH / 'test_data.parquet')

print(f"Train shape: {train_data.shape}")
print(f"Validation shape: {val_data.shape}")
print(f"Test shape: {test_data.shape}")

# Identificar features e target
feature_columns = [col for col in train_data.columns if col not in ['timestamp']]
target_column = None

# Procurar coluna de precipitação
for col in train_data.columns:
    if 'precipitacao' in col.lower() or 'chuva' in col.lower():
        target_column = col
        break

if target_column is None:
    target_column = feature_columns[0]
    print(f"ATENÇÃO: Usando {target_column} como target")

# Remover target das features
if target_column in feature_columns:
    feature_columns.remove(target_column)

print(f"Features: {len(feature_columns)} colunas")
print(f"Target: {target_column}")

# %% [markdown]
# ## 5. Experimento 1: Comparação de Arquiteturas

# %%
def run_architecture_experiment():
    """
    Executa experimento comparando diferentes arquiteturas
    """
    print("=== EXPERIMENTO 1: COMPARAÇÃO DE ARQUITETURAS ===")
    
    results = []
    
    for arch_name, arch_config in ARCHITECTURES.items():
        print(f"\nTestando arquitetura: {arch_name}")
        print(f"Configuração: {arch_config}")
        
        start_time = time.time()
        
        try:
            # Preparar dados com configuração base
            X_train, y_train = prepare_sequences(
                train_data, feature_columns, target_column,
                BASE_CONFIG['sequence_length'], BASE_CONFIG['forecast_horizon']
            )
            
            X_val, y_val = prepare_sequences(
                val_data, feature_columns, target_column,
                BASE_CONFIG['sequence_length'], BASE_CONFIG['forecast_horizon']
            )
            
            X_test, y_test = prepare_sequences(
                test_data, feature_columns, target_column,
                BASE_CONFIG['sequence_length'], BASE_CONFIG['forecast_horizon']
            )
            
            # Normalizar dados
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            
            feature_scaler = StandardScaler()
            X_train_scaled = feature_scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
            X_val_scaled = feature_scaler.transform(X_val_reshaped).reshape(X_val.shape)
            X_test_scaled = feature_scaler.transform(X_test_reshaped).reshape(X_test.shape)
            
            target_scaler = StandardScaler()
            y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
            y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
            
            # Criar modelo
            model = create_lstm_model(
                BASE_CONFIG['sequence_length'],
                len(feature_columns),
                arch_config['lstm_units'],
                arch_config['dropout_rate'],
                arch_config['learning_rate']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=BASE_CONFIG['patience'], 
                            restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, 
                                min_lr=1e-7, verbose=0)
            ]
            
            # Treinar
            history = model.fit(
                X_train_scaled, y_train_scaled,
                validation_data=(X_val_scaled, y_val_scaled),
                epochs=BASE_CONFIG['epochs'],
                batch_size=BASE_CONFIG['batch_size'],
                callbacks=callbacks,
                verbose=0
            )
            
            # Avaliar
            metrics = evaluate_model(model, X_test_scaled, y_test_scaled, target_scaler)
            
            # Calcular parâmetros do modelo
            total_params = model.count_params()
            
            training_time = time.time() - start_time
            
            result = {
                'architecture': arch_name,
                'lstm_units': arch_config['lstm_units'],
                'dropout_rate': arch_config['dropout_rate'],
                'learning_rate': arch_config['learning_rate'],
                'total_params': total_params,
                'training_time': training_time,
                'epochs_trained': len(history.history['loss']),
                'final_train_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'final_train_mae': history.history['mae'][-1],
                'final_val_mae': history.history['val_mae'][-1],
                **metrics
            }
            
            results.append(result)
            
            print(f"  ✓ Concluído em {training_time:.1f}s")
            print(f"  ✓ MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
            print(f"  ✓ Parâmetros: {total_params:,}")
            
        except Exception as e:
            print(f"  ✗ Erro: {str(e)}")
            continue
    
    # Salvar resultados
    results_df = pd.DataFrame(results)
    results_df.to_csv(EXPERIMENTS_PATH / 'architecture_comparison.csv', index=False)
    
    return results_df

# Executar experimento
architecture_results = run_architecture_experiment()

# %% [markdown]
# ## 6. Análise dos Resultados de Arquitetura

# %%
if not architecture_results.empty:
    print("\n=== ANÁLISE DOS RESULTADOS DE ARQUITETURA ===")
    
    # Ordenar por MAE
    architecture_results_sorted = architecture_results.sort_values('mae')
    
    print("\nTop 3 arquiteturas por MAE:")
    for i, (_, row) in enumerate(architecture_results_sorted.head(3).iterrows()):
        print(f"{i+1}. {row['architecture']}")
        print(f"   MAE: {row['mae']:.4f}, RMSE: {row['rmse']:.4f}")
        print(f"   Parâmetros: {row['total_params']:,}")
        print(f"   Tempo: {row['training_time']:.1f}s")
    
    # Visualizações
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # MAE vs RMSE
    axes[0, 0].scatter(architecture_results['mae'], architecture_results['rmse'])
    for i, row in architecture_results.iterrows():
        axes[0, 0].annotate(row['architecture'], 
                           (row['mae'], row['rmse']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 0].set_xlabel('MAE')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('MAE vs RMSE por Arquitetura')
    axes[0, 0].grid(True)
    
    # Parâmetros vs Performance
    axes[0, 1].scatter(architecture_results['total_params'], architecture_results['mae'])
    for i, row in architecture_results.iterrows():
        axes[0, 1].annotate(row['architecture'], 
                           (row['total_params'], row['mae']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 1].set_xlabel('Total Parameters')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Complexidade vs Performance')
    axes[0, 1].grid(True)
    
    # Tempo de treinamento
    axes[1, 0].bar(architecture_results['architecture'], architecture_results['training_time'])
    axes[1, 0].set_xlabel('Arquitetura')
    axes[1, 0].set_ylabel('Tempo (s)')
    axes[1, 0].set_title('Tempo de Treinamento')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Comparação de métricas
    metrics_comparison = architecture_results[['architecture', 'mae', 'rmse']].set_index('architecture')
    metrics_comparison.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Arquitetura')
    axes[1, 1].set_ylabel('Valor')
    axes[1, 1].set_title('Comparação MAE vs RMSE')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(EXPERIMENTS_PATH / 'architecture_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 7. Experimento 2: Grid Search de Hiperparâmetros

# %%
def run_grid_search_experiment(max_combinations=20):
    """
    Executa grid search limitado para hiperparâmetros
    """
    print(f"\n=== EXPERIMENTO 2: GRID SEARCH (máximo {max_combinations} combinações) ===")
    
    # Gerar todas as combinações
    param_combinations = list(itertools.product(
        GRID_SEARCH_PARAMS['sequence_lengths'],
        GRID_SEARCH_PARAMS['learning_rates'],
        GRID_SEARCH_PARAMS['batch_sizes'],
        GRID_SEARCH_PARAMS['dropout_rates']
    ))
    
    # Limitar número de combinações
    if len(param_combinations) > max_combinations:
        param_combinations = np.random.choice(
            len(param_combinations), max_combinations, replace=False
        )
        param_combinations = [list(itertools.product(
            GRID_SEARCH_PARAMS['sequence_lengths'],
            GRID_SEARCH_PARAMS['learning_rates'],
            GRID_SEARCH_PARAMS['batch_sizes'],
            GRID_SEARCH_PARAMS['dropout_rates']
        ))[i] for i in param_combinations]
    
    print(f"Testando {len(param_combinations)} combinações de hiperparâmetros...")
    
    results = []
    
    for i, (seq_len, lr, batch_size, dropout) in enumerate(param_combinations):
        print(f"\nCombinação {i+1}/{len(param_combinations)}")
        print(f"seq_len={seq_len}, lr={lr}, batch_size={batch_size}, dropout={dropout}")
        
        start_time = time.time()
        
        try:
            # Preparar dados com sequence_length específico
            X_train, y_train = prepare_sequences(
                train_data, feature_columns, target_column,
                seq_len, BASE_CONFIG['forecast_horizon']
            )
            
            X_val, y_val = prepare_sequences(
                val_data, feature_columns, target_column,
                seq_len, BASE_CONFIG['forecast_horizon']
            )
            
            X_test, y_test = prepare_sequences(
                test_data, feature_columns, target_column,
                seq_len, BASE_CONFIG['forecast_horizon']
            )
            
            # Normalizar
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            
            feature_scaler = StandardScaler()
            X_train_scaled = feature_scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
            X_val_scaled = feature_scaler.transform(X_val_reshaped).reshape(X_val.shape)
            X_test_scaled = feature_scaler.transform(X_test_reshaped).reshape(X_test.shape)
            
            target_scaler = StandardScaler()
            y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
            y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
            
            # Usar arquitetura padrão
            model = create_lstm_model(
                seq_len, len(feature_columns),
                [128, 64], dropout, lr
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, 
                            restore_best_weights=True, verbose=0)
            ]
            
            # Treinar
            history = model.fit(
                X_train_scaled, y_train_scaled,
                validation_data=(X_val_scaled, y_val_scaled),
                epochs=30,  # Reduzido para grid search
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # Avaliar
            metrics = evaluate_model(model, X_test_scaled, y_test_scaled, target_scaler)
            
            training_time = time.time() - start_time
            
            result = {
                'sequence_length': seq_len,
                'learning_rate': lr,
                'batch_size': batch_size,
                'dropout_rate': dropout,
                'training_time': training_time,
                'epochs_trained': len(history.history['loss']),
                'final_val_loss': history.history['val_loss'][-1],
                'final_val_mae': history.history['val_mae'][-1],
                **metrics
            }
            
            results.append(result)
            
            print(f"  ✓ MAE: {metrics['mae']:.4f} em {training_time:.1f}s")
            
        except Exception as e:
            print(f"  ✗ Erro: {str(e)}")
            continue
    
    # Salvar resultados
    results_df = pd.DataFrame(results)
    results_df.to_csv(EXPERIMENTS_PATH / 'grid_search_results.csv', index=False)
    
    return results_df

# Executar grid search
grid_search_results = run_grid_search_experiment(max_combinations=15)

# %% [markdown]
# ## 8. Análise do Grid Search

# %%
if not grid_search_results.empty:
    print("\n=== ANÁLISE DO GRID SEARCH ===")
    
    # Melhor combinação
    best_combination = grid_search_results.loc[grid_search_results['mae'].idxmin()]
    
    print("\nMelhor combinação de hiperparâmetros:")
    print(f"  Sequence Length: {best_combination['sequence_length']}")
    print(f"  Learning Rate: {best_combination['learning_rate']}")
    print(f"  Batch Size: {best_combination['batch_size']}")
    print(f"  Dropout Rate: {best_combination['dropout_rate']}")
    print(f"  MAE: {best_combination['mae']:.4f}")
    print(f"  RMSE: {best_combination['rmse']:.4f}")
    
    # Análise por parâmetro
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Sequence Length
    seq_len_analysis = grid_search_results.groupby('sequence_length')['mae'].agg(['mean', 'std'])
    seq_len_analysis['mean'].plot(kind='bar', ax=axes[0, 0], yerr=seq_len_analysis['std'])
    axes[0, 0].set_title('MAE por Sequence Length')
    axes[0, 0].set_ylabel('MAE')
    
    # Learning Rate
    lr_analysis = grid_search_results.groupby('learning_rate')['mae'].agg(['mean', 'std'])
    lr_analysis['mean'].plot(kind='bar', ax=axes[0, 1], yerr=lr_analysis['std'])
    axes[0, 1].set_title('MAE por Learning Rate')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_xscale('log')
    
    # Batch Size
    batch_analysis = grid_search_results.groupby('batch_size')['mae'].agg(['mean', 'std'])
    batch_analysis['mean'].plot(kind='bar', ax=axes[1, 0], yerr=batch_analysis['std'])
    axes[1, 0].set_title('MAE por Batch Size')
    axes[1, 0].set_ylabel('MAE')
    
    # Dropout Rate
    dropout_analysis = grid_search_results.groupby('dropout_rate')['mae'].agg(['mean', 'std'])
    dropout_analysis['mean'].plot(kind='bar', ax=axes[1, 1], yerr=dropout_analysis['std'])
    axes[1, 1].set_title('MAE por Dropout Rate')
    axes[1, 1].set_ylabel('MAE')
    
    plt.tight_layout()
    plt.savefig(EXPERIMENTS_PATH / 'grid_search_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 9. Relatório Final dos Experimentos

# %%
def generate_experiment_report():
    """
    Gera relatório final dos experimentos
    """
    report = {
        'experiment_date': datetime.now().isoformat(),
        'base_config': BASE_CONFIG,
        'architectures_tested': len(ARCHITECTURES),
        'grid_search_combinations': len(grid_search_results) if not grid_search_results.empty else 0,
        'best_architecture': None,
        'best_hyperparams': None,
        'recommendations': []
    }
    
    # Melhor arquitetura
    if not architecture_results.empty:
        best_arch = architecture_results.loc[architecture_results['mae'].idxmin()]
        report['best_architecture'] = {
            'name': best_arch['architecture'],
            'mae': float(best_arch['mae']),
            'rmse': float(best_arch['rmse']),
            'params': int(best_arch['total_params']),
            'config': {
                'lstm_units': best_arch['lstm_units'],
                'dropout_rate': float(best_arch['dropout_rate']),
                'learning_rate': float(best_arch['learning_rate'])
            }
        }
    
    # Melhores hiperparâmetros
    if not grid_search_results.empty:
        best_params = grid_search_results.loc[grid_search_results['mae'].idxmin()]
        report['best_hyperparams'] = {
            'sequence_length': int(best_params['sequence_length']),
            'learning_rate': float(best_params['learning_rate']),
            'batch_size': int(best_params['batch_size']),
            'dropout_rate': float(best_params['dropout_rate']),
            'mae': float(best_params['mae']),
            'rmse': float(best_params['rmse'])
        }
    
    # Recomendações
    recommendations = []
    
    if not architecture_results.empty:
        # Análise de complexidade vs performance
        complexity_performance = architecture_results['total_params'] / architecture_results['mae']
        best_efficiency = architecture_results.loc[complexity_performance.idxmax()]
        
        recommendations.append(f"Arquitetura mais eficiente: {best_efficiency['architecture']}")
        recommendations.append(f"Melhor performance: {architecture_results.loc[architecture_results['mae'].idxmin()]['architecture']}")
    
    if not grid_search_results.empty:
        # Análise de hiperparâmetros
        if len(grid_search_results['sequence_length'].unique()) > 1:
            best_seq_len = grid_search_results.groupby('sequence_length')['mae'].mean().idxmin()
            recommendations.append(f"Melhor sequence length: {best_seq_len}")
        
        if len(grid_search_results['learning_rate'].unique()) > 1:
            best_lr = grid_search_results.groupby('learning_rate')['mae'].mean().idxmin()
            recommendations.append(f"Melhor learning rate: {best_lr}")
    
    report['recommendations'] = recommendations
    
    # Salvar relatório
    with open(EXPERIMENTS_PATH / 'experiment_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

# Gerar relatório
final_report = generate_experiment_report()

print("\n=== RELATÓRIO FINAL DOS EXPERIMENTOS ===")
print(f"Data: {final_report['experiment_date']}")
print(f"Arquiteturas testadas: {final_report['architectures_tested']}")
print(f"Combinações de grid search: {final_report['grid_search_combinations']}")

if final_report['best_architecture']:
    print(f"\nMelhor arquitetura: {final_report['best_architecture']['name']}")
    print(f"  MAE: {final_report['best_architecture']['mae']:.4f}")
    print(f"  RMSE: {final_report['best_architecture']['rmse']:.4f}")

if final_report['best_hyperparams']:
    print(f"\nMelhores hiperparâmetros:")
    for param, value in final_report['best_hyperparams'].items():
        if param not in ['mae', 'rmse']:
            print(f"  {param}: {value}")

print(f"\nRecomendações:")
for rec in final_report['recommendations']:
    print(f"  - {rec}")

print(f"\nArquivos gerados em: {EXPERIMENTS_PATH}")
print("  - architecture_comparison.csv")
print("  - grid_search_results.csv")
print("  - experiment_report.json")
print("  - architecture_analysis.png")
print("  - grid_search_analysis.png")

# %%
