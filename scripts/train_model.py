#!/usr/bin/env python3
"""
Script de Treinamento Automatizado - Sistema de Alertas de Cheias
Sistema de treinamento do modelo LSTM para previsão meteorológica.

Baseado na documentação do projeto:
- Modelo LSTM com precisão > 75% para previsão de chuva 24h
- Features: 16+ variáveis meteorológicas do INMET (2000-2025)
- Arquiteturas configuráveis via linha de comando
- Suporte a grid search e experimentos

Uso:
    python scripts/train_model.py --architecture production
    python scripts/train_model.py --experiment --epochs 20
    python scripts/train_model.py --grid-search --max-combinations 10
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configurações
plt.style.use('seaborn-v0_8')
np.random.seed(42)
tf.random.set_seed(42)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
PROCESSED_DATA_PATH = DATA_PATH / 'processed'
MODELS_PATH = DATA_PATH / 'modelos_treinados'
CONFIGS_PATH = PROJECT_ROOT / 'configs'

# Criar diretórios se não existem
MODELS_PATH.mkdir(parents=True, exist_ok=True)
(MODELS_PATH / 'experiments').mkdir(exist_ok=True)
(MODELS_PATH / 'tensorboard_logs').mkdir(exist_ok=True)


# ========================================
# CONFIGURAÇÕES DE ARQUITETURAS
# ========================================

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
        'description': 'Arquitetura com 2 camadas LSTM - Recomendada para desenvolvimento'
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
        'description': 'Arquitetura balanceada para produção - PADRÃO'
    }
}

# Configuração base
BASE_CONFIG = {
    'sequence_length': 24,
    'forecast_horizon': 24,
    'batch_size': 32,
    'epochs': 100,
    'patience': 15,
    'validation_split': 0.2
}

# Features do INMET baseadas na documentação
FEATURE_COLUMNS = [
    'precipitation', 'pressure', 'temperature', 'dew_point',
    'humidity', 'wind_speed', 'wind_direction', 'radiation',
    'pressure_max', 'pressure_min', 'temp_max', 'temp_min',
    'humidity_max', 'humidity_min', 'dew_point_max', 'dew_point_min'
]

TARGET_COLUMN = 'precipitation'


# ========================================
# FUNÇÕES UTILITÁRIAS
# ========================================

def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega dados processados do INMET
    
    Returns:
        Tuple com (train_data, val_data, test_data)
    """
    logger.info("Carregando dados processados...")
    
    # Procurar arquivos de dados processados
    train_file = PROCESSED_DATA_PATH / 'train_data.csv'
    val_file = PROCESSED_DATA_PATH / 'val_data.csv'
    test_file = PROCESSED_DATA_PATH / 'test_data.csv'
    
    if not all([train_file.exists(), val_file.exists(), test_file.exists()]):
        logger.error("Dados processados não encontrados!")
        logger.error("Execute primeiro: python scripts/data_preprocessing.py")
        raise FileNotFoundError("Dados processados não encontrados")
    
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    test_data = pd.read_csv(test_file)
    
    logger.info(f"Dados carregados - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data


def prepare_sequences(data: pd.DataFrame, feature_cols: List[str], target_col: str, 
                     sequence_length: int, forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepara sequências temporais para treinamento do LSTM
    
    Args:
        data: DataFrame com dados temporais
        feature_cols: Lista de colunas de features
        target_col: Nome da coluna target
        sequence_length: Comprimento da sequência de entrada
        forecast_horizon: Horizonte de previsão
    
    Returns:
        Tuple com (X, y) arrays
    """
    # Verificar se colunas existem
    available_features = [col for col in feature_cols if col in data.columns]
    if len(available_features) != len(feature_cols):
        missing = set(feature_cols) - set(available_features)
        logger.warning(f"Features não encontradas: {missing}")
        logger.info(f"Usando features disponíveis: {available_features}")
        feature_cols = available_features
    
    if target_col not in data.columns:
        logger.error(f"Coluna target '{target_col}' não encontrada!")
        raise ValueError(f"Coluna target '{target_col}' não encontrada")
    
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


def create_lstm_model(sequence_length: int, features_count: int, 
                     lstm_units: List[int], dropout_rate: float, 
                     learning_rate: float) -> tf.keras.Model:
    """
    Cria modelo LSTM com configuração específica
    
    Args:
        sequence_length: Comprimento da sequência de entrada
        features_count: Número de features
        lstm_units: Lista com número de unidades por camada LSTM
        dropout_rate: Taxa de dropout
        learning_rate: Taxa de aprendizado
    
    Returns:
        Modelo Keras compilado
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


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray, 
                  target_scaler: StandardScaler) -> Dict[str, float]:
    """
    Avalia modelo e retorna métricas
    
    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Targets de teste
        target_scaler: Scaler do target
    
    Returns:
        Dicionário com métricas
    """
    # Predições
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Desnormalizar
    y_test_denorm = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_denorm = target_scaler.inverse_transform(y_pred_scaled).flatten()
    
    # Calcular métricas
    mae = mean_absolute_error(y_test_denorm, y_pred_denorm)
    rmse = np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm))
    
    # Classificação de eventos de chuva (threshold: 0.1mm)
    y_test_class = (y_test_denorm > 0.1).astype(int)
    y_pred_class = (y_pred_denorm > 0.1).astype(int)
    
    accuracy = np.mean(y_test_class == y_pred_class)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'accuracy': accuracy
    }


def save_model_artifacts(model: tf.keras.Model, config: Dict[str, Any], 
                        metrics: Dict[str, float], history: tf.keras.callbacks.History,
                        feature_cols: List[str], target_scaler: StandardScaler,
                        feature_scaler: StandardScaler, experiment_name: str = "default"):
    """
    Salva modelo e artefatos relacionados
    
    Args:
        model: Modelo treinado
        config: Configuração usada
        metrics: Métricas de avaliação
        history: Histórico de treinamento
        feature_cols: Colunas de features
        target_scaler: Scaler do target
        feature_scaler: Scaler das features
        experiment_name: Nome do experimento
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = MODELS_PATH / f"{experiment_name}_{timestamp}"
    experiment_dir.mkdir(exist_ok=True)
    
    # Salvar modelo
    model_path = experiment_dir / 'model.h5'
    model.save(str(model_path))
    logger.info(f"Modelo salvo em: {model_path}")
    
    # Salvar best model se for o melhor MAE
    best_model_path = MODELS_PATH / 'best_model.h5'
    if not best_model_path.exists() or is_best_model(metrics['mae']):
        model.save(str(best_model_path))
        logger.info(f"Melhor modelo atualizado: {best_model_path}")
    
    # Salvar metadados
    metadata = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'config': config,
        'metrics': metrics,
        'feature_columns': feature_cols,
        'target_column': TARGET_COLUMN,
        'tensorflow_version': tf.__version__,
        'total_params': int(model.count_params()),
        'epochs_trained': len(history.history['loss'])
    }
    
    with open(experiment_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Salvar histórico
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(experiment_dir / 'training_history.csv', index=False)
    
    # Salvar scalers
    import joblib
    joblib.dump(feature_scaler, experiment_dir / 'feature_scaler.pkl')
    joblib.dump(target_scaler, experiment_dir / 'target_scaler.pkl')
    
    # Atualizar metadata global
    update_global_metadata(metadata)
    
    logger.info(f"Artefatos salvos em: {experiment_dir}")


def is_best_model(mae: float) -> bool:
    """Verifica se é o melhor modelo baseado no MAE"""
    metadata_file = MODELS_PATH / 'model_metadata.json'
    if not metadata_file.exists():
        return True
    
    try:
        with open(metadata_file, 'r') as f:
            current_metadata = json.load(f)
        current_mae = current_metadata.get('model_metrics', {}).get('mae', float('inf'))
        return mae < current_mae
    except:
        return True


def update_global_metadata(metadata: Dict[str, Any]):
    """Atualiza metadata global do melhor modelo"""
    global_metadata = {
        'model_config': metadata['config'],
        'feature_columns': metadata['feature_columns'],
        'target_column': metadata['target_column'],
        'training_date': datetime.now().isoformat(),
        'tensorflow_version': metadata['tensorflow_version'],
        'model_metrics': metadata['metrics'],
        'total_params': metadata['total_params'],
        'epochs_trained': metadata['epochs_trained']
    }
    
    with open(MODELS_PATH / 'model_metadata.json', 'w') as f:
        json.dump(global_metadata, f, indent=2)


# ========================================
# FUNÇÕES DE TREINAMENTO
# ========================================

def train_single_model(architecture: str, config_overrides: Dict[str, Any] = None,
                      experiment: bool = False) -> Dict[str, Any]:
    """
    Treina um único modelo com arquitetura específica
    
    Args:
        architecture: Nome da arquitetura
        config_overrides: Overrides de configuração
        experiment: Se é modo experimental (menos epochs)
    
    Returns:
        Dicionário com resultados
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Arquitetura '{architecture}' não encontrada. Disponíveis: {list(ARCHITECTURES.keys())}")
    
    logger.info(f"=== TREINANDO ARQUITETURA: {architecture} ===")
    
    # Configuração
    arch_config = ARCHITECTURES[architecture].copy()
    config = BASE_CONFIG.copy()
    config.update(arch_config)
    
    if config_overrides:
        config.update(config_overrides)
    
    if experiment:
        config['epochs'] = min(config['epochs'], 30)
        config['patience'] = min(config['patience'], 10)
    
    logger.info(f"Configuração: {config}")
    
    # Carregar dados
    train_data, val_data, test_data = load_processed_data()
    
    # Preparar sequências
    logger.info("Preparando sequências temporais...")
    X_train, y_train = prepare_sequences(
        train_data, FEATURE_COLUMNS, TARGET_COLUMN,
        config['sequence_length'], config['forecast_horizon']
    )
    
    X_val, y_val = prepare_sequences(
        val_data, FEATURE_COLUMNS, TARGET_COLUMN,
        config['sequence_length'], config['forecast_horizon']
    )
    
    X_test, y_test = prepare_sequences(
        test_data, FEATURE_COLUMNS, TARGET_COLUMN,
        config['sequence_length'], config['forecast_horizon']
    )
    
    logger.info(f"Sequências criadas - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Normalizar dados
    logger.info("Normalizando dados...")
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
    logger.info("Criando modelo LSTM...")
    model = create_lstm_model(
        config['sequence_length'],
        len(FEATURE_COLUMNS),
        config['lstm_units'],
        config['dropout_rate'],
        config['learning_rate']
    )
    
    logger.info(f"Modelo criado com {model.count_params():,} parâmetros")
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_dir = MODELS_PATH / 'tensorboard_logs' / f"{architecture}_{timestamp}"
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config['patience']//2,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    # Treinamento
    logger.info("Iniciando treinamento...")
    start_time = time.time()
    
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    logger.info(f"Treinamento concluído em {training_time:.1f}s")
    
    # Avaliação
    logger.info("Avaliando modelo...")
    metrics = evaluate_model(model, X_test_scaled, y_test_scaled, target_scaler)
    
    logger.info(f"Métricas finais:")
    logger.info(f"  MAE: {metrics['mae']:.4f} mm/h")
    logger.info(f"  RMSE: {metrics['rmse']:.4f} mm/h")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    
    # Verificar critérios de sucesso
    success_criteria = {
        'MAE < 2.0': metrics['mae'] < 2.0,
        'RMSE < 3.0': metrics['rmse'] < 3.0,
        'Accuracy > 75%': metrics['accuracy'] > 0.75
    }
    
    logger.info("Critérios de sucesso:")
    for criterion, passed in success_criteria.items():
        status = "✓" if passed else "✗"
        logger.info(f"  {status} {criterion}")
    
    # Salvar artefatos
    save_model_artifacts(
        model, config, metrics, history,
        FEATURE_COLUMNS, target_scaler, feature_scaler,
        f"{architecture}_{'exp' if experiment else 'full'}"
    )
    
    return {
        'architecture': architecture,
        'config': config,
        'metrics': metrics,
        'training_time': training_time,
        'epochs_trained': len(history.history['loss']),
        'success_criteria': success_criteria
    }


def run_grid_search(architectures: List[str] = None, max_combinations: int = 10):
    """
    Executa grid search com múltiplas arquiteturas
    
    Args:
        architectures: Lista de arquiteturas para testar
        max_combinations: Máximo de combinações por arquitetura
    """
    if architectures is None:
        architectures = ['simple_2_layers', 'production']
    
    logger.info(f"=== GRID SEARCH: {architectures} ===")
    
    # Parâmetros para grid search
    param_grid = {
        'sequence_length': [12, 24, 48],
        'batch_size': [16, 32, 64],
        'learning_rate': [0.001, 0.0001]
    }
    
    # Gerar combinações
    import itertools
    param_combinations = list(itertools.product(*param_grid.values()))
    param_combinations = param_combinations[:max_combinations]
    
    logger.info(f"Testando {len(param_combinations)} combinações para {len(architectures)} arquiteturas")
    
    results = []
    
    for arch in architectures:
        for i, combination in enumerate(param_combinations):
            config_override = dict(zip(param_grid.keys(), combination))
            
            logger.info(f"\n--- {arch} - Combinação {i+1}/{len(param_combinations)} ---")
            logger.info(f"Parâmetros: {config_override}")
            
            try:
                result = train_single_model(arch, config_override, experiment=True)
                result['param_combination'] = config_override
                results.append(result)
            except Exception as e:
                logger.error(f"Erro na combinação {i+1}: {e}")
                continue
    
    # Salvar resultados do grid search
    results_df = pd.DataFrame([
        {
            'architecture': r['architecture'],
            'sequence_length': r['param_combination']['sequence_length'],
            'batch_size': r['param_combination']['batch_size'],
            'learning_rate': r['param_combination']['learning_rate'],
            'mae': r['metrics']['mae'],
            'rmse': r['metrics']['rmse'],
            'accuracy': r['metrics']['accuracy'],
            'training_time': r['training_time']
        }
        for r in results
    ])
    
    results_path = MODELS_PATH / 'experiments' / 'grid_search_results.csv'
    results_df.to_csv(results_path, index=False)
    
    # Melhor configuração
    best_result = min(results, key=lambda x: x['metrics']['mae'])
    logger.info(f"\n=== MELHOR CONFIGURAÇÃO ===")
    logger.info(f"Arquitetura: {best_result['architecture']}")
    logger.info(f"Parâmetros: {best_result['param_combination']}")
    logger.info(f"MAE: {best_result['metrics']['mae']:.4f}")
    
    return results


# ========================================
# FUNÇÃO PRINCIPAL
# ========================================

def main():
    """Função principal do script"""
    parser = argparse.ArgumentParser(
        description='Script de treinamento do modelo LSTM para previsão meteorológica'
    )
    
    parser.add_argument(
        '--architecture', '-a',
        default='production',
        choices=list(ARCHITECTURES.keys()),
        help='Arquitetura do modelo para treinar'
    )
    
    parser.add_argument(
        '--experiment', '-e',
        action='store_true',
        help='Modo experimental (menos epochs)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Número de epochs (override)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Tamanho do batch (override)'
    )
    
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        help='Taxa de aprendizado (override)'
    )
    
    parser.add_argument(
        '--sequence-length', '-seq',
        type=int,
        help='Comprimento da sequência (override)'
    )
    
    parser.add_argument(
        '--grid-search', '-g',
        action='store_true',
        help='Executar grid search'
    )
    
    parser.add_argument(
        '--max-combinations',
        type=int,
        default=10,
        help='Máximo de combinações no grid search'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Output verbose'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Verificar TensorFlow
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"GPU Available: {len(tf.config.list_physical_devices('GPU'))} devices")
    
    try:
        if args.grid_search:
            # Grid search
            run_grid_search(max_combinations=args.max_combinations)
        else:
            # Treinamento único
            config_overrides = {}
            
            if args.epochs:
                config_overrides['epochs'] = args.epochs
            if args.batch_size:
                config_overrides['batch_size'] = args.batch_size
            if args.learning_rate:
                config_overrides['learning_rate'] = args.learning_rate
            if args.sequence_length:
                config_overrides['sequence_length'] = args.sequence_length
            
            result = train_single_model(
                args.architecture, 
                config_overrides,
                args.experiment
            )
            
            logger.info("\n=== TREINAMENTO CONCLUÍDO COM SUCESSO! ===")
            logger.info(f"Arquitetura: {result['architecture']}")
            logger.info(f"MAE: {result['metrics']['mae']:.4f} mm/h")
            logger.info(f"RMSE: {result['metrics']['rmse']:.4f} mm/h")
            logger.info(f"Accuracy: {result['metrics']['accuracy']*100:.1f}%")
            
            # Verificar se atendeu critérios de sucesso
            success_count = sum(result['success_criteria'].values())
            total_criteria = len(result['success_criteria'])
            
            if success_count == total_criteria:
                logger.info("🎉 TODOS OS CRITÉRIOS DE SUCESSO FORAM ATENDIDOS!")
            else:
                logger.warning(f"⚠️  Atendeu {success_count}/{total_criteria} critérios de sucesso")
            
            logger.info(f"Modelo salvo em: {MODELS_PATH}")
            logger.info("Execute 'make tensorboard' para visualizar métricas")
    
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 