#!/usr/bin/env python3
"""
Fase 7: Treinamento do Modelo LSTM para Previsão de Cheias

Este script treina um modelo LSTM para previsão de precipitação baseado nos dados históricos preparados.

Funcionalidades:
- Carregamento dos dados processados
- Normalização das features
- Construção do modelo LSTM
- Treinamento com early stopping
- Validação e métricas de performance
- Salvamento do modelo treinado

Autor: Sistema de Alertas de Cheias - Rio Guaíba
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging
import joblib

# TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

# Scikit-learn
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurações
PROCESSED_DATA_PATH = Path("data/processed")
MODELS_PATH = Path("data/models")
LOGS_PATH = Path("logs/training")

# Criar diretórios
MODELS_PATH.mkdir(exist_ok=True, parents=True)
LOGS_PATH.mkdir(exist_ok=True, parents=True)


class LSTMTrainer:
    """Classe principal para treinamento do modelo LSTM"""
    
    def __init__(self):
        self.model = None
        self.scalers = {}
        self.training_history = None
        self.metadata = None
        
        # Verificar dependências
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow não está instalado. Execute: pip install tensorflow")
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn não está instalado. Execute: pip install scikit-learn")
        
        # Configurar TensorFlow
        self._setup_tensorflow()
    
    def _setup_tensorflow(self):
        """Configura TensorFlow para performance"""
        # Configurar GPU se disponível
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"🚀 GPU configurada: {len(gpus)} dispositivos")
            except RuntimeError as e:
                logger.warning(f"⚠️ Erro ao configurar GPU: {e}")
        else:
            logger.info("💻 Usando CPU para treinamento")
        
        # Configurar seeds para reproducibilidade
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def load_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Carrega dados processados"""
        logger.info("📂 Carregando dados processados...")
        
        # Verificar se arquivos existem
        required_files = [
            'train_X.npy', 'train_y.npy',
            'validation_X.npy', 'validation_y.npy', 
            'test_X.npy', 'test_y.npy',
            'metadata.json'
        ]
        
        for file in required_files:
            if not (PROCESSED_DATA_PATH / file).exists():
                raise FileNotFoundError(f"Arquivo necessário não encontrado: {file}")
        
        # Carregar arrays
        data = {
            'train': {
                'X': np.load(PROCESSED_DATA_PATH / 'train_X.npy'),
                'y': np.load(PROCESSED_DATA_PATH / 'train_y.npy')
            },
            'validation': {
                'X': np.load(PROCESSED_DATA_PATH / 'validation_X.npy'),
                'y': np.load(PROCESSED_DATA_PATH / 'validation_y.npy')
            },
            'test': {
                'X': np.load(PROCESSED_DATA_PATH / 'test_X.npy'),
                'y': np.load(PROCESSED_DATA_PATH / 'test_y.npy')
            }
        }
        
        # Carregar metadados
        with open(PROCESSED_DATA_PATH / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Log estatísticas
        for split_name, split_data in data.items():
            logger.info(f"📊 {split_name.upper()}: X={split_data['X'].shape}, y={split_data['y'].shape}")
        
        return data
    
    def normalize_data(self, data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Normaliza features e targets"""
        logger.info("📏 Normalizando dados...")
        
        # Normalizar features (X)
        X_train = data['train']['X']
        X_val = data['validation']['X']
        X_test = data['test']['X']
        
        # Reshape para normalização (samples * timesteps, features)
        original_shape_train = X_train.shape
        original_shape_val = X_val.shape
        original_shape_test = X_test.shape
        
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        
        # Scaler para features
        feature_scaler = StandardScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train_reshaped)
        X_val_scaled = feature_scaler.transform(X_val_reshaped)
        X_test_scaled = feature_scaler.transform(X_test_reshaped)
        
        # Reshape de volta
        X_train_scaled = X_train_scaled.reshape(original_shape_train)
        X_val_scaled = X_val_scaled.reshape(original_shape_val)
        X_test_scaled = X_test_scaled.reshape(original_shape_test)
        
        # Normalizar targets (y) 
        y_train = data['train']['y'].reshape(-1, 1)
        y_val = data['validation']['y'].reshape(-1, 1)
        y_test = data['test']['y'].reshape(-1, 1)
        
        # Scaler para target (usar RobustScaler para outliers de precipitação)
        target_scaler = RobustScaler()
        y_train_scaled = target_scaler.fit_transform(y_train).flatten()
        y_val_scaled = target_scaler.transform(y_val).flatten()
        y_test_scaled = target_scaler.transform(y_test).flatten()
        
        # Salvar scalers
        self.scalers = {
            'features': feature_scaler,
            'target': target_scaler
        }
        
        # Dados normalizados
        normalized_data = {
            'train': {'X': X_train_scaled, 'y': y_train_scaled},
            'validation': {'X': X_val_scaled, 'y': y_val_scaled},
            'test': {'X': X_test_scaled, 'y': y_test_scaled}
        }
        
        logger.info("✅ Normalização concluída")
        return normalized_data
    
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Constrói o modelo LSTM"""
        logger.info(f"🏗️ Construindo modelo LSTM para input {input_shape}...")
        
        model = Sequential([
            # Primeira camada LSTM
            LSTM(
                128, 
                return_sequences=True,
                input_shape=input_shape,
                dropout=0.2,
                recurrent_dropout=0.2,
                kernel_regularizer=l2(0.001),
                name='lstm_1'
            ),
            BatchNormalization(),
            
            # Segunda camada LSTM  
            LSTM(
                64,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2,
                kernel_regularizer=l2(0.001),
                name='lstm_2'
            ),
            BatchNormalization(),
            
            # Terceira camada LSTM
            LSTM(
                32,
                dropout=0.2,
                recurrent_dropout=0.2,
                kernel_regularizer=l2(0.001),
                name='lstm_3'
            ),
            BatchNormalization(),
            
            # Camadas densas
            Dense(50, activation='relu', kernel_regularizer=l2(0.001), name='dense_1'),
            Dropout(0.3),
            Dense(25, activation='relu', kernel_regularizer=l2(0.001), name='dense_2'),
            Dropout(0.2),
            
            # Saída (precipitação - não pode ser negativa)
            Dense(1, activation='linear', name='output')
        ])
        
        # Compilar modelo
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(
            optimizer=optimizer,
            loss='huber',  # Robusta para outliers
            metrics=['mae', 'mse']
        )
        
        # Resumo do modelo
        model.summary()
        logger.info("✅ Modelo construído")
        
        return model
    
    def create_callbacks(self, model_name: str) -> List:
        """Cria callbacks para treinamento"""
        
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Redução de learning rate
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Checkpoint do melhor modelo
            ModelCheckpoint(
                MODELS_PATH / f'{model_name}_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard
            TensorBoard(
                log_dir=LOGS_PATH / f'{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        return callbacks
    
    def train_model(self, data: Dict[str, Dict[str, np.ndarray]], 
                   epochs: int = 100, batch_size: int = 32, 
                   model_name: str = "lstm_precipitation") -> Dict[str, Any]:
        """Treina o modelo LSTM"""
        logger.info("🚂 Iniciando treinamento do modelo...")
        
        # Construir modelo
        input_shape = (data['train']['X'].shape[1], data['train']['X'].shape[2])
        self.model = self.build_model(input_shape)
        
        # Callbacks
        callbacks = self.create_callbacks(model_name)
        
        # Treinamento
        history = self.model.fit(
            data['train']['X'], data['train']['y'],
            validation_data=(data['validation']['X'], data['validation']['y']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        self.training_history = history.history
        
        logger.info("✅ Treinamento concluído!")
        return self.training_history
    
    def evaluate_model(self, data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Avalia o modelo treinado"""
        logger.info("📊 Avaliando modelo...")
        
        results = {}
        
        for split_name, split_data in data.items():
            # Predições
            y_pred_scaled = self.model.predict(split_data['X'], verbose=0)
            y_true_scaled = split_data['y']
            
            # Desnormalizar para métricas reais
            y_pred = self.scalers['target'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = self.scalers['target'].inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
            
            # Métricas
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            # Métricas específicas para precipitação
            # Acurácia para detecção de chuva (threshold = 0.1mm)
            rain_threshold = 0.1
            rain_true = (y_true >= rain_threshold).astype(int)
            rain_pred = (y_pred >= rain_threshold).astype(int)
            rain_accuracy = np.mean(rain_true == rain_pred)
            
            # Bias
            bias = np.mean(y_pred - y_true)
            
            results[split_name] = {
                'mae': float(mae),
                'mse': float(mse), 
                'rmse': float(rmse),
                'r2': float(r2),
                'rain_accuracy': float(rain_accuracy),
                'bias': float(bias),
                'n_samples': len(y_true)
            }
            
            logger.info(f"📈 {split_name.upper()}:")
            logger.info(f"   MAE: {mae:.3f} mm")
            logger.info(f"   RMSE: {rmse:.3f} mm")
            logger.info(f"   R²: {r2:.3f}")
            logger.info(f"   Rain Accuracy: {rain_accuracy:.3f}")
        
        return results
    
    def save_model(self, model_name: str = "lstm_precipitation"):
        """Salva modelo e componentes"""
        logger.info("💾 Salvando modelo...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = MODELS_PATH / f"{model_name}_{timestamp}"
        model_dir.mkdir(exist_ok=True)
        
        # Salvar modelo
        model_file = model_dir / "model.h5"
        self.model.save(model_file)
        
        # Salvar scalers
        scalers_file = model_dir / "scalers.joblib"
        joblib.dump(self.scalers, scalers_file)
        
        # Salvar metadados completos
        full_metadata = {
            'model_info': {
                'name': model_name,
                'timestamp': timestamp,
                'architecture': 'LSTM',
                'input_shape': list(self.model.input_shape[1:]),
                'output_shape': list(self.model.output_shape[1:]),
                'total_params': self.model.count_params()
            },
            'training_config': self.metadata['config'] if self.metadata else {},
            'data_stats': self.metadata['data_stats'] if self.metadata else {},
            'training_history': self.training_history
        }
        
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(full_metadata, f, indent=2, default=str)
        
        logger.info(f"✅ Modelo salvo em: {model_dir}")
        return model_dir
    
    def plot_training_history(self, save_path: Optional[Path] = None):
        """Plota histórico de treinamento"""
        if not self.training_history:
            logger.warning("Histórico de treinamento não disponível")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Histórico de Treinamento LSTM', fontsize=16)
        
        # Loss
        axes[0, 0].plot(self.training_history['loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE
        axes[0, 1].plot(self.training_history['mae'], label='Train MAE')
        axes[0, 1].plot(self.training_history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # MSE
        axes[1, 0].plot(self.training_history['mse'], label='Train MSE')
        axes[1, 0].plot(self.training_history['val_mse'], label='Val MSE')
        axes[1, 0].set_title('Mean Squared Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        if 'lr' in self.training_history:
            axes[1, 1].plot(self.training_history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path / 'training_history.png', dpi=300, bbox_inches='tight')
            logger.info(f"📊 Gráficos salvos: {save_path / 'training_history.png'}")
        
        plt.show()


def main():
    """Função principal"""
    logger.info("🚀 INICIANDO TREINAMENTO DO MODELO LSTM")
    logger.info("=" * 60)
    
    try:
        # Inicializar trainer
        trainer = LSTMTrainer()
        
        # 1. Carregar dados
        data = trainer.load_data()
        
        # 2. Normalizar dados
        normalized_data = trainer.normalize_data(data)
        
        # 3. Treinar modelo
        history = trainer.train_model(
            normalized_data,
            epochs=50,  # Reduzido para teste inicial
            batch_size=32,
            model_name="lstm_precipitation_v1"
        )
        
        # 4. Avaliar modelo
        results = trainer.evaluate_model(normalized_data)
        
        # 5. Salvar modelo
        model_dir = trainer.save_model("lstm_precipitation_v1")
        
        # 6. Plotar histórico
        trainer.plot_training_history(model_dir)
        
        # 7. Relatório final
        logger.info("\n" + "=" * 60)
        logger.info("✅ TREINAMENTO CONCLUÍDO!")
        logger.info(f"💾 Modelo salvo em: {model_dir}")
        logger.info("\n📊 RESULTADOS FINAIS:")
        
        for split_name, metrics in results.items():
            logger.info(f"\n🎯 {split_name.upper()}:")
            logger.info(f"   📈 MAE: {metrics['mae']:.3f} mm/h")
            logger.info(f"   📈 RMSE: {metrics['rmse']:.3f} mm/h")
            logger.info(f"   📈 R²: {metrics['r2']:.3f}")
            logger.info(f"   🌧️ Rain Detection: {metrics['rain_accuracy']:.1%}")
        
        # Verificar se atende critérios de qualidade
        test_metrics = results['test']
        quality_check = {
            'mae_target': 2.0,  # mm/h
            'r2_target': 0.6,   # correlação
            'rain_acc_target': 0.75  # 75% acurácia
        }
        
        logger.info("\n🎯 AVALIAÇÃO DE QUALIDADE:")
        mae_ok = test_metrics['mae'] <= quality_check['mae_target']
        r2_ok = test_metrics['r2'] >= quality_check['r2_target'] 
        rain_ok = test_metrics['rain_accuracy'] >= quality_check['rain_acc_target']
        
        logger.info(f"   MAE ≤ {quality_check['mae_target']} mm/h: {'✅' if mae_ok else '❌'} ({test_metrics['mae']:.3f})")
        logger.info(f"   R² ≥ {quality_check['r2_target']}: {'✅' if r2_ok else '❌'} ({test_metrics['r2']:.3f})")
        logger.info(f"   Rain Acc ≥ {quality_check['rain_acc_target']:.0%}: {'✅' if rain_ok else '❌'} ({test_metrics['rain_accuracy']:.1%})")
        
        if mae_ok and r2_ok and rain_ok:
            logger.info("\n🎉 MODELO APROVADO! Pronto para produção.")
        else:
            logger.info("\n⚠️ Modelo precisa de melhorias. Considere:")
            logger.info("   - Mais dados de treinamento")
            logger.info("   - Ajuste de hiperparâmetros")
            logger.info("   - Feature engineering adicional")
        
        logger.info(f"\n🚀 PRÓXIMO PASSO: Integrar modelo ao sistema de alertas")
        
    except Exception as e:
        logger.error(f"❌ Erro no treinamento: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 