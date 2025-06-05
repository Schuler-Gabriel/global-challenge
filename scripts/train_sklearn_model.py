#!/usr/bin/env python3
"""
Fase 7: Treinamento do Modelo Alternativo (scikit-learn) para Previsão de Cheias

Este script treina um modelo usando scikit-learn como alternativa ao LSTM,
adequado para quando TensorFlow não está disponível.

Funcionalidades:
- Carregamento dos dados processados
- Normalização das features
- Treinamento de modelos (Random Forest, XGBoost, etc.)
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

# Scikit-learn
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# XGBoost (opcional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

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


class SequenceProcessor:
    """Processa sequências para modelos tradicionais de ML"""
    
    def __init__(self, sequence_length: int = 24):
        self.sequence_length = sequence_length
    
    def flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Converte sequências 3D (samples, timesteps, features) para 2D (samples, features_flattened)
        para uso em modelos tradicionais de ML
        """
        n_samples, n_timesteps, n_features = X.shape
        return X.reshape(n_samples, n_timesteps * n_features)
    
    def create_features_from_sequences(self, X: np.ndarray) -> pd.DataFrame:
        """Cria features estatísticas a partir das sequências"""
        n_samples, n_timesteps, n_features = X.shape
        features = []
        feature_names = []
        
        # Para cada feature original
        for f in range(n_features):
            feature_data = X[:, :, f]  # (samples, timesteps)
            
            # Estatísticas agregadas
            features.append(np.mean(feature_data, axis=1))  # Média
            feature_names.append(f'feature_{f}_mean')
            
            features.append(np.std(feature_data, axis=1))   # Desvio padrão
            feature_names.append(f'feature_{f}_std')
            
            features.append(np.min(feature_data, axis=1))   # Mínimo
            feature_names.append(f'feature_{f}_min')
            
            features.append(np.max(feature_data, axis=1))   # Máximo
            feature_names.append(f'feature_{f}_max')
            
            # Tendência (diferença entre final e início)
            features.append(feature_data[:, -1] - feature_data[:, 0])
            feature_names.append(f'feature_{f}_trend')
            
            # Últimos valores (mais recentes)
            for i in range(min(3, n_timesteps)):  # Últimas 3 horas
                features.append(feature_data[:, -(i+1)])
                feature_names.append(f'feature_{f}_lag_{i+1}')
        
        # Transpor e criar DataFrame
        features_array = np.array(features).T
        return pd.DataFrame(features_array, columns=feature_names)


class SklearnTrainer:
    """Classe principal para treinamento usando scikit-learn"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.sequence_processor = SequenceProcessor()
        self.metadata = None
        self.best_model = None
        self.best_model_name = None
        
        # Verificar dependências
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn não está instalado. Execute: pip install scikit-learn")
    
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
    
    def prepare_features(self, data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Prepara features para modelos sklearn"""
        logger.info("🔧 Preparando features para modelos sklearn...")
        
        processed_data = {}
        
        for split_name, split_data in data.items():
            X_seq = split_data['X']  # (samples, timesteps, features)
            y = split_data['y']
            
            # Criar features estatísticas das sequências
            X_features = self.sequence_processor.create_features_from_sequences(X_seq)
            
            processed_data[split_name] = {
                'X': X_features.values,
                'y': y
            }
            
            logger.info(f"✅ {split_name}: {X_features.shape} features criadas")
        
        return processed_data
    
    def normalize_data(self, data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Normaliza features e targets"""
        logger.info("📏 Normalizando dados...")
        
        # Normalizar features (X)
        X_train = data['train']['X']
        X_val = data['validation']['X']
        X_test = data['test']['X']
        
        # Scaler para features
        feature_scaler = StandardScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_val_scaled = feature_scaler.transform(X_val)
        X_test_scaled = feature_scaler.transform(X_test)
        
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
    
    def build_models(self) -> Dict[str, Any]:
        """Constrói diferentes modelos para comparação"""
        logger.info("🏗️ Construindo modelos...")
        
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        }
        
        # XGBoost se disponível
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        
        logger.info(f"✅ {len(models)} modelos criados")
        return models
    
    def train_models(self, data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Treina todos os modelos"""
        logger.info("🚂 Iniciando treinamento dos modelos...")
        
        models = self.build_models()
        results = {}
        
        X_train = data['train']['X']
        y_train = data['train']['y']
        X_val = data['validation']['X']
        y_val = data['validation']['y']
        
        for model_name, model in models.items():
            logger.info(f"   🔄 Treinando {model_name}...")
            
            # Treinar modelo
            model.fit(X_train, y_train)
            
            # Predições de validação
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            
            # Métricas
            train_mae = mean_absolute_error(y_train, y_pred_train)
            train_r2 = r2_score(y_train, y_pred_train)
            val_mae = mean_absolute_error(y_val, y_pred_val)
            val_r2 = r2_score(y_val, y_pred_val)
            
            results[model_name] = {
                'model': model,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'val_mae': val_mae,
                'val_r2': val_r2
            }
            
            logger.info(f"   ✅ {model_name}: VAL MAE={val_mae:.3f}, R²={val_r2:.3f}")
        
        # Selecionar melhor modelo baseado em MAE de validação
        best_model_name = min(results.keys(), key=lambda k: results[k]['val_mae'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        logger.info(f"🏆 Melhor modelo: {best_model_name}")
        return results
    
    def evaluate_model(self, data: Dict[str, Dict[str, np.ndarray]], 
                      training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Avalia o melhor modelo em todos os splits"""
        logger.info("📊 Avaliando melhor modelo...")
        
        results = {}
        best_model = self.best_model
        
        for split_name, split_data in data.items():
            # Predições
            y_pred_scaled = best_model.predict(split_data['X'])
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
    
    def save_model(self, training_results: Dict[str, Any], 
                   evaluation_results: Dict[str, Any],
                   model_name: str = "sklearn_precipitation"):
        """Salva modelo e componentes"""
        logger.info("💾 Salvando modelo...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = MODELS_PATH / f"{model_name}_{timestamp}"
        model_dir.mkdir(exist_ok=True)
        
        # Salvar modelo
        model_file = model_dir / "model.joblib"
        joblib.dump(self.best_model, model_file)
        
        # Salvar scalers
        scalers_file = model_dir / "scalers.joblib"
        joblib.dump(self.scalers, scalers_file)
        
        # Salvar sequence processor
        processor_file = model_dir / "sequence_processor.joblib"
        joblib.dump(self.sequence_processor, processor_file)
        
        # Salvar metadados completos
        full_metadata = {
            'model_info': {
                'name': model_name,
                'timestamp': timestamp,
                'algorithm': self.best_model_name,
                'best_model': self.best_model_name,
                'sklearn_version': '1.3+',
                'n_features': self.best_model.n_features_in_ if hasattr(self.best_model, 'n_features_in_') else 'unknown'
            },
            'training_config': self.metadata['config'] if self.metadata else {},
            'data_stats': self.metadata['data_stats'] if self.metadata else {},
            'training_results': training_results,
            'evaluation_results': evaluation_results
        }
        
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(full_metadata, f, indent=2, default=str)
        
        logger.info(f"✅ Modelo salvo em: {model_dir}")
        return model_dir
    
    def plot_results(self, training_results: Dict[str, Any], 
                    evaluation_results: Dict[str, Any], 
                    save_path: Optional[Path] = None):
        """Plota resultados de treinamento"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Resultados do Modelo: {self.best_model_name}', fontsize=16)
        
        # Comparação de modelos (MAE de validação)
        model_names = list(training_results.keys())
        val_maes = [training_results[name]['val_mae'] for name in model_names]
        
        axes[0, 0].bar(model_names, val_maes)
        axes[0, 0].set_title('MAE de Validação por Modelo')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # R² por split para melhor modelo
        splits = list(evaluation_results.keys())
        r2_scores = [evaluation_results[split]['r2'] for split in splits]
        
        axes[0, 1].bar(splits, r2_scores)
        axes[0, 1].set_title('R² por Split (Melhor Modelo)')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].set_ylim(0, 1)
        
        # MAE por split
        maes = [evaluation_results[split]['mae'] for split in splits]
        
        axes[1, 0].bar(splits, maes)
        axes[1, 0].set_title('MAE por Split (mm)')
        axes[1, 0].set_ylabel('MAE (mm)')
        
        # Rain accuracy por split
        rain_accs = [evaluation_results[split]['rain_accuracy'] for split in splits]
        
        axes[1, 1].bar(splits, rain_accs)
        axes[1, 1].set_title('Acurácia de Detecção de Chuva')
        axes[1, 1].set_ylabel('Acurácia')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path / 'training_results.png', dpi=300, bbox_inches='tight')
            logger.info(f"📊 Gráficos salvos: {save_path / 'training_results.png'}")
        
        plt.show()


def main():
    """Função principal"""
    logger.info("🚀 INICIANDO TREINAMENTO DO MODELO SKLEARN")
    logger.info("=" * 60)
    
    try:
        # Inicializar trainer
        trainer = SklearnTrainer()
        
        # 1. Carregar dados
        data = trainer.load_data()
        
        # 2. Preparar features
        processed_data = trainer.prepare_features(data)
        
        # 3. Normalizar dados
        normalized_data = trainer.normalize_data(processed_data)
        
        # 4. Treinar modelos
        training_results = trainer.train_models(normalized_data)
        
        # 5. Avaliar melhor modelo
        evaluation_results = trainer.evaluate_model(normalized_data, training_results)
        
        # 6. Salvar modelo
        model_dir = trainer.save_model(training_results, evaluation_results, "sklearn_precipitation_v1")
        
        # 7. Plotar resultados
        trainer.plot_results(training_results, evaluation_results, model_dir)
        
        # 8. Relatório final
        logger.info("\n" + "=" * 60)
        logger.info("✅ TREINAMENTO CONCLUÍDO!")
        logger.info(f"🏆 Melhor modelo: {trainer.best_model_name}")
        logger.info(f"💾 Modelo salvo em: {model_dir}")
        logger.info("\n📊 RESULTADOS FINAIS:")
        
        for split_name, metrics in evaluation_results.items():
            logger.info(f"\n🎯 {split_name.upper()}:")
            logger.info(f"   📈 MAE: {metrics['mae']:.3f} mm")
            logger.info(f"   📈 RMSE: {metrics['rmse']:.3f} mm")
            logger.info(f"   📈 R²: {metrics['r2']:.3f}")
            logger.info(f"   🌧️ Rain Detection: {metrics['rain_accuracy']:.1%}")
        
        # Verificar se atende critérios de qualidade
        test_metrics = evaluation_results['test']
        quality_check = {
            'mae_target': 3.0,  # mm (relaxado para sklearn)
            'r2_target': 0.5,   # correlação
            'rain_acc_target': 0.70  # 70% acurácia
        }
        
        logger.info("\n🎯 AVALIAÇÃO DE QUALIDADE:")
        mae_ok = test_metrics['mae'] <= quality_check['mae_target']
        r2_ok = test_metrics['r2'] >= quality_check['r2_target'] 
        rain_ok = test_metrics['rain_accuracy'] >= quality_check['rain_acc_target']
        
        logger.info(f"   MAE ≤ {quality_check['mae_target']} mm: {'✅' if mae_ok else '❌'} ({test_metrics['mae']:.3f})")
        logger.info(f"   R² ≥ {quality_check['r2_target']}: {'✅' if r2_ok else '❌'} ({test_metrics['r2']:.3f})")
        logger.info(f"   Rain Acc ≥ {quality_check['rain_acc_target']:.0%}: {'✅' if rain_ok else '❌'} ({test_metrics['rain_accuracy']:.1%})")
        
        if mae_ok and r2_ok and rain_ok:
            logger.info("\n🎉 MODELO APROVADO! Pronto para produção.")
        else:
            logger.info("\n⚠️ Modelo precisa de melhorias. Considere:")
            logger.info("   - Mais dados de treinamento")
            logger.info("   - Feature engineering adicional")
            logger.info("   - Ajuste de hiperparâmetros")
        
        logger.info(f"\n🚀 PRÓXIMO PASSO: Integrar modelo ao sistema de alertas")
        logger.info("   O modelo pode ser carregado com: joblib.load('model.joblib')")
        
    except Exception as e:
        logger.error(f"❌ Erro no treinamento: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 