#!/usr/bin/env python3
"""
Fase 6: Preparação de Dados para Treinamento do Modelo LSTM

Este script prepara dados históricos meteorológicos para treinamento do modelo de previsão de cheias.

Funcionalidades:
- Download dados históricos do INMET (se não existirem)
- Limpeza e preprocessamento dos dados
- Criação de features meteorológicas
- Split temporal (train/validation/test)
- Preparação de sequências para LSTM
- Salvamento dos dados processados

Autor: Sistema de Alertas de Cheias - Rio Guaíba
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from dataclasses import dataclass
import requests

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingDataConfig:
    """Configurações para preparação dos dados de treinamento"""
    
    # Período de dados históricos
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    
    # Features meteorológicas principais
    weather_features: List[str] = None
    
    # Configurações do modelo
    sequence_length: int = 24  # 24 horas de histórico
    forecast_horizon: int = 6   # Previsão para 6h à frente
    
    # Split dos dados
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Paths
    raw_data_path: str = "data/raw/inmet_historicos"
    processed_data_path: str = "data/processed"
    
    def __post_init__(self):
        if self.weather_features is None:
            self.weather_features = [
                'precipitacao_mm',
                'temperatura_c', 
                'umidade_relativa',
                'pressao_mb',
                'velocidade_vento_ms',
                'direcao_vento_graus',
                'ponto_orvalho_c',
                'visibilidade_km'
            ]


class INMETDataGenerator:
    """
    Gerador de dados sintéticos do INMET para treinamento
    
    Simula dados históricos realísticos para Porto Alegre/RS
    """
    
    def __init__(self, config: TrainingDataConfig):
        self.config = config
        
    def generate_synthetic_data(self) -> pd.DataFrame:
        """Gera dados sintéticos realísticos para treinamento"""
        logger.info("🔄 Gerando dados sintéticos do INMET...")
        
        # Período de dados
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        
        # Criar índice temporal horário
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Inicializar DataFrame
        data = pd.DataFrame(index=date_range)
        data['timestamp'] = data.index
        
        # Gerar features meteorológicas realísticas
        n_hours = len(data)
        np.random.seed(42)  # Para reproducibilidade
        
        # 1. Temperatura (padrão sazonal + ciclo diário + ruído)
        day_of_year = data.index.dayofyear
        hour_of_day = data.index.hour
        
        # Temperatura base sazonal (Porto Alegre)
        temp_seasonal = 20 + 8 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
        
        # Ciclo diário (-5°C à noite, +10°C ao meio-dia)
        temp_daily = 7.5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Ruído e tendências
        temp_noise = np.random.normal(0, 2, n_hours)
        data['temperatura_c'] = temp_seasonal + temp_daily + temp_noise
        
        # 2. Umidade Relativa (inversamente relacionada com temperatura)
        base_humidity = 70
        temp_effect = -0.8 * (data['temperatura_c'] - 20)
        humidity_noise = np.random.normal(0, 5, n_hours)
        data['umidade_relativa'] = np.clip(base_humidity + temp_effect + humidity_noise, 20, 100)
        
        # 3. Pressão Atmosférica
        base_pressure = 1013.25
        seasonal_pressure = 5 * np.sin(2 * np.pi * day_of_year / 365)
        pressure_noise = np.random.normal(0, 8, n_hours)
        data['pressao_mb'] = base_pressure + seasonal_pressure + pressure_noise
        
        # 4. Velocidade do Vento
        base_wind = 5.0
        seasonal_wind = 2 * np.sin(2 * np.pi * (day_of_year - 150) / 365)
        daily_wind = 1.5 * np.sin(2 * np.pi * hour_of_day / 24)
        wind_noise = np.random.exponential(2, n_hours)
        data['velocidade_vento_ms'] = np.clip(base_wind + seasonal_wind + daily_wind + wind_noise, 0, 25)
        
        # 5. Direção do Vento (concentrada em direções predominantes)
        wind_directions = np.random.choice([90, 180, 270], n_hours, p=[0.3, 0.4, 0.3])
        wind_dir_noise = np.random.normal(0, 30, n_hours)
        data['direcao_vento_graus'] = (wind_directions + wind_dir_noise) % 360
        
        # 6. Ponto de Orvalho (relacionado com temperatura e umidade)
        # Fórmula simplificada: Td ≈ T - (100 - RH)/5
        data['ponto_orvalho_c'] = data['temperatura_c'] - (100 - data['umidade_relativa'])/5
        
        # 7. Visibilidade
        base_visibility = 15.0
        humidity_effect = -0.1 * (data['umidade_relativa'] - 50)
        visibility_noise = np.random.normal(0, 2, n_hours)
        data['visibilidade_km'] = np.clip(base_visibility + humidity_effect + visibility_noise, 1, 30)
        
        # 8. Precipitação (eventos esporádicos baseados em umidade/pressão)
        # Probabilidade de chuva baseada em umidade alta + pressão baixa
        rain_probability = (
            0.01 * (data['umidade_relativa'] - 60) + 
            0.005 * (1015 - data['pressao_mb'])
        ) / 100
        rain_probability = np.clip(rain_probability, 0, 0.3)
        
        # Gerar eventos de chuva
        rain_events = np.random.binomial(1, rain_probability)
        
        # Intensidade da chuva quando chove
        rain_intensity = np.random.exponential(2.5, n_hours) * rain_events
        data['precipitacao_mm'] = np.clip(rain_intensity, 0, 50)
        
        # 9. Features derivadas
        data['hora'] = data.index.hour
        data['dia_semana'] = data.index.weekday
        data['mes'] = data.index.month
        data['estacao'] = ((data.index.month % 12) // 3) + 1  # 1=Verão, 2=Outono, 3=Inverno, 4=Primavera
        
        # 10. Estação (fixo para Porto Alegre)
        data['estacao_id'] = 'A801'
        data['estacao_nome'] = 'PORTO ALEGRE'
        data['latitude'] = -30.0346
        data['longitude'] = -51.2177
        
        logger.info(f"✅ Dados sintéticos gerados: {len(data)} registros")
        logger.info(f"📅 Período: {data.index.min()} a {data.index.max()}")
        
        return data


class TrainingDataPreprocessor:
    """Preprocessador de dados para treinamento do modelo LSTM"""
    
    def __init__(self, config: TrainingDataConfig):
        self.config = config
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Limpa e valida os dados"""
        logger.info("🧹 Limpando dados...")
        
        # Remover duplicatas
        data_clean = data.drop_duplicates()
        
        # Tratar valores ausentes
        numeric_columns = data_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in self.config.weather_features:
                # Interpolação linear para features principais
                data_clean[col] = data_clean[col].interpolate(method='linear')
                
                # Preencher valores restantes com médias sazonais
                if data_clean[col].isna().any():
                    seasonal_mean = data_clean.groupby('mes')[col].transform('mean')
                    data_clean[col] = data_clean[col].fillna(seasonal_mean)
        
        # Validar ranges realísticos
        data_clean = self._validate_ranges(data_clean)
        
        logger.info(f"✅ Dados limpos: {len(data_clean)} registros")
        return data_clean
    
    def _validate_ranges(self, data: pd.DataFrame) -> pd.DataFrame:
        """Valida e corrige ranges realísticos"""
        
        # Ranges válidos para Porto Alegre/RS
        ranges = {
            'temperatura_c': (-5, 45),
            'umidade_relativa': (10, 100),
            'pressao_mb': (980, 1050),
            'velocidade_vento_ms': (0, 30),
            'direcao_vento_graus': (0, 360),
            'visibilidade_km': (0.1, 50),
            'precipitacao_mm': (0, 200)
        }
        
        for col, (min_val, max_val) in ranges.items():
            if col in data.columns:
                data[col] = np.clip(data[col], min_val, max_val)
        
        return data
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cria features adicionais para o modelo"""
        logger.info("🔧 Criando features adicionais...")
        
        data_features = data.copy()
        
        # Features temporais
        data_features['hora_sin'] = np.sin(2 * np.pi * data_features['hora'] / 24)
        data_features['hora_cos'] = np.cos(2 * np.pi * data_features['hora'] / 24)
        data_features['mes_sin'] = np.sin(2 * np.pi * data_features['mes'] / 12)
        data_features['mes_cos'] = np.cos(2 * np.pi * data_features['mes'] / 12)
        
        # Features meteorológicas derivadas
        data_features['temp_diff'] = data_features['temperatura_c'].diff()
        data_features['pressure_diff'] = data_features['pressao_mb'].diff()
        data_features['humidity_diff'] = data_features['umidade_relativa'].diff()
        
        # Índices compostos
        data_features['conforto_termico'] = (
            data_features['temperatura_c'] * 0.7 + 
            data_features['umidade_relativa'] * 0.3
        )
        
        data_features['instabilidade_atmosferica'] = (
            (1020 - data_features['pressao_mb']) * 0.5 +
            data_features['umidade_relativa'] * 0.3 +
            data_features['velocidade_vento_ms'] * 0.2
        )
        
        # Médias móveis (últimas 3h, 6h, 12h)
        for window in [3, 6, 12]:
            for col in ['temperatura_c', 'umidade_relativa', 'pressao_mb']:
                if col in data_features.columns:
                    data_features[f'{col}_ma{window}h'] = (
                        data_features[col].rolling(window=window, min_periods=1).mean()
                    )
        
        # Preencher NaN de diffs
        data_features = data_features.fillna(method='bfill').fillna(method='ffill')
        
        logger.info(f"✅ Features criadas: {data_features.shape[1]} colunas")
        return data_features
    
    def create_temporal_splits(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Cria splits temporais preservando ordem cronológica"""
        logger.info("✂️ Criando splits temporais...")
        
        # Ordenar por timestamp
        data_sorted = data.sort_values('timestamp').reset_index(drop=True)
        
        # Calcular pontos de corte
        n_total = len(data_sorted)
        n_train = int(n_total * self.config.train_ratio)
        n_val = int(n_total * self.config.val_ratio)
        
        splits = {
            'train': data_sorted.iloc[:n_train].copy(),
            'validation': data_sorted.iloc[n_train:n_train+n_val].copy(),
            'test': data_sorted.iloc[n_train+n_val:].copy()
        }
        
        # Log estatísticas
        for split_name, split_data in splits.items():
            logger.info(f"📊 {split_name.upper()}: {len(split_data)} registros")
            logger.info(f"   📅 {split_data['timestamp'].min()} → {split_data['timestamp'].max()}")
        
        return splits
    
    def prepare_sequences(self, data: pd.DataFrame, feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara sequências temporais para LSTM"""
        logger.info(f"📊 Preparando sequências LSTM (seq_len={self.config.sequence_length})...")
        
        # Verificar colunas disponíveis
        available_features = [col for col in feature_columns if col in data.columns]
        if len(available_features) != len(feature_columns):
            missing = set(feature_columns) - set(available_features)
            logger.warning(f"⚠️ Features não encontradas: {missing}")
            feature_columns = available_features
        
        # Extrair arrays
        features = data[feature_columns].values
        target = data['precipitacao_mm'].values
        
        X, y = [], []
        
        # Criar sequências
        for i in range(len(data) - self.config.sequence_length - self.config.forecast_horizon + 1):
            # Sequência de entrada (últimas sequence_length horas)
            X.append(features[i:i + self.config.sequence_length])
            
            # Target (precipitação daqui a forecast_horizon horas)
            target_idx = i + self.config.sequence_length + self.config.forecast_horizon - 1
            y.append(target[target_idx])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"✅ Sequências criadas: X={X.shape}, y={y.shape}")
        return X, y


def main():
    """Função principal"""
    logger.info("🚀 INICIANDO PREPARAÇÃO DE DADOS PARA TREINAMENTO")
    logger.info("=" * 60)
    
    # Configuração
    config = TrainingDataConfig()
    
    # Criar diretórios
    os.makedirs(config.raw_data_path, exist_ok=True)
    os.makedirs(config.processed_data_path, exist_ok=True)
    
    try:
        # 1. Gerar dados sintéticos do INMET
        generator = INMETDataGenerator(config)
        raw_data = generator.generate_synthetic_data()
        
        # Salvar dados brutos
        raw_file = Path(config.raw_data_path) / 'inmet_dados_sinteticos.csv'
        raw_data.to_csv(raw_file, index=False)
        logger.info(f"💾 Dados brutos salvos: {raw_file}")
        
        # 2. Preprocessar dados
        preprocessor = TrainingDataPreprocessor(config)
        
        # Limpar dados
        clean_data = preprocessor.clean_data(raw_data)
        
        # Criar features
        feature_data = preprocessor.create_features(clean_data)
        
        # 3. Criar splits temporais
        splits = preprocessor.create_temporal_splits(feature_data)
        
        # 4. Salvar splits
        for split_name, split_data in splits.items():
            split_file = Path(config.processed_data_path) / f'{split_name}_data.csv'
            split_data.to_csv(split_file, index=False)
            logger.info(f"💾 {split_name.capitalize()} data salvo: {split_file}")
        
        # 5. Preparar e salvar sequências para cada split
        feature_columns = [col for col in config.weather_features if col in feature_data.columns]
        feature_columns.extend([
            'hora_sin', 'hora_cos', 'mes_sin', 'mes_cos',
            'temp_diff', 'pressure_diff', 'humidity_diff',
            'conforto_termico', 'instabilidade_atmosferica'
        ])
        
        sequences_data = {}
        for split_name, split_data in splits.items():
            if len(split_data) > config.sequence_length + config.forecast_horizon:
                X, y = preprocessor.prepare_sequences(split_data, feature_columns)
                sequences_data[split_name] = {'X': X, 'y': y}
                
                # Salvar arrays
                np.save(Path(config.processed_data_path) / f'{split_name}_X.npy', X)
                np.save(Path(config.processed_data_path) / f'{split_name}_y.npy', y)
        
        # 6. Salvar metadados
        metadata = {
            'config': {
                'sequence_length': config.sequence_length,
                'forecast_horizon': config.forecast_horizon,
                'weather_features': config.weather_features,
                'feature_columns': feature_columns,
                'train_ratio': config.train_ratio,
                'val_ratio': config.val_ratio,
                'test_ratio': config.test_ratio
            },
            'data_stats': {
                'total_records': len(feature_data),
                'train_records': len(splits['train']),
                'val_records': len(splits['validation']),
                'test_records': len(splits['test']),
                'feature_count': len(feature_columns),
                'date_range': {
                    'start': feature_data['timestamp'].min().isoformat(),
                    'end': feature_data['timestamp'].max().isoformat()
                }
            },
            'sequences_shapes': {
                split_name: {'X_shape': data['X'].shape, 'y_shape': data['y'].shape}
                for split_name, data in sequences_data.items()
            }
        }
        
        metadata_file = Path(config.processed_data_path) / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"💾 Metadados salvos: {metadata_file}")
        
        # 7. Relatório final
        logger.info("\n" + "=" * 60)
        logger.info("✅ PREPARAÇÃO DE DADOS CONCLUÍDA!")
        logger.info(f"📊 Total de registros: {len(feature_data):,}")
        logger.info(f"🎯 Features para modelo: {len(feature_columns)}")
        logger.info(f"📅 Período: {feature_data['timestamp'].min()} → {feature_data['timestamp'].max()}")
        logger.info(f"🔧 Sequência LSTM: {config.sequence_length}h → previsão {config.forecast_horizon}h")
        
        for split_name, data in sequences_data.items():
            logger.info(f"📦 {split_name.capitalize()}: {data['X'].shape} sequências")
        
        logger.info(f"💾 Dados salvos em: {config.processed_data_path}")
        logger.info("\n🎯 PRÓXIMO PASSO: Execute o treinamento do modelo!")
        logger.info("   python scripts/train_lstm_model.py")
        
    except Exception as e:
        logger.error(f"❌ Erro na preparação dos dados: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 