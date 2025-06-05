#!/usr/bin/env python3
"""
Sistema de Alertas de Cheias - Rio Guaíba
Fase 2.2: Preprocessamento de Dados

Script para preprocessamento completo dos dados meteorológicos INMET (2000-2025)
Baseado nos resultados da análise exploratória da Fase 2.1

Funcionalidades:
- Padronização de formatos de data e timestamps
- Tratamento de valores missing/nulos
- Normalização e scaling de features
- Feature engineering (variáveis derivadas)
- Unificação de dados entre estações
- Pipeline de preprocessamento reutilizável

Autor: Sistema IA
Data: 2025-01-03
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração de warnings
warnings.filterwarnings('ignore')

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/processed/preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configurações para o preprocessamento"""
    # Caminhos
    raw_data_path: str = "data/raw/dados_historicos"
    processed_data_path: str = "data/processed"
    
    # Configurações de imputação
    max_gap_hours: int = 6  # Máximo gap para interpolação
    imputation_method: str = "knn"  # 'mean', 'median', 'knn', 'interpolate'
    
    # Configurações de normalização
    scaling_method: str = "standard"  # 'standard', 'minmax', 'robust'
    
    # Configurações de feature engineering
    create_derived_features: bool = True
    temporal_aggregations: List[str] = None
    
    # Configurações de validação
    min_data_coverage: float = 0.7  # Mínimo 70% de dados válidos
    
    def __post_init__(self):
        if self.temporal_aggregations is None:
            self.temporal_aggregations = ['3H', '6H', '12H', '24H']

class DataPreprocessor:
    """Classe principal para preprocessamento de dados meteorológicos"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.scalers = {}
        self.imputers = {}
        self.feature_columns = []
        self.preprocessing_stats = {}
        
        # Criar diretórios necessários
        os.makedirs(self.config.processed_data_path, exist_ok=True)
        os.makedirs(f"{self.config.processed_data_path}/unified", exist_ok=True)
        os.makedirs(f"{self.config.processed_data_path}/features", exist_ok=True)
        
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Carrega todos os dados brutos com tratamento robusto de encoding"""
        logger.info("🔄 Carregando dados brutos...")
        
        data_files = {}
        raw_path = Path(self.config.raw_data_path)
        
        # Padrões de encoding para tentar
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for file_path in raw_path.glob("*.CSV"):
            file_key = file_path.stem
            logger.info(f"Processando: {file_key}")
            
            # Tentar diferentes encodings
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(
                        file_path,
                        sep=';',
                        encoding=encoding,
                        low_memory=False,
                        na_values=['', ' ', 'null', 'NULL', '-', '--', '---']
                    )
                    logger.info(f"✅ Carregado com encoding: {encoding}")
                    break
                except Exception as e:
                    logger.warning(f"Falha com encoding {encoding}: {str(e)[:100]}")
                    continue
            
            if df is not None:
                data_files[file_key] = df
                logger.info(f"📊 Shape: {df.shape}")
            else:
                logger.error(f"❌ Falha ao carregar: {file_key}")
        
        logger.info(f"✅ Total de arquivos carregados: {len(data_files)}")
        return data_files
    
    def standardize_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Padroniza formatos de data e timestamps"""
        logger.info("🕐 Padronizando timestamps...")
        
        df_clean = df.copy()
        
        # Identificar colunas de data/hora
        date_columns = []
        for col in df_clean.columns:
            if any(keyword in col.lower() for keyword in ['data', 'hora', 'date', 'time']):
                date_columns.append(col)
        
        logger.info(f"Colunas de data identificadas: {date_columns}")
        
        # Tentar diferentes formatos de data
        date_formats = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%Y/%m/%d',
            '%Y-%m-%d %H:%M',
            '%d/%m/%Y %H:%M',
            '%Y-%m-%d %H:%M:%S'
        ]
        
        for col in date_columns:
            if col in df_clean.columns:
                # Tentar converter para datetime
                for fmt in date_formats:
                    try:
                        df_clean[col] = pd.to_datetime(df_clean[col], format=fmt, errors='coerce')
                        if not df_clean[col].isna().all():
                            logger.info(f"✅ {col} convertido com formato: {fmt}")
                            break
                    except:
                        continue
                
                # Se ainda não converteu, tentar inferir automaticamente
                if df_clean[col].dtype == 'object':
                    try:
                        df_clean[col] = pd.to_datetime(df_clean[col], infer_datetime_format=True, errors='coerce')
                        logger.info(f"✅ {col} convertido automaticamente")
                    except:
                        logger.warning(f"⚠️ Não foi possível converter {col}")
        
        # Criar timestamp unificado se possível
        if len(date_columns) >= 2:
            try:
                # Combinar data e hora se estiverem separadas
                date_col = next((col for col in date_columns if 'data' in col.lower()), None)
                time_col = next((col for col in date_columns if 'hora' in col.lower()), None)
                
                if date_col and time_col:
                    df_clean['timestamp'] = pd.to_datetime(
                        df_clean[date_col].astype(str) + ' ' + df_clean[time_col].astype(str),
                        errors='coerce'
                    )
                    logger.info("✅ Timestamp unificado criado")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao criar timestamp unificado: {e}")
        
        return df_clean
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Padroniza nomes de colunas"""
        logger.info("📝 Padronizando nomes de colunas...")
        
        df_clean = df.copy()
        
        # Mapeamento de colunas comuns
        column_mapping = {
            # Precipitação
            'precipitacao_total_horario_mm': 'precipitacao_mm',
            'precipitação total, horário (mm)': 'precipitacao_mm',
            'chuva_mm': 'precipitacao_mm',
            
            # Temperatura
            'temperatura_do_ar_bulbo_seco_c': 'temperatura_c',
            'temperatura do ar - bulbo seco (°c)': 'temperatura_c',
            'temp_c': 'temperatura_c',
            
            # Umidade
            'umidade_relativa_do_ar': 'umidade_relativa',
            'umidade relativa do ar (%)': 'umidade_relativa',
            'umidade_pct': 'umidade_relativa',
            
            # Pressão
            'pressao_atmosferica_ao_nivel_da_estacao_mb': 'pressao_mb',
            'pressão atmosférica ao nível da estação (mb)': 'pressao_mb',
            'pressao_mb': 'pressao_mb',
            
            # Vento
            'velocidade_do_vento_m_s': 'vento_velocidade_ms',
            'velocidade do vento (m/s)': 'vento_velocidade_ms',
            'direcao_do_vento_graus': 'vento_direcao_graus',
            'direção do vento (graus)': 'vento_direcao_graus',
            
            # Radiação
            'radiacao_global_kj_m2': 'radiacao_global_kjm2',
            'radiação global (kj/m²)': 'radiacao_global_kjm2'
        }
        
        # Aplicar mapeamento (case insensitive)
        for old_name, new_name in column_mapping.items():
            for col in df_clean.columns:
                if old_name.lower() in col.lower():
                    df_clean = df_clean.rename(columns={col: new_name})
                    logger.info(f"Renomeado: {col} -> {new_name}")
                    break
        
        # Limpar nomes de colunas
        df_clean.columns = [
            col.lower()
            .replace(' ', '_')
            .replace('(', '')
            .replace(')', '')
            .replace('°', '')
            .replace('²', '2')
            .replace('/', '_')
            .replace('-', '_')
            .replace(',', '')
            for col in df_clean.columns
        ]
        
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trata valores missing com diferentes estratégias"""
        logger.info("🔧 Tratando valores missing...")
        
        df_clean = df.copy()
        
        # Identificar colunas numéricas
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # Estatísticas de missing values
        missing_stats = {}
        for col in numeric_columns:
            missing_pct = df_clean[col].isna().sum() / len(df_clean) * 100
            missing_stats[col] = missing_pct
            
        logger.info(f"Colunas com missing values: {len([k for k, v in missing_stats.items() if v > 0])}")
        
        # Estratégias por tipo de variável
        for col in numeric_columns:
            missing_pct = missing_stats[col]
            
            if missing_pct == 0:
                continue
                
            logger.info(f"Tratando {col}: {missing_pct:.1f}% missing")
            
            if missing_pct > 50:
                # Muitos valores missing - remover coluna ou usar estratégia especial
                logger.warning(f"⚠️ {col} tem {missing_pct:.1f}% missing - considerando remoção")
                continue
            
            # Interpolação temporal para gaps pequenos
            if 'timestamp' in df_clean.columns and missing_pct < 20:
                try:
                    df_clean = df_clean.sort_values('timestamp')
                    df_clean[col] = df_clean[col].interpolate(method='time', limit=self.config.max_gap_hours)
                    logger.info(f"✅ {col}: interpolação temporal aplicada")
                except:
                    pass
            
            # Imputação baseada no método configurado
            remaining_missing = df_clean[col].isna().sum()
            if remaining_missing > 0:
                if self.config.imputation_method == 'knn':
                    # KNN Imputer para valores relacionados
                    imputer = KNNImputer(n_neighbors=5)
                    df_clean[col] = imputer.fit_transform(df_clean[[col]]).flatten()
                elif self.config.imputation_method == 'mean':
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif self.config.imputation_method == 'median':
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                
                logger.info(f"✅ {col}: imputação {self.config.imputation_method} aplicada")
        
        # Salvar estatísticas de missing values
        self.preprocessing_stats['missing_values'] = missing_stats
        
        return df_clean
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features derivadas e engenharia de características"""
        logger.info("🔬 Criando features derivadas...")
        
        df_enhanced = df.copy()
        
        # Features temporais
        if 'timestamp' in df_enhanced.columns:
            df_enhanced['hora'] = df_enhanced['timestamp'].dt.hour
            df_enhanced['dia_semana'] = df_enhanced['timestamp'].dt.dayofweek
            df_enhanced['mes'] = df_enhanced['timestamp'].dt.month
            df_enhanced['estacao'] = df_enhanced['mes'].map({
                12: 'verao', 1: 'verao', 2: 'verao',
                3: 'outono', 4: 'outono', 5: 'outono',
                6: 'inverno', 7: 'inverno', 8: 'inverno',
                9: 'primavera', 10: 'primavera', 11: 'primavera'
            })
            
            # Componentes cíclicas
            df_enhanced['hora_sin'] = np.sin(2 * np.pi * df_enhanced['hora'] / 24)
            df_enhanced['hora_cos'] = np.cos(2 * np.pi * df_enhanced['hora'] / 24)
            df_enhanced['mes_sin'] = np.sin(2 * np.pi * df_enhanced['mes'] / 12)
            df_enhanced['mes_cos'] = np.cos(2 * np.pi * df_enhanced['mes'] / 12)
            
            logger.info("✅ Features temporais criadas")
        
        # Features meteorológicas derivadas
        if 'temperatura_c' in df_enhanced.columns and 'umidade_relativa' in df_enhanced.columns:
            # Índice de calor aproximado
            df_enhanced['indice_calor'] = df_enhanced['temperatura_c'] + (df_enhanced['umidade_relativa'] / 100) * 2
            logger.info("✅ Índice de calor criado")
        
        if 'vento_velocidade_ms' in df_enhanced.columns and 'vento_direcao_graus' in df_enhanced.columns:
            # Componentes do vento
            df_enhanced['vento_u'] = -df_enhanced['vento_velocidade_ms'] * np.sin(np.radians(df_enhanced['vento_direcao_graus']))
            df_enhanced['vento_v'] = -df_enhanced['vento_velocidade_ms'] * np.cos(np.radians(df_enhanced['vento_direcao_graus']))
            logger.info("✅ Componentes do vento criadas")
        
        # Agregações temporais
        if 'timestamp' in df_enhanced.columns:
            df_enhanced = df_enhanced.sort_values('timestamp')
            
            for window in self.config.temporal_aggregations:
                try:
                    # Rolling statistics para variáveis chave
                    if 'precipitacao_mm' in df_enhanced.columns:
                        df_enhanced[f'precipitacao_sum_{window}'] = df_enhanced['precipitacao_mm'].rolling(window).sum()
                    
                    if 'temperatura_c' in df_enhanced.columns:
                        df_enhanced[f'temperatura_mean_{window}'] = df_enhanced['temperatura_c'].rolling(window).mean()
                        df_enhanced[f'temperatura_std_{window}'] = df_enhanced['temperatura_c'].rolling(window).std()
                    
                    if 'pressao_mb' in df_enhanced.columns:
                        df_enhanced[f'pressao_trend_{window}'] = df_enhanced['pressao_mb'].diff().rolling(window).mean()
                    
                    logger.info(f"✅ Agregações {window} criadas")
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao criar agregações {window}: {e}")
        
        # Features de eventos extremos
        if 'precipitacao_mm' in df_enhanced.columns:
            # Definir thresholds para eventos de chuva
            df_enhanced['chuva_leve'] = (df_enhanced['precipitacao_mm'] > 0) & (df_enhanced['precipitacao_mm'] <= 2.5)
            df_enhanced['chuva_moderada'] = (df_enhanced['precipitacao_mm'] > 2.5) & (df_enhanced['precipitacao_mm'] <= 10)
            df_enhanced['chuva_forte'] = (df_enhanced['precipitacao_mm'] > 10) & (df_enhanced['precipitacao_mm'] <= 50)
            df_enhanced['chuva_muito_forte'] = df_enhanced['precipitacao_mm'] > 50
            
            logger.info("✅ Features de eventos de chuva criadas")
        
        logger.info(f"📊 Features criadas: {df_enhanced.shape[1] - df.shape[1]} novas colunas")
        return df_enhanced
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza features numéricas"""
        logger.info("📏 Normalizando features...")
        
        df_normalized = df.copy()
        
        # Identificar colunas numéricas para normalização
        numeric_columns = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
        
        # Excluir colunas que não devem ser normalizadas
        exclude_columns = ['hora', 'dia_semana', 'mes', 'timestamp']
        numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        if self.config.scaling_method == 'standard':
            scaler = StandardScaler()
        elif self.config.scaling_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            logger.warning(f"Método de scaling não reconhecido: {self.config.scaling_method}")
            return df_normalized
        
        # Aplicar normalização
        df_normalized[numeric_columns] = scaler.fit_transform(df_normalized[numeric_columns])
        
        # Salvar scaler para uso posterior
        self.scalers['main'] = scaler
        self.feature_columns = numeric_columns
        
        logger.info(f"✅ {len(numeric_columns)} features normalizadas com {self.config.scaling_method}")
        return df_normalized
    
    def unify_datasets(self, data_files: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Unifica datasets de diferentes estações e períodos"""
        logger.info("🔗 Unificando datasets...")
        
        unified_data = []
        
        for file_key, df in data_files.items():
            logger.info(f"Processando: {file_key}")
            
            # Aplicar pipeline de preprocessamento
            df_processed = self.standardize_datetime(df)
            df_processed = self.standardize_columns(df_processed)
            df_processed = self.handle_missing_values(df_processed)
            
            # Adicionar metadados
            df_processed['source_file'] = file_key
            
            # Identificar estação
            if 'A801' in file_key:
                if 'JARDIM' in file_key.upper():
                    df_processed['station'] = 'A801_JARDIM_BOTANICO'
                else:
                    df_processed['station'] = 'A801_PORTO_ALEGRE'
            elif 'B807' in file_key:
                df_processed['station'] = 'B807_BELEM_NOVO'
            else:
                df_processed['station'] = 'UNKNOWN'
            
            # Validar cobertura de dados
            if 'timestamp' in df_processed.columns:
                valid_rows = df_processed.dropna(subset=['timestamp']).shape[0]
                coverage = valid_rows / df_processed.shape[0]
                
                if coverage >= self.config.min_data_coverage:
                    unified_data.append(df_processed)
                    logger.info(f"✅ {file_key}: {coverage:.1%} cobertura - incluído")
                else:
                    logger.warning(f"⚠️ {file_key}: {coverage:.1%} cobertura - excluído")
        
        if not unified_data:
            raise ValueError("Nenhum dataset válido encontrado para unificação")
        
        # Concatenar todos os datasets
        unified_df = pd.concat(unified_data, ignore_index=True, sort=False)
        
        # Remover duplicatas baseadas em timestamp e estação
        if 'timestamp' in unified_df.columns:
            unified_df = unified_df.drop_duplicates(subset=['timestamp', 'station'], keep='first')
            unified_df = unified_df.sort_values(['station', 'timestamp'])
        
        logger.info(f"✅ Dataset unificado: {unified_df.shape}")
        logger.info(f"Período: {unified_df['timestamp'].min()} a {unified_df['timestamp'].max()}")
        logger.info(f"Estações: {unified_df['station'].unique()}")
        
        return unified_df
    
    def create_temporal_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Cria splits temporais para treino/validação/teste"""
        logger.info("✂️ Criando splits temporais...")
        
        if 'timestamp' not in df.columns:
            raise ValueError("Coluna 'timestamp' necessária para splits temporais")
        
        df_sorted = df.sort_values('timestamp')
        
        # Definir pontos de corte (preservando sazonalidade)
        total_rows = len(df_sorted)
        train_end = int(total_rows * 0.7)  # 70% treino
        val_end = int(total_rows * 0.85)   # 15% validação, 15% teste
        
        splits = {
            'train': df_sorted.iloc[:train_end].copy(),
            'validation': df_sorted.iloc[train_end:val_end].copy(),
            'test': df_sorted.iloc[val_end:].copy()
        }
        
        # Log das informações dos splits
        for split_name, split_df in splits.items():
            logger.info(f"{split_name.upper()}: {split_df.shape[0]} registros")
            logger.info(f"  Período: {split_df['timestamp'].min()} a {split_df['timestamp'].max()}")
        
        return splits
    
    def generate_preprocessing_report(self, original_data: Dict[str, pd.DataFrame], 
                                    processed_data: pd.DataFrame) -> Dict:
        """Gera relatório detalhado do preprocessamento"""
        logger.info("📊 Gerando relatório de preprocessamento...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'original_files': len(original_data),
            'total_original_rows': sum(df.shape[0] for df in original_data.values()),
            'processed_rows': processed_data.shape[0],
            'processed_columns': processed_data.shape[1],
            'stations': processed_data['station'].unique().tolist() if 'station' in processed_data.columns else [],
            'date_range': {
                'start': processed_data['timestamp'].min().isoformat() if 'timestamp' in processed_data.columns else None,
                'end': processed_data['timestamp'].max().isoformat() if 'timestamp' in processed_data.columns else None
            },
            'preprocessing_stats': self.preprocessing_stats,
            'feature_columns': self.feature_columns,
            'data_quality': {}
        }
        
        # Estatísticas de qualidade dos dados
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            report['data_quality'][col] = {
                'missing_pct': processed_data[col].isna().sum() / len(processed_data) * 100,
                'mean': float(processed_data[col].mean()) if not processed_data[col].isna().all() else None,
                'std': float(processed_data[col].std()) if not processed_data[col].isna().all() else None,
                'min': float(processed_data[col].min()) if not processed_data[col].isna().all() else None,
                'max': float(processed_data[col].max()) if not processed_data[col].isna().all() else None
            }
        
        return report
    
    def save_processed_data(self, processed_data: pd.DataFrame, splits: Dict[str, pd.DataFrame], 
                          report: Dict) -> None:
        """Salva dados processados e artefatos"""
        logger.info("💾 Salvando dados processados...")
        
        # Salvar dataset unificado
        unified_path = f"{self.config.processed_data_path}/unified/meteorological_data_unified.parquet"
        processed_data.to_parquet(unified_path, index=False)
        logger.info(f"✅ Dataset unificado salvo: {unified_path}")
        
        # Salvar splits
        for split_name, split_df in splits.items():
            split_path = f"{self.config.processed_data_path}/unified/{split_name}_data.parquet"
            split_df.to_parquet(split_path, index=False)
            logger.info(f"✅ Split {split_name} salvo: {split_path}")
        
        # Salvar relatório
        report_path = f"{self.config.processed_data_path}/preprocessing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"✅ Relatório salvo: {report_path}")
        
        # Salvar scalers e outros artefatos
        import joblib
        if self.scalers:
            scaler_path = f"{self.config.processed_data_path}/scalers.joblib"
            joblib.dump(self.scalers, scaler_path)
            logger.info(f"✅ Scalers salvos: {scaler_path}")
    
    def run_preprocessing_pipeline(self) -> None:
        """Executa pipeline completo de preprocessamento"""
        logger.info("🚀 Iniciando pipeline de preprocessamento - Fase 2.2")
        logger.info("=" * 60)
        
        try:
            # 1. Carregar dados brutos
            original_data = self.load_raw_data()
            
            # 2. Unificar datasets
            unified_data = self.unify_datasets(original_data)
            
            # 3. Feature engineering
            if self.config.create_derived_features:
                unified_data = self.create_derived_features(unified_data)
            
            # 4. Normalização
            unified_data = self.normalize_features(unified_data)
            
            # 5. Criar splits temporais
            splits = self.create_temporal_splits(unified_data)
            
            # 6. Gerar relatório
            report = self.generate_preprocessing_report(original_data, unified_data)
            
            # 7. Salvar resultados
            self.save_processed_data(unified_data, splits, report)
            
            logger.info("=" * 60)
            logger.info("✅ Pipeline de preprocessamento concluído com sucesso!")
            logger.info(f"📊 Dados processados: {unified_data.shape}")
            logger.info(f"📅 Período: {report['date_range']['start']} a {report['date_range']['end']}")
            logger.info(f"🏢 Estações: {len(report['stations'])}")
            logger.info("🔄 Próximo passo: Fase 3 - Desenvolvimento do Modelo ML")
            
        except Exception as e:
            logger.error(f"❌ Erro no pipeline de preprocessamento: {e}")
            raise

def main():
    """Função principal"""
    print("🌦️ Sistema de Alertas de Cheias - Rio Guaíba")
    print("📋 Fase 2.2: Preprocessamento de Dados")
    print("=" * 60)
    
    # Configuração
    config = PreprocessingConfig()
    
    # Executar preprocessamento
    preprocessor = DataPreprocessor(config)
    preprocessor.run_preprocessing_pipeline()

if __name__ == "__main__":
    main() 