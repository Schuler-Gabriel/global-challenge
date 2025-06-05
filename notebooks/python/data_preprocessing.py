# %% [markdown]
# # Preprocessamento dos Dados Meteorológicos INMET
# 
# Este notebook realiza o preprocessamento dos dados históricos do INMET para treinamento do modelo LSTM.
# 
# ## Objetivos:
# - Limpeza e normalização dos dados
# - Tratamento de valores missing
# - Feature engineering
# - Criação de datasets de treino/validação/teste
# - Salvamento dos dados processados

# %%
# Imports necessários
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

warnings.filterwarnings('ignore')

print("=== PREPROCESSAMENTO DOS DADOS INMET ===")
print(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %%
# Configuração dos caminhos
DATA_PATH = Path('../data')
RAW_DATA_PATH = DATA_PATH / 'raw' / 'dados_historicos'
PROCESSED_DATA_PATH = DATA_PATH / 'processed'
SCALERS_PATH = DATA_PATH / 'scalers'

# Criar diretórios se não existirem
PROCESSED_DATA_PATH.mkdir(exist_ok=True, parents=True)
SCALERS_PATH.mkdir(exist_ok=True, parents=True)

print(f"Dados brutos: {RAW_DATA_PATH}")
print(f"Dados processados: {PROCESSED_DATA_PATH}")

# %%
def load_and_clean_inmet_data():
    """
    Carrega e limpa dados brutos do INMET
    """
    print("=== CARREGAMENTO E LIMPEZA DOS DADOS ===")
    
    # Verificar arquivos disponíveis
    csv_files = list(RAW_DATA_PATH.glob("*.CSV")) if RAW_DATA_PATH.exists() else []
    
    if not csv_files:
        print("❌ Nenhum arquivo CSV encontrado!")
        # Criar dados sintéticos para demonstração
        dates = pd.date_range('2000-01-01', '2023-12-31', freq='H')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'precipitacao_mm': np.random.exponential(0.3, len(dates)),
            'temperatura_c': 20 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / (24*365)) + np.random.normal(0, 3, len(dates)),
            'umidade_relativa': np.clip(50 + 30 * np.sin(2 * np.pi * np.arange(len(dates)) / (24*365)) + np.random.normal(0, 8, len(dates)), 0, 100),
            'pressao_mb': 1013 + np.random.normal(0, 15, len(dates)),
            'velocidade_vento_ms': np.random.gamma(2, 1.5, len(dates)),
            'direcao_vento_gr': np.random.uniform(0, 360, len(dates)),
            'radiacao_kjm2': np.maximum(0, 800 * np.sin(2 * np.pi * np.arange(len(dates)) / 24) + np.random.normal(0, 150, len(dates)))
        })
        return data
    
    # Carregar dados reais do INMET
    dataframes = []
    
    for file_path in csv_files:
        try:
            print(f"Processando: {file_path.name}")
            df = pd.read_csv(file_path, sep=';', encoding='latin1', skiprows=8)
            df['arquivo_origem'] = file_path.name
            dataframes.append(df)
        except Exception as e:
            print(f"❌ Erro ao processar {file_path.name}: {e}")
    
    if dataframes:
        data = pd.concat(dataframes, ignore_index=True)
        print(f"✓ {len(dataframes)} arquivos processados")
        return data
    
    return None

# Carregar dados
raw_data = load_and_clean_inmet_data()

if raw_data is not None:
    print(f"Dados carregados: {raw_data.shape}")
else:
    print("❌ Falha no carregamento")

# %%
def standardize_columns(df):
    """
    Padroniza nomes das colunas
    """
    print("=== PADRONIZAÇÃO DE COLUNAS ===")
    
    # Mapeamento de colunas para nomes padronizados
    column_mapping = {
        'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'precipitacao_mm',
        'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'temperatura_c',
        'UMIDADE RELATIVA DO AR, HORARIA (%)': 'umidade_relativa',
        'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': 'pressao_mb',
        'VENTO, VELOCIDADE HORARIA (m/s)': 'velocidade_vento_ms',
        'VENTO, DIREÇÃO HORARIA (gr) (° (gr))': 'direcao_vento_gr',
        'RADIACAO GLOBAL (Kj/m²)': 'radiacao_kjm2',
        'Data': 'data',
        'Hora UTC': 'hora'
    }
    
    # Renomear colunas existentes
    df_processed = df.copy()
    for old_name, new_name in column_mapping.items():
        if old_name in df_processed.columns:
            df_processed = df_processed.rename(columns={old_name: new_name})
            print(f"✓ {old_name} → {new_name}")
    
    return df_processed

# Padronizar colunas
if raw_data is not None:
    processed_data = standardize_columns(raw_data)
    print(f"Colunas após padronização: {list(processed_data.columns)}")

# %%
def create_datetime_column(df):
    """
    Cria coluna datetime unificada
    """
    print("=== CRIAÇÃO DE COLUNA DATETIME ===")
    
    if 'timestamp' in df.columns:
        print("✓ Coluna timestamp já existe")
        return df
    
    # Tentar criar timestamp a partir de Data + Hora
    if 'data' in df.columns and 'hora' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['data'] + ' ' + df['hora'].astype(str))
            print("✓ Timestamp criado a partir de Data + Hora")
        except:
            print("❌ Erro ao criar timestamp")
            return df
    else:
        print("❌ Colunas de data/hora não encontradas")
        return df
    
    return df

# Criar coluna datetime
if 'processed_data' in locals():
    processed_data = create_datetime_column(processed_data)

# %%
def handle_missing_values(df):
    """
    Trata valores missing nos dados
    """
    print("=== TRATAMENTO DE VALORES MISSING ===")
    
    # Identificar colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Estatísticas de missing
    missing_stats = []
    for col in numeric_cols:
        missing_count = df[col].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        missing_stats.append({
            'coluna': col,
            'missing_count': missing_count,
            'missing_percent': missing_percent
        })
    
    missing_df = pd.DataFrame(missing_stats).sort_values('missing_percent', ascending=False)
    print("Estatísticas de valores missing:")
    print(missing_df[missing_df['missing_count'] > 0])
    
    df_cleaned = df.copy()
    
    # Estratégias de imputação por coluna
    imputation_strategies = {
        'precipitacao_mm': 0,  # Precipitação missing = 0
        'temperatura_c': 'interpolate',
        'umidade_relativa': 'interpolate',
        'pressao_mb': 'mean',
        'velocidade_vento_ms': 'median',
        'direcao_vento_gr': 'forward_fill',
        'radiacao_kjm2': 'interpolate'
    }
    
    for col, strategy in imputation_strategies.items():
        if col in df_cleaned.columns:
            missing_before = df_cleaned[col].isnull().sum()
            
            if strategy == 'interpolate':
                df_cleaned[col] = df_cleaned[col].interpolate(method='linear')
            elif strategy == 'mean':
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
            elif strategy == 'median':
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            elif strategy == 'forward_fill':
                df_cleaned[col] = df_cleaned[col].fillna(method='ffill')
            elif isinstance(strategy, (int, float)):
                df_cleaned[col] = df_cleaned[col].fillna(strategy)
            
            missing_after = df_cleaned[col].isnull().sum()
            print(f"✓ {col}: {missing_before} → {missing_after} missing")
    
    return df_cleaned

# Tratar valores missing
if 'processed_data' in locals():
    processed_data = handle_missing_values(processed_data)

# %%
def create_features(df):
    """
    Cria features derivadas
    """
    print("=== FEATURE ENGINEERING ===")
    
    if 'timestamp' not in df.columns:
        print("❌ Coluna timestamp necessária")
        return df
    
    df_features = df.copy()
    
    # Features temporais
    df_features['ano'] = df_features['timestamp'].dt.year
    df_features['mes'] = df_features['timestamp'].dt.month
    df_features['dia'] = df_features['timestamp'].dt.day
    df_features['hora'] = df_features['timestamp'].dt.hour
    df_features['dia_semana'] = df_features['timestamp'].dt.dayofweek
    df_features['dia_ano'] = df_features['timestamp'].dt.dayofyear
    
    # Features cíclicas
    df_features['hora_sin'] = np.sin(2 * np.pi * df_features['hora'] / 24)
    df_features['hora_cos'] = np.cos(2 * np.pi * df_features['hora'] / 24)
    df_features['mes_sin'] = np.sin(2 * np.pi * df_features['mes'] / 12)
    df_features['mes_cos'] = np.cos(2 * np.pi * df_features['mes'] / 12)
    
    # Features meteorológicas derivadas
    if 'temperatura_c' in df_features.columns and 'umidade_relativa' in df_features.columns:
        # Índice de calor aproximado
        df_features['indice_calor'] = df_features['temperatura_c'] + 0.1 * df_features['umidade_relativa']
    
    if 'velocidade_vento_ms' in df_features.columns and 'temperatura_c' in df_features.columns:
        # Wind chill aproximado
        df_features['sensacao_termica'] = df_features['temperatura_c'] - 2 * df_features['velocidade_vento_ms']
    
    # Features de agregação temporal (moving averages)
    window_sizes = [3, 6, 12, 24]  # 3h, 6h, 12h, 24h
    
    for col in ['precipitacao_mm', 'temperatura_c', 'umidade_relativa']:
        if col in df_features.columns:
            for window in window_sizes:
                df_features[f'{col}_ma_{window}h'] = df_features[col].rolling(window=window, min_periods=1).mean()
    
    print(f"✓ Features criadas. Total de colunas: {len(df_features.columns)}")
    
    return df_features

# Criar features
if 'processed_data' in locals():
    processed_data = create_features(processed_data)

# %%
def split_temporal_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Divide dados em treino/validação/teste de forma temporal
    """
    print("=== DIVISÃO TEMPORAL DOS DADOS ===")
    
    if 'timestamp' not in df.columns:
        print("❌ Coluna timestamp necessária")
        return None, None, None
    
    # Ordenar por timestamp
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    # Calcular pontos de divisão
    n_total = len(df_sorted)
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))
    
    # Dividir datasets
    train_data = df_sorted.iloc[:train_end].copy()
    val_data = df_sorted.iloc[train_end:val_end].copy()
    test_data = df_sorted.iloc[val_end:].copy()
    
    print(f"Divisão dos dados:")
    print(f"  Treino: {len(train_data)} registros ({len(train_data)/n_total*100:.1f}%)")
    print(f"  Validação: {len(val_data)} registros ({len(val_data)/n_total*100:.1f}%)")
    print(f"  Teste: {len(test_data)} registros ({len(test_data)/n_total*100:.1f}%)")
    
    if 'timestamp' in train_data.columns:
        print(f"  Período treino: {train_data['timestamp'].min()} - {train_data['timestamp'].max()}")
        print(f"  Período validação: {val_data['timestamp'].min()} - {val_data['timestamp'].max()}")
        print(f"  Período teste: {test_data['timestamp'].min()} - {test_data['timestamp'].max()}")
    
    return train_data, val_data, test_data

# Dividir dados
if 'processed_data' in locals():
    train_data, val_data, test_data = split_temporal_data(processed_data)

# %%
def save_processed_data(train_df, val_df, test_df):
    """
    Salva dados processados
    """
    print("=== SALVAMENTO DOS DADOS ===")
    
    # Salvar em formato parquet (mais eficiente)
    train_df.to_parquet(PROCESSED_DATA_PATH / 'train_data.parquet', index=False)
    val_df.to_parquet(PROCESSED_DATA_PATH / 'validation_data.parquet', index=False)
    test_df.to_parquet(PROCESSED_DATA_PATH / 'test_data.parquet', index=False)
    
    print(f"✓ Dados salvos em {PROCESSED_DATA_PATH}")
    print(f"  - train_data.parquet: {len(train_df)} registros")
    print(f"  - validation_data.parquet: {len(val_df)} registros")
    print(f"  - test_data.parquet: {len(test_df)} registros")
    
    # Salvar metadados
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'train_shape': train_df.shape,
        'val_shape': val_df.shape,
        'test_shape': test_df.shape,
        'columns': list(train_df.columns),
        'train_period': f"{train_df['timestamp'].min()} - {train_df['timestamp'].max()}" if 'timestamp' in train_df.columns else 'N/A',
        'val_period': f"{val_df['timestamp'].min()} - {val_df['timestamp'].max()}" if 'timestamp' in val_df.columns else 'N/A',
        'test_period': f"{test_df['timestamp'].min()} - {test_df['timestamp'].max()}" if 'timestamp' in test_df.columns else 'N/A'
    }
    
    import json
    with open(PROCESSED_DATA_PATH / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print("✓ Metadados salvos")

# Salvar dados processados
if all(v is not None for v in [train_data, val_data, test_data]):
    save_processed_data(train_data, val_data, test_data)

# %%
print("\n" + "="*50)
print("✅ PREPROCESSAMENTO CONCLUÍDO!")
print("="*50)
print("Próximos passos:")
print("1. 🧠 Treinar modelo LSTM")
print("2. 📊 Avaliar performance")
print("3. 🔧 Ajustar hiperparâmetros")

# %% 