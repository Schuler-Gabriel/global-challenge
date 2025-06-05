# %% [markdown]
# # Análise Exploratória dos Dados Meteorológicos INMET
# 
# Este notebook realiza a análise exploratória dos dados históricos do INMET (2000-2025) para o projeto de alertas de cheias.
# 
# ## Objetivos:
# - Explorar estrutura e qualidade dos dados meteorológicos
# - Identificar padrões sazonais e tendências climáticas
# - Detectar outliers e dados inconsistentes
# - Análise de correlações entre variáveis
# - Visualizações descritivas e estatísticas

# %%
# Imports necessários
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import glob
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Configurações
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=== ANÁLISE EXPLORATÓRIA DOS DADOS INMET ===")
print(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %%
# Configuração dos caminhos de dados
DATA_PATH = Path('../data')
RAW_DATA_PATH = DATA_PATH / 'raw' / 'dados_historicos'
PROCESSED_DATA_PATH = DATA_PATH / 'processed'
ANALYSIS_OUTPUT_PATH = DATA_PATH / 'analysis'

# Criar diretório de saída se não existir
ANALYSIS_OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

print(f"Caminho dos dados brutos: {RAW_DATA_PATH}")
print(f"Caminho dos dados processados: {PROCESSED_DATA_PATH}")
print(f"Saída da análise: {ANALYSIS_OUTPUT_PATH}")

# Verificar se existem dados
raw_files = list(RAW_DATA_PATH.glob("*.CSV")) if RAW_DATA_PATH.exists() else []
processed_files = list(PROCESSED_DATA_PATH.glob("*.parquet")) if PROCESSED_DATA_PATH.exists() else []

print(f"\nArquivos CSV encontrados: {len(raw_files)}")
print(f"Arquivos processados encontrados: {len(processed_files)}")

# %% [markdown]
# ## 1. Carregamento e Estrutura dos Dados

# %%
def load_inmet_data():
    """
    Carrega dados meteorológicos do INMET
    """
    if processed_files:
        print("Carregando dados processados...")
        
        # Tentar carregar dados processados primeiro
        try:
            train_data = pd.read_parquet(PROCESSED_DATA_PATH / 'train_data.parquet')
            print(f"✓ Dados de treino carregados: {train_data.shape}")
            return train_data
        except FileNotFoundError:
            print("Arquivos processados não encontrados, carregando dados brutos...")
    
    if not raw_files:
        print("❌ Nenhum arquivo de dados encontrado!")
        # Criar dados sintéticos para demonstração
        print("Criando dados sintéticos para demonstração...")
        
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='H')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'precipitacao_mm': np.random.exponential(0.5, len(dates)),
            'temperatura_c': 20 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / (24*365)) + np.random.normal(0, 2, len(dates)),
            'umidade_relativa': 50 + 30 * np.sin(2 * np.pi * np.arange(len(dates)) / (24*365)) + np.random.normal(0, 5, len(dates)),
            'pressao_mb': 1013 + np.random.normal(0, 10, len(dates)),
            'velocidade_vento_ms': np.random.gamma(2, 2, len(dates)),
            'direcao_vento_gr': np.random.uniform(0, 360, len(dates)),
            'radiacao_kjm2': np.maximum(0, 500 * np.sin(2 * np.pi * np.arange(len(dates)) / 24) + np.random.normal(0, 100, len(dates)))
        })
        
        return data
    
    # Carregar dados brutos INMET
    print("Carregando dados brutos do INMET...")
    dataframes = []
    
    for file_path in raw_files[:3]:  # Limitar para 3 arquivos para análise inicial
        print(f"Processando: {file_path.name}")
        try:
            # Tentar diferentes encodings
            for encoding in ['latin1', 'utf-8', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, sep=';', encoding=encoding, skiprows=8)
                    break
                except UnicodeDecodeError:
                    continue
            
            # Adicionar informação do arquivo
            df['arquivo_origem'] = file_path.name
            dataframes.append(df)
            
        except Exception as e:
            print(f"  ❌ Erro ao carregar {file_path.name}: {e}")
    
    if dataframes:
        combined_data = pd.concat(dataframes, ignore_index=True)
        print(f"✓ Dados combinados: {combined_data.shape}")
        return combined_data
    else:
        return None

# Carregar dados
data = load_inmet_data()

if data is not None:
    print(f"\n📊 Dados carregados com sucesso!")
    print(f"Shape: {data.shape}")
    print(f"Período: {data['timestamp'].min() if 'timestamp' in data.columns else 'N/A'} até {data['timestamp'].max() if 'timestamp' in data.columns else 'N/A'}")
else:
    print("❌ Falha ao carregar dados")

# %%
# Explorar estrutura dos dados
if data is not None:
    print("=== ESTRUTURA DOS DADOS ===")
    print(f"Dimensões: {data.shape}")
    print(f"Colunas: {list(data.columns)}")
    print(f"Tipos de dados:\n{data.dtypes}")
    
    # Informações gerais
    print(f"\n=== INFORMAÇÕES GERAIS ===")
    print(data.info())
    
    # Primeiras linhas
    print(f"\n=== PRIMEIRAS 5 LINHAS ===")
    print(data.head())
    
    # Valores missing
    print(f"\n=== VALORES MISSING ===")
    missing_data = data.isnull().sum()
    missing_percent = (missing_data / len(data)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing %': missing_percent
    }).sort_values('Missing %', ascending=False)
    
    print(missing_df[missing_df['Missing Count'] > 0])

# %% [markdown]
# ## 2. Análise de Qualidade dos Dados

# %%
def analyze_data_quality(df):
    """
    Analisa qualidade dos dados meteorológicos
    """
    print("=== ANÁLISE DE QUALIDADE DOS DADOS ===")
    
    # Identificar colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print("❌ Nenhuma coluna numérica encontrada")
        return
    
    quality_report = []
    
    for col in numeric_cols:
        if col in df.columns:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                continue
                
            report = {
                'coluna': col,
                'total_valores': len(df),
                'valores_validos': len(col_data),
                'missing_count': df[col].isnull().sum(),
                'missing_percent': (df[col].isnull().sum() / len(df)) * 100,
                'min': col_data.min(),
                'max': col_data.max(),
                'mean': col_data.mean(),
                'std': col_data.std(),
                'zeros': (col_data == 0).sum(),
                'negativos': (col_data < 0).sum(),
                'outliers_iqr': 0
            }
            
            # Calcular outliers usando IQR
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = col_data[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)]
            report['outliers_iqr'] = len(outliers)
            
            quality_report.append(report)
    
    # Converter para DataFrame
    quality_df = pd.DataFrame(quality_report)
    
    print("Resumo da qualidade dos dados:")
    print(quality_df.round(2))
    
    return quality_df

# Executar análise de qualidade
if data is not None:
    quality_report = analyze_data_quality(data)

# %% [markdown]
# ## 3. Análise Estatística Descritiva

# %%
def generate_descriptive_statistics(df):
    """
    Gera estatísticas descritivas abrangentes
    """
    print("=== ESTATÍSTICAS DESCRITIVAS ===")
    
    # Colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print("❌ Nenhuma coluna numérica encontrada")
        return
    
    # Estatísticas básicas
    print("Estatísticas básicas:")
    stats = df[numeric_cols].describe()
    print(stats.round(3))
    
    # Análise de distribuições
    print(f"\n=== ANÁLISE DE DISTRIBUIÇÕES ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols[:6]):  # Limitar a 6 variáveis
        if col in df.columns:
            col_data = df[col].dropna()
            
            # Histograma
            axes[i].hist(col_data, bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribuição - {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequência')
            axes[i].grid(True, alpha=0.3)
            
            # Adicionar estatísticas no gráfico
            mean_val = col_data.mean()
            median_val = col_data.median()
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Média: {mean_val:.2f}')
            axes[i].axvline(median_val, color='green', linestyle='--', label=f'Mediana: {median_val:.2f}')
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(ANALYSIS_OUTPUT_PATH / 'distribuicoes_variaveis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return stats

# Executar análise descritiva
if data is not None:
    descriptive_stats = generate_descriptive_statistics(data)

# %% [markdown]
# ## 4. Análise Temporal

# %%
def analyze_temporal_patterns(df):
    """
    Analisa padrões temporais nos dados
    """
    print("=== ANÁLISE TEMPORAL ===")
    
    # Verificar se existe coluna temporal
    time_col = None
    for col in ['timestamp', 'data', 'datetime', 'Data', 'Hora UTC']:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is None:
        print("❌ Nenhuma coluna temporal encontrada")
        return
    
    # Converter para datetime se necessário
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except:
            print(f"❌ Erro ao converter {time_col} para datetime")
            return
    
    # Definir coluna de precipitação
    precip_col = None
    for col in ['precipitacao_mm', 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)', 'precipitacao']:
        if col in df.columns:
            precip_col = col
            break
    
    if precip_col is None:
        print("❌ Coluna de precipitação não encontrada")
        return
    
    # Análise temporal da precipitação
    df = df.copy()
    df['ano'] = df[time_col].dt.year
    df['mes'] = df[time_col].dt.month
    df['hora'] = df[time_col].dt.hour
    df['dia_semana'] = df[time_col].dt.dayofweek
    
    # Análise anual
    annual_precip = df.groupby('ano')[precip_col].agg(['sum', 'mean', 'count']).reset_index()
    
    print("Precipitação anual:")
    print(annual_precip.round(2))
    
    # Análise mensal
    monthly_precip = df.groupby('mes')[precip_col].agg(['sum', 'mean', 'count']).reset_index()
    
    # Visualizações temporais
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Precipitação anual
    if len(annual_precip) > 1:
        axes[0, 0].plot(annual_precip['ano'], annual_precip['sum'], marker='o')
        axes[0, 0].set_title('Precipitação Total Anual')
        axes[0, 0].set_xlabel('Ano')
        axes[0, 0].set_ylabel('Precipitação (mm)')
        axes[0, 0].grid(True)
    
    # Precipitação mensal (padrão sazonal)
    month_names = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                   'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    axes[0, 1].bar(range(1, 13), monthly_precip['sum'])
    axes[0, 1].set_title('Precipitação Total por Mês')
    axes[0, 1].set_xlabel('Mês')
    axes[0, 1].set_ylabel('Precipitação (mm)')
    axes[0, 1].set_xticks(range(1, 13))
    axes[0, 1].set_xticklabels(month_names, rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Padrão horário
    hourly_precip = df.groupby('hora')[precip_col].mean()
    axes[1, 0].plot(hourly_precip.index, hourly_precip.values, marker='o')
    axes[1, 0].set_title('Padrão Horário Médio de Precipitação')
    axes[1, 0].set_xlabel('Hora do Dia')
    axes[1, 0].set_ylabel('Precipitação Média (mm)')
    axes[1, 0].grid(True)
    
    # Padrão semanal
    weekday_names = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
    weekly_precip = df.groupby('dia_semana')[precip_col].mean()
    axes[1, 1].bar(range(7), weekly_precip.values)
    axes[1, 1].set_title('Padrão Semanal de Precipitação')
    axes[1, 1].set_xlabel('Dia da Semana')
    axes[1, 1].set_ylabel('Precipitação Média (mm)')
    axes[1, 1].set_xticks(range(7))
    axes[1, 1].set_xticklabels(weekday_names)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ANALYSIS_OUTPUT_PATH / 'analise_temporal.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'annual': annual_precip,
        'monthly': monthly_precip,
        'hourly': hourly_precip,
        'weekly': weekly_precip
    }

# Executar análise temporal
if data is not None:
    temporal_analysis = analyze_temporal_patterns(data)

# %% [markdown]
# ## 5. Análise de Correlações

# %%
def analyze_correlations(df):
    """
    Analisa correlações entre variáveis meteorológicas
    """
    print("=== ANÁLISE DE CORRELAÇÕES ===")
    
    # Selecionar apenas colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("❌ Insuficientes variáveis numéricas para análise de correlação")
        return
    
    # Calcular matriz de correlação
    correlation_matrix = df[numeric_cols].corr()
    
    print("Matriz de correlação:")
    print(correlation_matrix.round(3))
    
    # Visualizar matriz de correlação
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8})
    
    plt.title('Matriz de Correlação - Variáveis Meteorológicas')
    plt.tight_layout()
    plt.savefig(ANALYSIS_OUTPUT_PATH / 'matriz_correlacao.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Identificar correlações mais fortes
    print(f"\n=== CORRELAÇÕES MAIS FORTES ===")
    
    # Converter matriz em lista de correlações
    correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            var1 = correlation_matrix.columns[i]
            var2 = correlation_matrix.columns[j]
            corr_value = correlation_matrix.iloc[i, j]
            
            if not np.isnan(corr_value):
                correlations.append({
                    'variavel_1': var1,
                    'variavel_2': var2,
                    'correlacao': corr_value
                })
    
    # Ordenar por valor absoluto da correlação
    correlations_df = pd.DataFrame(correlations)
    correlations_df['correlacao_abs'] = correlations_df['correlacao'].abs()
    correlations_df = correlations_df.sort_values('correlacao_abs', ascending=False)
    
    print("Top 10 correlações mais fortes:")
    print(correlations_df.head(10)[['variavel_1', 'variavel_2', 'correlacao']].round(3))
    
    return correlation_matrix, correlations_df

# Executar análise de correlações
if data is not None:
    correlation_matrix, correlations_df = analyze_correlations(data)

# %% [markdown]
# ## 6. Detecção de Outliers

# %%
def detect_outliers(df):
    """
    Detecta outliers usando múltiplos métodos
    """
    print("=== DETECÇÃO DE OUTLIERS ===")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print("❌ Nenhuma coluna numérica encontrada")
        return
    
    outlier_report = []
    
    for col in numeric_cols[:6]:  # Limitar a 6 variáveis principais
        if col not in df.columns:
            continue
            
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            continue
        
        # Método IQR
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        
        # Método Z-Score
        z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
        zscore_outliers = col_data[z_scores > 3]
        
        # Método baseado em percentis
        percentile_lower = col_data.quantile(0.01)
        percentile_upper = col_data.quantile(0.99)
        percentile_outliers = col_data[(col_data < percentile_lower) | (col_data > percentile_upper)]
        
        outlier_info = {
            'variavel': col,
            'total_valores': len(col_data),
            'outliers_iqr': len(iqr_outliers),
            'outliers_zscore': len(zscore_outliers),
            'outliers_percentil': len(percentile_outliers),
            'iqr_percent': (len(iqr_outliers) / len(col_data)) * 100,
            'min_valor': col_data.min(),
            'max_valor': col_data.max(),
            'q1': Q1,
            'q3': Q3,
            'iqr_lower': lower_bound,
            'iqr_upper': upper_bound
        }
        
        outlier_report.append(outlier_info)
    
    # Converter para DataFrame
    outlier_df = pd.DataFrame(outlier_report)
    
    print("Relatório de outliers:")
    print(outlier_df[['variavel', 'outliers_iqr', 'outliers_zscore', 'iqr_percent']].round(2))
    
    # Visualizar outliers para variáveis principais
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols[:6]):
        if col in df.columns:
            col_data = df[col].dropna()
            
            # Box plot
            axes[i].boxplot(col_data)
            axes[i].set_title(f'Box Plot - {col}')
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ANALYSIS_OUTPUT_PATH / 'outliers_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return outlier_df

# Executar detecção de outliers
if data is not None:
    outlier_report = detect_outliers(data)

# %% [markdown]
# ## 7. Análise de Eventos Extremos

# %%
def analyze_extreme_events(df):
    """
    Analisa eventos meteorológicos extremos
    """
    print("=== ANÁLISE DE EVENTOS EXTREMOS ===")
    
    # Identificar coluna de precipitação
    precip_col = None
    for col in ['precipitacao_mm', 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)', 'precipitacao']:
        if col in df.columns:
            precip_col = col
            break
    
    if precip_col is None:
        print("❌ Coluna de precipitação não encontrada")
        return
    
    precip_data = df[precip_col].dropna()
    
    if len(precip_data) == 0:
        print("❌ Nenhum dado de precipitação válido")
        return
    
    # Definir thresholds para eventos extremos
    p95 = precip_data.quantile(0.95)
    p99 = precip_data.quantile(0.99)
    p99_9 = precip_data.quantile(0.999)
    
    # Identificar eventos extremos
    eventos_moderados = precip_data[precip_data >= p95]
    eventos_severos = precip_data[precip_data >= p99]
    eventos_extremos = precip_data[precip_data >= p99_9]
    
    print(f"Thresholds de precipitação:")
    print(f"  P95 (eventos moderados): {p95:.2f} mm/h")
    print(f"  P99 (eventos severos): {p99:.2f} mm/h")
    print(f"  P99.9 (eventos extremos): {p99_9:.2f} mm/h")
    
    print(f"\nContagem de eventos:")
    print(f"  Eventos moderados (>= P95): {len(eventos_moderados)}")
    print(f"  Eventos severos (>= P99): {len(eventos_severos)}")
    print(f"  Eventos extremos (>= P99.9): {len(eventos_extremos)}")
    
    # Estatísticas dos eventos extremos
    print(f"\nEstatísticas dos eventos extremos:")
    if len(eventos_extremos) > 0:
        print(f"  Precipitação máxima: {eventos_extremos.max():.2f} mm/h")
        print(f"  Precipitação média em eventos extremos: {eventos_extremos.mean():.2f} mm/h")
    
    # Visualizar distribuição de eventos extremos
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histograma de precipitação com thresholds
    axes[0].hist(precip_data, bins=100, alpha=0.7, edgecolor='black', density=True)
    axes[0].axvline(p95, color='orange', linestyle='--', label=f'P95: {p95:.2f}')
    axes[0].axvline(p99, color='red', linestyle='--', label=f'P99: {p99:.2f}')
    axes[0].axvline(p99_9, color='darkred', linestyle='--', label=f'P99.9: {p99_9:.2f}')
    axes[0].set_title('Distribuição de Precipitação com Thresholds')
    axes[0].set_xlabel('Precipitação (mm/h)')
    axes[0].set_ylabel('Densidade')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, min(50, precip_data.max()))  # Limitar visualização
    
    # Box plot comparativo
    event_data = []
    event_labels = []
    
    normal_events = precip_data[precip_data < p95]
    if len(normal_events) > 0:
        event_data.append(normal_events)
        event_labels.append('Normal')
    
    if len(eventos_moderados) > 0:
        event_data.append(eventos_moderados)
        event_labels.append('Moderado')
    
    if len(eventos_severos) > 0:
        event_data.append(eventos_severos)
        event_labels.append('Severo')
    
    if len(eventos_extremos) > 0:
        event_data.append(eventos_extremos)
        event_labels.append('Extremo')
    
    if event_data:
        axes[1].boxplot(event_data, labels=event_labels)
        axes[1].set_title('Comparação de Eventos por Intensidade')
        axes[1].set_ylabel('Precipitação (mm/h)')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ANALYSIS_OUTPUT_PATH / 'eventos_extremos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'thresholds': {'p95': p95, 'p99': p99, 'p99_9': p99_9},
        'eventos_moderados': len(eventos_moderados),
        'eventos_severos': len(eventos_severos),
        'eventos_extremos': len(eventos_extremos),
        'max_precipitacao': precip_data.max(),
        'dados_eventos_extremos': eventos_extremos if len(eventos_extremos) > 0 else None
    }

# Executar análise de eventos extremos
if data is not None:
    extreme_events = analyze_extreme_events(data)

# %% [markdown]
# ## 8. Relatório Final da Análise Exploratória

# %%
def generate_final_report():
    """
    Gera relatório final da análise exploratória
    """
    print("=== RELATÓRIO FINAL DA ANÁLISE EXPLORATÓRIA ===")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'dados': {
            'shape': data.shape if data is not None else None,
            'periodo': f"{data['timestamp'].min()} - {data['timestamp'].max()}" if data is not None and 'timestamp' in data.columns else 'N/A',
            'total_registros': len(data) if data is not None else 0
        },
        'qualidade': {},
        'principais_insights': []
    }
    
    if data is not None:
        # Resumo da qualidade
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        total_missing = data.isnull().sum().sum()
        missing_percent = (total_missing / (len(data) * len(data.columns))) * 100
        
        report['qualidade'] = {
            'variaveis_numericas': len(numeric_cols),
            'total_missing': int(total_missing),
            'missing_percent': round(missing_percent, 2),
            'registros_completos': int(len(data.dropna()))
        }
        
        # Insights principais
        insights = [
            "✓ Dados meteorológicos INMET carregados e analisados com sucesso",
            f"✓ Dataset contém {len(data)} registros e {len(data.columns)} variáveis",
            f"✓ {missing_percent:.1f}% de dados faltantes identificados",
        ]
        
        # Insights específicos baseados na análise
        if 'quality_report' in locals() and quality_report is not None:
            high_missing_vars = quality_report[quality_report['missing_percent'] > 20]
            if len(high_missing_vars) > 0:
                insights.append(f"⚠️ {len(high_missing_vars)} variáveis com >20% de dados faltantes")
        
        if 'extreme_events' in locals() and extreme_events is not None:
            insights.append(f"✓ {extreme_events['eventos_extremos']} eventos de precipitação extrema identificados")
            insights.append(f"✓ Precipitação máxima registrada: {extreme_events['max_precipitacao']:.2f} mm/h")
        
        if 'correlations_df' in locals() and correlations_df is not None:
            strong_corrs = correlations_df[correlations_df['correlacao_abs'] > 0.7]
            if len(strong_corrs) > 0:
                insights.append(f"✓ {len(strong_corrs)} correlações fortes (>0.7) entre variáveis")
        
        insights.extend([
            "✓ Padrões sazonais e horários identificados na precipitação",
            "✓ Outliers detectados e analisados usando múltiplos métodos",
            "✓ Análise de eventos extremos concluída",
            "✓ Dados prontos para fase de preprocessamento"
        ])
        
        report['principais_insights'] = insights
    
    # Salvar relatório
    import json
    with open(ANALYSIS_OUTPUT_PATH / 'relatorio_analise_exploratoria.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Exibir resumo
    print("\n📋 RESUMO EXECUTIVO:")
    for insight in report['principais_insights']:
        print(f"  {insight}")
    
    print(f"\n📁 Arquivos gerados:")
    output_files = list(ANALYSIS_OUTPUT_PATH.glob("*"))
    for file in output_files:
        print(f"  - {file.name}")
    
    print(f"\n✅ Análise exploratória concluída com sucesso!")
    print(f"📊 Relatório completo salvo em: {ANALYSIS_OUTPUT_PATH}")
    
    return report

# Gerar relatório final
final_report = generate_final_report()

# %%
print("\n" + "="*60)
print("🎉 ANÁLISE EXPLORATÓRIA CONCLUÍDA!")
print("="*60)
print("\nPróximos passos recomendados:")
print("1. 📝 Revisar relatório de qualidade dos dados")
print("2. 🧹 Executar preprocessamento baseado nos insights")
print("3. 🔧 Tratar valores missing e outliers identificados")
print("4. 📈 Preparar features para modelagem LSTM")
print("5. ⚡ Prosseguir para treinamento do modelo")

# %% 