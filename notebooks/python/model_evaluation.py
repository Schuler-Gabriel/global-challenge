# %% [markdown]
# # Avaliação do Modelo LSTM - Previsão Meteorológica
# 
# Este notebook avalia a performance do modelo LSTM treinado para previsão de chuva.
# 
# ## Objetivos:
# - Carregar modelo treinado
# - Avaliar métricas de performance
# - Análise de erros e casos extremos
# - Visualizações de resultados
# - Relatório final de avaliação

# %%
# Imports necessários
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

warnings.filterwarnings('ignore')

print("=== AVALIAÇÃO DO MODELO LSTM ===")
print(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %%
# Configuração dos caminhos
DATA_PATH = Path('../data')
PROCESSED_DATA_PATH = DATA_PATH / 'processed'
MODELS_PATH = DATA_PATH / 'modelos_treinados'
EVALUATION_PATH = DATA_PATH / 'evaluation'

EVALUATION_PATH.mkdir(exist_ok=True, parents=True)

print(f"Dados processados: {PROCESSED_DATA_PATH}")
print(f"Modelos: {MODELS_PATH}")
print(f"Avaliação: {EVALUATION_PATH}")

# %%
def load_test_data():
    """
    Carrega dados de teste
    """
    print("=== CARREGAMENTO DOS DADOS DE TESTE ===")
    
    try:
        test_data = pd.read_parquet(PROCESSED_DATA_PATH / 'test_data.parquet')
        print(f"✓ Dados de teste carregados: {test_data.shape}")
        return test_data
    except FileNotFoundError:
        print("❌ Dados de teste não encontrados!")
        # Criar dados sintéticos para demonstração
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='H')
        np.random.seed(42)
        
        test_data = pd.DataFrame({
            'timestamp': dates,
            'precipitacao_mm': np.random.exponential(0.3, len(dates)),
            'temperatura_c': 20 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / (24*365)) + np.random.normal(0, 3, len(dates)),
            'umidade_relativa': np.clip(50 + 30 * np.sin(2 * np.pi * np.arange(len(dates)) / (24*365)) + np.random.normal(0, 8, len(dates)), 0, 100),
            'pressao_mb': 1013 + np.random.normal(0, 15, len(dates)),
            'velocidade_vento_ms': np.random.gamma(2, 1.5, len(dates)),
            'direcao_vento_gr': np.random.uniform(0, 360, len(dates)),
            'radiacao_kjm2': np.maximum(0, 800 * np.sin(2 * np.pi * np.arange(len(dates)) / 24) + np.random.normal(0, 150, len(dates)))
        })
        
        print("📊 Dados sintéticos criados para demonstração")
        return test_data

# Carregar dados de teste
test_data = load_test_data()

# %%
def load_trained_model():
    """
    Carrega modelo LSTM treinado
    """
    print("=== CARREGAMENTO DO MODELO ===")
    
    # Procurar modelos disponíveis
    if MODELS_PATH.exists():
        model_files = list(MODELS_PATH.glob("*.h5")) + list(MODELS_PATH.glob("*.keras"))
        
        if model_files:
            # Usar o modelo mais recente
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            print(f"Carregando modelo: {latest_model.name}")
            
            try:
                model = tf.keras.models.load_model(latest_model)
                print(f"✓ Modelo carregado com sucesso")
                print(f"Arquitetura: {model.summary()}")
                return model
            except Exception as e:
                print(f"❌ Erro ao carregar modelo: {e}")
        else:
            print("❌ Nenhum modelo encontrado!")
    else:
        print("❌ Diretório de modelos não existe!")
    
    # Criar modelo dummy para demonstração
    print("🔧 Criando modelo dummy para demonstração...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(24, 8)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("✓ Modelo dummy criado")
    
    return model

# Carregar modelo
model = load_trained_model()

# %%
def prepare_sequences_for_prediction(data, sequence_length=24):
    """
    Prepara sequências para predição
    """
    print("=== PREPARAÇÃO DE SEQUÊNCIAS ===")
    
    # Selecionar features numéricas
    feature_cols = ['precipitacao_mm', 'temperatura_c', 'umidade_relativa', 'pressao_mb',
                   'velocidade_vento_ms', 'direcao_vento_gr', 'radiacao_kjm2']
    
    # Usar apenas colunas disponíveis
    available_cols = [col for col in feature_cols if col in data.columns]
    
    if len(available_cols) < 3:
        print("❌ Insuficientes features numéricas")
        return None, None, None
    
    print(f"Features utilizadas: {available_cols}")
    
    # Extrair features
    features = data[available_cols].values
    
    # Preencher com colunas extras se necessário (para compatibilidade)
    if features.shape[1] < 8:
        padding = np.zeros((features.shape[0], 8 - features.shape[1]))
        features = np.hstack([features, padding])
    
    # Criar sequências
    X, y = [], []
    
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(features[i, 0])  # Predizer precipitação (primeira coluna)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✓ Sequências criadas: X={X.shape}, y={y.shape}")
    
    return X, y, available_cols

# Preparar sequências
X_test, y_test, feature_names = prepare_sequences_for_prediction(test_data)

# %%
def evaluate_regression_metrics(model, X_test, y_test):
    """
    Avalia métricas de regressão
    """
    print("=== MÉTRICAS DE REGRESSÃO ===")
    
    if X_test is None or y_test is None:
        print("❌ Dados de teste não disponíveis")
        return None
    
    # Fazer predições
    print("Fazendo predições...")
    y_pred = model.predict(X_test, verbose=0)
    y_pred = y_pred.flatten()
    
    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Métricas específicas para precipitação
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 0.01))) * 100
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape
    }
    
    print("Métricas de regressão:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Verificar se atende critérios
    print(f"\n📊 Avaliação dos critérios:")
    print(f"  MAE < 2.0 mm/h: {'✅' if mae < 2.0 else '❌'} (atual: {mae:.3f})")
    print(f"  RMSE < 3.0 mm/h: {'✅' if rmse < 3.0 else '❌'} (atual: {rmse:.3f})")
    print(f"  R² > 0.5: {'✅' if r2 > 0.5 else '❌'} (atual: {r2:.3f})")
    
    return metrics, y_pred

# Avaliar métricas de regressão
if model is not None:
    regression_metrics, predictions = evaluate_regression_metrics(model, X_test, y_test)

# %%
def evaluate_classification_metrics(y_true, y_pred, threshold=1.0):
    """
    Avalia métricas de classificação para eventos de chuva
    """
    print("=== MÉTRICAS DE CLASSIFICAÇÃO ===")
    
    # Converter para classificação binária (chuva/sem chuva)
    y_true_class = (y_true >= threshold).astype(int)
    y_pred_class = (y_pred >= threshold).astype(int)
    
    # Métricas de classificação
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true_class, y_pred_class)
    precision = precision_score(y_true_class, y_pred_class, zero_division=0)
    recall = recall_score(y_true_class, y_pred_class, zero_division=0)
    f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
    
    print(f"Métricas de classificação (threshold={threshold} mm/h):")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")
    
    # Verificar critério de accuracy > 75%
    print(f"\n📊 Critério de classificação:")
    print(f"  Accuracy > 75%: {'✅' if accuracy > 0.75 else '❌'} (atual: {accuracy:.1%})")
    
    # Matriz de confusão
    cm = confusion_matrix(y_true_class, y_pred_class)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Sem Chuva', 'Com Chuva'],
                yticklabels=['Sem Chuva', 'Com Chuva'])
    plt.title('Matriz de Confusão - Eventos de Chuva')
    plt.ylabel('Valores Reais')
    plt.xlabel('Predições')
    plt.tight_layout()
    plt.savefig(EVALUATION_PATH / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

# Avaliar métricas de classificação
if 'predictions' in locals() and y_test is not None:
    classification_metrics = evaluate_classification_metrics(y_test, predictions)

# %%
def analyze_prediction_errors(y_true, y_pred):
    """
    Analisa erros de predição
    """
    print("=== ANÁLISE DE ERROS ===")
    
    # Calcular erros
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    # Estatísticas dos erros
    print("Estatísticas dos erros:")
    print(f"  Erro médio: {np.mean(errors):.4f}")
    print(f"  Erro absoluto médio: {np.mean(abs_errors):.4f}")
    print(f"  Desvio padrão dos erros: {np.std(errors):.4f}")
    print(f"  Erro máximo: {np.max(abs_errors):.4f}")
    
    # Percentis dos erros
    print(f"\nPercentis dos erros absolutos:")
    for p in [50, 75, 90, 95, 99]:
        print(f"  P{p}: {np.percentile(abs_errors, p):.4f}")
    
    # Visualizações dos erros
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scatter plot: predito vs real
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Valores Reais')
    axes[0, 0].set_ylabel('Predições')
    axes[0, 0].set_title('Predito vs Real')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Distribuição dos erros
    axes[0, 1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Erro (Predito - Real)')
    axes[0, 1].set_ylabel('Frequência')
    axes[0, 1].set_title('Distribuição dos Erros')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Erros absolutos ao longo do tempo
    axes[1, 0].plot(abs_errors[:1000])  # Primeiro 1000 pontos
    axes[1, 0].set_xlabel('Índice Temporal')
    axes[1, 0].set_ylabel('Erro Absoluto')
    axes[1, 0].set_title('Erros Absolutos ao Longo do Tempo')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot dos erros
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot dos Erros')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(EVALUATION_PATH / 'error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'mean_error': np.mean(errors),
        'mean_abs_error': np.mean(abs_errors),
        'std_error': np.std(errors),
        'max_abs_error': np.max(abs_errors),
        'percentiles': {p: np.percentile(abs_errors, p) for p in [50, 75, 90, 95, 99]}
    }

# Analisar erros
if 'predictions' in locals() and y_test is not None:
    error_analysis = analyze_prediction_errors(y_test, predictions)

# %%
def analyze_extreme_events(y_true, y_pred, threshold_percentile=95):
    """
    Analisa performance em eventos extremos
    """
    print("=== ANÁLISE DE EVENTOS EXTREMOS ===")
    
    # Definir threshold para eventos extremos
    threshold = np.percentile(y_true, threshold_percentile)
    
    # Identificar eventos extremos
    extreme_mask = y_true >= threshold
    
    if np.sum(extreme_mask) == 0:
        print("❌ Nenhum evento extremo encontrado")
        return None
    
    print(f"Threshold P{threshold_percentile}: {threshold:.3f} mm/h")
    print(f"Eventos extremos: {np.sum(extreme_mask)} ({np.mean(extreme_mask)*100:.1f}%)")
    
    # Métricas específicas para eventos extremos
    y_true_extreme = y_true[extreme_mask]
    y_pred_extreme = y_pred[extreme_mask]
    
    mae_extreme = mean_absolute_error(y_true_extreme, y_pred_extreme)
    rmse_extreme = np.sqrt(mean_squared_error(y_true_extreme, y_pred_extreme))
    r2_extreme = r2_score(y_true_extreme, y_pred_extreme)
    
    print(f"\nMétricas para eventos extremos:")
    print(f"  MAE: {mae_extreme:.4f}")
    print(f"  RMSE: {rmse_extreme:.4f}")
    print(f"  R²: {r2_extreme:.4f}")
    
    # Análise de detecção de eventos extremos
    y_pred_extreme_class = y_pred >= threshold
    precision_extreme = np.sum(extreme_mask & y_pred_extreme_class) / np.sum(y_pred_extreme_class) if np.sum(y_pred_extreme_class) > 0 else 0
    recall_extreme = np.sum(extreme_mask & y_pred_extreme_class) / np.sum(extreme_mask)
    
    print(f"\nDetecção de eventos extremos:")
    print(f"  Precision: {precision_extreme:.3f}")
    print(f"  Recall: {recall_extreme:.3f}")
    
    # Visualização
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(y_true_extreme, y_pred_extreme, alpha=0.7, color='red')
    plt.plot([y_true_extreme.min(), y_true_extreme.max()], 
             [y_true_extreme.min(), y_true_extreme.max()], 'k--')
    plt.xlabel('Valores Reais')
    plt.ylabel('Predições')
    plt.title('Eventos Extremos - Predito vs Real')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    errors_extreme = y_pred_extreme - y_true_extreme
    plt.hist(errors_extreme, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Erro')
    plt.ylabel('Frequência')
    plt.title('Distribuição dos Erros - Eventos Extremos')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(EVALUATION_PATH / 'extreme_events_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'threshold': threshold,
        'count': np.sum(extreme_mask),
        'mae': mae_extreme,
        'rmse': rmse_extreme,
        'r2': r2_extreme,
        'precision': precision_extreme,
        'recall': recall_extreme
    }

# Analisar eventos extremos
if 'predictions' in locals() and y_test is not None:
    extreme_events_analysis = analyze_extreme_events(y_test, predictions)

# %%
def generate_evaluation_report():
    """
    Gera relatório final de avaliação
    """
    print("=== RELATÓRIO FINAL DE AVALIAÇÃO ===")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'architecture': 'LSTM',
            'test_samples': len(y_test) if y_test is not None else 0
        },
        'regression_metrics': regression_metrics if 'regression_metrics' in locals() else {},
        'classification_metrics': classification_metrics if 'classification_metrics' in locals() else {},
        'error_analysis': error_analysis if 'error_analysis' in locals() else {},
        'extreme_events': extreme_events_analysis if 'extreme_events_analysis' in locals() else {},
        'criteria_evaluation': {}
    }
    
    # Avaliar critérios de sucesso
    if 'regression_metrics' in locals() and regression_metrics:
        mae_ok = regression_metrics['MAE'] < 2.0
        rmse_ok = regression_metrics['RMSE'] < 3.0
        r2_ok = regression_metrics['R²'] > 0.5
        
        report['criteria_evaluation']['mae_criterion'] = {
            'target': '< 2.0 mm/h',
            'actual': regression_metrics['MAE'],
            'passed': mae_ok
        }
        
        report['criteria_evaluation']['rmse_criterion'] = {
            'target': '< 3.0 mm/h',
            'actual': regression_metrics['RMSE'],
            'passed': rmse_ok
        }
        
        report['criteria_evaluation']['r2_criterion'] = {
            'target': '> 0.5',
            'actual': regression_metrics['R²'],
            'passed': r2_ok
        }
    
    if 'classification_metrics' in locals() and classification_metrics:
        accuracy_ok = classification_metrics['accuracy'] > 0.75
        
        report['criteria_evaluation']['accuracy_criterion'] = {
            'target': '> 75%',
            'actual': classification_metrics['accuracy'],
            'passed': accuracy_ok
        }
    
    # Salvar relatório
    import json
    with open(EVALUATION_PATH / 'evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Exibir resumo
    print("\n📋 RESUMO DA AVALIAÇÃO:")
    
    if 'regression_metrics' in locals():
        print("✓ Métricas de regressão calculadas")
    
    if 'classification_metrics' in locals():
        print("✓ Métricas de classificação calculadas")
    
    if 'error_analysis' in locals():
        print("✓ Análise de erros concluída")
    
    if 'extreme_events_analysis' in locals():
        print("✓ Análise de eventos extremos concluída")
    
    print(f"\n📁 Arquivos gerados:")
    output_files = list(EVALUATION_PATH.glob("*"))
    for file in output_files:
        print(f"  - {file.name}")
    
    print(f"\n✅ Avaliação concluída!")
    print(f"📊 Relatório completo salvo em: {EVALUATION_PATH}")
    
    return report

# Gerar relatório final
final_report = generate_evaluation_report()

# %%
print("\n" + "="*60)
print("🎉 AVALIAÇÃO DO MODELO CONCLUÍDA!")
print("="*60)
print("\nPróximos passos recomendados:")
print("1. 📊 Revisar métricas de performance")
print("2. 🔧 Ajustar arquitetura se necessário")
print("3. 📈 Otimizar hiperparâmetros")
print("4. 🚀 Preparar para deploy em produção")

# %% 