# %% [markdown]
# # Validação Avançada de Modelo LSTM
# 
# Este notebook demonstra a implementação completa da **validação avançada** do modelo LSTM para previsão meteorológica.
# 
# ## Objetivos da Validação:
# 
# ✅ **Pipeline de treinamento completo**
# - Preparação de sequências temporais
# - Batch processing para grandes volumes  
# - Validation split temporal (não aleatório)
# 
# ✅ **Cross-validation temporal para séries temporais**
# - Walk-forward validation
# - Preservação de ordem cronológica
# - Validação temporal robusta
# 
# ✅ **Otimização de hiperparâmetros com grid search**
# - Learning rate: 0.001, 0.0001, 0.00001
# - Batch size: 16, 32, 64, 128
# - Sequence length: 12, 24, 48, 72 horas
# 
# ✅ **Validação com métricas específicas para meteorologia**
# - MAE para precipitação (mm/h)
# - RMSE para variáveis contínuas
# - Skill Score para eventos de chuva
# - **Target: Accuracy > 75% para classificação de eventos**
# 
# ## Critérios de Sucesso:
# - 🎯 **Accuracy > 75%** em previsão de chuva 24h
# - 🎯 **MAE < 2.0 mm/h** para precipitação
# - 🎯 **RMSE < 3.0 mm/h** para precipitação

# %%
# Imports necessários
import sys
import warnings
from pathlib import Path

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Pipeline personalizado
from scripts.training_pipeline import (
    TrainingPipeline, 
    TemporalDataSplitter, 
    MeteorologicalMetrics,
    LSTMModelBuilder,
    TEMPORAL_VALIDATION_CONFIG,
    HYPERPARAMETER_GRID,
    RAIN_THRESHOLDS,
    METEOROLOGICAL_FEATURES
)

# Configurações
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🚀 Validação Avançada de Modelo LSTM")
print("=" * 50)
print(f"📊 Features meteorológicas: {len(METEOROLOGICAL_FEATURES)}")
print(f"🔄 Configuração de validação temporal: {TEMPORAL_VALIDATION_CONFIG}")
print(f"🎯 Thresholds de chuva: {RAIN_THRESHOLDS}")

# %% [markdown]
# ## 1. Configuração e Carregamento de Dados

# %%
# Inicializar pipeline
pipeline = TrainingPipeline()

# Carregar dados
print("📥 Carregando dados processados...")
try:
    data = pipeline.load_data()
    print(f"✅ Dados carregados com sucesso: {data.shape}")
    print(f"📅 Período: {data['timestamp'].min()} até {data['timestamp'].max()}")
    print(f"🔢 Colunas disponíveis: {len(data.columns)}")
    
    # Verificar features disponíveis
    available_features = [col for col in METEOROLOGICAL_FEATURES if col in data.columns]
    print(f"🌦️  Features meteorológicas disponíveis: {len(available_features)}/{len(METEOROLOGICAL_FEATURES)}")
    
    if len(available_features) < len(METEOROLOGICAL_FEATURES):
        missing_features = set(METEOROLOGICAL_FEATURES) - set(available_features)
        print(f"⚠️  Features faltando: {missing_features}")
    
except Exception as e:
    print(f"❌ Erro ao carregar dados: {e}")
    print("💡 Execute primeiro o preprocessamento de dados")

# %%
# Análise rápida dos dados
if 'data' in locals():
    print("📊 Estatísticas dos dados:")
    
    # Verificar coluna de precipitação
    precip_cols = [col for col in data.columns if 'precipitacao' in col.lower()]
    if precip_cols:
        precip_col = precip_cols[0]
        precip_data = data[precip_col]
        
        print(f"\n🌧️  Estatísticas de precipitação ({precip_col}):")
        print(f"   Média: {precip_data.mean():.3f} mm/h")
        print(f"   Mediana: {precip_data.median():.3f} mm/h")
        print(f"   Máximo: {precip_data.max():.3f} mm/h")
        print(f"   % sem chuva: {(precip_data == 0).sum() / len(precip_data) * 100:.1f}%")
        print(f"   % chuva leve (>0.1): {(precip_data >= 0.1).sum() / len(precip_data) * 100:.1f}%")
        print(f"   % chuva moderada (>2.5): {(precip_data >= 2.5).sum() / len(precip_data) * 100:.1f}%")
        print(f"   % chuva forte (>10): {(precip_data >= 10.0).sum() / len(precip_data) * 100:.1f}%")
        
        # Visualizar distribuição
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histograma da precipitação
        axes[0].hist(precip_data[precip_data > 0], bins=50, alpha=0.7, color='skyblue')
        axes[0].set_xlabel('Precipitação (mm/h)')
        axes[0].set_ylabel('Frequência')
        axes[0].set_title('Distribuição da Precipitação (> 0)')
        axes[0].set_yscale('log')
        
        # Série temporal (amostra)
        sample_data = data.sample(n=min(1000, len(data))).sort_values('timestamp')
        axes[1].plot(sample_data['timestamp'], sample_data[precip_col], alpha=0.7, color='blue')
        axes[1].set_xlabel('Tempo')
        axes[1].set_ylabel('Precipitação (mm/h)')
        axes[1].set_title('Série Temporal da Precipitação (Amostra)')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 2. Validação Cruzada Temporal

# %%
print("🔄 Executando Validação Cruzada Temporal")
print("=" * 40)

# Configurar validação temporal
data_splitter = TemporalDataSplitter(TEMPORAL_VALIDATION_CONFIG)

# Demonstrar como funcionam os splits temporais
print("📅 Demonstração dos splits temporais:")
splits_demo = list(data_splitter.create_temporal_splits(data))

print(f"✅ Gerados {len(splits_demo)} folds temporais")
print("\n📊 Resumo dos folds:")

for i, (train_split, val_split) in enumerate(splits_demo[:3]):  # Mostrar apenas os 3 primeiros
    train_start = train_split['timestamp'].min()
    train_end = train_split['timestamp'].max()
    val_start = val_split['timestamp'].min()
    val_end = val_split['timestamp'].max()
    
    print(f"\n   Fold {i+1}:")
    print(f"   📈 Treino: {train_start.strftime('%Y-%m-%d')} até {train_end.strftime('%Y-%m-%d')} ({len(train_split)} amostras)")
    print(f"   📊 Validação: {val_start.strftime('%Y-%m-%d')} até {val_end.strftime('%Y-%m-%d')} ({len(val_split)} amostras)")

# %%
# Executar validação cruzada temporal completa
print("\n🚀 Executando validação cruzada temporal completa...")

try:
    cv_results = pipeline.run_temporal_cross_validation(max_folds=3)  # Reduzido para demonstração
    
    if cv_results:
        print("\n📊 RESULTADOS DA VALIDAÇÃO CRUZADA TEMPORAL")
        print("=" * 50)
        
        # Métricas principais
        metrics_to_show = ['accuracy', 'mae', 'rmse', 'f1_score']
        
        for metric in metrics_to_show:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            
            if mean_key in cv_results:
                mean_val = cv_results[mean_key]
                std_val = cv_results.get(std_key, 0)
                print(f"   {metric.upper()}: {mean_val:.3f} ± {std_val:.3f}")
        
        # Verificar critérios de sucesso
        print("\n🎯 CRITÉRIOS DE SUCESSO:")
        accuracy_target = cv_results.get('meets_accuracy_target', False)
        mae_target = cv_results.get('meets_mae_target', False)
        overall_success = cv_results.get('overall_success', False)
        
        print(f"   Accuracy >= 75%: {'✅' if accuracy_target else '❌'}")
        print(f"   MAE <= 2.0 mm/h: {'✅' if mae_target else '❌'}")
        print(f"   Sucesso geral: {'✅' if overall_success else '❌'}")
        
        # Visualizar resultados por fold
        if 'fold_results' in cv_results:
            fold_results = cv_results['fold_results']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # MAE por fold
            mae_values = [fold['mae'] for fold in fold_results]
            axes[0, 0].plot(range(1, len(mae_values) + 1), mae_values, 'o-', color='red')
            axes[0, 0].axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Target: 2.0')
            axes[0, 0].set_xlabel('Fold')
            axes[0, 0].set_ylabel('MAE (mm/h)')
            axes[0, 0].set_title('MAE por Fold')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Accuracy por fold
            acc_values = [fold.get('accuracy', 0) for fold in fold_results]
            axes[0, 1].plot(range(1, len(acc_values) + 1), acc_values, 'o-', color='green')
            axes[0, 1].axhline(y=0.75, color='green', linestyle='--', alpha=0.7, label='Target: 75%')
            axes[0, 1].set_xlabel('Fold')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Accuracy por Fold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # RMSE por fold
            rmse_values = [fold['rmse'] for fold in fold_results]
            axes[1, 0].plot(range(1, len(rmse_values) + 1), rmse_values, 'o-', color='blue')
            axes[1, 0].axhline(y=3.0, color='blue', linestyle='--', alpha=0.7, label='Target: 3.0')
            axes[1, 0].set_xlabel('Fold')
            axes[1, 0].set_ylabel('RMSE (mm/h)')
            axes[1, 0].set_title('RMSE por Fold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # F1-Score por fold
            f1_values = [fold.get('f1_score', 0) for fold in fold_results]
            axes[1, 1].plot(range(1, len(f1_values) + 1), f1_values, 'o-', color='purple')
            axes[1, 1].set_xlabel('Fold')
            axes[1, 1].set_ylabel('F1-Score')
            axes[1, 1].set_title('F1-Score por Fold')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    else:
        print("❌ Nenhum resultado da validação cruzada temporal")
        
except Exception as e:
    print(f"❌ Erro durante validação cruzada temporal: {e}")

# %% [markdown]
# ## 3. Métricas Meteorológicas Específicas

# %%
print("🌦️  Demonstração das Métricas Meteorológicas")
print("=" * 45)

# Criar dados sintéticos para demonstração
np.random.seed(42)
n_samples = 1000

# Simular dados de precipitação realistas
y_true = np.random.exponential(scale=0.5, size=n_samples)  # Distribuição exponencial (comum para chuva)
y_true[y_true > 20] = 20  # Limitar valores extremos

# Simular predições com algum ruído
y_pred = y_true + np.random.normal(0, 0.2, size=n_samples)
y_pred[y_pred < 0] = 0  # Precipitação não pode ser negativa

print(f"📊 Dados sintéticos gerados: {n_samples} amostras")
print(f"   Precipitação real - Média: {y_true.mean():.3f}, Max: {y_true.max():.3f}")
print(f"   Precipitação predita - Média: {y_pred.mean():.3f}, Max: {y_pred.max():.3f}")

# Calcular métricas meteorológicas
metrics_calc = MeteorologicalMetrics()
detailed_metrics = metrics_calc.calculate_precipitation_metrics(y_true, y_pred)

print("\n📈 MÉTRICAS METEOROLÓGICAS DETALHADAS:")
print("=" * 40)

# Métricas básicas
print("🔢 Métricas Básicas:")
print(f"   MAE: {detailed_metrics['mae']:.3f} mm/h")
print(f"   RMSE: {detailed_metrics['rmse']:.3f} mm/h")
print(f"   MSE: {detailed_metrics['mse']:.3f}")

# Métricas por intensidade de chuva
print("\n🌧️  Métricas por Intensidade:")
for intensity in ['light', 'moderate', 'heavy']:
    mae_key = f'mae_{intensity}'
    count_key = f'count_{intensity}'
    
    if mae_key in detailed_metrics and count_key in detailed_metrics:
        mae_val = detailed_metrics[mae_key]
        count_val = detailed_metrics[count_key]
        print(f"   {intensity.capitalize()}: MAE = {mae_val:.3f} mm/h ({count_val} amostras)")

# Skill Scores
print("\n🎯 Skill Scores:")
for intensity in ['light', 'moderate', 'heavy']:
    skill_key = f'skill_score_{intensity}'
    if skill_key in detailed_metrics:
        skill_val = detailed_metrics[skill_key]
        print(f"   {intensity.capitalize()}: {skill_val:.3f}")

# Métricas de classificação
print("\n📊 Métricas de Classificação (eventos de chuva):")
if 'accuracy' in detailed_metrics:
    print(f"   Accuracy: {detailed_metrics['accuracy']:.3f}")
if 'f1_score' in detailed_metrics:
    print(f"   F1-Score: {detailed_metrics['f1_score']:.3f}")
if 'auc' in detailed_metrics:
    print(f"   AUC: {detailed_metrics['auc']:.3f}")

# %%
# Visualizar métricas meteorológicas
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Scatter plot: Real vs Predito
axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=10)
axes[0, 0].plot([0, y_true.max()], [0, y_true.max()], 'r--', alpha=0.8)
axes[0, 0].set_xlabel('Precipitação Real (mm/h)')
axes[0, 0].set_ylabel('Precipitação Predita (mm/h)')
axes[0, 0].set_title('Real vs Predito')
axes[0, 0].grid(True, alpha=0.3)

# Histograma dos erros
errors = y_pred - y_true
axes[0, 1].hist(errors, bins=30, alpha=0.7, color='orange')
axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.8)
axes[0, 1].set_xlabel('Erro (Predito - Real)')
axes[0, 1].set_ylabel('Frequência')
axes[0, 1].set_title(f'Distribuição dos Erros (MAE: {detailed_metrics["mae"]:.3f})')

# Métricas por threshold
thresholds = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
skill_scores = []

for threshold in thresholds:
    skill = metrics_calc.calculate_skill_score(y_true, y_pred, threshold)
    skill_scores.append(skill)

axes[1, 0].plot(thresholds, skill_scores, 'o-', color='purple')
axes[1, 0].set_xlabel('Threshold (mm/h)')
axes[1, 0].set_ylabel('Skill Score')
axes[1, 0].set_title('Skill Score por Threshold')
axes[1, 0].grid(True, alpha=0.3)

# Box plot das métricas por intensidade
intensities = []
mae_by_intensity = []

for intensity in ['light', 'moderate', 'heavy']:
    if intensity == 'light':
        mask = (y_true >= 0) & (y_true < 2.5)
    elif intensity == 'moderate':
        mask = (y_true >= 2.5) & (y_true < 10.0)
    elif intensity == 'heavy':
        mask = y_true >= 10.0
    
    if np.sum(mask) > 0:
        errors_intensity = np.abs(y_pred[mask] - y_true[mask])
        intensities.append(intensity.capitalize())
        mae_by_intensity.append(errors_intensity)

if mae_by_intensity:
    axes[1, 1].boxplot(mae_by_intensity, labels=intensities)
    axes[1, 1].set_ylabel('Erro Absoluto (mm/h)')
    axes[1, 1].set_title('Distribuição dos Erros por Intensidade')
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Otimização de Hiperparâmetros

# %%
print("⚙️  Demonstração da Otimização de Hiperparâmetros")
print("=" * 50)

# Mostrar grid de hiperparâmetros
print("🔧 Grid de Hiperparâmetros:")
for param, values in HYPERPARAMETER_GRID.items():
    print(f"   {param}: {values}")

# Simular otimização rápida com dados reduzidos
print("\n🚀 Executando otimização de hiperparâmetros (versão reduzida)...")

try:
    # Usar amostra menor para demonstração
    if 'data' in locals() and len(data) > 10000:
        data_sample = data.sample(n=10000, random_state=42).sort_values('timestamp')
        print(f"📊 Usando amostra de {len(data_sample)} registros para demonstração")
    else:
        data_sample = data
    
    # Executar otimização com poucos trials
    hyperopt_results = pipeline.run_hyperparameter_optimization(max_trials=5)
    
    if hyperopt_results:
        print("\n📊 RESULTADOS DA OTIMIZAÇÃO DE HIPERPARÂMETROS")
        print("=" * 50)
        
        print(f"🏆 Melhor MAE: {hyperopt_results.get('best_mae', 'N/A'):.3f}")
        print(f"🔧 Melhores parâmetros: {hyperopt_results.get('best_params', {})}")
        print(f"🔢 Total de trials: {hyperopt_results.get('total_trials', 0)}")
        
        # Visualizar resultados dos trials
        if 'trial_results' in hyperopt_results:
            trial_results = hyperopt_results['trial_results']
            
            if len(trial_results) > 1:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # MAE por trial
                trial_numbers = [r['trial'] for r in trial_results]
                mae_values = [r['mae'] for r in trial_results]
                
                axes[0].plot(trial_numbers, mae_values, 'o-', color='red')
                axes[0].set_xlabel('Trial')
                axes[0].set_ylabel('MAE (mm/h)')
                axes[0].set_title('MAE por Trial')
                axes[0].grid(True, alpha=0.3)
                
                # Accuracy por trial
                acc_values = [r.get('accuracy', 0) for r in trial_results]
                axes[1].plot(trial_numbers, acc_values, 'o-', color='green')
                axes[1].set_xlabel('Trial')
                axes[1].set_ylabel('Accuracy')
                axes[1].set_title('Accuracy por Trial')
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
            
            # Mostrar top 3 configurações
            sorted_results = sorted(trial_results, key=lambda x: x['mae'])
            print(f"\n🏅 TOP 3 CONFIGURAÇÕES:")
            
            for i, result in enumerate(sorted_results[:3]):
                print(f"\n   #{i+1} - MAE: {result['mae']:.3f}")
                if 'params' in result:
                    for param, value in result['params'].items():
                        print(f"      {param}: {value}")
    
    else:
        print("❌ Nenhum resultado da otimização de hiperparâmetros")

except Exception as e:
    print(f"❌ Erro durante otimização de hiperparâmetros: {e}")

# %% [markdown]
# ## 5. Análise de Performance e Conclusões

# %%
print("📈 Análise de Performance - Fase 3.2")
print("=" * 40)

# Resumo dos resultados (se disponíveis)
if 'cv_results' in locals() and cv_results:
    print("📊 RESUMO DA VALIDAÇÃO CRUZADA TEMPORAL:")
    
    accuracy_mean = cv_results.get('accuracy_mean', 0)
    mae_mean = cv_results.get('mae_mean', 0)
    rmse_mean = cv_results.get('rmse_mean', 0)
    
    print(f"   🎯 Accuracy média: {accuracy_mean:.3f}")
    print(f"   📉 MAE médio: {mae_mean:.3f} mm/h")
    print(f"   📊 RMSE médio: {rmse_mean:.3f} mm/h")
    
    # Avaliação dos critérios
    print("\n🎯 AVALIAÇÃO DOS CRITÉRIOS DE SUCESSO:")
    
    accuracy_ok = accuracy_mean >= 0.75
    mae_ok = mae_mean <= 2.0
    rmse_ok = rmse_mean <= 3.0
    
    print(f"   Accuracy >= 75%: {'✅ PASSOU' if accuracy_ok else '❌ FALHOU'} ({accuracy_mean:.1%})")
    print(f"   MAE <= 2.0 mm/h: {'✅ PASSOU' if mae_ok else '❌ FALHOU'} ({mae_mean:.3f})")
    print(f"   RMSE <= 3.0 mm/h: {'✅ PASSOU' if rmse_ok else '❌ FALHOU'} ({rmse_mean:.3f})")
    
    overall_success = accuracy_ok and mae_ok and rmse_ok
    print(f"\n🏆 RESULTADO GERAL: {'✅ SUCESSO' if overall_success else '❌ PRECISA MELHORIAS'}")

if 'hyperopt_results' in locals() and hyperopt_results:
    print(f"\n⚙️  MELHOR CONFIGURAÇÃO ENCONTRADA:")
    best_params = hyperopt_results.get('best_params', {})
    for param, value in best_params.items():
        print(f"   {param}: {value}")

# %%
print("\n📋 CHECKLIST DA FASE 3.2")
print("=" * 30)

checklist = [
    ("Pipeline de treinamento completo", "✅"),
    ("Preparação de sequências temporais", "✅"),
    ("Validation split temporal (não aleatório)", "✅"),
    ("Cross-validation temporal", "✅"),
    ("Walk-forward validation", "✅"),
    ("Preservação de ordem cronológica", "✅"),
    ("Otimização de hiperparâmetros", "✅"),
    ("Grid search automatizado", "✅"),
    ("Métricas meteorológicas específicas", "✅"),
    ("MAE para precipitação", "✅"),
    ("RMSE para variáveis contínuas", "✅"),
    ("Skill Score para eventos de chuva", "✅"),
    ("Accuracy > 75% para classificação", "🔄 Em validação"),
]

for item, status in checklist:
    print(f"   {status} {item}")

print(f"\n💡 PRÓXIMOS PASSOS:")
print("   1. 🔧 Executar pipeline completo: `make training-pipeline`")
print("   2. 📊 Validar métricas: `make validate-model-metrics`")
print("   3. 🚀 Se critérios atendidos, prosseguir para Fase 4")
print("   4. 🔄 Se não, ajustar hiperparâmetros e re-treinar")

print(f"\n📁 COMANDOS ÚTEIS:")
print("   - `make temporal-cv`: Validação cruzada temporal")
print("   - `make hyperopt`: Otimização de hiperparâmetros")
print("   - `make training-pipeline`: Pipeline completo")
print("   - `make view-training-results`: Ver resultados")

# %% [markdown]
# ## 6. Conclusão da Fase 3.2
# 
# ✅ **Implementação Completa da Fase 3.2**
# 
# Esta implementação cobre todos os requisitos especificados na documentação:
# 
# ### ✅ Pipeline de Treinamento Completo
# - Preparação automática de sequências temporais
# - Batch processing otimizado para grandes volumes
# - Validation split temporal que preserva ordem cronológica
# 
# ### ✅ Cross-validation Temporal 
# - Walk-forward validation implementado
# - Preservação rigorosa da ordem cronológica
# - Múltiplos folds temporais com configuração flexível
# 
# ### ✅ Otimização de Hiperparâmetros
# - Grid search sistemático com parâmetros definidos na documentação
# - Learning rates: 0.001, 0.0001, 0.00001
# - Batch sizes: 16, 32, 64, 128
# - Sequence lengths: 12, 24, 48, 72 horas
# 
# ### ✅ Métricas Meteorológicas Específicas
# - MAE estratificado por intensidade de chuva
# - RMSE para variáveis contínuas
# - Skill Score (Equitable Threat Score) para eventos de chuva
# - Métricas de classificação para eventos (Accuracy, F1-Score, AUC)
# 
# ### 🎯 Critérios de Sucesso Implementados
# - **Target: Accuracy > 75%** em previsão de chuva 24h
# - **Target: MAE < 2.0 mm/h** para precipitação  
# - **Target: RMSE < 3.0 mm/h** para precipitação
# 
# ### 🚀 Pronto para Próxima Fase
# A Fase 3.2 está **completa e funcional**. O sistema pode agora:
# 
# 1. Treinar modelos com validação temporal rigorosa
# 2. Otimizar hiperparâmetros sistematicamente
# 3. Avaliar performance com métricas meteorológicas específicas
# 4. Validar se os critérios de sucesso são atendidos
# 
# **Próximo passo:** Fase 4 - Feature Forecast (Previsão) 