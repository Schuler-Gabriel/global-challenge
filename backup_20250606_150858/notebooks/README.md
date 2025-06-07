# 📓 Notebooks Jupyter - Sistema de Alertas de Cheias

Esta pasta contém os notebooks Jupyter para desenvolvimento, treinamento e análise do modelo LSTM híbrido de previsão de cheias.

## 📋 Lista de Notebooks

### 🔄 0. Pipeline de Dados

**`0_data_pipeline.ipynb`**

- Executa pipeline completo de coleta de dados
- Integração com APIs Open-Meteo (Historical Forecast + Weather)
- Processamento e validação dos dados atmosféricos
- **Execute primeiro** para preparar os dados

### 📊 1. Análise Exploratória

**`1_exploratory_analysis.ipynb`**

- Análise exploratória dos dados atmosféricos
- Visualizações das 149 variáveis meteorológicas
- Análise de níveis de pressão (850hPa, 500hPa)
- Padrões sazonais e correlações
- Comparação Open-Meteo vs INMET

### 🧠 2. Treinamento do Modelo

**`2_model_training.ipynb`**

- Implementação do modelo LSTM híbrido
- Arquitetura com dois componentes (149 + 25 variáveis)
- Treinamento com ensemble weighted (0.7 + 0.3)
- Callbacks e otimização
- Salvamento do modelo treinado

### 📊 3. Avaliação do Modelo

**`3_model_evaluation.ipynb`**

- Métricas de performance atmosférica
- Análise de casos extremos (>20mm/h)
- Detecção de frentes frias e vórtices
- Comparação com modelos tradicionais
- Visualizações de resultados

### ✅ 4. Validação Cruzada

**`4_model_validation.ipynb`**

- Validação cruzada temporal
- Walk-forward validation
- Testes out-of-time
- Análise de robustez sazonal
- Métricas de estabilidade

### 🔬 5. Experimentos de Arquitetura

**`5_model_architecture_experiments.ipynb`**

- Otimização de hiperparâmetros
- Comparação de arquiteturas
- Ablation studies
- Feature importance
- Análise de diferentes ensembles

### 🌊 6. Exemplos Práticos

**`6_practical_examples.ipynb`**

- Casos de estudo reais
- Análise da enchente de Maio 2024
- Detecção de sistemas meteorológicos
- Alertas automáticos
- Validação com dados observados

### 🌐 7. Integração com API

**`7_api_integration.ipynb`**

- Demonstração da API FastAPI
- Endpoints de previsão
- Sistema de alertas
- Integração em produção
- Exemplos de uso

## 🚀 Como Executar

### Pré-requisitos

```bash
# Instalar dependências
pip install -r requirements.txt

# Ativar ambiente virtual
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### Ordem de Execução Recomendada

1. **📋 Preparação dos Dados**

   ```bash
   jupyter notebook 0_data_pipeline.ipynb
   ```

2. **🔍 Análise Exploratória**

   ```bash
   jupyter notebook 1_exploratory_analysis.ipynb
   ```

3. **🧠 Treinamento**

   ```bash
   jupyter notebook 2_model_training.ipynb
   ```

4. **📊 Avaliação**

   ```bash
   jupyter notebook 3_model_evaluation.ipynb
   ```

5. **✅ Validação**
   ```bash
   jupyter notebook 4_model_validation.ipynb
   ```

### Execução via Linha de Comando

```bash
# Executar todos os notebooks sequencialmente
make run-notebooks

# Executar notebook específico
jupyter nbconvert --execute --to notebook --inplace 1_exploratory_analysis.ipynb
```

## 📁 Estrutura de Dados Esperada

```
data/
├── processed/
│   ├── atmospheric_features_149vars.csv      # Dados principais (Componente 1)
│   ├── surface_features_25vars.csv           # Dados de superfície (Componente 2)
│   ├── openmeteo_historical_forecast.csv     # Dados brutos Historical Forecast
│   ├── openmeteo_historical_weather.csv      # Dados brutos Historical Weather
│   └── dados_inmet_processados.csv           # Dados INMET para validação
└── raw/
    └── dados_historicos/                      # Dados INMET brutos
```

## 🎯 Objetivos dos Notebooks

### Científicos

- **Precisão >82%** vs ~70% de modelos tradicionais
- **Análise sinótica** com níveis de pressão atmosférica
- **Detecção de frentes frias** via 850hPa
- **Identificação de vórtices** via 500hPa

### Técnicos

- Modelo LSTM híbrido reproduzível
- Pipeline de dados automatizado
- Validação rigorosa temporal
- Integração com API de produção

### Práticos

- Sistema de alertas automático
- Casos de uso reais
- Métricas interpretáveis
- Interface para usuários finais

## 🛠️ Dependências Principais

```python
# Machine Learning
tensorflow >= 2.12.0
scikit-learn >= 1.3.0
pandas >= 2.0.0
numpy >= 1.24.0

# Visualização
matplotlib >= 3.7.0
seaborn >= 0.12.0
plotly >= 5.14.0

# APIs e dados
requests >= 2.31.0
aiohttp >= 3.8.0

# Processamento
joblib >= 1.3.0
```

## 📈 Métricas de Performance

| Métrica                  | Target   | Status |
| ------------------------ | -------- | ------ |
| **Accuracy Global**      | >82%     | 🎯     |
| **MAE Precipitação**     | <1.5mm/h | 🎯     |
| **RMSE Precipitação**    | <2.5mm/h | 🎯     |
| **Frontal Detection**    | >90%     | 🎯     |
| **Synoptic Recognition** | >85%     | 🎯     |

## 🐛 Troubleshooting

### Erro: Dados não encontrados

```python
# Execute primeiro o pipeline de dados
python scripts/collect_openmeteo_hybrid_data.py
python scripts/atmospheric_feature_engineering.py
```

### Erro: GPU não detectada

```python
# Verificar instalação CUDA
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Erro: Memória insuficiente

```python
# Reduzir batch size no treinamento
BATCH_SIZE = 16  # em vez de 32
```

## 📞 Suporte

Para dúvidas sobre os notebooks:

1. Consulte a documentação em `docs/`
2. Verifique logs em `logs/`
3. Execute testes com `pytest tests/`

## 🔗 Links Úteis

- **Documentação Completa**: `../docs/`
- **API Documentation**: `../docs/API.md`
- **Modelo Documentation**: `../docs/MODEL.md`
- **Dados Documentation**: `../docs/DATA.md`
