# Sistema de Alertas de Cheias - Rio Guaíba

## 🌊 Visão Geral

Sistema inteligente de previsão meteorológica e alertas de cheias para Porto Alegre, utilizando dados históricos meteorológicos do INMET (2000-2025) e APIs em tempo real do nível do Rio Guaíba.

### 🎯 Principais Objetivos

- **IA Preditiva**: Modelo LSTM com precisão > 75% para previsão de chuva 24h
- **API Robusta**: FastAPI com alta disponibilidade e resposta rápida
- **Alertas Inteligentes**: Sistema automatizado baseado em matriz de risco
- **Arquitetura Limpa**: Clean Architecture organizada por features
- **Monitoramento**: Logs estruturados e métricas de performance

## 🏗️ Arquitetura

```
projeto_alerta_cheias/
├── app/                           # Aplicação principal (Clean Architecture)
│   ├── core/                      # Domínio compartilhado
│   └── features/                  # Features organizadas por domínio
│       ├── forecast/              # Previsão meteorológica
│       └── alerts/                # Sistema de alertas
├── data/                          # Dados do projeto
│   ├── raw/dados_historicos/      # CSVs INMET (2000-2025)
│   ├── processed/                 # Dados processados
│   └── modelos_treinados/         # Modelos salvos
├── notebooks/                     # Análise e experimentação
│   ├── python/                    # Arquivos Python (.py) - FONTE
│   └── jupyter/                   # Notebooks (.ipynb) - GERADOS
├── scripts/                       # Scripts utilitários
├── tests/                         # Testes automatizados
├── docker/                        # Configurações Docker
└── requirements/                  # Dependências por ambiente
```

## 🚀 Quick Start

### 1. Configuração do Ambiente

```bash
# Clonar repositório
git clone <repository-url>
cd Challenge

# Instalar dependências
pip install -r requirements/base.txt

# Para desenvolvimento
pip install -r requirements/development.txt
```

### 2. Dados Meteorológicos

O projeto utiliza dados históricos do INMET (Instituto Nacional de Meteorologia):

- **Período**: 2000-2025 (25+ anos)
- **Estações**: A801 (Porto Alegre), B807 (Belém Novo)
- **Frequência**: Observações horárias
- **Variáveis**: 16+ features meteorológicas

```bash
# Organizar dados iniciais
python scripts/setup_data.py

# Validar qualidade dos dados
python scripts/validate_data.py

# Preprocessar dados
python scripts/data_preprocessing.py
```

### 3. Notebooks de Análise

Os notebooks seguem uma metodologia específica com arquivos Python como fonte:

```bash
# Instalar jupytext
pip install jupytext

# Converter notebook Python para Jupyter
cd notebooks/python/
jupytext --to notebook exploratory_analysis.py
mv exploratory_analysis.ipynb ../jupyter/

# Executar análise exploratória
jupyter notebook ../jupyter/exploratory_analysis.ipynb
```

**Notebooks Disponíveis:**

1. **`exploratory_analysis`** - Análise exploratória dos dados INMET
2. **`data_preprocessing`** - Limpeza e normalização de dados
3. **`model_training`** - Treinamento do modelo LSTM
4. **`model_architecture_experiments`** - Experimentos de arquitetura
5. **`model_evaluation`** - Avaliação de métricas de performance
6. **`model_validation`** - Validação cruzada temporal e otimização

### 4. Treinamento do Modelo

```bash
# Treinamento básico
make train-model

# Experimentos rápidos
make train-experiment

# Grid search completo
make train-full-grid

# Monitoramento com TensorBoard
make tensorboard
```

### 5. Validação Avançada

```bash
# Validação cruzada temporal
make temporal-cv

# Otimização de hiperparâmetros
make hyperopt

# Pipeline completo de treinamento
make training-pipeline

# Validar métricas do modelo
make validate-model-metrics
```

## 📊 Status do Desenvolvimento

### ✅ Implementado

1. **Configuração e Estrutura Base**

   - Clean Architecture com features organizadas
   - Docker setup completo
   - Configurações por ambiente

2. **Análise e Preparação de Dados**

   - Análise exploratória completa (25 anos de dados INMET)
   - Pipeline de preprocessamento robusto
   - Scripts de validação e organização

3. **Desenvolvimento do Modelo ML**

   - Arquitetura LSTM otimizada para meteorologia
   - 6 configurações diferentes testadas
   - Grid search automatizado
   - TensorBoard integrado

4. **Validação Avançada**
   - Cross-validation temporal (walk-forward)
   - Métricas meteorológicas específicas
   - Otimização de hiperparâmetros
   - Critérios de sucesso automatizados

### 🔄 Próximos Passos

1. **Feature Forecast** - API de previsão meteorológica
2. **APIs Externas** - Integração CPTEC e Guaíba
3. **Feature Alerts** - Sistema de alertas inteligente
4. **Testes e Qualidade** - Cobertura > 80%
5. **Monitoramento** - Logs estruturados e métricas

## 🎯 Critérios de Sucesso

### Modelo de ML ✅

- **Accuracy > 75%** em previsões de 24h ✅
- **MAE < 2.0 mm/h** para precipitação ✅
- **RMSE < 3.0 mm/h** para precipitação ✅
- Tempo de inferência < 100ms

### API Performance (Próximo)

- Latência média < 200ms
- Disponibilidade > 99.5%
- Rate limiting: 1000 req/min por IP

### Qualidade de Código (Próximo)

- Cobertura de testes > 80%
- Type hints em 100% das funções
- Zero warnings no mypy

## 🛠️ Comandos Úteis

### Makefile Commands

```bash
# Setup e dados
make setup              # Configuração inicial
make setup-data         # Organizar dados INMET

# Análise e preprocessamento
make explore-data       # Análise exploratória
make preprocess-data    # Preprocessamento

# Treinamento
make train-model        # Treinamento básico
make train-experiment   # Experimentos rápidos
make train-full-grid    # Grid search completo

# Validação
make temporal-cv        # Cross-validation temporal
make hyperopt          # Otimização de hiperparâmetros
make training-pipeline  # Pipeline completo

# Monitoramento
make tensorboard        # TensorBoard
make view-results       # Visualizar resultados

# Docker
make docker-build       # Build containers
make docker-run         # Executar containers
make docker-training    # Treinamento em Docker

# Notebooks
make convert-notebooks  # Converter Python → Jupyter
make jupyter           # Iniciar Jupyter Lab

# Testes
make test              # Executar testes
make test-validation   # Testar validação de modelos
make lint              # Linting
make format            # Formatação de código
```

### Scripts Diretos

```bash
# Teste rápido da validação
python3 scripts/test_model_validation.py

# Pipeline de treinamento
python3 scripts/training_pipeline.py

# Configuração de dados
python3 scripts/setup_data.py
```

## 📚 Documentação Completa

Para documentação detalhada, consulte:

- [`PROJETO_DOCUMENTACAO.md`](PROJETO_DOCUMENTACAO.md) - Documentação completa
- [`notebooks/`](notebooks/) - Notebooks de análise e experimentação
- [`docs/`](docs/) - Documentação técnica específica

## 🤝 Contribuição

Este projeto segue as melhores práticas de Clean Architecture e desenvolvimento orientado por testes. Consulte a documentação para detalhes sobre:

- Estrutura do projeto
- Padrões de código
- Workflow de notebooks
- Critérios de qualidade

## 📄 Licença

[Adicionar informações de licença conforme necessário]
# global-challenge
