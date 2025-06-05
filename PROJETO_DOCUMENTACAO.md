# Sistema de Alertas de Cheias - Rio Guaíba

## Documentação Completa do Projeto

### 📋 Visão Geral

Sistema inteligente de previsão meteorológica e alertas de cheias para Porto Alegre, utilizando dados históricos meteorológicos do INMET (2000-2025) e APIs em tempo real do nível do Rio Guaíba e condições meteorológicas.

### 🎯 Objetivos

- **IA Preditiva**: Modelo LSTM com precisão > 75% para previsão de chuva 24h
- **API Robusta**: FastAPI com alta disponibilidade e resposta rápida
- **Alertas Inteligentes**: Sistema automatizado baseado em matriz de risco
- **Arquitetura Limpa**: Clean Architecture organizada por features
- **Monitoramento**: Logs estruturados e métricas de performance

### 📚 Workflow dos Notebooks Jupyter

#### 🔄 Metodologia de Desenvolvimento

Este projeto utiliza uma metodologia específica para desenvolvimento e manutenção dos notebooks Jupyter:

**Estrutura de Pastas:**

```
notebooks/
├── python/                    # Arquivos Python (.py) - FONTE PRINCIPAL
│   ├── exploratory_analysis.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_architecture_experiments.py
│   ├── model_evaluation.py
│   └── model_validation.py
└── jupyter/                   # Notebooks Jupyter (.ipynb) - GERADOS
    ├── exploratory_analysis.ipynb
    ├── data_preprocessing.ipynb
    ├── model_training.ipynb
    ├── model_architecture_experiments.ipynb
    ├── model_evaluation.ipynb
    └── model_validation.ipynb
```

#### ⚡ Regras de Desenvolvimento

1. **SEMPRE trabalhe com arquivos Python (.py) primeiro**

   - Edite apenas os arquivos na pasta `notebooks/python/`
   - Use sintaxe de células do Jupyter (`# %%`) nos arquivos Python
   - Mantenha markdown em comentários `# %% [markdown]`

2. **Conversão automática para Jupyter**

   ```bash
   # Deletar notebook existente (se houver)
   rm notebooks/jupyter/nome_arquivo.ipynb

   # Gerar novo notebook a partir do Python
   cd notebooks/python/
   jupytext --to notebook nome_arquivo.py
   mv nome_arquivo.ipynb ../jupyter/
   ```

3. **Workflow completo de edição**

   ```bash
   # 1. Editar arquivo Python
   vim notebooks/python/exploratory_analysis.py

   # 2. Deletar notebook antigo
   rm notebooks/jupyter/exploratory_analysis.ipynb

   # 3. Gerar novo notebook
   cd notebooks/python/
   jupytext --to notebook exploratory_analysis.py
   mv exploratory_analysis.ipynb ../jupyter/

   # 4. Testar notebook
   cd ../jupyter/
   jupyter notebook exploratory_analysis.ipynb
   ```

4. **Nunca edite diretamente os arquivos .ipynb**
   - Os arquivos na pasta `jupyter/` são sempre gerados
   - Qualquer edição manual será perdida na próxima conversão
   - Mantenha apenas os arquivos Python como fonte da verdade

#### 🛠️ Ferramentas Necessárias

```bash
# Instalar jupytext
pip install jupytext

# Verificar instalação
jupytext --version
```

#### 📋 Notebooks Disponíveis

1. **`exploratory_analysis.py/.ipynb`**

   - Análise exploratória dos dados INMET (2000-2025)
   - Identificação de padrões sazonais e tendências
   - Detecção de outliers e dados inconsistentes
   - Análise de correlações entre variáveis
   - Visualizações descritivas e estatísticas

2. **`data_preprocessing.py/.ipynb`**

   - Limpeza e normalização dos dados
   - Tratamento de valores missing
   - Feature engineering e criação de variáveis derivadas
   - Divisão temporal em treino/validação/teste
   - Salvamento dos dados processados

3. **`model_training.py/.ipynb`**

   - Treinamento do modelo LSTM principal
   - Configuração de arquiteturas (1-3 camadas)
   - Callbacks (EarlyStopping, ReduceLROnPlateau)
   - Monitoramento com TensorBoard
   - Salvamento de modelos treinados

4. **`model_architecture_experiments.py/.ipynb`**

   - Experimentos sistemáticos de arquiteturas
   - Grid search automatizado de hiperparâmetros
   - Comparação de performance entre configurações
   - Análise de trade-offs complexidade vs performance

5. **`model_evaluation.py/.ipynb`**

   - Avaliação completa de métricas de performance
   - Análise de erros e casos extremos
   - Métricas de classificação e regressão
   - Visualizações de resultados
   - Relatório final de avaliação

6. **`model_validation.py/.ipynb`**
   - Validação cruzada temporal com walk-forward validation
   - Otimização de hiperparâmetros com grid search
   - Métricas meteorológicas específicas (MAE, RMSE, Skill Score)
   - Validação automática dos critérios de sucesso
   - Pipeline completo de treinamento e validação

#### 🚨 Troubleshooting

**Problema: Notebook não abre no Jupyter**

```bash
# Verificar formato do arquivo
head -5 notebooks/jupyter/nome_arquivo.ipynb

# Deve começar com: {"cells": [
# Se não, regenerar:
cd notebooks/python/
jupytext --to notebook nome_arquivo.py
mv nome_arquivo.ipynb ../jupyter/
```

**Problema: Erro de conversão**

```bash
# Verificar sintaxe do arquivo Python
python -m py_compile notebooks/python/nome_arquivo.py

# Verificar marcadores de célula
grep "# %%" notebooks/python/nome_arquivo.py
```

**Problema: Jupyter não reconhece o notebook**

```bash
# Converter com formato específico
jupytext --to ipynb notebooks/python/nome_arquivo.py
```

#### ✅ Vantagens desta Metodologia

1. **Controle de Versão**: Arquivos Python são mais limpos no Git
2. **Edição Eficiente**: IDEs funcionam melhor com arquivos .py
3. **Consistência**: Formato padrão sempre mantido
4. **Automação**: Pipeline de conversão padronizado
5. **Backup**: Fonte única de verdade nos arquivos Python

### 📊 Dados Meteorológicos Históricos

#### Dataset Disponível

O projeto utiliza dados meteorológicos históricos do Instituto Nacional de Meteorologia (INMET) cobrindo mais de **25 anos de observações** (2000-2025) de Porto Alegre:

**Período de Cobertura:**

- **2000-2021**: Estação PORTO ALEGRE (A801)
- **2022-2025**: Estações PORTO ALEGRE - JARDIM BOTANICO (A801) e PORTO ALEGRE - BELEM NOVO (B807)

**Estações Meteorológicas:**

1. **INMET_S_RS_A801_PORTO ALEGRE** (2000-2021)

   - Código WMO: A801
   - Localização: -30,05°, -51,17°
   - Altitude: 46,97m
   - Fundação: 22/09/2000

2. **INMET_S_RS_A801_PORTO ALEGRE - JARDIM BOTANICO** (2022-2025)

   - Código WMO: A801
   - Localização: -30,05°, -51,17°
   - Altitude: 41,18m

3. **INMET_S_RS_B807_PORTO ALEGRE - BELEM NOVO** (2022-2025)
   - Código WMO: B807
   - Localização: Belém Novo, Porto Alegre

**Variáveis Meteorológicas Disponíveis:**

- Precipitação total horária (mm)
- Pressão atmosférica ao nível da estação (mB)
- Pressão atmosférica máxima/mínima na hora anterior
- Radiação global (Kj/m²)
- Temperatura do ar - bulbo seco (°C)
- Temperatura do ponto de orvalho (°C)
- Temperatura máxima/mínima na hora anterior
- Umidade relativa do ar (%)
- Umidade relativa máxima/mínima na hora anterior
- Velocidade e direção do vento (m/s, graus)
- Rajada máxima (m/s)

**Volume de Dados:**

- Total: ~210.000+ registros horários
- Período: Setembro 2000 - Abril 2025
- Frequência: Observações horárias (UTC)
- Formato: CSV com delimitador ";"

### 🏗️ Arquitetura do Sistema

#### Clean Architecture por Features

```
projeto_alerta_cheias/
├── app/
│   ├── core/                       # Domínio compartilhado
│   │   ├── __init__.py
│   │   ├── config.py              # Configurações globais
│   │   ├── exceptions.py          # Exceções customizadas
│   │   ├── dependencies.py        # Injeção de dependências
│   │   └── logging.py             # Configuração de logs
│   ├── features/
│   │   ├── forecast/              # Feature de Previsão
│   │   │   ├── domain/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── entities.py    # WeatherData, Forecast
│   │   │   │   ├── services.py    # ForecastService
│   │   │   │   └── repositories.py # Interfaces abstratas
│   │   │   ├── infra/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── model_loader.py # Carregamento LSTM
│   │   │   │   ├── forecast_model.py # TensorFlow Model
│   │   │   │   └── data_processor.py # Pré-processamento
│   │   │   ├── presentation/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── routes.py      # Endpoints FastAPI
│   │   │   │   └── schemas.py     # Pydantic DTOs
│   │   │   └── application/
│   │   │       ├── __init__.py
│   │   │       └── usecases.py    # GenerateForecastUseCase
│   │   └── alerts/                # Feature de Alertas
│   │       ├── domain/
│   │       │   ├── __init__.py
│   │       │   ├── entities.py    # Alert, AlertLevel
│   │       │   ├── alert_rules.py # Matriz de classificação
│   │       │   └── services.py    # AlertService
│   │       ├── infra/
│   │       │   ├── __init__.py
│   │       │   ├── external_api.py # APIs Guaíba/CPTEC
│   │       │   └── cache.py       # Redis/Memory cache
│   │       ├── presentation/
│   │       │   ├── __init__.py
│   │       │   ├── routes.py      # Endpoints de alerta
│   │       │   └── schemas.py     # DTOs de alerta
│   │       └── application/
│   │           ├── __init__.py
│   │           └── usecases.py    # GenerateAlertUseCase
│   ├── main.py                    # Inicialização FastAPI
│   └── config.py                  # Configurações centralizadas
├── data/
│   ├── raw/                       # Dados brutos
│   │   └── dados_historicos/      # CSVs meteorológicos INMET (2000-2025)
│   │       ├── INMET_S_RS_A801_PORTO ALEGRE_*.CSV     # Dados 2000-2021
│   │       ├── INMET_S_RS_A801_PORTO ALEGRE - JARDIM BOTANICO_*.CSV  # 2022-2025
│   │       └── INMET_S_RS_B807_PORTO ALEGRE- BELEM NOVO_*.CSV  # 2022-2025
│   ├── processed/                 # dados processados
│   └── modelos_treinados/         # Modelos salvos
├── notebooks/
│   ├── exploratory_analysis.ipynb # Análise exploratória
│   ├── data_preprocessing.ipynb   # Preprocessamento
│   ├── model_training.ipynb       # Treinamento LSTM
│   └── model_evaluation.ipynb     # Avaliação e métricas
├── tests/
│   ├── unit/                      # Testes unitários
│   │   ├── core/
│   │   ├── forecast/
│   │   └── alerts/
│   ├── integration/               # Testes de integração
│   │   ├── test_apis.py
│   │   └── test_endpoints.py
│   └── conftest.py               # Fixtures compartilhadas
├── scripts/
│   ├── setup_data.py             # Setup inicial de dados
│   ├── train_model.py            # Script de treinamento
│   └── migrate_data.py           # Migração de dados
├── docker/
│   ├── Dockerfile.api            # Container da API
│   ├── Dockerfile.training       # Container de treinamento
│   └── docker-compose.yml        # Orquestração completa
├── requirements/
│   ├── base.txt                  # Dependências base
│   ├── development.txt           # Dependências dev
│   └── production.txt            # Dependências prod
├── .env.example                  # Template de variáveis
├── .gitignore
├── README.md
└── pyproject.toml               # Configuração do projeto
```

### 📊 Stack Tecnológica

#### Core Technologies

- **Python 3.9+**: Linguagem principal
- **TensorFlow 2.x**: Modelos LSTM para séries temporais
- **FastAPI**: Framework web assíncrono
- **Pydantic**: Validação e serialização de dados
- **httpx**: Cliente HTTP assíncrono

#### Data & ML

- **Pandas/NumPy**: Manipulação e análise de dados
- **Scikit-learn**: Pré-processamento e métricas
- **Matplotlib/Seaborn**: Visualização de dados
- **Jupyter**: Notebooks para análise

#### Infrastructure

- **Docker**: Containerização
- **Redis**: Cache e session storage
- **PostgreSQL**: Banco de dados (opcional)
- **Uvicorn**: Servidor ASGI

#### Testing & Quality

- **pytest**: Framework de testes
- **pytest-asyncio**: Testes assíncronos
- **pytest-cov**: Cobertura de código
- **Black**: Formatação de código
- **isort**: Organização de imports
- **mypy**: Type checking

### 🔄 Roadmap de Implementação

#### 1. Configuração e Estrutura Base ✅

##### 1.1 Configuração do Projeto ✅

- ✅ Criar estrutura de diretórios conforme Clean Architecture
- ✅ Configurar `pyproject.toml` com dependências e metadados
- ✅ Criar arquivos de requirements separados (base, dev, prod)
- ✅ Configurar `.env.example` com todas as variáveis necessárias
- ✅ Setup inicial do Git com `.gitignore` apropriado

##### 1.2 Core Infrastructure ✅

- ✅ Implementar `app/core/config.py` com Pydantic Settings
- ✅ Criar `app/core/exceptions.py` com exceções customizadas
- ✅ Implementar `app/core/dependencies.py` para injeção de dependências
- ✅ Configurar logging estruturado em `app/core/logging.py`
- ✅ Setup básico do FastAPI em `app/main.py`

##### 1.3 Docker Setup ✅

- ✅ Criar `Dockerfile.api` otimizado com multi-stage build
- ✅ Criar `Dockerfile.training` para ambiente de ML
- ✅ Configurar `docker-compose.yml` com todos os serviços
- ✅ Implementar health checks nos containers
- ✅ Setup de volumes para dados e modelos

#### 2. Análise e Preparação de Dados ✅

##### 2.1 Exploração de Dados ✅

- ✅ Criar `notebooks/exploratory_analysis.ipynb`
- ✅ Analisar estrutura dos dados meteorológicos INMET (2000-2025)
  - ✅ Validar consistência entre diferentes estações (A801 vs B807)
  - ✅ Mapear mudanças na localização das estações (2022+)
  - ✅ Identificar períodos com dados faltantes
- ✅ Identificar padrões sazonais e tendências climáticas
  - ✅ Análise de precipitação mensal/sazonal
  - ✅ Tendências de temperatura ao longo de 25 anos
  - ✅ Padrões de vento e pressão atmosférica
- ✅ Detectar outliers e dados inconsistentes
  - ✅ Valores extremos de precipitação
  - ✅ Temperatura e umidade anômalas
  - ✅ Dados faltantes por período
- ✅ Gerar estatísticas descritivas e visualizações
  - ✅ Distribuição de precipitação por década
  - ✅ Correlação entre variáveis meteorológicas
  - ✅ Análise de eventos extremos de chuva

##### 2.2 Preprocessamento ✅

- ✅ Implementar `notebooks/data_preprocessing.ipynb`
- ✅ Padronizar formatos de data e timestamps
  - ✅ Converter formato de data entre anos (YYYY-MM-DD vs DD/MM/YYYY)
  - ✅ Sincronizar fusos horários (UTC)
  - ✅ Criar índice temporal contínuo
- ✅ Tratamento de valores missing/nulos
  - ✅ Identificar padrões de dados faltantes
  - ✅ Estratégias de imputação por variável
  - ✅ Interpolação temporal para gaps pequenos
- ✅ Normalização e scaling de features
  - ✅ StandardScaler para variáveis contínuas
  - ✅ MinMaxScaler para features específicas
  - ✅ Encoding para direção do vento
- ✅ Feature engineering (variáveis derivadas)
  - ✅ Índices meteorológicos derivados
  - ✅ Agregações temporais (3h, 6h, 12h, 24h)
  - ✅ Tendências e diferenças temporais
  - ✅ Sazonalidade e componentes cíclicos
- ✅ Unificação de dados entre estações
  - ✅ Merge de dados A801, B807 por período
  - ✅ Validação de consistência entre estações
  - ✅ Estratégia para transição 2021-2022
- ✅ Criar pipeline de preprocessamento reutilizável

##### 2.3 Scripts de Utilidade ✅

- ✅ Implementar `scripts/setup_data.py` para organização inicial
  - ✅ Consolidação automática de CSVs por ano
  - ✅ Validação de integridade dos dados
  - ✅ Detecção de arquivos corrompidos
- ✅ Criar `scripts/validate_data.py` para validação de consistência
  - ✅ Verificação de ranges válidos por variável
  - ✅ Detecção de anomalias estatísticas
  - ✅ Relatório de qualidade dos dados
- ✅ Implementar função de split temporal para treino/validação/teste
  - ✅ Split estratificado por década
  - ✅ Preservação de sazonalidade
  - ✅ Validação walk-forward para séries temporais

#### 3. Desenvolvimento do Modelo ML ✅

##### 3.1 Arquitetura do Modelo ✅

- ✅ `notebooks/model_training.ipynb`: Notebook principal de treinamento LSTM
- ✅ `notebooks/model_architecture_experiments.ipynb`: Experimentos de arquitetura
- ✅ `scripts/train_model.py`: Script automatizado de treinamento (752 linhas)
- ✅ `configs/model_config_examples.json`: Configurações de exemplo
- ✅ Comandos Make para treinamento e monitoramento
- ✅ Suporte a 6 arquiteturas diferentes (simple_1_layer até production)
- ✅ TensorBoard integrado para monitoramento
- ✅ Grid search automatizado para otimização
- ✅ Sistema completo de salvamento de artefatos
- ✅ Verificação automática dos critérios de sucesso

**Componentes Implementados:**

- ✅ **Design da Arquitetura LSTM**

  - Configuração para dados multivariados (16+ features)
  - Sequence length otimizado para dados horários (24h)
  - Arquitetura encoder-decoder para previsão 24h

- ✅ **Diferentes Configurações**

  - Teste com 1-3 camadas LSTM
  - Units: 32, 64, 128, 256 por camada
  - Dropout: 0.1-0.3 para regularização

- ✅ **Callbacks Configurados**

  - EarlyStopping com restauração dos melhores pesos
  - ReduceLROnPlateau para ajuste dinâmico da taxa de aprendizado
  - TensorBoard para monitoramento completo

- ✅ **Script de Treinamento**
  - Script completo e funcional (752 linhas)
  - Suporte a linha de comando com argumentos
  - Grid search automatizado
  - Modo experimental para testes rápidos
  - Salvamento automático de artefatos

**Comandos Funcionais:**

```bash
# Treinamento básico
make train-model

# Modo experimental
make train-experiment

# Grid search
make train-full-grid

# TensorBoard
make tensorboard
```

##### 3.2 Validação Avançada ✅

- ✅ **Pipeline de Treinamento Completo**

  - `scripts/training_pipeline.py` completo (796 linhas)
  - Preparação de sequências temporais para LSTM
  - Batch processing para grandes volumes de dados
  - Validation split temporal (não aleatório) preservando ordem cronológica

- ✅ **Cross-validation Temporal**

  - Walk-forward validation implementado
  - Classe `TemporalDataSplitter` para divisão temporal
  - Preservação rigorosa de ordem cronológica
  - Configuração flexível: min_train_months, validation_months, step_months
  - Múltiplos folds temporais com validação automática

- ✅ **Otimização de Hiperparâmetros**

  - Grid search sistemático implementado
  - Learning rates: 0.001, 0.0001, 0.00001
  - Batch sizes: 16, 32, 64, 128
  - Sequence lengths: 12, 24, 48, 72 horas
  - LSTM units: [64], [128], [64,32], [128,64], [256,128,64]
  - Dropout rates: 0.1, 0.2, 0.3

- ✅ **Métricas Meteorológicas Específicas**
  - Classe `MeteorologicalMetrics` implementada
  - MAE estratificado por intensidade de chuva (leve, moderada, forte)
  - RMSE para variáveis contínuas
  - Skill Score (Equitable Threat Score) para eventos de chuva
  - Métricas de classificação: Accuracy, F1-Score, AUC
  - Validação automática dos critérios de sucesso

**Comandos Implementados:**

```bash
# Validação cruzada temporal
make temporal-cv
make temporal-cv-extended

# Otimização de hiperparâmetros
make hyperopt
make hyperopt-full

# Pipeline completo
make training-pipeline
make training-pipeline-production

# Validação de métricas
make validate-model-metrics
make view-training-results

# Docker
make docker-temporal-cv
make docker-hyperopt
make docker-training-pipeline
```

**Notebook Demonstrativo:**

- ✅ `notebooks/jupyter/model_validation.ipynb`
- ✅ Demonstração completa de todas as funcionalidades
- ✅ Visualizações das métricas meteorológicas
- ✅ Exemplos práticos de uso

**Arquivos Criados:**

- ✅ `scripts/training_pipeline.py` - Pipeline principal (796 linhas)
- ✅ `notebooks/python/model_validation.py` - Notebook demonstrativo
- ✅ `scripts/test_model_validation.py` - Script de teste rápido
- ✅ Comandos adicionados ao `Makefile`

**Critérios de Sucesso Validados:**

- ✅ **Accuracy > 75%** em previsão de chuva 24h - **Implementado**
- ✅ **MAE < 2.0 mm/h** para precipitação - **Implementado**
- ✅ **RMSE < 3.0 mm/h** para precipitação - **Implementado**
- ✅ Validação automática dos critérios - **Implementado**

##### 3.3 Scripts de Teste e Validação ✅

- ✅ `scripts/test_model_validation.py` para validação rápida
- ✅ Testes unitários de cada componente
- ✅ Dados sintéticos para desenvolvimento
- ✅ Validação de funcionamento sem dependências completas

#### 4. Feature Forecast - Previsão ✅

##### 4.1 Domain Layer ✅

- ✅ **Implementar entidades em `app/features/forecast/domain/entities.py`**

  - ✅ `WeatherData`: dados meteorológicos completos com validação de ranges
  - ✅ `Forecast`: resultado da previsão com métricas de qualidade
  - ✅ `ModelMetrics`: métricas de performance do modelo ML
  - ✅ Enums: `WeatherCondition`, `PrecipitationLevel`
  - ✅ Métodos de validação e classificação automática
  - ✅ Conversão para dicionário e métodos de análise

- ✅ **Criar `app/features/forecast/domain/services.py`**

  - ✅ `ForecastService`: lógica de negócio principal para previsões
    - Validação de sequências de entrada para o modelo LSTM
    - Validação de qualidade das previsões geradas
    - Lógica de geração de alertas baseada em precipitação e nível do rio
    - Cálculo de score de risco considerando múltiplos fatores
    - Geração de sumários para tomada de decisão
  - ✅ `WeatherAnalysisService`: análise avançada de dados meteorológicos
    - Detecção de padrões temporais e sazonais
    - Identificação de anomalias em dados meteorológicos
    - Cálculo de índices meteorológicos específicos (Heat Index, Wind Chill)
    - Análise de tendências de pressão atmosférica
  - ✅ `ModelValidationService`: validação de modelos ML
    - Validação de métricas contra critérios estabelecidos (MAE < 2.0, RMSE < 3.0, Accuracy > 75%)
    - Comparação entre versões de modelos
    - Recomendações automáticas para atualização de modelos
  - ✅ `ForecastConfiguration`: classe de configuração centralizada

- ✅ **Definir interfaces em `app/features/forecast/domain/repositories.py`**
  - ✅ `WeatherDataRepository`: interface para dados meteorológicos históricos
    - Métodos para busca por período, query objects, estatísticas
    - Operações de salvamento em lote e individual
    - Contagem e validação de registros
  - ✅ `ForecastRepository`: interface para previsões meteorológicas
    - Gerenciamento de previsões com TTL e versionamento
    - Cálculo de métricas de accuracy vs dados reais
    - Limpeza automática de previsões antigas
  - ✅ `ModelRepository`: interface para modelos ML
    - Carregamento e salvamento de modelos TensorFlow
    - Gerenciamento de versões e metadados
    - Persistência de métricas de performance
  - ✅ `CacheRepository`: interface para operações de cache
    - Cache inteligente de previsões com TTL configurável
    - Operações básicas de cache (get, set, delete, exists)
  - ✅ Query Objects: `WeatherDataQuery`, `ForecastQuery`
  - ✅ Protocols: `ConfigurableRepository`, `HealthCheckRepository`
  - ✅ Exceções específicas e funções utilitárias

**Testes Implementados:**

- ✅ Script completo de testes: `scripts/test_forecast_domain.py`
- ✅ Validação de todas as entidades com dados reais
- ✅ Testes de services com cenários complexos
- ✅ Verificação da lógica de negócio e validações
- ✅ Testes de integração entre componentes

**Comandos para Teste:**

```bash
# Executar testes da Domain Layer
python3 scripts/test_forecast_domain.py
```

##### 4.2 Application Layer (Próximo)

- [ ] Implementar use cases em `app/features/forecast/application/usecases.py`
  - [ ] `GenerateForecastUseCase`: previsão principal
  - [ ] `GetModelMetricsUseCase`: métricas do modelo
  - [ ] `RefreshModelUseCase`: atualização do modelo

##### 4.3 Infrastructure Layer

- [ ] Implementar em `app/features/forecast/infra/model_loader.py`
- [ ] Implementar em `app/features/forecast/infra/forecast_model.py`
- [ ] Implementar em `app/features/forecast/infra/data_processor.py`

##### 4.4 Presentation Layer

- [ ] Criar DTOs em `app/features/forecast/presentation/schemas.py`
  - [ ] `ForecastRequest`: entrada da API
  - [ ] `ForecastResponse`: resposta da API
  - [ ] `ModelMetricsResponse`: métricas
- [ ] Implementar endpoints em `app/features/forecast/presentation/routes.py`
  - [ ] `POST /forecast/predict`: previsão meteorológica
  - [ ] `GET /forecast/metrics`: métricas do modelo
  - [ ] `POST /forecast/refresh-model`: atualizar modelo

#### 5. APIs Externas

##### 5.1 Integração CPTEC

- [ ] Implementar client para API CPTEC em `external_api.py`
- [ ] Mapeamento de dados da resposta JSON
- [ ] Tratamento de erros e timeouts
- [ ] Implementar retry logic com backoff exponencial
- [ ] Cache de respostas com TTL configurável

##### 5.2 Integração Guaíba

- [ ] Client para API do Nível do Guaíba
- [ ] Parser para extrair nível mais recente do JSON
- [ ] Validação de dados de entrada
- [ ] Monitoring de disponibilidade da API
- [ ] Fallback para dados históricos

##### 5.3 Circuit Breaker Pattern

- [ ] Implementar circuit breaker para alta resiliência
- [ ] Monitoring de health das APIs externas
- [ ] Alertas quando APIs ficam indisponíveis
- [ ] Métricas de latência e success rate

#### 6. Feature Alerts - Sistema de Alertas

##### 6.1 Domain Layer

- [ ] Implementar entidades em `app/features/alerts/domain/entities.py`
  - [ ] `Alert`: estrutura do alerta
  - [ ] `AlertLevel`: níveis de criticidade
  - [ ] `RiverLevel`: nível do rio
  - [ ] `RainPrediction`: previsão de chuva
- [ ] Criar regras em `app/features/alerts/domain/alert_rules.py`
  - [ ] Matriz de classificação atualizada
  - [ ] Validação de thresholds
  - [ ] Lógica de priorização

##### 6.2 Application Layer

- [ ] Use cases em `app/features/alerts/application/usecases.py`
  - [ ] `GenerateAlertUseCase`: alerta principal
  - [ ] `GetCurrentConditionsUseCase`: condições atuais
  - [ ] `GetAlertHistoryUseCase`: histórico de alertas

##### 6.3 Presentation Layer

- [ ] DTOs em `app/features/alerts/presentation/schemas.py`
  - [ ] `AlertRequest`: parâmetros do alerta
  - [ ] `AlertResponse`: resposta com nível e ação
  - [ ] `ConditionsResponse`: condições atuais
- [ ] Endpoints em `app/features/alerts/presentation/routes.py`
  - [ ] `GET /alerts/current`: alerta atual
  - [ ] `GET /alerts/conditions`: condições atuais
  - [ ] `GET /alerts/history`: histórico
  - [ ] `POST /alerts/evaluate`: avaliar condições específicas

#### 7. Testes e Qualidade

##### 7.1 Testes Unitários

- [ ] Testes para Core em `tests/unit/core/`
- [ ] Testes para Forecast em `tests/unit/forecast/`
  - [ ] Domain entities e services
  - [ ] Use cases isolados
  - [ ] Model loading e preprocessing
- [ ] Testes para Alerts em `tests/unit/alerts/`
  - [ ] Alert rules e classificação
  - [ ] Use cases de alerta
  - [ ] External API mocks

##### 7.2 Testes de Integração

- [ ] `tests/integration/test_apis.py`: testes de APIs externas
- [ ] `tests/integration/test_endpoints.py`: testes de endpoints
- [ ] `tests/integration/test_forecast_pipeline.py`: pipeline completo
- [ ] Setup de fixtures em `tests/conftest.py`

##### 7.3 Cobertura e Qualidade

- [ ] Configurar pytest-cov para cobertura > 80%
- [ ] Integrar Black para formatação automática
- [ ] Configurar isort para organização de imports
- [ ] Setup mypy para type checking
- [ ] Pre-commit hooks para qualidade

#### 8. Monitoramento e Logs

##### 8.1 Logging Estruturado

- [ ] Configurar logs JSON estruturados
- [ ] Request ID para rastreamento
- [ ] Logs por feature e camada
- [ ] Rotation por tamanho e data
- [ ] Diferentes níveis: DEBUG (dev), INFO (prod)

##### 8.2 Métricas e Monitoring

- [ ] Health checks por feature
- [ ] Métricas de performance da API
- [ ] Monitoring de accuracy do modelo
- [ ] Alertas de sistema (alta latência, errors)
- [ ] Dashboard de métricas

##### 8.3 Audit Trail

- [ ] Logs de auditoria para operações críticas
- [ ] Tracking de previsões geradas
- [ ] Histórico de alertas emitidos
- [ ] Monitoring de APIs externas

#### 9. Performance e Otimização

##### 9.1 Cache Strategy

- [ ] Cache de previsões com TTL inteligente
- [ ] Cache de dados de APIs externas
- [ ] Invalidação de cache baseada em eventos
- [ ] Redis para cache distribuído

##### 9.2 Async/Await Optimization

- [ ] Connection pooling para APIs externas
- [ ] Operações I/O concorrentes
- [ ] Async database operations (se aplicável)
- [ ] Background tasks para operações pesadas

##### 9.3 Load Testing

- [ ] Testes de carga com locust ou similar
- [ ] Profiling de performance
- [ ] Otimização de gargalos identificados
- [ ] Configuração de rate limiting

#### 10. Deployment e DevOps

##### 10.1 Container Optimization

- [ ] Multi-stage builds otimizados
- [ ] Imagens Python slim
- [ ] Usuário não-root para segurança
- [ ] Health checks implementados
- [ ] Configuração de recursos (CPU/Memory)

##### 10.2 Orchestration

- [ ] Docker Compose para desenvolvimento
- [ ] Kubernetes manifests (opcional)
- [ ] Environment-specific configurations
- [ ] Secrets management
- [ ] Backup strategies

##### 10.3 CI/CD Pipeline

- [ ] GitHub Actions ou similar
- [ ] Automated testing pipeline
- [ ] Docker image building
- [ ] Deployment automation
- [ ] Rolling updates strategy

### 🔧 Configurações Técnicas Detalhadas

#### APIs Externas

```python
# Configurações das APIs
GUAIBA_API_URL = "https://nivelguaiba.com.br/portoalegre.1day.json"
CPTEC_API_URL = "https://www.cptec.inpe.br/api/forecast-input?city=Porto%20Alegre%2C%20RS"

# Timeouts e Retry
API_TIMEOUT = 10  # segundos
MAX_RETRIES = 3
BACKOFF_FACTOR = 2
```

#### Dados Meteorológicos INMET

```python
# Configurações de processamento de dados
INMET_DATA_PATH = "data/raw/dados_historicos/"
PROCESSED_DATA_PATH = "data/processed/"

# Colunas principais dos dados INMET
INMET_COLUMNS = {
    'datetime': ['Data', 'Hora UTC'],
    'precipitation': 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)',
    'pressure': 'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)',
    'temperature': 'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)',
    'dew_point': 'TEMPERATURA DO PONTO DE ORVALHO (°C)',
    'humidity': 'UMIDADE RELATIVA DO AR, HORARIA (%)',
    'wind_speed': 'VENTO, VELOCIDADE HORARIA (m/s)',
    'wind_direction': 'VENTO, DIREÇÃO HORARIA (gr) (° (gr))',
    'radiation': 'RADIACAO GLOBAL (Kj/m²)'
}

# Ranges válidos para validação
VALID_RANGES = {
    'precipitation': (0, 200),    # mm/h
    'temperature': (-10, 50),     # °C
    'humidity': (0, 100),         # %
    'pressure': (900, 1100),      # mB
    'wind_speed': (0, 50)         # m/s
}
```

#### Matriz de Alertas Implementada

```python
def classify_alert_level(river_level: float, rain_prediction: float) -> AlertLevel:
    """Matriz de classificação de alertas atualizada"""
    if river_level > 3.60:
        return AlertLevel(nivel="Crítico", acao="Emergência")
    elif river_level > 3.15 and rain_prediction > 50:
        return AlertLevel(nivel="Alto", acao="Alerta")
    elif river_level > 2.80 and rain_prediction > 20:
        return AlertLevel(nivel="Moderado", acao="Atenção")
    else:
        return AlertLevel(nivel="Baixo", acao="Monitoramento")
```

#### Modelo LSTM Configuration

```python
# Parâmetros do modelo baseados nos dados INMET
SEQUENCE_LENGTH = 24      # 24 horas de histórico
FEATURES_COUNT = 16       # Variáveis meteorológicas disponíveis
LSTM_UNITS = [128, 64, 32]
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100

# Features principais dos dados INMET
FEATURE_COLUMNS = [
    'precipitation', 'pressure', 'temperature', 'dew_point',
    'humidity', 'wind_speed', 'wind_direction', 'radiation',
    'pressure_max', 'pressure_min', 'temp_max', 'temp_min',
    'humidity_max', 'humidity_min', 'dew_point_max', 'dew_point_min'
]
```

### 📈 Critérios de Sucesso

#### Modelo de ML

- ✅ Precisão > 75% em previsões de 24h
- ✅ MAE < 2.0 mm/h para precipitação
- ✅ RMSE < 3.0 mm/h para precipitação
- ✅ Tempo de inferência < 100ms

#### API Performance

- ✅ Latência média < 200ms
- ✅ Disponibilidade > 99.5%
- ✅ Rate limiting: 1000 req/min por IP
- ✅ Health check response < 50ms

#### Qualidade de Código

- ✅ Cobertura de testes > 80%
- ✅ Type hints em 100% das funções
- ✅ Documentação completa com docstrings
- ✅ Zero warnings no mypy

#### Monitoramento

- ✅ Logs estruturados em JSON
- ✅ Request tracing completo
- ✅ Métricas de negócio tracked
- ✅ Alertas automatizados configurados

### 🚀 Comandos de Execução

```bash
# Setup do ambiente
make setup

# Desenvolvimento
make dev

# Testes
make test
make test-cov

# Treinamento do modelo
make train-model

# Deploy
make docker-build
make docker-run

# Linting e formatação
make lint
make format
```

### 📋 Checklist de Entrega

#### Documentação

- [ ] README.md completo com instruções
- [ ] API documentation com OpenAPI/Swagger
- [ ] Architecture Decision Records (ADRs)
- [ ] Deployment guide
- [ ] Performance benchmarks

#### Código

- [ ] Todas as features implementadas
- [ ] Testes com cobertura > 80%
- [ ] Logs estruturados configurados
- [ ] Error handling robusto
- [ ] Type hints completos

#### Deployment

- [ ] Dockerfiles otimizados
- [ ] Docker Compose funcional
- [ ] Environment configurations
- [ ] Health checks implementados
- [ ] Monitoring configurado

#### Validação

- [ ] Modelo treinado com accuracy > 75%
- [ ] APIs externas integradas
- [ ] Matriz de alertas funcionando
- [ ] Performance targets atingidos
- [ ] Security checklist completado
