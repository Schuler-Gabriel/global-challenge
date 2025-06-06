# 🏗️ Nova Arquitetura do Projeto - Proposta de Refatoração

## 🎯 Objetivos da Refatoração

### Problemas Identificados

- ❌ Notebooks distantes dos scripts relacionados
- ❌ Scripts amontoados numa única pasta sem organização
- ❌ Dados longe dos scripts e notebooks que os utilizam
- ❌ Mistura entre componentes de ML/Data Science e API Backend
- ❌ Falta de modularidade e clean architecture

### Solução Proposta

- ✅ **Separação clara entre ML/DS e Backend**
- ✅ **Organização modular por domínio**
- ✅ **Proximidade entre dados, notebooks e scripts relacionados**
- ✅ **Clean Architecture inspirada em TensorFlow Project Template**
- ✅ **Facilidade de desenvolvimento e manutenção**

## 📁 Nova Estrutura Proposta

```
sistema-alertas-cheias/
│
├── 🧠 ml-platform/                          # PLATAFORMA ML/DATA SCIENCE
│   ├── data/                               # Dados centralizados para ML
│   │   ├── raw/                           # Dados brutos
│   │   ├── processed/                     # Dados processados
│   │   ├── features/                      # Features engineered
│   │   ├── models/                        # Modelos treinados salvos
│   │   └── validation/                    # Dados de validação
│   │
│   ├── notebooks/                         # Notebooks de análise/experimentação
│   │   ├── 01-data-collection/            # Coleta de dados
│   │   ├── 02-exploratory-analysis/       # Análise exploratória
│   │   ├── 03-feature-engineering/        # Engenharia de features
│   │   ├── 04-model-development/          # Desenvolvimento de modelos
│   │   ├── 05-model-evaluation/           # Avaliação de modelos
│   │   └── 06-model-validation/           # Validação de modelos
│   │
│   ├── src/                               # Código fonte ML
│   │   ├── data/                          # Módulos de dados
│   │   │   ├── collectors/                # Coletores de dados
│   │   │   ├── processors/                # Processadores de dados
│   │   │   └── validators/                # Validadores de dados
│   │   │
│   │   ├── features/                      # Feature engineering
│   │   │   ├── atmospheric/               # Features atmosféricas
│   │   │   ├── temporal/                  # Features temporais
│   │   │   └── transforms/                # Transformações
│   │   │
│   │   ├── models/                        # Modelos ML
│   │   │   ├── base/                      # Classes base
│   │   │   ├── lstm/                      # Modelos LSTM
│   │   │   ├── ensemble/                  # Modelos ensemble
│   │   │   └── hybrid/                    # Modelos híbridos
│   │   │
│   │   ├── training/                      # Pipeline de treinamento
│   │   │   ├── trainers/                  # Classes de treinamento
│   │   │   ├── optimizers/                # Otimizadores
│   │   │   └── schedulers.py              # Agendadores
│   │   │
│   │   ├── evaluation/                    # Avaliação de modelos
│   │   │   ├── metrics/                   # Métricas customizadas
│   │   │   ├── validators/                # Validadores
│   │   │   └── reports/                   # Relatórios
│   │   │
│   │   └── utils/                         # Utilitários ML
│   │       ├── config.py                  # Configurações
│   │       ├── logging.py                 # Logging
│   │       └── visualization.py           # Visualizações
│   │
│   ├── scripts/                           # Scripts de ML
│   │   ├── data/                          # Scripts de dados
│   │   │   ├── collect_openmeteo.py       # Coleta Open-Meteo
│   │   │   ├── process_inmet.py           # Processamento INMET
│   │   │   └── validate_quality.py       # Validação qualidade
│   │   │
│   │   ├── training/                      # Scripts de treinamento
│   │   │   ├── train_lstm.py              # Treinar LSTM
│   │   │   ├── train_hybrid.py            # Treinar híbrido
│   │   │   └── hyperparameter_tuning.py   # Tuning
│   │   │
│   │   ├── evaluation/                    # Scripts de avaliação
│   │   │   ├── evaluate_model.py          # Avaliação geral
│   │   │   ├── cross_validation.py        # Validação cruzada
│   │   │   └── benchmark.py               # Benchmark
│   │   │
│   │   └── deployment/                    # Scripts de deploy ML
│   │       ├── export_model.py            # Exportar modelo
│   │       └── model_serving.py           # Servir modelo
│   │
│   ├── configs/                           # Configurações ML
│   │   ├── data/                          # Configs de dados
│   │   ├── models/                        # Configs de modelos
│   │   └── training/                      # Configs de treinamento
│   │
│   ├── experiments/                       # Experimentos ML
│   │   ├── runs/                          # Execuções experimentais
│   │   └── results/                       # Resultados
│   │
│   ├── tests/                             # Testes ML
│   │   ├── test_data/                     # Testes de dados
│   │   ├── test_models/                   # Testes de modelos
│   │   └── test_training/                 # Testes de treinamento
│   │
│   ├── Makefile                           # Comandos ML
│   ├── pyproject.toml                     # Dependências ML
│   └── README.md                          # Documentação ML
│
├── 🌐 api-backend/                         # BACKEND API/SERVIÇOS
│   ├── src/                               # Código fonte API
│   │   ├── core/                          # Core da aplicação
│   │   │   ├── config.py                  # Configurações
│   │   │   ├── dependencies.py            # Dependências
│   │   │   └── middleware.py              # Middlewares
│   │   │
│   │   ├── api/                           # Endpoints API
│   │   │   ├── v1/                        # Versão 1 da API
│   │   │   │   ├── weather/               # Endpoints meteorológicos
│   │   │   │   ├── alerts/                # Endpoints de alertas
│   │   │   │   └── predictions/           # Endpoints de previsões
│   │   │   └── health/                    # Health checks
│   │   │
│   │   ├── services/                      # Serviços de negócio
│   │   │   ├── weather_service.py         # Serviço meteorológico
│   │   │   ├── alert_service.py           # Serviço de alertas
│   │   │   └── prediction_service.py      # Serviço de previsões
│   │   │
│   │   ├── repositories/                  # Repositórios de dados
│   │   │   ├── weather_repo.py            # Repositório meteorológico
│   │   │   └── alert_repo.py              # Repositório de alertas
│   │   │
│   │   ├── models/                        # Modelos de dados (Pydantic)
│   │   │   ├── weather.py                 # Modelos meteorológicos
│   │   │   ├── alerts.py                  # Modelos de alertas
│   │   │   └── predictions.py             # Modelos de previsões
│   │   │
│   │   ├── integrations/                  # Integrações externas
│   │   │   ├── openmeteo/                 # Integração Open-Meteo
│   │   │   ├── ml_platform/               # Integração com ML Platform
│   │   │   └── notifications/             # Sistemas de notificação
│   │   │
│   │   └── utils/                         # Utilitários API
│   │       ├── logging.py                 # Logging
│   │       ├── cache.py                   # Cache
│   │       └── security.py                # Segurança
│   │
│   ├── tests/                             # Testes API
│   │   ├── unit/                          # Testes unitários
│   │   ├── integration/                   # Testes de integração
│   │   └── e2e/                           # Testes end-to-end
│   │
│   ├── alembic/                           # Migrações de BD
│   ├── configs/                           # Configurações API
│   ├── main.py                            # Entry point
│   ├── pyproject.toml                     # Dependências API
│   └── README.md                          # Documentação API
│
├── 🔗 shared/                             # COMPONENTES COMPARTILHADOS
│   ├── schemas/                           # Schemas compartilhados
│   ├── utils/                             # Utilitários comuns
│   ├── configs/                           # Configurações globais
│   └── types/                             # Tipos de dados
│
├── 🐳 infrastructure/                      # INFRAESTRUTURA E DEPLOY
│   ├── docker/                            # Containers Docker
│   │   ├── ml-platform/                   # Container ML
│   │   ├── api-backend/                   # Container API
│   │   └── docker-compose.yml             # Orquestração
│   │
│   ├── k8s/                               # Kubernetes manifests
│   ├── terraform/                         # Infrastructure as Code
│   └── monitoring/                        # Monitoramento
│
├── 📖 docs/                               # DOCUMENTAÇÃO GERAL
│   ├── architecture/                      # Documentação arquitetura
│   ├── api/                              # Documentação API
│   ├── ml/                               # Documentação ML
│   └── deployment/                       # Documentação deploy
│
├── .github/                               # GitHub Actions
├── .gitignore                             # Git ignore global
├── Makefile                               # Comandos globais
├── docker-compose.yml                     # Orquestração local
└── README.md                              # Documentação principal
```

## 🎯 Vantagens da Nova Arquitetura

### 🧠 ML Platform (Independente)

- ✅ **Auto-suficiente**: Notebooks, dados e scripts no mesmo contexto
- ✅ **Experimentação**: Facilita pesquisa e desenvolvimento de modelos
- ✅ **Reproducibilidade**: Configs e experimentos organizados
- ✅ **Colaboração**: Data Scientists trabalham de forma independente

### 🌐 API Backend (Independente)

- ✅ **Clean Architecture**: Separação clara de responsabilidades
- ✅ **Testabilidade**: Testes unitários e de integração
- ✅ **Escalabilidade**: Fácil de escalar e manter
- ✅ **Performance**: Otimizado para produção

### 🔗 Integração Inteligente

- ✅ **Desacoplamento**: Componentes independentes
- ✅ **Flexibilidade**: ML pode evoluir sem afetar API
- ✅ **Deploy Independente**: Pipelines de CI/CD separados
- ✅ **Monitoramento**: Métricas específicas por componente

## 🚀 Benefícios Imediatos

### Para Data Scientists

- 📊 **Notebooks próximos aos dados e scripts**
- 🧪 **Experimentação facilitada**
- 📈 **Versionamento de experimentos**
- 🔄 **Pipeline de ML completo**

### Para Desenvolvedores Backend

- 🏗️ **Arquitetura limpa e testável**
- ⚡ **APIs otimizadas para produção**
- 🔒 **Segurança e monitoramento**
- 📱 **Integrações externas organizadas**

### Para DevOps

- 🐳 **Containers especializados**
- 🚀 **Deploy independente**
- 📊 **Monitoramento granular**
- 🔧 **Configuração centralizada**

## 📋 Plano de Migração

### Fase 1: Preparação (1-2 dias)

1. Criar nova estrutura de diretórios
2. Definir configurações base
3. Configurar ambientes virtuais separados

### Fase 2: Migração ML (3-4 dias)

1. Mover notebooks organizadamente
2. Refatorar scripts em módulos
3. Reorganizar dados e configs
4. Atualizar imports e caminhos

### Fase 3: Migração API (2-3 dias)

1. Refatorar código FastAPI
2. Implementar clean architecture
3. Criar integração com ML Platform
4. Atualizar testes

### Fase 4: Integração (1-2 dias)

1. Configurar comunicação entre componentes
2. Testar pipelines completos
3. Atualizar documentação
4. Configurar CI/CD

## 🛠️ Comandos Principais

```bash
# ML Platform
cd ml-platform/
make setup-ml-env
make collect-data
make train-model
make evaluate-model

# API Backend
cd api-backend/
make setup-api-env
make run-api
make test-api

# Infraestrutura
make docker-build-all
make docker-run-local
make deploy-staging
```

## 🔍 Próximos Passos

1. **Validar Proposta**: Revisar com equipe
2. **Criar Estrutura**: Implementar nova organização
3. **Migrar Gradualmente**: Mover componentes por partes
4. **Testar Integração**: Validar funcionamento
5. **Atualizar Documentação**: Manter docs atualizadas

Esta arquitetura segue as melhores práticas de projetos ML/TensorFlow e garante escalabilidade, manutenibilidade e facilidade de desenvolvimento.
