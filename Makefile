# Sistema de Alertas de Cheias - Rio Guaíba
# Makefile para automação de tarefas

.PHONY: help setup dev test test-cov lint format train-model docker-build docker-run clean install-dev install-prod

# Configurações
PYTHON := python3
PIP := pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose

# Cores para output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Mostra esta mensagem de ajuda
	@echo "$(BLUE)Sistema de Alertas de Cheias - Rio Guaíba$(NC)"
	@echo "=========================================="
	@echo ""
	@echo "Comandos disponíveis:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

setup: ## Setup inicial do projeto
	@echo "$(YELLOW)Configurando ambiente de desenvolvimento...$(NC)"
	$(PYTHON) -m venv venv
	@echo "$(GREEN)Ambiente virtual criado!$(NC)"
	@echo "$(YELLOW)Ative o ambiente virtual com:$(NC) source venv/bin/activate"

install-dev: ## Instala dependências de desenvolvimento
	@echo "$(YELLOW)Instalando dependências de desenvolvimento...$(NC)"
	$(PIP) install -r requirements/development.txt
	@echo "$(GREEN)Dependências de desenvolvimento instaladas!$(NC)"

install-prod: ## Instala dependências de produção
	@echo "$(YELLOW)Instalando dependências de produção...$(NC)"
	$(PIP) install -r requirements/production.txt
	@echo "$(GREEN)Dependências de produção instaladas!$(NC)"

dev: ## Executa a aplicação em modo desenvolvimento
	@echo "$(YELLOW)Iniciando aplicação em modo desenvolvimento...$(NC)"
	fastapi dev app/main.py --reload

run: ## Executa a aplicação em modo produção
	@echo "$(YELLOW)Iniciando aplicação em modo produção...$(NC)"
	fastapi run app/main.py

test: ## Executa todos os testes
	@echo "$(YELLOW)Executando testes...$(NC)"
	pytest tests/ -v

test-cov: ## Executa testes com cobertura
	@echo "$(YELLOW)Executando testes com cobertura...$(NC)"
	pytest tests/ --cov=app --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Relatório de cobertura gerado em htmlcov/index.html$(NC)"

test-unit: ## Executa apenas testes unitários
	@echo "$(YELLOW)Executando testes unitários...$(NC)"
	pytest tests/unit/ -v

test-integration: ## Executa apenas testes de integração
	@echo "$(YELLOW)Executando testes de integração...$(NC)"
	pytest tests/integration/ -v

lint: ## Executa linting do código
	@echo "$(YELLOW)Executando linting...$(NC)"
	flake8 app/ tests/
	mypy app/
	@echo "$(GREEN)Linting concluído!$(NC)"

format: ## Formata o código
	@echo "$(YELLOW)Formatando código...$(NC)"
	black app/ tests/ scripts/
	isort app/ tests/ scripts/
	@echo "$(GREEN)Código formatado!$(NC)"

format-check: ## Verifica formatação sem alterar arquivos
	@echo "$(YELLOW)Verificando formatação...$(NC)"
	black --check app/ tests/ scripts/
	isort --check-only app/ tests/ scripts/

pre-commit: ## Executa verificações pre-commit
	@echo "$(YELLOW)Executando verificações pre-commit...$(NC)"
	pre-commit run --all-files

install-hooks: ## Instala git hooks
	@echo "$(YELLOW)Instalando git hooks...$(NC)"
	pre-commit install
	@echo "$(GREEN)Git hooks instalados!$(NC)"

train-model: ## Treina o modelo de ML com configuração padrão
	@echo "$(YELLOW)Iniciando treinamento do modelo LSTM (configuração padrão)...$(NC)"
	$(PYTHON) scripts/train_model.py --architecture production
	@echo "$(GREEN)Treinamento concluído!$(NC)"

train-experiment: ## Treina modelo em modo experimental (epochs reduzidos)
	@echo "$(YELLOW)Iniciando treinamento experimental...$(NC)"
	$(PYTHON) scripts/train_model.py --experiment --architecture simple_2_layers
	@echo "$(GREEN)Treinamento experimental concluído!$(NC)"

train-architectures: ## Testa diferentes arquiteturas LSTM
	@echo "$(YELLOW)Testando diferentes arquiteturas LSTM...$(NC)"
	@echo "$(BLUE)Treinando simple_1_layer...$(NC)"
	$(PYTHON) scripts/train_model.py --experiment --architecture simple_1_layer --epochs 20
	@echo "$(BLUE)Treinando simple_2_layers...$(NC)"
	$(PYTHON) scripts/train_model.py --experiment --architecture simple_2_layers --epochs 20
	@echo "$(BLUE)Treinando simple_3_layers...$(NC)"
	$(PYTHON) scripts/train_model.py --experiment --architecture simple_3_layers --epochs 20
	@echo "$(BLUE)Treinando heavy_2_layers...$(NC)"
	$(PYTHON) scripts/train_model.py --experiment --architecture heavy_2_layers --epochs 20
	@echo "$(BLUE)Treinando light_3_layers...$(NC)"
	$(PYTHON) scripts/train_model.py --experiment --architecture light_3_layers --epochs 20
	@echo "$(GREEN)Teste de arquiteturas concluído! Verifique resultados em data/modelos_treinados/$(NC)"

train-sequence-lengths: ## Testa diferentes sequence lengths
	@echo "$(YELLOW)Testando diferentes sequence lengths...$(NC)"
	@echo "$(BLUE)Sequence length: 12 horas...$(NC)"
	$(PYTHON) scripts/train_model.py --experiment --sequence-length 12 --epochs 15
	@echo "$(BLUE)Sequence length: 24 horas...$(NC)"
	$(PYTHON) scripts/train_model.py --experiment --sequence-length 24 --epochs 15
	@echo "$(BLUE)Sequence length: 48 horas...$(NC)"
	$(PYTHON) scripts/train_model.py --experiment --sequence-length 48 --epochs 15
	@echo "$(GREEN)Teste de sequence lengths concluído!$(NC)"

train-learning-rates: ## Testa diferentes learning rates
	@echo "$(YELLOW)Testando diferentes learning rates...$(NC)"
	@echo "$(BLUE)Learning rate: 0.001...$(NC)"
	$(PYTHON) scripts/train_model.py --experiment --learning-rate 0.001 --epochs 15
	@echo "$(BLUE)Learning rate: 0.0001...$(NC)"
	$(PYTHON) scripts/train_model.py --experiment --learning-rate 0.0001 --epochs 15
	@echo "$(BLUE)Learning rate: 0.01...$(NC)"
	$(PYTHON) scripts/train_model.py --experiment --learning-rate 0.01 --epochs 15
	@echo "$(GREEN)Teste de learning rates concluído!$(NC)"

train-full-grid: ## Executa grid search completo de hiperparâmetros
	@echo "$(YELLOW)Executando grid search completo...$(NC)"
	@echo "$(RED)ATENÇÃO: Este comando pode demorar várias horas!$(NC)"
	@echo "$(BLUE)Iniciando notebook de experimentos...$(NC)"
	$(PYTHON) -c "import nbformat; import nbconvert; print('Executando notebook de experimentos...')"
	jupyter nbconvert --to notebook --execute notebooks/model_architecture_experiments.ipynb --output experiments_results.ipynb
	@echo "$(GREEN)Grid search concluído! Resultados em notebooks/experiments_results.ipynb$(NC)"

tensorboard: ## Inicia TensorBoard para monitorar treinamento
	@echo "$(YELLOW)Iniciando TensorBoard...$(NC)"
	tensorboard --logdir=data/modelos_treinados/tensorboard_logs --port=6006
	@echo "$(BLUE)TensorBoard disponível em: http://localhost:6006$(NC)"

model-info: ## Mostra informações dos modelos treinados
	@echo "$(YELLOW)Informações dos modelos treinados:$(NC)"
	@if [ -f "data/modelos_treinados/model_metadata.json" ]; then \
		$(PYTHON) -c "import json; data=json.load(open('data/modelos_treinados/model_metadata.json')); print(f'Modelo: {data[\"model_config\"][\"lstm_units\"]}'); print(f'MAE: {data[\"model_metrics\"][\"mae\"]:.4f}'); print(f'RMSE: {data[\"model_metrics\"][\"rmse\"]:.4f}'); print(f'Data: {data[\"training_date\"]}')"; \
	else \
		echo "$(RED)Nenhum modelo encontrado. Execute 'make train-model' primeiro.$(NC)"; \
	fi

model-compare: ## Compara resultados de diferentes experimentos
	@echo "$(YELLOW)Comparando resultados de experimentos...$(NC)"
	@if [ -f "data/modelos_treinados/experiments/experiment_results.csv" ]; then \
		$(PYTHON) -c "import pandas as pd; df=pd.read_csv('data/modelos_treinados/experiments/experiment_results.csv'); print('=== TOP 5 MODELOS ==='); print(df.nsmallest(5, 'mae')[['architecture', 'sequence_length', 'learning_rate', 'mae', 'rmse']].to_string(index=False))"; \
	else \
		echo "$(RED)Nenhum resultado de experimento encontrado. Execute 'make train-full-grid' primeiro.$(NC)"; \
	fi

validate-model: ## Valida modelo treinado
	@echo "$(YELLOW)Validando modelo treinado...$(NC)"
	@if [ -f "data/modelos_treinados/best_model.h5" ]; then \
		$(PYTHON) -c "import tensorflow as tf; model=tf.keras.models.load_model('data/modelos_treinados/best_model.h5'); print(f'Modelo carregado com {model.count_params():,} parâmetros'); model.summary()"; \
	else \
		echo "$(RED)Modelo não encontrado. Execute 'make train-model' primeiro.$(NC)"; \
	fi

clean-models: ## Remove modelos treinados
	@echo "$(YELLOW)Removendo modelos treinados...$(NC)"
	@read -p "Tem certeza que deseja remover todos os modelos? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/modelos_treinados/*; \
		echo "$(GREEN)Modelos removidos!$(NC)"; \
	else \
		echo "$(BLUE)Operação cancelada.$(NC)"; \
	fi

# ===============================
# Data Analysis and Notebooks
# ===============================

notebook-training: ## Abre notebook de treinamento
	@echo "$(YELLOW)Abrindo notebook de treinamento...$(NC)"
	jupyter notebook notebooks/jupyter/model_training.ipynb

notebook-experiments: ## Abre notebook de experimentos
	@echo "$(YELLOW)Abrindo notebook de experimentos...$(NC)"
	jupyter notebook notebooks/jupyter/model_architecture_experiments.ipynb

notebook-analysis: ## Abre notebook de análise exploratória
	@echo "$(YELLOW)Abrindo notebook de análise exploratória...$(NC)"
	jupyter notebook notebooks/jupyter/exploratory_analysis.ipynb

notebook-preprocessing: ## Abre notebook de preprocessamento
	@echo "$(YELLOW)Abrindo notebook de preprocessamento...$(NC)"
	jupyter notebook notebooks/jupyter/data_preprocessing.ipynb

notebook-evaluation: ## Abre notebook de avaliação de modelo
	@echo "$(YELLOW)Abrindo notebook de avaliação...$(NC)"
	jupyter notebook notebooks/jupyter/model_evaluation.ipynb

# ===============================
# Notebook Conversion Workflow
# ===============================

notebooks-list: ## Lista status de todos os notebooks
	@echo "$(YELLOW)Listando notebooks disponíveis...$(NC)"
	$(PYTHON) scripts/convert_notebooks.py --list

notebooks-convert-all: ## Converte todos os arquivos Python para notebooks
	@echo "$(YELLOW)Convertendo todos os notebooks...$(NC)"
	$(PYTHON) scripts/convert_notebooks.py --all --force
	@echo "$(GREEN)Todos os notebooks foram convertidos!$(NC)"

notebooks-convert: ## Converte um notebook específico (uso: make notebooks-convert NOTEBOOK=nome)
	@if [ -z "$(NOTEBOOK)" ]; then \
		echo "$(RED)Erro: Especifique o nome do notebook com NOTEBOOK=nome$(NC)"; \
		echo "$(BLUE)Exemplo: make notebooks-convert NOTEBOOK=model_training$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Convertendo notebook: $(NOTEBOOK)$(NC)"
	$(PYTHON) scripts/convert_notebooks.py $(NOTEBOOK) --force
	@echo "$(GREEN)Notebook $(NOTEBOOK) convertido com sucesso!$(NC)"

notebooks-install: ## Instala dependências necessárias para notebooks
	@echo "$(YELLOW)Instalando dependências dos notebooks...$(NC)"
	$(PIP) install jupyter jupytext nbformat nbconvert
	@echo "$(GREEN)Dependências dos notebooks instaladas!$(NC)"

notebooks-check: ## Verifica se notebooks estão atualizados
	@echo "$(YELLOW)Verificando status dos notebooks...$(NC)"
	@$(PYTHON) scripts/convert_notebooks.py --list | grep "⚠️" && \
		echo "$(RED)Alguns notebooks estão desatualizados! Execute 'make notebooks-convert-all'$(NC)" || \
		echo "$(GREEN)Todos os notebooks estão atualizados!$(NC)"

# ===============================
# Workflow Helpers
# ===============================

edit-notebook: ## Abre arquivo Python para edição (uso: make edit-notebook NOTEBOOK=nome)
	@if [ -z "$(NOTEBOOK)" ]; then \
		echo "$(RED)Erro: Especifique o nome do notebook com NOTEBOOK=nome$(NC)"; \
		echo "$(BLUE)Exemplo: make edit-notebook NOTEBOOK=model_training$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Abrindo para edição: notebooks/python/$(NOTEBOOK).py$(NC)"
	@if command -v code > /dev/null; then \
		code notebooks/python/$(NOTEBOOK).py; \
	elif command -v vim > /dev/null; then \
		vim notebooks/python/$(NOTEBOOK).py; \
	else \
		echo "$(RED)Editor não encontrado. Abra manualmente: notebooks/python/$(NOTEBOOK).py$(NC)"; \
	fi

sync-notebook: ## Sincroniza notebook específico (edita Python + converte para Jupyter)
	@if [ -z "$(NOTEBOOK)" ]; then \
		echo "$(RED)Erro: Especifique o nome do notebook com NOTEBOOK=nome$(NC)"; \
		echo "$(BLUE)Exemplo: make sync-notebook NOTEBOOK=model_training$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Sincronizando notebook: $(NOTEBOOK)$(NC)"
	$(PYTHON) scripts/convert_notebooks.py $(NOTEBOOK) --force
	@echo "$(GREEN)Notebook $(NOTEBOOK) sincronizado!$(NC)"
	@echo "$(BLUE)Para abrir no Jupyter: make notebook-$(NOTEBOOK)$(NC)"

# Aliases para comandos comuns
nb-list: notebooks-list
nb-convert: notebooks-convert-all
nb-check: notebooks-check
nb-sync: sync-notebook

# ===============================
# Docker Commands
# ===============================

docker-build: ## Constrói imagem Docker da API
	@echo "$(YELLOW)Construindo imagem Docker da API...$(NC)"
	$(DOCKER) build -f docker/Dockerfile.api -t alerta-cheias-api:latest .
	@echo "$(GREEN)Imagem Docker da API construída!$(NC)"

docker-build-training: ## Constrói imagem Docker para treinamento
	@echo "$(YELLOW)Construindo imagem Docker para treinamento...$(NC)"
	$(DOCKER) build -f docker/Dockerfile.training -t alerta-cheias-training:latest .
	@echo "$(GREEN)Imagem Docker de treinamento construída!$(NC)"

docker-build-all: ## Constrói todas as imagens Docker
	@echo "$(YELLOW)Construindo todas as imagens Docker...$(NC)"
	$(MAKE) docker-build
	$(MAKE) docker-build-training
	@echo "$(GREEN)Todas as imagens Docker construídas!$(NC)"

docker-run: ## Executa aplicação via Docker (desenvolvimento)
	@echo "$(YELLOW)Executando aplicação via Docker...$(NC)"
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml up -d api redis postgres
	@echo "$(GREEN)Aplicação executando em Docker!$(NC)"
	@echo "$(BLUE)API disponível em: http://localhost:8000$(NC)"
	@echo "$(BLUE)Documentação em: http://localhost:8000/docs$(NC)"

docker-run-training: ## Executa ambiente de treinamento
	@echo "$(YELLOW)Executando ambiente de treinamento...$(NC)"
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml --profile training up -d
	@echo "$(GREEN)Ambiente de treinamento executando!$(NC)"
	@echo "$(BLUE)Jupyter Lab: http://localhost:8888 (token: alerta_cheias_dev)$(NC)"
	@echo "$(BLUE)TensorBoard: http://localhost:6006$(NC)"
	@echo "$(BLUE)MLflow: http://localhost:5000$(NC)"

docker-run-prod: ## Executa em modo produção com Nginx
	@echo "$(YELLOW)Executando em modo produção...$(NC)"
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml --profile production up -d
	@echo "$(GREEN)Aplicação em produção executando!$(NC)"
	@echo "$(BLUE)Aplicação disponível em: http://localhost$(NC)"

docker-run-monitoring: ## Executa com monitoramento (Prometheus + Grafana)
	@echo "$(YELLOW)Executando com monitoramento...$(NC)"
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml --profile monitoring up -d
	@echo "$(GREEN)Monitoramento executando!$(NC)"
	@echo "$(BLUE)Prometheus: http://localhost:9090$(NC)"
	@echo "$(BLUE)Grafana: http://localhost:3000 (admin/admin)$(NC)"

docker-run-all: ## Executa todos os serviços
	@echo "$(YELLOW)Executando todos os serviços...$(NC)"
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml --profile training --profile production --profile monitoring up -d
	@echo "$(GREEN)Todos os serviços executando!$(NC)"

docker-stop: ## Para containers Docker
	@echo "$(YELLOW)Parando containers Docker...$(NC)"
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml down
	@echo "$(GREEN)Containers parados!$(NC)"

docker-restart: ## Reinicia containers Docker
	@echo "$(YELLOW)Reiniciando containers Docker...$(NC)"
	$(MAKE) docker-stop
	$(MAKE) docker-run
	@echo "$(GREEN)Containers reiniciados!$(NC)"

docker-logs: ## Mostra logs dos containers
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml logs -f

docker-logs-api: ## Mostra logs apenas da API
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml logs -f api

docker-logs-training: ## Mostra logs do ambiente de treinamento
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml logs -f training

docker-status: ## Mostra status dos containers
	@echo "$(YELLOW)Status dos containers:$(NC)"
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml ps

docker-exec-api: ## Acessa terminal da API
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml exec api /bin/bash

docker-exec-training: ## Acessa terminal do ambiente de treinamento
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml exec training /bin/bash

docker-exec-postgres: ## Acessa terminal do PostgreSQL
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml exec postgres psql -U postgres -d alerta_cheias

docker-exec-redis: ## Acessa terminal do Redis
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml exec redis redis-cli

docker-volumes: ## Lista volumes Docker
	@echo "$(YELLOW)Volumes Docker:$(NC)"
	$(DOCKER) volume ls | grep alerta_cheias

docker-clean: ## Remove containers e imagens não utilizadas
	@echo "$(YELLOW)Limpando containers e imagens não utilizadas...$(NC)"
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml down --volumes --remove-orphans
	$(DOCKER) system prune -f
	$(DOCKER) volume prune -f
	@echo "$(GREEN)Limpeza concluída!$(NC)"

docker-clean-all: ## Remove TUDO relacionado ao Docker (cuidado!)
	@echo "$(RED)ATENÇÃO: Isso removerá TODOS os dados persistentes!$(NC)"
	@read -p "Tem certeza? (y/N) " confirm && [ "$$confirm" = "y" ] || exit 1
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml down --volumes --remove-orphans
	$(DOCKER) volume rm $$($(DOCKER) volume ls -q | grep alerta_cheias) 2>/dev/null || true
	$(DOCKER) image rm alerta-cheias-api:latest alerta-cheias-training:latest 2>/dev/null || true
	@echo "$(GREEN)Limpeza completa concluída!$(NC)"

# ===============================
# Comandos específicos do projeto
# ===============================

docker-train-model: ## Treina modelo via Docker
	@echo "$(YELLOW)Treinando modelo via Docker...$(NC)"
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml --profile training run --rm training python scripts/train_model.py
	@echo "$(GREEN)Treinamento via Docker concluído!$(NC)"

docker-setup-data: ## Configura dados via Docker
	@echo "$(YELLOW)Configurando dados via Docker...$(NC)"
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml run --rm api python scripts/setup_data.py
	@echo "$(GREEN)Dados configurados via Docker!$(NC)"

docker-migrate-db: ## Executa migrações do banco de dados
	@echo "$(YELLOW)Executando migrações do banco de dados...$(NC)"
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml exec postgres psql -U postgres -d alerta_cheias -f /docker-entrypoint-initdb.d/init.sql
	@echo "$(GREEN)Migrações concluídas!$(NC)"

docker-backup-db: ## Faz backup do banco de dados
	@echo "$(YELLOW)Fazendo backup do banco de dados...$(NC)"
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml exec postgres pg_dump -U postgres alerta_cheias > backup_db_$$(date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)Backup do banco concluído!$(NC)"

docker-health: ## Verifica saúde dos containers
	@echo "$(YELLOW)Verificando saúde dos containers...$(NC)"
	@echo "API Health:"
	@curl -f http://localhost:8000/health 2>/dev/null && echo "$(GREEN)✓ API OK$(NC)" || echo "$(RED)✗ API Failed$(NC)"
	@echo "Redis Health:"
	@$(DOCKER_COMPOSE) -f docker/docker-compose.yml exec redis redis-cli ping 2>/dev/null && echo "$(GREEN)✓ Redis OK$(NC)" || echo "$(RED)✗ Redis Failed$(NC)"
	@echo "PostgreSQL Health:"
	@$(DOCKER_COMPOSE) -f docker/docker-compose.yml exec postgres pg_isready -U postgres 2>/dev/null && echo "$(GREEN)✓ PostgreSQL OK$(NC)" || echo "$(RED)✗ PostgreSQL Failed$(NC)"

# ===============================
# Fase 3.2 - Treinamento e Validação
# ===============================

temporal-cv: ## Executa validação cruzada temporal (Fase 3.2)
	@echo "$(YELLOW)Executando validação cruzada temporal...$(NC)"
	$(PYTHON) scripts/training_pipeline.py --mode temporal-cv --max-folds 5
	@echo "$(GREEN)Validação cruzada temporal concluída!$(NC)"

temporal-cv-extended: ## Validação cruzada temporal com mais folds
	@echo "$(YELLOW)Executando validação cruzada temporal estendida...$(NC)"
	$(PYTHON) scripts/training_pipeline.py --mode temporal-cv --max-folds 10
	@echo "$(GREEN)Validação cruzada temporal estendida concluída!$(NC)"

hyperopt: ## Otimização de hiperparâmetros (Fase 3.2)
	@echo "$(YELLOW)Executando otimização de hiperparâmetros...$(NC)"
	$(PYTHON) scripts/training_pipeline.py --mode hyperopt --max-trials 20
	@echo "$(GREEN)Otimização de hiperparâmetros concluída!$(NC)"

hyperopt-full: ## Otimização completa de hiperparâmetros
	@echo "$(YELLOW)Executando otimização completa de hiperparâmetros...$(NC)"
	$(PYTHON) scripts/training_pipeline.py --mode hyperopt --max-trials 50
	@echo "$(GREEN)Otimização completa concluída!$(NC)"

training-pipeline: ## Pipeline completo de treinamento e validação (Fase 3.2)
	@echo "$(YELLOW)Executando pipeline completo de treinamento...$(NC)"
	$(PYTHON) scripts/training_pipeline.py --mode full-pipeline --max-folds 5 --max-trials 20
	@echo "$(GREEN)Pipeline completo de treinamento concluído!$(NC)"

training-pipeline-production: ## Pipeline completo para produção
	@echo "$(YELLOW)Executando pipeline completo para produção...$(NC)"
	$(PYTHON) scripts/training_pipeline.py --mode full-pipeline --max-folds 10 --max-trials 50
	@echo "$(GREEN)Pipeline de produção concluído!$(NC)"

validate-model-metrics: ## Valida se o modelo atende critérios de sucesso
	@echo "$(YELLOW)Validando métricas do modelo...$(NC)"
	@echo "$(BLUE)Critérios de sucesso:$(NC)"
	@echo "• Accuracy >= 75% para classificação de eventos de chuva"
	@echo "• MAE <= 2.0 mm/h para precipitação"
	@echo "• RMSE <= 3.0 mm/h para precipitação"
	@if [ -f "data/modelos_treinados/temporal_validation/cv_results_*.json" ]; then \
		echo "$(GREEN)✓ Arquivos de validação encontrados$(NC)"; \
		$(PYTHON) -c "import json; import glob; \
		files = glob.glob('data/modelos_treinados/temporal_validation/cv_results_*.json'); \
		if files: \
			with open(sorted(files)[-1]) as f: data = json.load(f); \
			acc = data.get('accuracy_mean', 0); mae = data.get('mae_mean', 0); rmse = data.get('rmse_mean', 0); \
			print(f'Accuracy: {acc:.3f} ({\"✅\" if acc >= 0.75 else \"❌\"})'); \
			print(f'MAE: {mae:.3f} ({\"✅\" if mae <= 2.0 else \"❌\"})'); \
			print(f'RMSE: {rmse:.3f} ({\"✅\" if rmse <= 3.0 else \"❌\"})'); \
		else: print('❌ Nenhum resultado encontrado')"; \
	else \
		echo "$(RED)❌ Execute primeiro: make temporal-cv$(NC)"; \
	fi

view-training-results: ## Visualiza resultados de treinamento
	@echo "$(YELLOW)Resultados de treinamento:$(NC)"
	@echo "$(BLUE)Validação cruzada temporal:$(NC)"
	@ls -la data/modelos_treinados/temporal_validation/ 2>/dev/null || echo "  Nenhum resultado de validação temporal"
	@echo "$(BLUE)Otimização de hiperparâmetros:$(NC)"
	@ls -la data/modelos_treinados/hyperopt/ 2>/dev/null || echo "  Nenhum resultado de otimização"
	@echo "$(BLUE)Logs de treinamento:$(NC)"
	@ls -la training_pipeline.log 2>/dev/null || echo "  Nenhum log de pipeline"

clean-training-results: ## Remove resultados de treinamento
	@echo "$(YELLOW)Removendo resultados de treinamento...$(NC)"
	rm -rf data/modelos_treinados/temporal_validation/*
	rm -rf data/modelos_treinados/hyperopt/*
	rm -f training_pipeline.log
	@echo "$(GREEN)Resultados de treinamento removidos!$(NC)"

# ===============================
# Docker para Fase 3.2
# ===============================

docker-temporal-cv: ## Validação cruzada temporal via Docker
	@echo "$(YELLOW)Executando validação cruzada temporal via Docker...$(NC)"
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml --profile training run --rm training python scripts/training_pipeline.py --mode temporal-cv --max-folds 5
	@echo "$(GREEN)Validação cruzada temporal via Docker concluída!$(NC)"

docker-hyperopt: ## Otimização de hiperparâmetros via Docker
	@echo "$(YELLOW)Executando otimização via Docker...$(NC)"
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml --profile training run --rm training python scripts/training_pipeline.py --mode hyperopt --max-trials 20
	@echo "$(GREEN)Otimização via Docker concluída!$(NC)"

docker-training-pipeline: ## Pipeline completo via Docker
	@echo "$(YELLOW)Executando pipeline completo via Docker...$(NC)"
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml --profile training run --rm training python scripts/training_pipeline.py --mode full-pipeline --max-folds 5 --max-trials 20
	@echo "$(GREEN)Pipeline completo via Docker concluído!$(NC)"

# ===============================
# Comandos originais
# ===============================

clean: ## Remove arquivos temporários e cache
	@echo "$(YELLOW)Removendo arquivos temporários...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	rm -rf build/ dist/
	@echo "$(GREEN)Limpeza concluída!$(NC)"

check: ## Executa todas as verificações (lint, format-check, test)
	@echo "$(YELLOW)Executando todas as verificações...$(NC)"
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) test
	@echo "$(GREEN)Todas as verificações concluídas!$(NC)"

ci: ## Executa pipeline de CI
	@echo "$(YELLOW)Executando pipeline de CI...$(NC)"
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) test-cov
	@echo "$(GREEN)Pipeline de CI concluída!$(NC)"

docs: ## Gera documentação
	@echo "$(YELLOW)Gerando documentação...$(NC)"
	mkdocs build
	@echo "$(GREEN)Documentação gerada em site/$(NC)"

docs-serve: ## Serve documentação localmente
	@echo "$(YELLOW)Servindo documentação localmente...$(NC)"
	mkdocs serve

env-check: ## Verifica configuração do ambiente
	@echo "$(YELLOW)Verificando configuração do ambiente...$(NC)"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "Virtual environment: $$VIRTUAL_ENV"
	@if [ -f ".env" ]; then echo "$(GREEN).env file exists$(NC)"; else echo "$(RED).env file missing$(NC)"; fi

deps-check: ## Verifica dependências desatualizadas
	@echo "$(YELLOW)Verificando dependências desatualizadas...$(NC)"
	$(PIP) list --outdated

deps-update: ## Atualiza dependências (cuidado!)
	@echo "$(RED)ATENÇÃO: Isso pode quebrar compatibilidade!$(NC)"
	@echo "$(YELLOW)Atualizando dependências...$(NC)"
	$(PIP) install --upgrade -r requirements/development.txt

backup-data: ## Faz backup dos dados
	@echo "$(YELLOW)Fazendo backup dos dados...$(NC)"
	tar -czf backup_data_$$(date +%Y%m%d_%H%M%S).tar.gz data/
	@echo "$(GREEN)Backup concluído!$(NC)"

health: ## Verifica saúde da aplicação
	@echo "$(YELLOW)Verificando saúde da aplicação...$(NC)"
	curl -f http://localhost:8000/health || echo "$(RED)Aplicação não está respondendo$(NC)"

# ===============================
# Comandos para desenvolvimento rápido
# ===============================

quick-start: setup install-dev ## Setup rápido para desenvolvimento
	@echo "$(GREEN)Setup rápido concluído!$(NC)"
	@echo "$(YELLOW)Próximos passos:$(NC)"
	@echo "1. Ative o ambiente virtual: source venv/bin/activate"
	@echo "2. Configure .env: cp .env.example .env"
	@echo "3. Execute: make dev"

quick-docker: docker-build docker-run ## Setup rápido com Docker
	@echo "$(GREEN)Setup Docker concluído!$(NC)"
	@echo "$(YELLOW)Serviços disponíveis:$(NC)"
	@echo "• API: http://localhost:8000"
	@echo "• Docs: http://localhost:8000/docs"
	@echo "• Redis: localhost:6379"
	@echo "• PostgreSQL: localhost:5432"

quick-training: docker-build-training docker-run-training ## Setup rápido para treinamento
	@echo "$(GREEN)Setup de treinamento concluído!$(NC)"
	@echo "$(YELLOW)Serviços disponíveis:$(NC)"
	@echo "• Jupyter Lab: http://localhost:8888"
	@echo "• TensorBoard: http://localhost:6006"
	@echo "• MLflow: http://localhost:5000"

all: clean format lint test ## Executa limpeza, formatação, lint e testes 