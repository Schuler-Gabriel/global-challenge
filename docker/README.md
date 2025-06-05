# Docker Setup - Sistema de Alertas de Cheias

Este diretório contém toda a configuração Docker para o Sistema de Alertas de Cheias do Rio Guaíba.

## 📋 Arquivos

- `Dockerfile.api` - Container da API FastAPI para produção
- `Dockerfile.training` - Container para ambiente de ML/treinamento
- `docker-compose.yml` - Orquestração completa dos serviços
- `nginx.conf` - Configuração do Nginx como reverse proxy
- `prometheus.yml` - Configuração do Prometheus para métricas

## 🚀 Quick Start

### Desenvolvimento Básico

```bash
# Construir e executar API + banco + cache
make docker-build
make docker-run

# Acessar serviços
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
# PostgreSQL: localhost:5432
# Redis: localhost:6379
```

### Ambiente de Treinamento ML

```bash
# Construir e executar ambiente completo de ML
make docker-build-training
make docker-run-training

# Acessar serviços
# Jupyter Lab: http://localhost:8888 (token: alerta_cheias_dev)
# TensorBoard: http://localhost:6006
# MLflow: http://localhost:5000
```

### Produção Completa

```bash
# Executar com Nginx, SSL e monitoramento
make docker-run-prod
make docker-run-monitoring

# Acessar serviços
# Aplicação: http://localhost (ou https se SSL configurado)
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

## 🏗️ Arquitetura dos Containers

### API Container (`Dockerfile.api`)

- **Base**: `python:3.9-slim`
- **Multi-stage build** para otimização
- **Usuário não-root** para segurança
- **Health checks** implementados
- **Volumes**: dados, modelos, logs

### Training Container (`Dockerfile.training`)

- **Base**: `python:3.9-slim`
- **Inclui**: Jupyter Lab, TensorBoard, MLflow
- **Ferramentas ML**: TensorFlow, scikit-learn, optuna
- **Ports**: 8888 (Jupyter), 6006 (TensorBoard), 5000 (MLflow)

## 🔧 Profiles do Docker Compose

O `docker-compose.yml` usa profiles para diferentes cenários:

### Default (sem profile)

- `api` - FastAPI application
- `redis` - Cache Redis
- `postgres` - Banco de dados PostgreSQL

### Profile: `training`

- `training` - Ambiente Jupyter + ML tools

### Profile: `production`

- `nginx` - Reverse proxy com SSL

### Profile: `monitoring`

- `prometheus` - Métricas
- `grafana` - Dashboards

## 📊 Volumes Persistentes

```yaml
volumes:
  postgres_data: # Dados do PostgreSQL
  redis_data: # Cache Redis
  model_artifacts: # Modelos treinados
  training_logs: # Logs de treinamento
  api_logs: # Logs da API
  prometheus_data: # Dados do Prometheus
  grafana_data: # Configurações Grafana
```

## 🌐 Portas Utilizadas

| Serviço     | Porta  | Descrição           |
| ----------- | ------ | ------------------- |
| API         | 8000   | FastAPI application |
| PostgreSQL  | 5432   | Banco de dados      |
| Redis       | 6379   | Cache               |
| Jupyter     | 8888   | Jupyter Lab         |
| TensorBoard | 6006   | Visualização ML     |
| MLflow      | 5000   | Tracking ML         |
| Nginx       | 80/443 | Reverse proxy       |
| Prometheus  | 9090   | Métricas            |
| Grafana     | 3000   | Dashboards          |

## 🔒 Configurações de Segurança

### API Container

- Usuário não-root (`appuser`)
- Apenas portas necessárias expostas
- Health checks configurados
- Volumes read-only quando possível

### Nginx

- Headers de segurança (HSTS, CSP, etc.)
- Rate limiting configurado
- SSL/TLS configurado (produção)
- Compression habilitada

### Database

- Credenciais via environment variables
- Conexões limitadas
- Schema isolado (`alerta_cheias`)

## 🚀 Comandos Make Úteis

### Build

```bash
make docker-build              # Build API
make docker-build-training     # Build training
make docker-build-all          # Build tudo
```

### Execução

```bash
make docker-run                # Desenvolvimento
make docker-run-training       # ML environment
make docker-run-prod          # Produção
make docker-run-monitoring    # Monitoramento
make docker-run-all           # Todos serviços
```

### Operações

```bash
make docker-logs              # Logs de todos
make docker-logs-api          # Logs apenas API
make docker-status            # Status containers
make docker-health            # Health check
```

### Acesso

```bash
make docker-exec-api          # Terminal API
make docker-exec-training     # Terminal training
make docker-exec-postgres     # Terminal PostgreSQL
make docker-exec-redis        # Terminal Redis
```

### Manutenção

```bash
make docker-stop              # Parar containers
make docker-restart           # Reiniciar
make docker-clean             # Limpeza básica
make docker-clean-all         # Limpeza completa
```

### Específicos do Projeto

```bash
make docker-train-model       # Treinar modelo
make docker-setup-data        # Configurar dados
make docker-backup-db         # Backup PostgreSQL
make docker-migrate-db        # Executar migrações
```

## 🔧 Variáveis de Ambiente

### API Container

```env
ENVIRONMENT=production
REDIS_URL=redis://redis:6379/0
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/alerta_cheias
GUAIBA_API_URL=https://nivelguaiba.com.br/portoalegre.1day.json
CPTEC_API_URL=https://www.cptec.inpe.br/api/forecast-input?city=Porto%20Alegre%2C%20RS
API_TIMEOUT=10
MAX_RETRIES=3
```

### Training Container

```env
JUPYTER_TOKEN=alerta_cheias_dev
JUPYTER_ALLOW_ROOT=1
PYTHONPATH=/workspace
```

### PostgreSQL

```env
POSTGRES_DB=alerta_cheias
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

## 🔍 Troubleshooting

### Container não inicia

```bash
# Verificar logs
make docker-logs

# Verificar status
make docker-status

# Verificar health
make docker-health
```

### Problemas de Porta

```bash
# Verificar portas em uso
netstat -tlnp | grep :8000

# Parar containers conflitantes
docker stop $(docker ps -q)
```

### Problemas de Volume

```bash
# Listar volumes
make docker-volumes

# Limpar volumes órfãos
docker volume prune
```

### Problemas de Rede

```bash
# Verificar rede
docker network ls

# Recrear rede
docker network rm alerta_cheias_network
make docker-run
```

### Reset Completo

```bash
# ATENÇÃO: Remove todos os dados
make docker-clean-all
```

## 📋 Health Checks

Todos os containers têm health checks configurados:

### API

```bash
curl -f http://localhost:8000/health
```

### Redis

```bash
redis-cli ping
```

### PostgreSQL

```bash
pg_isready -U postgres -d alerta_cheias
```

### Jupyter

```bash
curl -f http://localhost:8888/api
```

## 🔄 CI/CD Integration

### Build Testing

```bash
# Testar build em CI
docker build -f docker/Dockerfile.api -t test .
docker build -f docker/Dockerfile.training -t test-training .
```

### Security Scanning

```bash
# Scan de vulnerabilidades (se disponível)
docker scan alerta-cheias-api:latest
docker scan alerta-cheias-training:latest
```

## 📚 Recursos Adicionais

- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Multi-stage Builds](https://docs.docker.com/develop/dev-best-practices/)
- [Docker Security](https://docs.docker.com/engine/security/)

## 🆘 Suporte

Para problemas específicos do Docker:

1. Verificar logs: `make docker-logs`
2. Verificar health: `make docker-health`
3. Consultar documentação do projeto
4. Abrir issue no repositório
