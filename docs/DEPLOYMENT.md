# 🐳 Deployment e Docker

## Visão Geral

Sistema containerizado com **Docker** para desenvolvimento e produção, incluindo orquestração completa com **Docker Compose**.

## 🏗️ Estrutura Docker

```
docker/
├── Dockerfile.api            # Container da API FastAPI
├── Dockerfile.training       # Container de treinamento ML
├── Dockerfile.jupyter        # Container Jupyter para análise
├── docker-compose.yml        # Orquestração desenvolvimento
├── docker-compose.prod.yml   # Orquestração produção
└── nginx/
    └── nginx.conf            # Configuração proxy reverso
```

## 🚀 API Container (Dockerfile.api)

### Multi-stage Build Otimizado

```dockerfile
# Build stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements/production.txt requirements/
RUN pip install --no-cache-dir --user -r requirements/production.txt

# Production stage
FROM python:3.9-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY app/ app/
COPY data/modelos_treinados/ data/modelos_treinados/

# Set permissions
RUN chown -R appuser:appuser /app
USER appuser

# Environment
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Build da API

```bash
# Build local
docker build -f docker/Dockerfile.api -t alerta-cheias-api:latest .

# Build com cache otimizado
docker build -f docker/Dockerfile.api \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  -t alerta-cheias-api:latest .

# Build para produção
docker build -f docker/Dockerfile.api \
  --target production \
  -t alerta-cheias-api:prod .
```

## 🧠 Training Container (Dockerfile.training)

### Container Específico para ML

```dockerfile
FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /app

# Install system dependencies for data processing
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python ML dependencies
COPY requirements/training.txt requirements/
RUN pip install --no-cache-dir -r requirements/training.txt

# Install Jupyter and extensions
RUN pip install jupyterlab jupyter-tensorboard

# Copy training scripts and notebooks
COPY scripts/ scripts/
COPY notebooks/ notebooks/
COPY data/ data/

# Jupyter configuration
RUN jupyter lab --generate-config
COPY docker/jupyter_lab_config.py /root/.jupyter/

# Environment variables
ENV JUPYTER_ENABLE_LAB=yes
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8888 6006

# Start command (can be overridden)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
```

### Comandos de Treinamento

```bash
# Build training container
docker build -f docker/Dockerfile.training -t alerta-cheias-training:latest .

# Executar treinamento
docker run --rm \
  --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  alerta-cheias-training:latest \
  python scripts/train_hybrid_model.py

# Jupyter Lab para análise
docker run -p 8888:8888 -p 6006:6006 \
  -v $(pwd):/app \
  alerta-cheias-training:latest
```

## 🔄 Docker Compose

### Desenvolvimento (docker-compose.yml)

```yaml
version: "3.8"

services:
  # API FastAPI
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=DEBUG
      - API_RELOAD=true
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./app:/app/app
      - ./data:/app/data
    depends_on:
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Jupyter Lab (desenvolvimento)
  jupyter:
    build:
      context: .
      dockerfile: docker/Dockerfile.training
    ports:
      - "8888:8888"
      - "6006:6006" # TensorBoard
    volumes:
      - ./notebooks:/app/notebooks
      - ./data:/app/data
      - ./scripts:/app/scripts
    environment:
      - JUPYTER_TOKEN=alerta-cheias-dev
    restart: unless-stopped

  # PostgreSQL (opcional)
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=alerta_cheias
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:

networks:
  default:
    name: alerta-cheias-network
```

### Produção (docker-compose.prod.yml)

```yaml
version: "3.8"

services:
  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./docker/nginx/ssl:/etc/ssl/certs
    depends_on:
      - api
    restart: always

  # API FastAPI (produção)
  api:
    image: alerta-cheias-api:prod
    environment:
      - LOG_LEVEL=INFO
      - API_RELOAD=false
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/alerta_cheias
    depends_on:
      - redis
      - postgres
    restart: always
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: "1"
          memory: 1G
        reservations:
          cpus: "0.5"
          memory: 512M

  # Redis Cache (produção)
  redis:
    image: redis:7-alpine
    volumes:
      - redis_prod_data:/data
    restart: always
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}

  # PostgreSQL (produção)
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=alerta_cheias
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_prod_data:/var/lib/postgresql/data
    restart: always

  # Monitoring with Prometheus
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./docker/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: always

  # Grafana Dashboard
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
    restart: always

volumes:
  redis_prod_data:
  postgres_prod_data:
  grafana_data:
```

## 🔧 Makefile para Automação

```makefile
# Makefile para automação Docker

.PHONY: build up down logs test clean

# Environment
ENV ?= dev
COMPOSE_FILE = docker-compose.yml
ifeq ($(ENV),prod)
    COMPOSE_FILE = docker-compose.prod.yml
endif

# Build images
build:
	docker-compose -f $(COMPOSE_FILE) build

# Build sem cache
build-no-cache:
	docker-compose -f $(COMPOSE_FILE) build --no-cache

# Start services
up:
	docker-compose -f $(COMPOSE_FILE) up -d

# Start com logs
up-logs:
	docker-compose -f $(COMPOSE_FILE) up

# Stop services
down:
	docker-compose -f $(COMPOSE_FILE) down

# Stop com volumes
down-volumes:
	docker-compose -f $(COMPOSE_FILE) down -v

# View logs
logs:
	docker-compose -f $(COMPOSE_FILE) logs -f

# API logs
logs-api:
	docker-compose -f $(COMPOSE_FILE) logs -f api

# Restart API
restart-api:
	docker-compose -f $(COMPOSE_FILE) restart api

# Execute comando na API
exec-api:
	docker-compose -f $(COMPOSE_FILE) exec api bash

# Health check
health:
	curl -f http://localhost:8000/health

# Test API
test-api:
	docker-compose -f $(COMPOSE_FILE) exec api pytest tests/

# Clean containers e images
clean:
	docker-compose -f $(COMPOSE_FILE) down -v --rmi all
	docker system prune -f

# Training commands
train-model:
	docker-compose -f $(COMPOSE_FILE) run --rm jupyter \
		python scripts/train_hybrid_model.py

# Collect data
collect-data:
	docker-compose -f $(COMPOSE_FILE) run --rm api \
		python scripts/collect_openmeteo_hybrid_data.py

# Backup database
backup-db:
	docker-compose -f $(COMPOSE_FILE) exec postgres \
		pg_dump -U postgres alerta_cheias > backup_$(shell date +%Y%m%d_%H%M%S).sql

# Production deployment
deploy-prod:
	ENV=prod $(MAKE) build
	ENV=prod $(MAKE) up

# Security scan
security-scan:
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		aquasec/trivy image alerta-cheias-api:latest
```

## 🌐 Nginx Configuration

### Reverse Proxy (nginx.conf)

```nginx
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        server api:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    server {
        listen 80;
        server_name localhost;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        # API proxy
        location /api/ {
            limit_req zone=api burst=20 nodelay;

            proxy_pass http://api_backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }

        # Health check
        location /health {
            proxy_pass http://api_backend/health;
            access_log off;
        }

        # Static files (if any)
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

## 📊 Monitoring e Logs

### Prometheus Configuration

```yaml
# docker/prometheus/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "api"
    static_configs:
      - targets: ["api:8000"]
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: "redis"
    static_configs:
      - targets: ["redis:6379"]

  - job_name: "postgres"
    static_configs:
      - targets: ["postgres:5432"]
```

### Docker Logs Configuration

```yaml
# Em docker-compose.yml
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## 🔐 Security

### Production Security Checklist

```bash
# 1. Use secrets para senhas
echo "POSTGRES_PASSWORD=$(openssl rand -base64 32)" > .env.prod
echo "REDIS_PASSWORD=$(openssl rand -base64 32)" >> .env.prod

# 2. Scan de vulnerabilidades
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy image alerta-cheias-api:latest

# 3. User não-root nos containers
USER appuser  # No Dockerfile

# 4. Read-only filesystem (quando possível)
docker run --read-only --tmpfs /tmp alerta-cheias-api:latest

# 5. Resource limits
deploy:
  resources:
    limits:
      cpus: '1'
      memory: 1G
```

## 🚀 Comandos de Deploy

### Desenvolvimento

```bash
# Setup inicial
make build
make up

# Verificar status
make health
make logs

# Desenvolvimento ativo
make logs-api  # Acompanhar logs da API
```

### Produção

```bash
# Deploy produção
make deploy-prod

# Monitoramento
docker-compose -f docker-compose.prod.yml ps
docker-compose -f docker-compose.prod.yml logs -f

# Backup
make backup-db

# Update da aplicação
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d --no-deps api
```

### CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build and push images
        run: |
          docker build -f docker/Dockerfile.api -t alerta-cheias-api:${{ github.sha }} .
          docker push alerta-cheias-api:${{ github.sha }}

      - name: Deploy to production
        run: |
          ENV=prod make deploy-prod
```

## 📋 Troubleshooting

### Problemas Comuns

```bash
# Container não inicia
docker-compose logs api

# Problemas de rede
docker network ls
docker network inspect alerta-cheias-network

# Problemas de volume
docker volume ls
docker volume inspect alerta-cheias_redis_data

# Reset completo
make down-volumes
make clean
make build
make up
```

### Performance Tuning

```bash
# CPU e Memory usage
docker stats

# Otimização da imagem
docker images
docker system df

# Análise de layers
docker history alerta-cheias-api:latest
```

O sistema Docker garante **deploy consistente**, **escalabilidade** e **fácil manutenção** do sistema de alertas de cheias.
