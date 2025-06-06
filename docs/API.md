# ⚡ API FastAPI

## Visão Geral

API REST robusta construída com **FastAPI** para previsões meteorológicas e alertas de cheias em tempo real, seguindo **Clean Architecture**.

## 🏗️ Arquitetura da API

### Estrutura por Features

```
app/
├── core/                    # Configurações compartilhadas
│   ├── config.py           # Settings com Pydantic
│   ├── exceptions.py       # Exceções customizadas
│   ├── dependencies.py     # Injeção de dependências
│   └── logging.py          # Logs estruturados
├── features/
│   ├── forecast/           # Feature de Previsão
│   │   ├── domain/         # Regras de negócio
│   │   ├── application/    # Use cases
│   │   ├── infra/          # Modelo ML, APIs externas
│   │   └── presentation/   # Endpoints, schemas
│   └── alerts/             # Feature de Alertas
│       ├── domain/         # Matriz de classificação
│       ├── application/    # Use cases de alerta
│       ├── infra/          # APIs externas (Open-Meteo, Guaíba)
│       └── presentation/   # Endpoints de alerta
└── main.py                 # Aplicação FastAPI
```

## 🌐 Endpoints Disponíveis

### 🔮 Forecast - Previsão Meteorológica

#### `POST /forecast/predict`

Gera previsão meteorológica usando modelo LSTM híbrido.

**Request:**

```json
{
  "hours_ahead": 24,
  "include_atmospheric": true,
  "location": {
    "latitude": -30.0331,
    "longitude": -51.23
  }
}
```

**Response:**

```json
{
  "forecast": {
    "precipitation_24h": 15.2,
    "temperature": 28.5,
    "pressure": 1013.2,
    "wind_speed": 12.8,
    "humidity": 68
  },
  "atmospheric_analysis": {
    "frontal_activity": "cold_front_approaching",
    "synoptic_pattern": "cyclonic_vortex_detected",
    "instability_index": 0.75
  },
  "confidence": 0.87,
  "model_version": "hybrid_v2.1",
  "timestamp": "2025-01-06T15:30:00-03:00"
}
```

#### `GET /forecast/metrics`

Métricas de performance do modelo ML.

**Response:**

```json
{
  "model_metrics": {
    "accuracy": 0.847,
    "mae_precipitation": 1.42,
    "rmse_precipitation": 2.38,
    "frontal_detection_accuracy": 0.912
  },
  "ensemble_weights": {
    "historical_forecast": 0.7,
    "historical_weather": 0.3
  },
  "last_training": "2025-01-01T00:00:00Z"
}
```

### 🚨 Alerts - Sistema de Alertas

#### `GET /alerts/current`

Alerta atual baseado em condições meteorológicas e nível do rio.

**Response:**

```json
{
  "alert": {
    "level": "Moderado",
    "action": "Atenção",
    "risk_score": 0.65,
    "expires_at": "2025-01-06T18:00:00-03:00"
  },
  "conditions": {
    "river_level": 2.95,
    "rain_prediction_24h": 25.3,
    "current_precipitation": 3.2
  },
  "reasoning": "Nível do rio em 2.95m com previsão de 25mm de chuva nas próximas 24h",
  "recommendations": [
    "Monitorar níveis do rio de perto",
    "Preparar planos de contingência"
  ]
}
```

#### `GET /alerts/conditions`

Condições meteorológicas atuais em tempo real.

**Response:**

```json
{
  "weather": {
    "temperature": 28.5,
    "humidity": 68,
    "precipitation": 0.0,
    "pressure": 1013.2,
    "wind_speed": 12.8,
    "weather_description": "Parcialmente nublado"
  },
  "river": {
    "level": 2.95,
    "trend": "stable",
    "last_update": "2025-01-06T15:00:00-03:00"
  },
  "atmospheric": {
    "frontal_activity": "stable",
    "pressure_levels": {
      "850hPa": { "temperature": 15.2, "wind_speed": 45.1 },
      "500hPa": { "geopotential": 5820, "wind_speed": 62.3 }
    }
  }
}
```

#### `POST /alerts/evaluate`

Avalia condições específicas fornecidas pelo usuário.

**Request:**

```json
{
  "river_level": 3.2,
  "rain_prediction": 45.0,
  "current_weather": {
    "temperature": 22.1,
    "pressure": 1008.5
  }
}
```

## 📊 Schemas e DTOs

### Forecast Schemas

```python
from pydantic import BaseModel
from typing import Optional

class ForecastRequest(BaseModel):
    hours_ahead: int = 24
    include_atmospheric: bool = True
    location: Optional[dict] = None

class ForecastResponse(BaseModel):
    forecast: dict
    atmospheric_analysis: Optional[dict]
    confidence: float
    model_version: str
    timestamp: str

class ModelMetricsResponse(BaseModel):
    model_metrics: dict
    ensemble_weights: dict
    last_training: str
```

### Alert Schemas

```python
class AlertResponse(BaseModel):
    alert: dict
    conditions: dict
    reasoning: str
    recommendations: list[str]

class CurrentConditionsResponse(BaseModel):
    weather: dict
    river: dict
    atmospheric: dict

class AlertEvaluationRequest(BaseModel):
    river_level: float
    rain_prediction: float
    current_weather: Optional[dict] = None
```

## 🔧 Configuração e Middleware

### Environment Configuration

```python
# .env
API_PORT=8000
API_HOST=0.0.0.0
LOG_LEVEL=INFO
REDIS_URL=redis://localhost:6379
OPENMETEO_CACHE_TTL=3600
GUAIBA_CACHE_TTL=1800
MAX_REQUEST_SIZE=10MB
```

### Middleware Stack

- **CORS**: Configurado para desenvolvimento e produção
- **Request Logging**: Logs estruturados de todas as requisições
- **Rate Limiting**: 1000 req/minuto por IP
- **Error Handling**: Tratamento global de exceções
- **Request ID**: Tracking de requisições para debug

### Health Checks

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0",
        "services": {
            "model": "loaded",
            "openmeteo_api": "available",
            "guaiba_api": "available",
            "redis": "connected"
        }
    }
```

## 🚀 Performance

### Métricas Esperadas

- **Latência média**: < 200ms
- **Disponibilidade**: > 99.5%
- **Throughput**: 1000+ req/min
- **Timeout**: 30s para operações ML

### Otimizações

- **Async/Await**: Operações I/O não-bloqueantes
- **Connection Pooling**: Pool de conexões para APIs externas
- **Cache Inteligente**: Redis com TTL otimizado
- **Background Tasks**: Processamento assíncrono

## 📝 Logging e Monitoramento

### Logs Estruturados

```json
{
  "timestamp": "2025-01-06T15:30:00Z",
  "level": "INFO",
  "request_id": "req_123456",
  "endpoint": "/alerts/current",
  "method": "GET",
  "status_code": 200,
  "response_time_ms": 145,
  "user_agent": "curl/7.68.0"
}
```

### Métricas de Negócio

- Total de previsões geradas
- Accuracy em tempo real
- Alertas emitidos por nível
- Taxa de erro das APIs externas

## 🐳 Docker e Deploy

### Dockerfile Otimizado

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements/ requirements/
RUN pip install -r requirements/production.txt

COPY app/ app/
COPY data/modelos_treinados/ data/modelos_treinados/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

## 🔧 Comandos Úteis

```bash
# Desenvolvimento local
make dev                    # FastAPI com reload
make test-api              # Testes da API

# Docker
make docker-api            # Build da API
make docker-run            # Executar com compose

# Testes
make test-endpoints        # Testes de endpoints
make test-integration      # Testes de integração

# Monitoramento
make logs-api              # Visualizar logs
make health-check          # Status da aplicação
```

## 📚 Documentação Interativa

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`
