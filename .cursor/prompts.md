# Prompts Específicos - Projeto Alertas de Cheias

# Clean Architecture organizada por Features

## Contexto do Projeto

Sistema de previsão meteorológica para alertas de cheias do Rio Guaíba em Porto Alegre, usando TensorFlow e FastAPI com Clean Architecture organizada por features (forecast e alerts).

## APIs Externas

- **Nível do Guaíba**: https://nivelguaiba.com.br/portoalegre.1day.json
- **Meteorologia CPTEC**: https://www.cptec.inpe.br/api/forecast-input?city=Porto%20Alegre%2C%20RS

## Prompts Úteis

### Para Clean Architecture

```
Você está implementando Clean Architecture organizada por features. Estruture o código em:
- Domain: Entidades e regras de negócio puras
- Application: Use cases que coordenam operações
- Infrastructure: Implementações concretas (APIs, modelos ML)
- Presentation: Controllers FastAPI e DTOs

Features: forecast (previsão meteorológica) e alerts (sistema de alertas)
Evite dependências entre features. Use app/core/ para lógica compartilhada.
```

### Para Feature de Forecast

```
Você está desenvolvendo a feature de previsão meteorológica com:
- Domain: WeatherData, ForecastResult entities
- Application: GenerateForecastUseCase
- Infrastructure: LSTMModel, ModelLoader
- Presentation: ForecastController com endpoints FastAPI

Use dados históricos 2000-2025 para treinar LSTM que prevê chuva em 24h.
Acurácia mínima: 75%. Métricas: MAE, RMSE.
```

### Para Feature de Alerts

```
Você está desenvolvendo a feature de alertas com:
- Domain: AlertLevel, AlertRule entities
- Application: GenerateAlertUseCase
- Infrastructure: GuaibaApiClient, CptecApiClient
- Presentation: AlertController

Matriz de alertas:
- Rio < 2,80m + Chuva < 20mm = Baixo/Monitoramento
- Rio 2,80-3,15m + Chuva 20-50mm = Moderado/Atenção
- Rio 3,15-3,60m + Chuva > 50mm = Alto/Alerta
- Rio > 3,60m + Qualquer chuva = Crítico/Emergência
```

### Para APIs Externas

```
Implemente clientes para APIs externas usando httpx:

1. GuaibaApiClient para https://nivelguaiba.com.br/portoalegre.1day.json
   - Resposta: {"2025-06-03 19:15": 1.93, "2025-06-03 19:00": 1.94, ...}
   - Extrair valor mais recente (último timestamp)

2. CptecApiClient para https://www.cptec.inpe.br/api/forecast-input?city=Porto%20Alegre%2C%20RS
   - Resposta: {"probabilidade_chuva": 37, "precipitacao_acumulada": "0", ...}

Use timeout 10s, retry logic, async/await, tratamento de erros.
```

### Para Domain Layer

```
Crie entidades de domínio puras (sem dependências externas):

WeatherData:
- temperature, humidity, pressure, precipitation
- timestamp

ForecastResult:
- predicted_rainfall_24h
- confidence_score
- generated_at

AlertLevel:
- nivel (Baixo/Moderado/Alto/Crítico)
- acao (Monitoramento/Atenção/Alerta/Emergência)
- river_level, rain_prediction

Use dataclasses ou Pydantic BaseModel.
```

### Para Application Layer

```
Implemente use cases que coordenam operações:

GenerateForecastUseCase:
- Input: current weather conditions
- Output: ForecastResult
- Coordena: model loading, preprocessing, prediction

GenerateAlertUseCase:
- Input: location (Porto Alegre)
- Output: AlertLevel
- Coordena: fetch river level, get forecast, apply rules

Use cases não devem conter lógica de negócio, apenas coordenação.
```

### Para Infrastructure Layer

```
Implemente adaptadores para recursos externos:

ModelLoader:
- Carrega modelo LSTM do arquivo
- Preprocessing pipeline
- Versionamento de modelos

ExternalApiClients:
- GuaibaApiClient, CptecApiClient
- Connection pooling, circuit breaker
- Cache com TTL apropriado

Use interfaces/abstractions no domain, implementações concretas aqui.
```

### Para Presentation Layer

```
Crie controllers FastAPI para cada feature:

ForecastController (/forecast):
- POST /forecast/predict
- GET /forecast/health

AlertController (/alerts):
- GET /alerts/current
- GET /alerts/status

Use Pydantic para DTOs, dependency injection, tratamento de exceções HTTP.
Documentação OpenAPI completa.
```

### Para Testes por Camada

```
Estruture testes por camada e feature:

tests/unit/forecast/:
- test_domain.py (entities, business rules)
- test_application.py (use cases with mocks)
- test_infrastructure.py (adapters)

tests/integration/:
- test_forecast_api.py (endpoints E2E)
- test_external_apis.py (real API calls)

Use pytest, mocks para dependencies, fixtures para setup.
```

### Para Configurações

````
Centralize configurações em app/core/config.py:

```python
class Settings(BaseSettings):
    # APIs Externas
    guaiba_api_url: str = "https://nivelguaiba.com.br/portoalegre.1day.json"
    cptec_api_url: str = "https://www.cptec.inpe.br/api/forecast-input"

    # Modelo ML
    model_path: str = "data/modelos_treinados/lstm_model"

    # Cache
    cache_ttl_seconds: int = 300

    class Config:
        env_file = ".env"
````

Use diferentes configs para dev/test/prod.

````

## Comandos Úteis

### Setup do Ambiente
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
````

### Desenvolvimento por Feature

```bash
# Testar feature específica
pytest tests/unit/forecast/ -v
pytest tests/unit/alerts/ -v

# Executar API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Treinar modelo LSTM
python -c "from app.features.forecast.infra.train_model import train_lstm; train_lstm()"
```

### Testes

```bash
# Todos os testes
pytest tests/ -v --cov=app

# Por camada
pytest tests/unit/ -v         # Unitários
pytest tests/integration/ -v  # Integração

# Por feature
pytest tests/ -k "forecast" -v
pytest tests/ -k "alerts" -v
```

### Docker

```bash
docker build -t alerta-cheias .
docker run -p 8000:8000 --env-file .env alerta-cheias

# Com docker-compose
docker-compose up --build
```

### APIs de Teste

```bash
# Testar previsão
curl -X POST "http://localhost:8000/forecast/predict" \
  -H "Content-Type: application/json" \
  -d '{"temperature": 25.5, "humidity": 70, "pressure": 1013.25}'

# Testar alertas
curl "http://localhost:8000/alerts/current"

# Health check
curl "http://localhost:8000/health"
```

## Estrutura de Arquivos Detalhada

```
projeto_alerta_cheias/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI app + routes registration
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                # Settings centralizadas
│   │   ├── exceptions.py            # Exceções customizadas
│   │   ├── dependencies.py          # DI container
│   │   └── logging.py               # Configuração de logs
│   └── features/
│       ├── __init__.py
│       ├── forecast/
│       │   ├── __init__.py
│       │   ├── domain/
│       │   │   ├── __init__.py
│       │   │   ├── entities.py      # WeatherData, ForecastResult
│       │   │   ├── services.py      # ForecastService
│       │   │   └── interfaces.py    # Repository interfaces
│       │   ├── application/
│       │   │   ├── __init__.py
│       │   │   └── usecases.py      # GenerateForecastUseCase
│       │   ├── infrastructure/
│       │   │   ├── __init__.py
│       │   │   ├── model_loader.py  # ModelLoader
│       │   │   ├── forecast_model.py # LSTM implementation
│       │   │   └── repositories.py  # Repository implementations
│       │   └── presentation/
│       │       ├── __init__.py
│       │       ├── routes.py        # FastAPI routes
│       │       └── schemas.py       # Request/Response DTOs
│       └── alerts/
│           ├── __init__.py
│           ├── domain/
│           │   ├── __init__.py
│           │   ├── entities.py      # AlertLevel, AlertRule
│           │   ├── alert_rules.py   # Business rules
│           │   └── interfaces.py    # API client interfaces
│           ├── application/
│           │   ├── __init__.py
│           │   └── usecases.py      # GenerateAlertUseCase
│           ├── infrastructure/
│           │   ├── __init__.py
│           │   ├── external_api.py  # GuaibaClient, CptecClient
│           │   └── repositories.py  # Alert data persistence
│           └── presentation/
│               ├── __init__.py
│               ├── routes.py        # Alert endpoints
│               └── schemas.py       # Alert DTOs
├── data/
│   ├── dados_historicos/            # CSV files 2000-2025
│   │   ├── weather_2000.csv
│   │   ├── weather_2001.csv
│   │   └── ...
│   └── modelos_treinados/           # Saved ML models
│       ├── lstm_model/
│       │   ├── saved_model.pb
│       │   └── variables/
│       └── preprocessor.pkl
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures
│   ├── unit/
│   │   ├── forecast/
│   │   │   ├── test_domain.py
│   │   │   ├── test_application.py
│   │   │   └── test_infrastructure.py
│   │   └── alerts/
│   │       ├── test_domain.py
│   │       ├── test_application.py
│   │       └── test_infrastructure.py
│   └── integration/
│       ├── test_forecast_api.py
│       ├── test_alerts_api.py
│       └── test_external_apis.py
├── .env.example                     # Environment variables template
├── .gitignore
├── requirements.txt
├── requirements-dev.txt             # Development dependencies
├── Dockerfile
├── docker-compose.yml
└── README.md
```
