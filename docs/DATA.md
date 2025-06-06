# 📊 Dados Meteorológicos

## Visão Geral

O projeto utiliza uma **estratégia híbrida** de dados meteorológicos que combina múltiplas fontes para maximizar a precisão das previsões.

## 🌟 Estratégia Híbrida Open-Meteo

### Fontes Primárias de Dados

| API                     | Período   | Resolução | Variáveis     | Uso Principal        |
| ----------------------- | --------- | --------- | ------------- | -------------------- |
| **Historical Forecast** | 2022-2025 | 2-25km    | 149 variáveis | **Modelo principal** |
| **Historical Weather**  | 2000-2024 | 25km      | 25 variáveis  | Extensão temporal    |
| **INMET Porto Alegre**  | 2000-2025 | Pontual   | ~10 variáveis | Validação local      |

### Historical Forecast API (Fonte Principal) ⭐

**Endpoint:** `https://historical-forecast-api.open-meteo.com/v1/forecast`

**Características:**

- **Período**: 2022-2025 (3+ anos)
- **Modelos**: ECMWF IFS, DWD ICON, Météo-France AROME
- **Resolução**: 2-25km (alta resolução)
- **Atualização**: Diária com delay de 2 dias

**Dados de Níveis de Pressão:**

```python
PRESSURE_LEVELS = {
    '1000hPa': '110m above sea level',    # Camada de mistura
    '850hPa': '1500m above sea level',    # ⭐ FRENTES FRIAS
    '700hPa': '3000m above sea level',    # Nível médio
    '500hPa': '5600m above sea level',    # ⭐ VÓRTICES
    '300hPa': '9200m above sea level',    # Corrente de jato
    '200hPa': '11800m above sea level'    # Alta troposfera
}

VARIABLES_PER_LEVEL = [
    'temperature',           # Análise térmica
    'relative_humidity',     # Umidade em altitude
    'cloud_cover',          # Cobertura de nuvens
    'wind_speed',           # Vento em altitude
    'wind_direction',       # Direção do vento
    'geopotential_height'   # Altura real dos níveis
]

# Total: 6 níveis × 6 variáveis = 36 variáveis de pressão
# + 35 variáveis de superfície = 71 variáveis por hora
```

### Historical Weather API (Extensão Temporal)

**Endpoint:** `https://archive-api.open-meteo.com/v1/archive`

**Características:**

- **Período**: 2000-2024 (24+ anos)
- **Modelo**: ERA5 Reanalysis (ECMWF)
- **Resolução**: 25km global
- **Variáveis**: 25 de superfície

## 📍 Coordenadas Porto Alegre

```python
PORTO_ALEGRE_COORDS = {
    'latitude': -30.0331,
    'longitude': -51.2300,
    'timezone': 'America/Sao_Paulo',
    'elevation': 46.0  # metros
}
```

## 🌦️ Variáveis Atmosféricas Coletadas

### Dados de Superfície (35 variáveis)

```python
SURFACE_VARIABLES = [
    # Temperatura e umidade
    'temperature_2m',
    'relative_humidity_2m',
    'dewpoint_2m',
    'apparent_temperature',

    # Precipitação
    'precipitation',
    'snowfall',
    'rain',
    'showers',

    # Pressão e vento
    'pressure_msl',
    'surface_pressure',
    'wind_speed_10m',
    'wind_direction_10m',
    'wind_gusts_10m',

    # Radiação e nuvens
    'shortwave_radiation',
    'direct_radiation',
    'diffuse_radiation',
    'cloud_cover',
    'cloud_cover_low',
    'cloud_cover_mid',
    'cloud_cover_high',

    # Índices atmosféricos
    'et0_fao_evapotranspiration',
    'vapour_pressure_deficit',
    'cape',  # Energia potencial convectiva
    'lifted_index',  # Índice de instabilidade

    # Solo e extras
    'soil_temperature_0cm',
    'soil_moisture_0_1cm',
    'is_day',
    'sunshine_duration'
]
```

### Dados de Níveis de Pressão (114 variáveis)

```python
# 6 níveis × 6 variáveis = 36 variáveis por hora
# Exemplo para 850hPa (detecção de frentes frias):
PRESSURE_850HPA = {
    'temperature_850hPa': 'Temperatura em 850hPa (°C)',
    'relative_humidity_850hPa': 'Umidade relativa em 850hPa (%)',
    'wind_speed_850hPa': 'Velocidade do vento em 850hPa (km/h)',
    'wind_direction_850hPa': 'Direção do vento em 850hPa (°)',
    'geopotential_height_850hPa': 'Altura geopotencial 850hPa (m)',
    'cloud_cover_850hPa': 'Cobertura de nuvens 850hPa (%)'
}

# Exemplo para 500hPa (análise de vórtices):
PRESSURE_500HPA = {
    'temperature_500hPa': 'Temperatura em 500hPa (°C)',
    'geopotential_height_500hPa': 'Altura geopotencial 500hPa (m)',
    'wind_speed_500hPa': 'Velocidade do vento em 500hPa (km/h)',
    # ... outras variáveis
}
```

## 🔄 Pipeline de Coleta

### Scripts de Coleta Implementados

```bash
# 1. Análise comparativa das APIs
python scripts/analyze_openmeteo_apis.py

# 2. Coleta híbrida (principal)
python scripts/collect_openmeteo_hybrid_data.py

# 3. Coleta específica forecast
python scripts/collect_openmeteo_forecast.py

# 4. Validação de qualidade
python scripts/validate_collected_data.py
```

### Exemplo de Coleta

```python
import asyncio
import aiohttp

async def collect_openmeteo_data():
    """Coleta dados Open-Meteo Historical Forecast"""

    params = {
        'latitude': -30.0331,
        'longitude': -51.2300,
        'start_date': '2022-01-01',
        'end_date': '2025-06-30',
        'timezone': 'America/Sao_Paulo',

        # Variáveis de superfície
        'hourly': [
            'temperature_2m', 'precipitation', 'pressure_msl',
            'relative_humidity_2m', 'wind_speed_10m', 'cape'
        ],

        # Níveis de pressão críticos
        'pressure_level': [1000, 850, 700, 500, 300],
        'pressure_level_variables': [
            'temperature', 'relative_humidity', 'wind_speed',
            'wind_direction', 'geopotential_height'
        ]
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(
            'https://historical-forecast-api.open-meteo.com/v1/forecast',
            params=params
        ) as response:
            data = await response.json()
            return process_openmeteo_data(data)
```

## 🧮 Feature Engineering Atmosférica

### Features Sinóticas Derivadas

```python
def create_atmospheric_features(data):
    """Cria features atmosféricas derivadas"""

    features = {}

    # Gradiente térmico vertical (instabilidade)
    features['thermal_gradient_850_500'] = (
        data['temperature_850hPa'] - data['temperature_500hPa']
    )

    # Advecção de temperatura (frentes frias)
    features['temp_advection_850'] = calculate_temperature_advection(
        data['temperature_850hPa'],
        data['wind_speed_850hPa'],
        data['wind_direction_850hPa']
    )

    # Vorticidade (vórtices ciclônicos)
    features['vorticity_500'] = calculate_vorticity(
        data['wind_speed_500hPa'],
        data['wind_direction_500hPa']
    )

    # Wind shear vertical
    features['wind_shear_850_500'] = (
        data['wind_speed_500hPa'] - data['wind_speed_850hPa']
    )

    return features
```

### Agregações Temporais

```python
TEMPORAL_AGGREGATIONS = {
    '3h': ['mean', 'max', 'min', 'std'],
    '6h': ['mean', 'max', 'min', 'trend'],
    '12h': ['mean', 'max', 'min', 'trend'],
    '24h': ['mean', 'max', 'min', 'trend', 'range']
}

def create_temporal_features(df):
    """Cria features com agregações temporais"""

    features = {}

    for window in ['3h', '6h', '12h', '24h']:
        for var in ['temperature_2m', 'pressure_msl', 'precipitation']:
            for agg in TEMPORAL_AGGREGATIONS[window]:
                col_name = f"{var}_{window}_{agg}"
                features[col_name] = getattr(
                    df[var].rolling(window), agg
                )()

    return features
```

## 📊 Dados INMET (Validação Local)

### Estações Meteorológicas

1. **A801 - PORTO ALEGRE** (2000-2021)

   - Localização: -30,05°, -51,17°
   - Altitude: 46,97m

2. **A801 - PORTO ALEGRE JARDIM BOTANICO** (2022-2025)

   - Localização: -30,05°, -51,17°
   - Altitude: 41,18m

3. **B807 - PORTO ALEGRE BELEM NOVO** (2022-2025)
   - Localização: Belém Novo

### Variáveis INMET Disponíveis

```python
INMET_VARIABLES = {
    'precipitation': 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)',
    'pressure': 'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)',
    'temperature': 'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)',
    'dew_point': 'TEMPERATURA DO PONTO DE ORVALHO (°C)',
    'humidity': 'UMIDADE RELATIVA DO AR, HORARIA (%)',
    'wind_speed': 'VENTO, VELOCIDADE HORARIA (m/s)',
    'wind_direction': 'VENTO, DIREÇÃO HORARIA (gr) (° (gr))',
    'radiation': 'RADIACAO GLOBAL (Kj/m²)'
}
```

## 📈 Vantagens da Estratégia Híbrida

### Open-Meteo vs INMET Tradicional

| Aspecto                   | Open-Meteo        | INMET             |
| ------------------------- | ----------------- | ----------------- |
| **Variáveis**             | 149 atmosféricas  | ~10 básicas       |
| **Níveis de pressão**     | ✅ 500hPa, 850hPa | ❌ Não disponível |
| **Resolução espacial**    | 2-25km            | Pontual           |
| **Consistência temporal** | ⭐⭐⭐⭐⭐        | ⭐⭐⭐⭐          |
| **Custo**                 | Gratuito          | Gratuito          |
| **API**                   | REST moderna      | CSV manual        |

### Melhoria Esperada

- **+10-15% accuracy** com dados atmosféricos
- **Detecção de frentes frias** via 850hPa
- **Análise de vórtices** via 500hPa
- **Consistency** entre histórico e tempo real

## 🔧 Comandos de Dados

```bash
# Coleta de dados híbrida
make collect-hybrid-data

# Análise de qualidade
make validate-data-quality

# Feature engineering atmosférica
make atmospheric-features

# Comparação INMET vs Open-Meteo
make compare-data-sources

# Preprocessamento completo
make preprocess-atmospheric-data
```

## 📁 Estrutura de Dados

```
data/
├── raw/
│   ├── openmeteo_historical_forecast_2022_2025.json    # Dados principais
│   ├── openmeteo_historical_weather_2000_2024.json    # Extensão temporal
│   └── dados_historicos/                              # INMET (validação)
├── processed/
│   ├── atmospheric_features_149vars.csv              # Features atmosféricas
│   ├── surface_features_25vars.csv                   # Features de superfície
│   └── inmet_validation_data.csv                     # Dados INMET processados
└── analysis/
    ├── openmeteo_apis_analysis.json                  # Análise comparativa
    └── data_quality_report.json                      # Relatório de qualidade
```
