"""
Cliente para coleta de dados históricos da Open-Meteo API
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OpenMeteoConfig:
    """Configuração para coleta de dados da Open-Meteo"""

    # Coordenadas de Porto Alegre
    latitude: float = -30.0346
    longitude: float = -51.2177
    timezone: str = "America/Sao_Paulo"

    # Variáveis meteorológicas essenciais para previsão de enchentes
    hourly_variables: List[str] = None
    daily_variables: List[str] = None

    def __post_init__(self):
        if self.hourly_variables is None:
            self.hourly_variables = [
                # Variáveis básicas de superfície
                "temperature_2m",
                "relative_humidity_2m",
                "dew_point_2m",
                "precipitation",
                "rain",
                "snowfall",
                "pressure_msl",
                "surface_pressure",
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
                # Dados atmosféricos para detectar sistemas sinóticos
                "vapour_pressure_deficit",
                "et0_fao_evapotranspiration",
                "shortwave_radiation",
                "weather_code",
                # Dados de solo para análise de saturação
                "soil_temperature_0_to_7cm",
                "soil_temperature_7_to_28cm",
                "soil_moisture_0_to_7cm",
                "soil_moisture_7_to_28cm",
            ]

        if self.daily_variables is None:
            self.daily_variables = [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "rain_sum",
                "snowfall_sum",
                "wind_speed_10m_max",
                "wind_gusts_10m_max",
                "wind_direction_10m_dominant",
                "pressure_msl_mean",
                "shortwave_radiation_sum",
                "precipitation_hours",
                "et0_fao_evapotranspiration",
            ]


class OpenMeteoHistoricalClient:
    """Cliente para coleta de dados históricos da Open-Meteo"""

    def __init__(self, config: OpenMeteoConfig = None):
        self.config = config or OpenMeteoConfig()
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        self.timeout = 30

    async def fetch_year_data(self, year: int) -> Optional[Dict]:
        """
        Coleta dados de um ano específico da API Open-Meteo

        Args:
            year: Ano para coletar dados (2000-2025)

        Returns:
            Dict com dados meteorológicos do ano ou None se erro
        """
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        params = {
            "latitude": self.config.latitude,
            "longitude": self.config.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": self.config.timezone,
            "hourly": ",".join(self.config.hourly_variables),
            "daily": ",".join(self.config.daily_variables),
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"Coletando dados do ano {year}...")

                response = await client.get(self.base_url, params=params)
                response.raise_for_status()

                data = response.json()
                logger.info(
                    f"Dados de {year} coletados com sucesso. "
                    f"Registros horários: {len(data.get('hourly', {}).get('time', []))}"
                )

                return data

        except httpx.HTTPError as e:
            logger.error(f"Erro HTTP ao coletar dados de {year}: {e}")
            if hasattr(e, "response") and e.response:
                logger.error(f"Status: {e.response.status_code}")
                logger.error(f"Resposta: {e.response.text[:500]}")
            return None
        except Exception as e:
            logger.error(f"Erro inesperado ao coletar dados de {year}: {e}")
            return None

    def save_data_to_csv(self, data: Dict, year: int, output_dir: str):
        """
        Salva dados em arquivos CSV separados para dados horários e diários

        Args:
            data: Dados retornados pela API
            year: Ano dos dados
            output_dir: Diretório para salvar os arquivos
        """
        try:
            # Dados horários
            if "hourly" in data and data["hourly"]:
                hourly_df = pd.DataFrame(data["hourly"])
                hourly_df["time"] = pd.to_datetime(hourly_df["time"])

                hourly_file = f"{output_dir}/open_meteo_hourly_{year}.csv"
                hourly_df.to_csv(hourly_file, index=False)
                logger.info(f"Dados horários salvos em: {hourly_file}")

            # Dados diários
            if "daily" in data and data["daily"]:
                daily_df = pd.DataFrame(data["daily"])
                daily_df["time"] = pd.to_datetime(daily_df["time"])

                daily_file = f"{output_dir}/open_meteo_daily_{year}.csv"
                daily_df.to_csv(daily_file, index=False)
                logger.info(f"Dados diários salvos em: {daily_file}")

            # Metadados
            metadata = {
                "latitude": data.get("latitude"),
                "longitude": data.get("longitude"),
                "elevation": data.get("elevation"),
                "timezone": data.get("timezone"),
                "year": year,
                "collection_date": datetime.now().isoformat(),
            }

            metadata_file = f"{output_dir}/open_meteo_metadata_{year}.json"
            import json

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.error(f"Erro ao salvar dados de {year}: {e}")


async def collect_historical_data_2000_2025():
    """
    Função principal para coletar todos os dados históricos de 2000-2025
    """
    config = OpenMeteoConfig()
    client = OpenMeteoHistoricalClient(config)

    output_dir = "data/raw/open_meteo"
    years = range(2000, 2026)  # 2000 até 2025

    logger.info(f"Iniciando coleta de dados históricos para {len(years)} anos")
    logger.info(f"Coordenadas: {config.latitude}, {config.longitude}")
    logger.info(f"Variáveis horárias: {len(config.hourly_variables)}")
    logger.info(f"Variáveis diárias: {len(config.daily_variables)}")

    success_count = 0
    error_count = 0

    for year in years:
        try:
            data = await client.fetch_year_data(year)

            if data:
                client.save_data_to_csv(data, year, output_dir)
                success_count += 1
                logger.info(f"✅ Ano {year} processado com sucesso")
            else:
                error_count += 1
                logger.error(f"❌ Falha ao processar ano {year}")

            # Pausa entre requisições para evitar rate limiting
            await asyncio.sleep(2)

        except Exception as e:
            error_count += 1
            logger.error(f"❌ Erro ao processar ano {year}: {e}")

    logger.info(f"Coleta finalizada. Sucessos: {success_count}, Erros: {error_count}")

    if success_count > 0:
        logger.info("✅ Dados históricos da Open-Meteo coletados com sucesso!")
        logger.info(f"Arquivos salvos em: {output_dir}")
    else:
        logger.error("❌ Nenhum dado foi coletado com sucesso")


if __name__ == "__main__":
    asyncio.run(collect_historical_data_2000_2025())
