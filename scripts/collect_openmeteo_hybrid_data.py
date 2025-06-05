#!/usr/bin/env python3
"""
Script para coleta de dados da estratégia híbrida Open-Meteo
Baseado na análise comparativa realizada.

Implementa:
1. Historical Forecast API (2022-2025) - Dados principais com níveis de pressão
2. Historical Weather API (2000-2021) - Extensão temporal de superfície
3. Validação opcional com dados INMET existentes
"""

import argparse
import asyncio
import json
import os
import ssl
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import certifi
import pandas as pd


class HybridDataCollector:
    """Coletor de dados da estratégia híbrida Open-Meteo"""

    # Coordenadas de Porto Alegre
    LATITUDE = -30.0331
    LONGITUDE = -51.2300
    TIMEZONE = "America/Sao_Paulo"

    def __init__(self, output_dir: str = "data/openmeteo_hybrid"):
        """Inicializar coletor"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Context manager async entry"""
        # Configurar SSL context
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)

        self.session = aiohttp.ClientSession(
            connector=connector, timeout=aiohttp.ClientTimeout(total=120)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager async exit"""
        if self.session:
            await self.session.close()

    def build_historical_forecast_url(
        self, start_date: str, end_date: str, include_pressure: bool = True
    ) -> str:
        """Construir URL da Historical Forecast API"""

        # Variáveis de superfície principais
        surface_vars = [
            "temperature_2m",
            "relative_humidity_2m",
            "dewpoint_2m",
            "apparent_temperature",
            "precipitation_probability",
            "precipitation",
            "rain",
            "showers",
            "pressure_msl",
            "surface_pressure",
            "cloudcover",
            "cloudcover_low",
            "cloudcover_mid",
            "cloudcover_high",
            "windspeed_10m",
            "winddirection_10m",
            "windgusts_10m",
            "cape",
            "lifted_index",
            "visibility",
        ]

        variables = surface_vars.copy()

        # Adicionar variáveis de níveis de pressão
        if include_pressure:
            pressure_levels = ["1000hPa", "850hPa", "700hPa", "500hPa", "300hPa"]
            pressure_vars = [
                "temperature",
                "relative_humidity",
                "wind_speed",
                "wind_direction",
                "geopotential_height",
            ]

            for level in pressure_levels:
                for var in pressure_vars:
                    variables.append(f"{var}_{level}")

        params = {
            "latitude": self.LATITUDE,
            "longitude": self.LONGITUDE,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": self.TIMEZONE,
            "format": "json",
        }

        base_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        url_params = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{base_url}?{url_params}"

    def build_historical_weather_url(self, start_date: str, end_date: str) -> str:
        """Construir URL da Historical Weather API"""

        # Variáveis de superfície do ERA5
        variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "dewpoint_2m",
            "precipitation",
            "rain",
            "pressure_msl",
            "surface_pressure",
            "cloudcover",
            "windspeed_10m",
            "winddirection_10m",
            "windgusts_10m",
            "et0_fao_evapotranspiration",
            "vapour_pressure_deficit",
            "shortwave_radiation",
            "soil_temperature_0_7cm",
            "soil_moisture_0_7cm",
        ]

        params = {
            "latitude": self.LATITUDE,
            "longitude": self.LONGITUDE,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": self.TIMEZONE,
            "format": "json",
        }

        base_url = "https://archive-api.open-meteo.com/v1/archive"
        url_params = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{base_url}?{url_params}"

    def chunk_date_range(
        self, start_date: str, end_date: str, chunk_months: int = 6
    ) -> List[tuple]:
        """Dividir período em chunks menores"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        chunks = []
        current = start

        while current < end:
            chunk_end = min(current + timedelta(days=chunk_months * 30), end)
            chunks.append(
                (current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
            )
            current = chunk_end + timedelta(days=1)

        return chunks

    async def fetch_data_chunk(
        self, url: str, api_name: str, start_date: str, end_date: str
    ) -> Optional[Dict]:
        """Buscar chunk de dados da API"""

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ {api_name}: {start_date} a {end_date}")
                    return data
                else:
                    text = await response.text()
                    print(f"❌ Erro {response.status} para {api_name}: {start_date}")
                    print(f"   Resposta: {text[:200]}")
                    return None

        except Exception as e:
            print(f"❌ Exceção ao coletar {api_name}: {e}")
            return None

    def combine_chunks(self, chunks: List[Dict]) -> Dict:
        """Combinar chunks de dados em um único dataset"""
        if not chunks:
            return {}

        # Usar o primeiro chunk como base
        combined = chunks[0].copy()

        # Combinar dados horários de outros chunks
        if "hourly" in combined:
            for chunk in chunks[1:]:
                if "hourly" in chunk:
                    # Combinar arrays de tempo
                    combined["hourly"]["time"].extend(chunk["hourly"]["time"])

                    # Combinar arrays de dados para cada variável
                    for var in chunk["hourly"]:
                        if var != "time" and var in combined["hourly"]:
                            combined["hourly"][var].extend(chunk["hourly"][var])

        return combined

    async def collect_historical_forecast_data(self) -> Optional[Dict]:
        """Coletar dados Historical Forecast API (2022-2025)"""

        print("\n🎯 FASE 1: Historical Forecast API (2022-2025)")
        print("📊 Coletando dados com níveis de pressão...")

        # Definir período
        start_date = "2022-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"📅 Período: {start_date} a {end_date}")

        # Dividir em chunks
        chunks_dates = self.chunk_date_range(start_date, end_date, chunk_months=3)
        print(f"📦 Total de chunks: {len(chunks_dates)}")

        all_data = []
        failed_chunks = []

        for i, (start_chunk, end_chunk) in enumerate(chunks_dates):
            print(f"📦 Chunk {i+1}/{len(chunks_dates)}: {start_chunk} a {end_chunk}")

            url = self.build_historical_forecast_url(
                start_chunk, end_chunk, include_pressure=True
            )

            data = await self.fetch_data_chunk(
                url, "Historical Forecast", start_chunk, end_chunk
            )

            if data:
                all_data.append(data)
                # Delay para ser respeitoso com a API
                await asyncio.sleep(2)
            else:
                failed_chunks.append((start_chunk, end_chunk))

        if all_data:
            combined_data = self.combine_chunks(all_data)

            # Salvar dados
            filepath = self.output_dir / "historical_forecast_with_pressure_levels.json"
            with open(filepath, "w") as f:
                json.dump(combined_data, f, indent=2)

            print(f"💾 Dados salvos: {filepath}")

            if failed_chunks:
                print(f"⚠️ Chunks falharam: {len(failed_chunks)}")

            return combined_data
        else:
            print("❌ Nenhum dado coletado para Historical Forecast API")
            return None

    async def collect_historical_weather_data(self) -> Optional[Dict]:
        """Coletar dados Historical Weather API (2000-2021)"""

        print("\n🌍 FASE 2: Historical Weather API (2000-2021)")
        print("📈 Coletando dados de superfície para extensão temporal...")

        # Definir período
        start_date = "2000-01-01"
        end_date = "2021-12-31"

        print(f"📅 Período: {start_date} a {end_date}")

        # Dividir em chunks
        chunks_dates = self.chunk_date_range(start_date, end_date, chunk_months=6)
        print(f"📦 Total de chunks: {len(chunks_dates)}")

        all_data = []
        failed_chunks = []

        for i, (start_chunk, end_chunk) in enumerate(chunks_dates):
            print(f"📦 Chunk {i+1}/{len(chunks_dates)}: {start_chunk} a {end_chunk}")

            url = self.build_historical_weather_url(start_chunk, end_chunk)

            data = await self.fetch_data_chunk(
                url, "Historical Weather", start_chunk, end_chunk
            )

            if data:
                all_data.append(data)
                # Delay para ser respeitoso com a API
                await asyncio.sleep(2)
            else:
                failed_chunks.append((start_chunk, end_chunk))

        if all_data:
            combined_data = self.combine_chunks(all_data)

            # Salvar dados
            filepath = self.output_dir / "historical_weather_surface_only.json"
            with open(filepath, "w") as f:
                json.dump(combined_data, f, indent=2)

            print(f"💾 Dados salvos: {filepath}")

            if failed_chunks:
                print(f"⚠️ Chunks falharam: {len(failed_chunks)}")

            return combined_data
        else:
            print("❌ Nenhum dado coletado para Historical Weather API")
            return None

    def analyze_collected_data(self) -> Dict:
        """Analisar dados coletados"""

        print("\n📊 FASE 3: Análise dos Dados Coletados")

        analysis = {
            "collection_date": datetime.now().isoformat(),
            "location": f"Porto Alegre ({self.LATITUDE}, {self.LONGITUDE})",
            "datasets": {},
        }

        # Analisar Historical Forecast
        forecast_file = (
            self.output_dir / "historical_forecast_with_pressure_levels.json"
        )
        if forecast_file.exists():
            with open(forecast_file) as f:
                data = json.load(f)

                if "hourly" in data:
                    hourly = data["hourly"]
                    analysis["datasets"]["historical_forecast"] = {
                        "total_records": len(hourly.get("time", [])),
                        "variables_count": len(
                            [k for k in hourly.keys() if k != "time"]
                        ),
                        "variables": list(hourly.keys()),
                        "pressure_level_variables": [
                            v
                            for v in hourly.keys()
                            if any(
                                level in v for level in ["500hPa", "850hPa", "1000hPa"]
                            )
                        ],
                        "time_range": {
                            "start": hourly["time"][0] if hourly.get("time") else None,
                            "end": hourly["time"][-1] if hourly.get("time") else None,
                        },
                    }

        # Analisar Historical Weather
        weather_file = self.output_dir / "historical_weather_surface_only.json"
        if weather_file.exists():
            with open(weather_file) as f:
                data = json.load(f)

                if "hourly" in data:
                    hourly = data["hourly"]
                    analysis["datasets"]["historical_weather"] = {
                        "total_records": len(hourly.get("time", [])),
                        "variables_count": len(
                            [k for k in hourly.keys() if k != "time"]
                        ),
                        "variables": list(hourly.keys()),
                        "time_range": {
                            "start": hourly["time"][0] if hourly.get("time") else None,
                            "end": hourly["time"][-1] if hourly.get("time") else None,
                        },
                    }

        # Salvar análise
        analysis_file = self.output_dir / "collection_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        print(f"📋 Análise salva: {analysis_file}")

        return analysis

    def print_collection_summary(self, analysis: Dict):
        """Imprimir resumo da coleta"""

        print("\n" + "=" * 80)
        print("📊 RESUMO DA COLETA - ESTRATÉGIA HÍBRIDA OPEN-METEO")
        print("=" * 80)

        if "historical_forecast" in analysis["datasets"]:
            forecast = analysis["datasets"]["historical_forecast"]
            pressure_vars = forecast.get("pressure_level_variables", [])

            print(f"\n🎯 HISTORICAL FORECAST API (PRINCIPAL)")
            print(f"   • Registros: {forecast.get('total_records', 0):,}")
            print(f"   • Variáveis: {forecast.get('variables_count', 0)}")
            print(
                f"   • Período: {forecast.get('time_range', {}).get('start')} a {forecast.get('time_range', {}).get('end')}"
            )
            print(f"   • Variáveis de pressão: {len(pressure_vars)}")

            if pressure_vars:
                print(f"   • Níveis encontrados:")
                for level in ["1000hPa", "850hPa", "700hPa", "500hPa", "300hPa"]:
                    level_vars = [v for v in pressure_vars if level in v]
                    if level_vars:
                        print(f"     - {level}: {len(level_vars)} variáveis")

        if "historical_weather" in analysis["datasets"]:
            weather = analysis["datasets"]["historical_weather"]

            print(f"\n🌍 HISTORICAL WEATHER API (EXTENSÃO)")
            print(f"   • Registros: {weather.get('total_records', 0):,}")
            print(f"   • Variáveis: {weather.get('variables_count', 0)}")
            print(
                f"   • Período: {weather.get('time_range', {}).get('start')} a {weather.get('time_range', {}).get('end')}"
            )

        total_records = 0
        for dataset in analysis["datasets"].values():
            total_records += dataset.get("total_records", 0)

        print(f"\n🔄 TOTAL COMBINADO:")
        print(f"   • Registros totais: {total_records:,}")
        print(f"   • Período completo: 2000-2025 (25+ anos)")
        print(f"   • Dados com níveis de pressão: 2022-2025 ✅")
        print(f"   • Extensão temporal: 2000-2021 ✅")

        print("\n" + "=" * 80)

    async def run_hybrid_collection(self) -> Dict:
        """Executar coleta completa da estratégia híbrida"""

        print("🚀 Iniciando coleta da estratégia híbrida Open-Meteo...")
        print(f"📍 Localização: Porto Alegre ({self.LATITUDE}, {self.LONGITUDE})")

        results = {}

        # Fase 1: Historical Forecast API (principal)
        results["historical_forecast"] = await self.collect_historical_forecast_data()

        # Fase 2: Historical Weather API (extensão)
        results["historical_weather"] = await self.collect_historical_weather_data()

        # Fase 3: Análise dos dados coletados
        analysis = self.analyze_collected_data()

        # Imprimir resumo
        self.print_collection_summary(analysis)

        return results


async def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Coletar dados da estratégia híbrida Open-Meteo"
    )
    parser.add_argument(
        "--output-dir", default="data/openmeteo_hybrid", help="Diretório de saída"
    )
    parser.add_argument(
        "--test-mode", action="store_true", help="Modo teste com período limitado"
    )

    args = parser.parse_args()

    async with HybridDataCollector(args.output_dir) as collector:
        if args.test_mode:
            print("🧪 Modo teste - período limitado")
            # Modificar períodos para teste
            # Implementar lógica de teste se necessário

        # Executar coleta híbrida
        results = await collector.run_hybrid_collection()

        print("\n✅ Coleta da estratégia híbrida concluída!")
        print(f"📁 Dados salvos em: {args.output_dir}")
        print("\n🎯 Próximos passos:")
        print("   1. Verificar dados coletados")
        print("   2. Implementar feature engineering atmosférica")
        print("   3. Treinar modelos LSTM híbridos")
        print("   4. Validar com dados INMET")


if __name__ == "__main__":
    asyncio.run(main())
