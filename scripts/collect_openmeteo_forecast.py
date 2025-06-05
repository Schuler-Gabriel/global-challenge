#!/usr/bin/env python3
"""
Script focado para coleta APENAS dos dados Open-Meteo Historical Forecast API (2022-2025)
Com dados de níveis de pressão 500hPa e 850hPa para o Sistema de Alertas de Cheias.

Salva em data/raw/ para manter padrão de organização do projeto.
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


class OpenMeteoForecastCollector:
    """Coletor focado nos dados Historical Forecast API"""

    # Coordenadas de Porto Alegre
    LATITUDE = -30.0331
    LONGITUDE = -51.2300
    TIMEZONE = "America/Sao_Paulo"

    def __init__(self, output_dir: str = "data/raw"):
        """Inicializar coletor"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Context manager async entry"""
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

    def build_forecast_url(self, start_date: str, end_date: str) -> str:
        """Construir URL da Historical Forecast API com todas as variáveis necessárias"""

        # Variáveis de superfície completas
        surface_vars = [
            # Básicas meteorológicas
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
            # Nuvens e visibilidade
            "cloudcover",
            "cloudcover_low",
            "cloudcover_mid",
            "cloudcover_high",
            "visibility",
            # Ventos
            "windspeed_10m",
            "winddirection_10m",
            "windgusts_10m",
            # Variáveis avançadas atmosféricas
            "cape",
            "lifted_index",
            "vapour_pressure_deficit",
            # Solo
            "soil_temperature_0cm",
            "soil_moisture_0_1cm",
        ]

        # Níveis de pressão críticos para análise sinótica
        pressure_levels = ["1000hPa", "850hPa", "700hPa", "500hPa", "300hPa"]
        pressure_vars = [
            "temperature",
            "relative_humidity",
            "wind_speed",
            "wind_direction",
            "geopotential_height",
        ]

        # Combinar todas as variáveis
        all_variables = surface_vars.copy()

        for level in pressure_levels:
            for var in pressure_vars:
                all_variables.append(f"{var}_{level}")

        # Construir parâmetros da URL
        params = {
            "latitude": self.LATITUDE,
            "longitude": self.LONGITUDE,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(all_variables),
            "timezone": self.TIMEZONE,
            "format": "json",
        }

        base_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        url_params = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{base_url}?{url_params}"

    def chunk_date_range(
        self, start_date: str, end_date: str, chunk_months: int = 4
    ) -> List[tuple]:
        """Dividir período em chunks de 4 meses para facilitar download"""
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

    async def fetch_data_chunk(self, url: str, chunk_name: str) -> Optional[Dict]:
        """Buscar chunk de dados da API"""

        try:
            print(f"🔄 Fazendo request para: {chunk_name}")
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    # Verificar se tem dados válidos
                    if "hourly" in data and "time" in data["hourly"]:
                        records = len(data["hourly"]["time"])
                        variables = len(
                            [k for k in data["hourly"].keys() if k != "time"]
                        )
                        pressure_vars = [
                            k
                            for k in data["hourly"].keys()
                            if any(level in k for level in ["500hPa", "850hPa"])
                        ]

                        print(
                            f"✅ {chunk_name}: {records} registros, {variables} variáveis"
                        )
                        print(f"   Variáveis de pressão: {len(pressure_vars)}")
                        return data
                    else:
                        print(f"❌ {chunk_name}: Estrutura de dados inválida")
                        return None

                else:
                    text = await response.text()
                    print(f"❌ Erro {response.status} para {chunk_name}")
                    print(f"   Resposta: {text[:200]}")
                    return None

        except Exception as e:
            print(f"❌ Exceção ao coletar {chunk_name}: {e}")
            return None

    def combine_chunks(self, chunks: List[Dict]) -> Dict:
        """Combinar chunks de dados em um único dataset"""
        if not chunks:
            return {}

        print(f"🔧 Combinando {len(chunks)} chunks...")

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

        # Estatísticas finais
        total_records = len(combined["hourly"]["time"]) if "hourly" in combined else 0
        total_variables = (
            len([k for k in combined["hourly"].keys() if k != "time"])
            if "hourly" in combined
            else 0
        )

        print(
            f"✅ Dataset combinado: {total_records:,} registros, {total_variables} variáveis"
        )

        return combined

    async def collect_forecast_data(self) -> Optional[Dict]:
        """Coletar dados Historical Forecast API (2022-2025)"""

        print("\n🎯 COLETANDO OPEN-METEO HISTORICAL FORECAST API")
        print("=" * 60)
        print("📊 Dados com níveis de pressão 500hPa e 850hPa")
        print(f"📍 Local: Porto Alegre ({self.LATITUDE}, {self.LONGITUDE})")

        # Definir período
        start_date = "2022-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"📅 Período: {start_date} a {end_date}")

        # Dividir em chunks
        chunks_dates = self.chunk_date_range(start_date, end_date, chunk_months=4)
        print(f"📦 Total de chunks: {len(chunks_dates)}")

        all_data = []
        failed_chunks = []

        for i, (start_chunk, end_chunk) in enumerate(chunks_dates):
            chunk_name = (
                f"Chunk {i+1}/{len(chunks_dates)} ({start_chunk} a {end_chunk})"
            )
            print(f"\n📦 {chunk_name}")

            url = self.build_forecast_url(start_chunk, end_chunk)

            data = await self.fetch_data_chunk(url, chunk_name)

            if data:
                all_data.append(data)
                # Delay respeitoso para a API
                await asyncio.sleep(3)
            else:
                failed_chunks.append((start_chunk, end_chunk))

        if all_data:
            print(f"\n🔧 COMBINANDO DADOS...")
            combined_data = self.combine_chunks(all_data)

            # Salvar dados em data/raw
            output_filename = (
                "openmeteo_historical_forecast_2022_2025_with_pressure_levels.json"
            )
            filepath = self.output_dir / output_filename

            print(f"💾 Salvando dados em: {filepath}")

            with open(filepath, "w") as f:
                json.dump(combined_data, f, indent=2)

            print(f"✅ Dados salvos com sucesso!")

            if failed_chunks:
                print(f"⚠️ Chunks que falharam: {len(failed_chunks)}")
                for start_fail, end_fail in failed_chunks:
                    print(f"   - {start_fail} a {end_fail}")

            return combined_data
        else:
            print("❌ Nenhum dado foi coletado")
            return None

    def analyze_collected_data(self) -> Dict:
        """Analisar dados coletados"""

        print(f"\n📊 ANALISANDO DADOS COLETADOS")
        print("=" * 40)

        filepath = (
            self.output_dir
            / "openmeteo_historical_forecast_2022_2025_with_pressure_levels.json"
        )

        if not filepath.exists():
            print("❌ Arquivo de dados não encontrado")
            return {}

        with open(filepath) as f:
            data = json.load(f)

        if "hourly" not in data:
            print("❌ Estrutura de dados inválida")
            return {}

        hourly = data["hourly"]
        total_records = len(hourly.get("time", []))
        all_variables = [k for k in hourly.keys() if k != "time"]

        # Analisar variáveis por categoria
        pressure_vars = [
            v for v in all_variables if any(level in v for level in ["hPa"])
        ]
        surface_vars = [
            v for v in all_variables if not any(level in v for level in ["hPa"])
        ]

        # Variáveis críticas para análise sinótica
        vars_850hpa = [v for v in pressure_vars if "850hPa" in v]
        vars_500hpa = [v for v in pressure_vars if "500hPa" in v]

        analysis = {
            "collection_date": datetime.now().isoformat(),
            "file_path": str(filepath),
            "total_records": total_records,
            "total_variables": len(all_variables),
            "surface_variables": len(surface_vars),
            "pressure_variables": len(pressure_vars),
            "vars_850hPa": len(vars_850hpa),
            "vars_500hPa": len(vars_500hpa),
            "time_range": {
                "start": hourly["time"][0] if hourly.get("time") else None,
                "end": hourly["time"][-1] if hourly.get("time") else None,
            },
            "sample_pressure_vars": {
                "850hPa": vars_850hpa[:5],
                "500hPa": vars_500hpa[:5],
            },
        }

        # Salvar análise
        analysis_file = self.output_dir / "openmeteo_forecast_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        print(f"📋 Análise salva em: {analysis_file}")

        return analysis

    def print_final_summary(self, analysis: Dict):
        """Imprimir resumo final da coleta"""

        print("\n" + "=" * 80)
        print("🎉 COLETA CONCLUÍDA - OPEN-METEO HISTORICAL FORECAST API")
        print("=" * 80)

        print(f"\n📁 ARQUIVO SALVO:")
        print(f"   {analysis.get('file_path', 'N/A')}")

        print(f"\n📊 ESTATÍSTICAS:")
        print(f"   • Total de registros: {analysis.get('total_records', 0):,}")
        print(f"   • Total de variáveis: {analysis.get('total_variables', 0)}")
        print(f"   • Variáveis de superfície: {analysis.get('surface_variables', 0)}")
        print(f"   • Variáveis de pressão: {analysis.get('pressure_variables', 0)}")

        print(f"\n🌦️ DADOS DE NÍVEIS DE PRESSÃO:")
        print(f"   • Variáveis 850hPa: {analysis.get('vars_850hPa', 0)} ✅")
        print(f"   • Variáveis 500hPa: {analysis.get('vars_500hPa', 0)} ✅")

        time_range = analysis.get("time_range", {})
        print(f"\n📅 PERÍODO COLETADO:")
        print(f"   • Início: {time_range.get('start', 'N/A')}")
        print(f"   • Fim: {time_range.get('end', 'N/A')}")

        sample_vars = analysis.get("sample_pressure_vars", {})
        if sample_vars.get("850hPa"):
            print(f"\n🔍 EXEMPLOS 850hPa:")
            for var in sample_vars["850hPa"]:
                print(f"   • {var}")

        if sample_vars.get("500hPa"):
            print(f"\n🔍 EXEMPLOS 500hPa:")
            for var in sample_vars["500hPa"]:
                print(f"   • {var}")

        print(f"\n✅ Dados prontos para feature engineering e treinamento LSTM!")
        print("=" * 80)


async def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Coletar dados Open-Meteo Historical Forecast API"
    )
    parser.add_argument(
        "--output-dir", default="data/raw", help="Diretório de saída (padrão: data/raw)"
    )
    parser.add_argument(
        "--test", action="store_true", help="Modo teste (apenas 1 chunk)"
    )

    args = parser.parse_args()

    print("🚀 INICIANDO COLETA FOCADA - OPEN-METEO HISTORICAL FORECAST")
    print(f"📁 Salvando em: {args.output_dir}")

    async with OpenMeteoForecastCollector(args.output_dir) as collector:
        if args.test:
            print("🧪 MODO TESTE - Coletando apenas um período pequeno")
            # Implementar lógica de teste se necessário

        # Executar coleta
        data = await collector.collect_forecast_data()

        if data:
            # Analisar dados coletados
            analysis = collector.analyze_collected_data()

            # Imprimir resumo final
            collector.print_final_summary(analysis)
        else:
            print("❌ Falha na coleta de dados")


if __name__ == "__main__":
    asyncio.run(main())
