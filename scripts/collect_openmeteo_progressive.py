#!/usr/bin/env python3
"""
Script PROGRESSIVO para coleta dos dados Open-Meteo Historical Forecast API (2022-2025)
SALVA CADA CHUNK INDIVIDUALMENTE para não perder dados se interrompido.

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


class ProgressiveOpenMeteoCollector:
    """Coletor que salva progressivamente cada chunk"""

    # Coordenadas de Porto Alegre
    LATITUDE = -30.0331
    LONGITUDE = -51.2300
    TIMEZONE = "America/Sao_Paulo"

    def __init__(self, output_dir: str = "data/raw"):
        """Inicializar coletor"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir = self.output_dir / "openmeteo_chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
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

        # Variáveis de superfície essenciais
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
            "vapour_pressure_deficit",
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
        self, start_date: str, end_date: str, chunk_months: int = 3
    ) -> List[tuple]:
        """Dividir período em chunks menores (3 meses) para garantir sucesso"""
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

    def get_chunk_filename(self, start_date: str, end_date: str) -> str:
        """Gerar nome do arquivo para o chunk"""
        start_clean = start_date.replace("-", "")
        end_clean = end_date.replace("-", "")
        return f"chunk_{start_clean}_to_{end_clean}.json"

    def chunk_exists(self, start_date: str, end_date: str) -> bool:
        """Verificar se chunk já foi baixado"""
        filename = self.get_chunk_filename(start_date, end_date)
        filepath = self.chunks_dir / filename
        return filepath.exists()

    async def fetch_and_save_chunk(
        self, start_date: str, end_date: str, chunk_number: int, total_chunks: int
    ) -> bool:
        """Buscar chunk de dados da API e salvar imediatamente"""

        chunk_name = f"Chunk {chunk_number}/{total_chunks} ({start_date} a {end_date})"

        # Verificar se já existe
        if self.chunk_exists(start_date, end_date):
            print(f"⏭️ {chunk_name}: Já existe, pulando...")
            return True

        try:
            print(f"🔄 {chunk_name}: Fazendo request...")
            url = self.build_forecast_url(start_date, end_date)

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

                        # Salvar chunk imediatamente
                        filename = self.get_chunk_filename(start_date, end_date)
                        filepath = self.chunks_dir / filename

                        with open(filepath, "w") as f:
                            json.dump(data, f)

                        print(
                            f"✅ {chunk_name}: {records} registros, {variables} variáveis"
                        )
                        print(f"   Variáveis de pressão: {len(pressure_vars)}")
                        print(f"   💾 Salvo em: {filename}")
                        return True
                    else:
                        print(f"❌ {chunk_name}: Estrutura de dados inválida")
                        return False

                else:
                    text = await response.text()
                    print(f"❌ {chunk_name}: Erro {response.status}")
                    print(f"   Resposta: {text[:200]}")
                    return False

        except Exception as e:
            print(f"❌ {chunk_name}: Exceção - {e}")
            return False

    def list_existing_chunks(self) -> List[str]:
        """Listar chunks já baixados"""
        if not self.chunks_dir.exists():
            return []
        return [f.name for f in self.chunks_dir.glob("chunk_*.json")]

    def combine_all_chunks(self) -> Optional[Dict]:
        """Combinar todos os chunks em um único dataset"""

        print(f"\n🔧 COMBINANDO CHUNKS SALVOS...")

        chunk_files = sorted(self.chunks_dir.glob("chunk_*.json"))

        if not chunk_files:
            print("❌ Nenhum chunk encontrado para combinar")
            return None

        print(f"📦 Encontrados {len(chunk_files)} chunks para combinar")

        combined = None
        total_records = 0

        for i, chunk_file in enumerate(chunk_files):
            print(f"   📄 {i+1}/{len(chunk_files)}: {chunk_file.name}")

            try:
                with open(chunk_file) as f:
                    chunk_data = json.load(f)

                if "hourly" not in chunk_data:
                    print(f"      ⚠️ Chunk inválido, pulando...")
                    continue

                if combined is None:
                    # Primeiro chunk
                    combined = chunk_data.copy()
                    total_records = len(combined["hourly"]["time"])
                else:
                    # Combinar com dados existentes
                    chunk_records = len(chunk_data["hourly"]["time"])

                    # Combinar arrays de tempo
                    combined["hourly"]["time"].extend(chunk_data["hourly"]["time"])

                    # Combinar arrays de dados para cada variável
                    for var in chunk_data["hourly"]:
                        if var != "time" and var in combined["hourly"]:
                            combined["hourly"][var].extend(chunk_data["hourly"][var])

                    total_records += chunk_records

            except Exception as e:
                print(f"      ❌ Erro ao processar chunk: {e}")
                continue

        if combined:
            print(f"✅ Dataset combinado: {total_records:,} registros")
            return combined
        else:
            print("❌ Falha ao combinar chunks")
            return None

    async def collect_progressive_data(self) -> Optional[Dict]:
        """Coletar dados progressivamente, salvando cada chunk"""

        print("\n🎯 COLETA PROGRESSIVA - OPEN-METEO HISTORICAL FORECAST API")
        print("=" * 70)
        print("📊 Dados com níveis de pressão 500hPa e 850hPa")
        print(f"📍 Local: Porto Alegre ({self.LATITUDE}, {self.LONGITUDE})")
        print(f"📁 Chunks salvos em: {self.chunks_dir}")

        # Verificar chunks existentes
        existing_chunks = self.list_existing_chunks()
        if existing_chunks:
            print(f"\n📋 Chunks já existentes: {len(existing_chunks)}")
            for chunk in existing_chunks[:5]:  # Mostrar apenas os primeiros 5
                print(f"   • {chunk}")
            if len(existing_chunks) > 5:
                print(f"   • ... e mais {len(existing_chunks) - 5} chunks")

        # Definir período
        start_date = "2022-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"\n📅 Período total: {start_date} a {end_date}")

        # Dividir em chunks
        chunks_dates = self.chunk_date_range(start_date, end_date, chunk_months=3)
        print(f"📦 Total de chunks: {len(chunks_dates)}")

        successful_chunks = 0
        failed_chunks = 0

        for i, (start_chunk, end_chunk) in enumerate(chunks_dates):
            success = await self.fetch_and_save_chunk(
                start_chunk, end_chunk, i + 1, len(chunks_dates)
            )

            if success:
                successful_chunks += 1
            else:
                failed_chunks += 1

            # Delay respeitoso para a API
            await asyncio.sleep(2)

        print(f"\n📊 RESULTADO DA COLETA:")
        print(f"   ✅ Chunks bem-sucedidos: {successful_chunks}")
        print(f"   ❌ Chunks falharam: {failed_chunks}")

        # Combinar todos os chunks
        if successful_chunks > 0:
            return self.combine_all_chunks()
        else:
            return None

    def save_final_dataset(self, combined_data: Dict) -> str:
        """Salvar dataset final combinado"""

        output_filename = (
            "openmeteo_historical_forecast_2022_2025_with_pressure_levels.json"
        )
        filepath = self.output_dir / output_filename

        print(f"💾 Salvando dataset final em: {filepath}")

        with open(filepath, "w") as f:
            json.dump(combined_data, f, indent=2)

        # Criar análise simples
        hourly = combined_data.get("hourly", {})
        total_records = len(hourly.get("time", []))
        all_variables = [k for k in hourly.keys() if k != "time"]
        pressure_vars = [
            v for v in all_variables if any(level in v for level in ["hPa"])
        ]

        analysis = {
            "collection_date": datetime.now().isoformat(),
            "file_path": str(filepath),
            "total_records": total_records,
            "total_variables": len(all_variables),
            "pressure_variables": len(pressure_vars),
            "time_range": {
                "start": hourly["time"][0] if hourly.get("time") else None,
                "end": hourly["time"][-1] if hourly.get("time") else None,
            },
        }

        # Salvar análise
        analysis_file = self.output_dir / "openmeteo_forecast_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def print_final_summary(self, filepath: str, analysis: Dict):
        """Imprimir resumo final"""

        print("\n" + "=" * 80)
        print("🎉 COLETA PROGRESSIVA CONCLUÍDA!")
        print("=" * 80)

        print(f"\n📁 DATASET FINAL:")
        print(f"   {filepath}")

        print(f"\n📊 ESTATÍSTICAS:")
        print(f"   • Total de registros: {analysis.get('total_records', 0):,}")
        print(f"   • Total de variáveis: {analysis.get('total_variables', 0)}")
        print(f"   • Variáveis de pressão: {analysis.get('pressure_variables', 0)}")

        time_range = analysis.get("time_range", {})
        print(f"\n📅 PERÍODO:")
        print(f"   • Início: {time_range.get('start', 'N/A')}")
        print(f"   • Fim: {time_range.get('end', 'N/A')}")

        print(f"\n✅ Dados com níveis de pressão 500hPa e 850hPa coletados!")
        print(f"✅ Pronto para feature engineering e treinamento LSTM!")
        print("=" * 80)


async def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Coleta progressiva Open-Meteo Historical Forecast API"
    )
    parser.add_argument(
        "--output-dir", default="data/raw", help="Diretório de saída (padrão: data/raw)"
    )
    parser.add_argument(
        "--combine-only", action="store_true", help="Apenas combinar chunks existentes"
    )

    args = parser.parse_args()

    print("🚀 COLETA PROGRESSIVA - OPEN-METEO HISTORICAL FORECAST")
    print(f"📁 Salvando em: {args.output_dir}")

    async with ProgressiveOpenMeteoCollector(args.output_dir) as collector:
        if args.combine_only:
            print("🔧 Apenas combinando chunks existentes...")
            combined_data = collector.combine_all_chunks()
        else:
            # Executar coleta progressiva
            combined_data = await collector.collect_progressive_data()

        if combined_data:
            # Salvar dataset final
            filepath = collector.save_final_dataset(combined_data)

            # Criar análise
            analysis = {
                "total_records": len(combined_data.get("hourly", {}).get("time", [])),
                "total_variables": len(
                    [k for k in combined_data.get("hourly", {}).keys() if k != "time"]
                ),
                "pressure_variables": len(
                    [v for v in combined_data.get("hourly", {}).keys() if "hPa" in v]
                ),
                "time_range": {
                    "start": combined_data.get("hourly", {}).get("time", [None])[0],
                    "end": combined_data.get("hourly", {}).get("time", [None])[-1],
                },
            }

            # Imprimir resumo final
            collector.print_final_summary(filepath, analysis)
        else:
            print("❌ Falha na coleta/combinação de dados")


if __name__ == "__main__":
    asyncio.run(main())
