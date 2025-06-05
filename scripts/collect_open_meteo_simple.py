#!/usr/bin/env python3
"""
Script simplificado para coletar dados históricos da Open-Meteo (2000-2025)
"""

import requests
import json
import time
import os
from datetime import datetime

def create_directories():
    """Cria diretórios necessários"""
    os.makedirs("data/raw/open_meteo", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def collect_year_data(year):
    """
    Coleta dados de um ano específico da Open-Meteo
    """
    print(f"Coletando dados do ano {year}...")
    
    # Parâmetros da API
    params = {
        "latitude": -30.0346,
        "longitude": -51.2177,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "timezone": "America/Sao_Paulo",
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m", 
            "dew_point_2m",
            "precipitation",
            "rain",
            "pressure_msl",
            "surface_pressure",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "weather_code"
        ],
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "wind_direction_10m_dominant",
            "pressure_msl_mean"
        ],
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm"
    }
    
    # Converter lista de variáveis para string
    params["hourly"] = ",".join(params["hourly"])
    params["daily"] = ",".join(params["daily"])
    
    try:
        # Fazer requisição
        response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=params,
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        # Salvar dados
        output_file = f"data/raw/open_meteo/open_meteo_{year}.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Log de sucesso
        hourly_records = len(data.get('hourly', {}).get('time', []))
        daily_records = len(data.get('daily', {}).get('time', []))
        
        print(f"✅ Ano {year}: {hourly_records} registros horários, {daily_records} registros diários")
        
        # Log para arquivo
        with open("logs/open_meteo_collection.log", "a") as f:
            f.write(f"{datetime.now().isoformat()} - SUCCESS - {year}: {hourly_records}h, {daily_records}d\n")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro ao coletar {year}: {e}")
        
        # Log de erro
        with open("logs/open_meteo_collection.log", "a") as f:
            f.write(f"{datetime.now().isoformat()} - ERROR - {year}: {str(e)}\n")
        
        return False

def main():
    """Função principal"""
    print("=== Coleta de Dados Históricos Open-Meteo ===")
    print("Período: 2000-2025")
    print("Localização: Porto Alegre (-30.0346, -51.2177)")
    print()
    
    # Criar diretórios
    create_directories()
    
    # Inicializar log
    with open("logs/open_meteo_collection.log", "w") as f:
        f.write(f"{datetime.now().isoformat()} - Iniciando coleta Open-Meteo\n")
    
    # Coletar dados ano por ano
    success_count = 0
    error_count = 0
    
    for year in range(2000, 2026):  # 2000 até 2025
        success = collect_year_data(year)
        
        if success:
            success_count += 1
        else:
            error_count += 1
        
        # Pausa entre requisições
        time.sleep(2)
    
    # Resultado final
    print()
    print("=== Coleta Finalizada ===")
    print(f"Sucessos: {success_count}")
    print(f"Erros: {error_count}")
    print(f"Total: {success_count + error_count}")
    
    if success_count > 0:
        print("✅ Dados salvos em: data/raw/open_meteo/")
    
    # Log final
    with open("logs/open_meteo_collection.log", "a") as f:
        f.write(f"{datetime.now().isoformat()} - Coleta finalizada: {success_count} sucessos, {error_count} erros\n")

if __name__ == "__main__":
    main() 