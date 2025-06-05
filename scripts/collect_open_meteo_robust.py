#!/usr/bin/env python3
"""
Script robusto para coletar dados históricos da Open-Meteo (2000-2025)
Com tratamento de rate limiting e retry automático
"""

import requests
import json
import time
import os
from datetime import datetime
import sys

def create_directories():
    """Cria diretórios necessários"""
    os.makedirs("data/raw/open_meteo", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def log_message(message, level="INFO"):
    """Log com timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")
    
    # Também salva no arquivo
    with open("logs/open_meteo_robust.log", "a") as f:
        f.write(f"{timestamp} - {level} - {message}\n")

def check_existing_data():
    """Verifica quais anos já foram coletados"""
    existing_years = set()
    
    if os.path.exists("data/raw/open_meteo"):
        for filename in os.listdir("data/raw/open_meteo"):
            if filename.startswith("open_meteo_") and filename.endswith(".json"):
                try:
                    year_str = filename.replace("open_meteo_", "").replace(".json", "")
                    year = int(year_str)
                    existing_years.add(year)
                except ValueError:
                    continue
    
    return existing_years

def collect_year_data_with_retry(year, max_retries=3):
    """
    Coleta dados de um ano com retry automático em caso de rate limiting
    """
    log_message(f"Coletando dados do ano {year}...")
    
    # Parâmetros da API - versão mais simples para reduzir carga
    params = {
        "latitude": -30.0346,
        "longitude": -51.2177,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "timezone": "America/Sao_Paulo",
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m", 
            "precipitation",
            "rain",
            "pressure_msl",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m"
        ]),
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "wind_speed_10m_max",
            "wind_direction_10m_dominant"
        ]),
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm"
    }
    
    for attempt in range(max_retries):
        try:
            log_message(f"Tentativa {attempt + 1}/{max_retries} para o ano {year}")
            
            # Fazer requisição
            response = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params=params,
                timeout=60  # Timeout maior
            )
            
            if response.status_code == 429:
                # Rate limiting - aguardar mais tempo
                wait_time = (attempt + 1) * 60  # 60, 120, 180 segundos
                log_message(f"Rate limit atingido. Aguardando {wait_time} segundos...", "WARNING")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            data = response.json()
            
            # Verificar se há erro na resposta
            if data.get("error"):
                log_message(f"Erro na API: {data.get('reason', 'Erro desconhecido')}", "ERROR")
                return False
            
            # Salvar dados
            output_file = f"data/raw/open_meteo/open_meteo_{year}.json"
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Log de sucesso
            hourly_records = len(data.get('hourly', {}).get('time', []))
            daily_records = len(data.get('daily', {}).get('time', []))
            
            log_message(f"✅ Ano {year}: {hourly_records} registros horários, {daily_records} registros diários")
            return True
            
        except requests.exceptions.Timeout:
            log_message(f"Timeout na tentativa {attempt + 1} para {year}", "WARNING")
            if attempt < max_retries - 1:
                time.sleep(30)
        except requests.exceptions.RequestException as e:
            log_message(f"Erro na tentativa {attempt + 1} para {year}: {e}", "WARNING")
            if attempt < max_retries - 1:
                time.sleep(30)
    
    log_message(f"❌ Falha ao coletar dados de {year} após {max_retries} tentativas", "ERROR")
    return False

def main():
    """Função principal"""
    print("=" * 60)
    print("🌤️  COLETA ROBUSTA DE DADOS OPEN-METEO")
    print("=" * 60)
    print("Período: 2000-2025")
    print("Localização: Porto Alegre (-30.0346, -51.2177)")
    print("Estratégia: Um ano por vez com retry automático")
    print()
    
    # Criar diretórios
    create_directories()
    
    # Verificar dados existentes
    existing_years = check_existing_data()
    all_years = set(range(2000, 2026))
    remaining_years = sorted(all_years - existing_years)
    
    if existing_years:
        print(f"📁 Dados já coletados: {len(existing_years)} anos")
        print(f"🎯 Anos restantes: {len(remaining_years)} anos")
        print(f"📋 Lista: {remaining_years}")
    else:
        print(f"🎯 Coletando todos os {len(all_years)} anos")
    
    if not remaining_years:
        print("✅ Todos os dados já foram coletados!")
        return
    
    print()
    input("⏸️  Pressione ENTER para continuar ou Ctrl+C para cancelar...")
    print()
    
    # Inicializar log
    log_message("=== Iniciando coleta robusta Open-Meteo ===")
    
    # Coletar dados ano por ano
    success_count = 0
    error_count = 0
    
    for i, year in enumerate(remaining_years, 1):
        print(f"\n📅 [{i}/{len(remaining_years)}] Processando ano {year}")
        
        success = collect_year_data_with_retry(year)
        
        if success:
            success_count += 1
            print(f"✅ Sucesso!")
        else:
            error_count += 1
            print(f"❌ Erro!")
        
        # Pausa entre anos (importante para evitar rate limiting)
        if i < len(remaining_years):  # Não pausar no último
            wait_time = 10  # 10 segundos entre anos
            print(f"⏳ Aguardando {wait_time} segundos antes do próximo ano...")
            time.sleep(wait_time)
    
    # Resultado final
    print("\n" + "=" * 60)
    print("📊 RESUMO FINAL")
    print("=" * 60)
    print(f"✅ Sucessos: {success_count}")
    print(f"❌ Erros: {error_count}")
    print(f"📁 Total de arquivos: {len(existing_years) + success_count}")
    
    if success_count > 0:
        print(f"💾 Dados salvos em: data/raw/open_meteo/")
    
    if error_count > 0:
        print(f"🔄 Execute novamente para tentar os anos com erro")
    
    # Log final
    log_message(f"Coleta finalizada: {success_count} sucessos, {error_count} erros")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Coleta interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Erro inesperado: {e}")
        sys.exit(1) 