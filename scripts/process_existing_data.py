#!/usr/bin/env python3
"""
Script para processar dados já existentes em data/raw/
Converte dados brutos em formato processado para os notebooks
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def setup_directories():
    """Criar diretórios necessários"""
    # Detectar se estamos executando do notebook ou da raiz
    current_dir = Path.cwd()
    if current_dir.name == "notebooks":
        base_path = current_dir.parent
    else:
        base_path = current_dir
    
    dirs = [
        base_path / "data/processed",
        base_path / "data/analysis", 
        base_path / "models"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("✅ Diretórios configurados")
    return base_path

def process_openmeteo_forecast(base_path):
    """Processar dados Open-Meteo Historical Forecast (JSON)"""
    print("\n🔄 Processando Open-Meteo Historical Forecast...")
    
    forecast_dir = base_path / "data/raw/Open-Meteo Historical Forecast"
    if not forecast_dir.exists():
        print("❌ Diretório não encontrado")
        return None
    
    # Encontrar arquivos JSON
    json_files = list(forecast_dir.glob("*.json"))
    
    if not json_files:
        print("❌ Nenhum arquivo JSON encontrado")
        return None
    
    print(f"📁 Encontrados {len(json_files)} arquivos JSON")
    
    # Carregar e combinar arquivos JSON
    all_hourly_data = []
    
    for file in json_files[:5]:  # Limitar a 5 arquivos para performance
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            # Extrair dados horárias
            if 'hourly' in data:
                hourly = data['hourly']
                
                # Converter em DataFrame
                df_hourly = pd.DataFrame(hourly)
                
                # Garantir coluna datetime
                if 'time' in df_hourly.columns:
                    df_hourly['datetime'] = pd.to_datetime(df_hourly['time'])
                
                all_hourly_data.append(df_hourly)
                print(f"✅ Carregado: {file.name} - {df_hourly.shape}")
                
        except Exception as e:
            print(f"❌ Erro em {file.name}: {e}")
    
    if not all_hourly_data:
        print("❌ Nenhum dado processado com sucesso")
        return None
    
    # Combinar todos os DataFrames
    df_combined = pd.concat(all_hourly_data, ignore_index=True)
    
    # Limpar e organizar
    if 'datetime' not in df_combined.columns and 'time' in df_combined.columns:
        df_combined['datetime'] = pd.to_datetime(df_combined['time'])
    
    # Remover duplicatas e ordenar
    if 'datetime' in df_combined.columns:
        df_combined = df_combined.drop_duplicates(subset=['datetime']).sort_values('datetime')
    
    # Salvar processado
    output_file = base_path / "data/processed/openmeteo_historical_forecast.csv"
    df_combined.to_csv(output_file, index=False)
    
    print(f"✅ Salvo: {output_file}")
    print(f"📊 Shape final: {df_combined.shape}")
    if 'datetime' in df_combined.columns:
        print(f"🗓️ Período: {df_combined['datetime'].min()} até {df_combined['datetime'].max()}")
    
    return df_combined

def process_openmeteo_weather(base_path):
    """Processar dados Open-Meteo Historical Weather (CSV)"""
    print("\n🔄 Processando Open-Meteo Historical Weather...")
    
    weather_dir = base_path / "data/raw/Open-Meteo Historical Weather"
    if not weather_dir.exists():
        print("❌ Diretório não encontrado")
        return None
    
    # Priorizar arquivos horárias (mais dados)
    hourly_files = list(weather_dir.glob("*hourly*.csv"))
    daily_files = list(weather_dir.glob("*daily*.csv"))
    
    files_to_process = hourly_files if hourly_files else daily_files
    
    if not files_to_process:
        print("❌ Nenhum arquivo CSV encontrado")
        return None
    
    print(f"📁 Encontrados {len(files_to_process)} arquivos ({'horárias' if hourly_files else 'diárias'})")
    
    # Carregar e combinar múltiplos arquivos
    dfs = []
    for file in files_to_process[:10]:  # Limitar a 10 arquivos para performance
        try:
            df = pd.read_csv(file)
            
            # Garantir coluna datetime
            datetime_cols = ['datetime', 'time', 'date', 'timestamp']
            datetime_col = None
            
            for col in datetime_cols:
                if col in df.columns:
                    datetime_col = col
                    break
            
            if datetime_col:
                df['datetime'] = pd.to_datetime(df[datetime_col])
            else:
                print(f"⚠️ {file.name}: Coluna de datetime não encontrada")
                # Tentar criar baseado no nome do arquivo
                year_match = str(file.name)
                if any(year in year_match for year in ['2020', '2021', '2022', '2023', '2024']):
                    year = int([y for y in ['2020', '2021', '2022', '2023', '2024'] if y in year_match][0])
                    df['datetime'] = pd.date_range(f'{year}-01-01', periods=len(df), freq='H')
            
            dfs.append(df)
            print(f"✅ Carregado: {file.name} - {df.shape}")
            
        except Exception as e:
            print(f"❌ Erro em {file.name}: {e}")
    
    if not dfs:
        return None
    
    # Combinar DataFrames
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Remover duplicatas se temos datetime
    if 'datetime' in df_combined.columns:
        df_combined = df_combined.drop_duplicates(subset=['datetime']).sort_values('datetime')
    
    # Salvar processado
    output_file = base_path / "data/processed/openmeteo_historical_weather.csv"
    df_combined.to_csv(output_file, index=False)
    
    print(f"✅ Salvo: {output_file}")
    print(f"📊 Shape final: {df_combined.shape}")
    
    return df_combined

def process_inmet_data(base_path):
    """Processar dados INMET"""
    print("\n🔄 Processando dados INMET...")
    
    inmet_dir = base_path / "data/raw/INMET"
    if not inmet_dir.exists():
        print("❌ Diretório INMET não encontrado")
        return None
    
    # Procurar arquivos CSV (maiúsculo e minúsculo)
    csv_files = list(inmet_dir.glob("*.csv")) + list(inmet_dir.glob("*.CSV"))
    
    if not csv_files:
        print("❌ Nenhum arquivo INMET encontrado")
        return None
    
    print(f"📁 Encontrados {len(csv_files)} arquivos INMET")
    
    # Combinar múltiplos arquivos INMET
    all_dfs = []
    
    for file in csv_files[:15]:  # Aumentar para 15 arquivos
        try:
            # INMET usa encoding latin-1, separador ; e tem cabeçalho de 8 linhas
            df = pd.read_csv(file, encoding='latin-1', sep=';', skiprows=8, 
                           decimal=',', na_values=['', ' ', 'null', 'NULL', '-9999', '-9999.0'])
            
            # Verificar se conseguiu carregar dados válidos
            if df.shape[0] > 0 and df.shape[1] > 3:
                print(f"✅ Carregado: {file.name} - {df.shape}")
            else:
                print(f"⚠️ Dados insuficientes: {file.name} - {df.shape}")
                continue
            
            # Limpar o DataFrame
            df = df.dropna(how='all')  # Remover linhas completamente vazias
            df = df.dropna(axis=1, how='all')  # Remover colunas completamente vazias
            
            # Remover colunas unnamed
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            if df.empty:
                print(f"⚠️ DataFrame vazio após limpeza: {file.name}")
                continue
            
            # Extrair ano do nome do arquivo
            file_name = str(file.name)
            year = 2020  # padrão
            for y in range(2000, 2026):
                if str(y) in file_name:
                    year = y
                    break
            
            # Identificar coluna de data
            date_col = None
            possible_date_cols = ['Data', 'DATA (YYYY-MM-DD)', 'data']
            
            for col in df.columns:
                if col in possible_date_cols or 'data' in col.lower():
                    date_col = col
                    break
            
            if not date_col and len(df.columns) > 0:
                # A primeira coluna normalmente é a data
                date_col = df.columns[0]
            
            if date_col and date_col in df.columns:
                try:
                    # Tentar diferentes formatos de data
                    if 'YYYY-MM-DD' in date_col:
                        df['datetime'] = pd.to_datetime(df[date_col], format='%Y-%m-%d', errors='coerce')
                    else:
                        df['datetime'] = pd.to_datetime(df[date_col], format='%Y/%m/%d', errors='coerce')
                    
                    # Se tem coluna de hora, tentar combinar
                    hora_col = None
                    for col in df.columns:
                        if 'hora' in col.lower() and 'utc' in col.lower():
                            hora_col = col
                            break
                    
                    if hora_col and hora_col in df.columns:
                        # Converter hora UTC para datetime
                        df['hora_str'] = df[hora_col].astype(str).str.replace(' UTC', '').str.replace('UTC', '').str.zfill(4)
                        df['hora_time'] = pd.to_datetime(df['hora_str'], format='%H%M', errors='coerce').dt.time
                        
                        # Combinar data e hora
                        valid_dates = df['datetime'].notna()
                        valid_times = df['hora_time'].notna()
                        valid_both = valid_dates & valid_times
                        
                        if valid_both.sum() > 0:
                            df.loc[valid_both, 'datetime'] = pd.to_datetime(
                                df.loc[valid_both, 'datetime'].dt.date.astype(str) + ' ' + 
                                df.loc[valid_both, 'hora_time'].astype(str), 
                                errors='coerce'
                            )
                    
                    valid_count = df['datetime'].notna().sum()
                    print(f"✅ Data processada - período válido: {valid_count}/{len(df)} registros")
                    
                except Exception as e:
                    print(f"⚠️ Erro convertendo data, criando sequencial: {str(e)[:50]}...")
                    df['datetime'] = pd.date_range(f'{year}-01-01', periods=len(df), freq='H')
            else:
                print("⚠️ Coluna de data não encontrada, criando sequencial")
                df['datetime'] = pd.date_range(f'{year}-01-01', periods=len(df), freq='H')
            
            # Padronizar nomes de colunas principais antes de combinar
            standardized_cols = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'temperatura do ar' in col_lower and 'bulbo seco' in col_lower:
                    standardized_cols[col] = 'temperatura'
                elif 'precipitação total' in col_lower or 'precipita' in col_lower:
                    standardized_cols[col] = 'precipitacao'  
                elif 'umidade relativa do ar' in col_lower and 'horaria' in col_lower:
                    standardized_cols[col] = 'umidade'
                elif 'pressao atmosferica' in col_lower and 'estacao' in col_lower:
                    standardized_cols[col] = 'pressao'
                elif 'vento' in col_lower and 'velocidade' in col_lower:
                    standardized_cols[col] = 'vento_velocidade'
                elif 'vento' in col_lower and ('direção' in col_lower or 'direcao' in col_lower):
                    standardized_cols[col] = 'vento_direcao'
            
            df = df.rename(columns=standardized_cols)
            
            # Adicionar metadados
            df['ano_arquivo'] = year
            df['fonte_arquivo'] = file.name
            
            # Selecionar apenas colunas essenciais para evitar conflitos
            essential_cols = ['datetime', 'temperatura', 'precipitacao', 'umidade', 'pressao', 
                            'vento_velocidade', 'vento_direcao', 'ano_arquivo', 'fonte_arquivo']
            
            available_cols = [col for col in essential_cols if col in df.columns]
            df_clean = df[available_cols].copy()
            
            all_dfs.append(df_clean)
            
        except Exception as e:
            print(f"❌ Erro processando {file.name}: {str(e)[:100]}...")
    
    if not all_dfs:
        print("❌ Nenhum arquivo INMET processado com sucesso")
        return None
    
    print(f"📊 Processados {len(all_dfs)} arquivos com sucesso")
    
    # Combinar todos os DataFrames
    df_combined = pd.concat(all_dfs, ignore_index=True, sort=False)
    
    # Remover duplicatas se temos datetime
    if 'datetime' in df_combined.columns:
        initial_size = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['datetime']).sort_values('datetime')
        removed = initial_size - len(df_combined)
        if removed > 0:
            print(f"🔄 Removidas {removed} duplicatas")
    
    # Converter valores -9999 restantes para NaN
    numeric_cols = ['temperatura', 'precipitacao', 'umidade', 'pressao', 'vento_velocidade', 'vento_direcao']
    for col in numeric_cols:
        if col in df_combined.columns:
            df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
            # Converter -9999 para NaN
            df_combined.loc[df_combined[col] == -9999, col] = pd.NA
            df_combined.loc[df_combined[col] < -999, col] = pd.NA  # Outros valores extremos
    
    # Salvar processado
    output_file = base_path / "data/processed/dados_inmet_processados.csv"
    df_combined.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"✅ Salvo: {output_file}")
    print(f"📊 Shape final: {df_combined.shape}")
    print(f"📋 Colunas finais: {list(df_combined.columns)}")
    
    if 'datetime' in df_combined.columns:
        valid_dates = df_combined['datetime'].notna()
        if valid_dates.sum() > 0:
            print(f"🗓️ Período: {df_combined.loc[valid_dates, 'datetime'].min()} até {df_combined.loc[valid_dates, 'datetime'].max()}")
        print(f"📅 Registros com data válida: {valid_dates.sum()}/{len(df_combined)}")
    
    # Estatísticas de qualidade dos dados
    print("\n📈 Qualidade dos dados:")
    for col in numeric_cols:
        if col in df_combined.columns:
            valid_count = df_combined[col].notna().sum()
            total_count = len(df_combined)
            pct = (valid_count / total_count) * 100
            print(f"  • {col}: {valid_count}/{total_count} ({pct:.1f}%) valores válidos")
    
    return df_combined

def create_atmospheric_features(df_forecast, base_path):
    """Criar features atmosféricas básicas"""
    print("\n🧮 Criando features atmosféricas...")
    
    if df_forecast is None:
        print("❌ Dados forecast não disponíveis")
        return None
    
    df_features = df_forecast.copy()
    
    # Identificar colunas de níveis de pressão
    pressure_levels = ['850hPa', '500hPa', '700hPa', '1000hPa']
    temp_cols = []
    
    for level in pressure_levels:
        # Diferentes formatos possíveis
        possible_cols = [
            f"temperature_{level}",
            f"temp_{level}",
            f"T_{level}",
            f"temperature{level}"
        ]
        
        for temp_col in possible_cols:
            if temp_col in df_features.columns:
                temp_cols.append(temp_col)
                break
    
    print(f"🌡️ Colunas de temperatura encontradas: {temp_cols}")
    
    # Criar gradiente térmico se temos pelo menos 2 níveis
    if len(temp_cols) >= 2:
        # Procurar especificamente 850hPa e 500hPa
        temp_850 = None
        temp_500 = None
        
        for col in temp_cols:
            if '850' in col:
                temp_850 = col
            elif '500' in col:
                temp_500 = col
        
        if temp_850 and temp_500:
            df_features['thermal_gradient_850_500'] = (
                df_features[temp_850] - df_features[temp_500]
            )
            print("✅ Gradiente térmico 850-500hPa criado")
    
    # Features temporais básicas
    if 'datetime' in df_features.columns:
        df_features['hour'] = pd.to_datetime(df_features['datetime']).dt.hour
        df_features['day_of_year'] = pd.to_datetime(df_features['datetime']).dt.dayofyear
        df_features['month'] = pd.to_datetime(df_features['datetime']).dt.month
        df_features['weekday'] = pd.to_datetime(df_features['datetime']).dt.weekday
        print("✅ Features temporais criadas")
    
    # Salvar features
    output_file = base_path / "data/processed/atmospheric_features_149vars.csv"
    df_features.to_csv(output_file, index=False)
    
    print(f"✅ Features salvas: {output_file}")
    print(f"📊 Total de features: {df_features.shape[1]}")
    
    return df_features

def create_surface_features(df_weather, base_path):
    """Criar features de superfície"""
    print("\n🌍 Criando features de superfície...")
    
    if df_weather is None:
        print("❌ Dados weather não disponíveis")
        return None
    
    df_surface = df_weather.copy()
    
    # Features temporais
    if 'datetime' in df_surface.columns:
        df_surface['hour'] = pd.to_datetime(df_surface['datetime']).dt.hour
        df_surface['day_of_year'] = pd.to_datetime(df_surface['datetime']).dt.dayofyear
        df_surface['month'] = pd.to_datetime(df_surface['datetime']).dt.month
        df_surface['weekday'] = pd.to_datetime(df_surface['datetime']).dt.weekday
        print("✅ Features temporais de superfície criadas")
    
    # Features meteorológicas derivadas se existem as colunas
    if 'temperature_2m' in df_surface.columns and 'relative_humidity_2m' in df_surface.columns:
        # Índice de calor simplificado
        df_surface['heat_index'] = df_surface['temperature_2m'] * (1 + df_surface['relative_humidity_2m'] / 100)
        print("✅ Índice de calor criado")
    
    # Salvar features de superfície
    output_file = base_path / "data/processed/surface_features_25vars.csv"
    df_surface.to_csv(output_file, index=False)
    
    print(f"✅ Features de superfície salvas: {output_file}")
    print(f"📊 Total de features: {df_surface.shape[1]}")
    
    return df_surface

def create_summary_report(df_forecast, df_weather, df_inmet, base_path):
    """Criar relatório resumo"""
    print("\n📋 Criando relatório resumo...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "data_sources": {},
        "total_features": 0,
        "date_ranges": {},
        "processing_status": "completed"
    }
    
    if df_forecast is not None:
        report["data_sources"]["openmeteo_forecast"] = {
            "shape": list(df_forecast.shape),
            "features": df_forecast.shape[1],
            "date_range": [
                str(df_forecast['datetime'].min()) if 'datetime' in df_forecast.columns else None,
                str(df_forecast['datetime'].max()) if 'datetime' in df_forecast.columns else None
            ],
            "columns_sample": list(df_forecast.columns[:10])
        }
        report["total_features"] += df_forecast.shape[1]
    
    if df_weather is not None:
        report["data_sources"]["openmeteo_weather"] = {
            "shape": list(df_weather.shape),
            "features": df_weather.shape[1],
            "date_range": [
                str(df_weather['datetime'].min()) if 'datetime' in df_weather.columns else None,
                str(df_weather['datetime'].max()) if 'datetime' in df_weather.columns else None
            ],
            "columns_sample": list(df_weather.columns[:10])
        }
    
    if df_inmet is not None:
        report["data_sources"]["inmet"] = {
            "shape": list(df_inmet.shape),
            "features": df_inmet.shape[1],
            "date_range": [
                str(df_inmet['datetime'].min()) if 'datetime' in df_inmet.columns else None,
                str(df_inmet['datetime'].max()) if 'datetime' in df_inmet.columns else None
            ],
            "columns_sample": list(df_inmet.columns[:10])
        }
    
    # Salvar relatório
    report_file = base_path / "data/analysis/processing_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Relatório salvo: {report_file}")
    return report

def main():
    """Executar pipeline completo"""
    print("🚀 INICIANDO PROCESSAMENTO DOS DADOS EXISTENTES")
    print("=" * 50)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Setup
    base_path = setup_directories()
    
    # 2. Processar dados Open-Meteo
    df_forecast = process_openmeteo_forecast(base_path)
    df_weather = process_openmeteo_weather(base_path)
    
    # 3. Processar dados INMET
    df_inmet = process_inmet_data(base_path)
    
    # 4. Criar features
    df_features = create_atmospheric_features(df_forecast, base_path)
    df_surface = create_surface_features(df_weather, base_path)
    
    # 5. Relatório
    report = create_summary_report(df_forecast, df_weather, df_inmet, base_path)
    
    print("\n🎯 PROCESSAMENTO CONCLUÍDO!")
    print("=" * 30)
    print(f"✅ Dados processados e salvos em data/processed/")
    print(f"📊 Total de features disponíveis: {report.get('total_features', 0)}")
    print(f"📋 Relatório: data/analysis/processing_report.json")
    
    # Resumo dos dados processados
    processed_count = len([ds for ds in report['data_sources'].values() if ds])
    if processed_count > 0:
        print(f"\n📈 Fontes de dados processadas: {processed_count}")
        for source, data in report['data_sources'].items():
            if data:
                print(f"   • {source}: {data['shape'][0]:,} registros, {data['features']} features")
    
    print("\n🔗 Próximos passos:")
    print("   1. Execute: notebooks/1_exploratory_analysis.ipynb")
    print("   2. Execute: notebooks/2_model_training.ipynb")

if __name__ == "__main__":
    main() 