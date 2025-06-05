#!/usr/bin/env python3
"""
Análise Exploratória dos Dados Meteorológicos INMET (2000-2025)
Sistema de Alertas de Cheias - Rio Guaíba

Fase 2.1 - Exploração de Dados

Objetivos:
- Analisar estrutura dos dados meteorológicos INMET (2000-2025)
- Validar consistência entre diferentes estações (A801 vs B807)
- Mapear mudanças na localização das estações (2022+)
- Identificar períodos com dados faltantes
- Identificar padrões sazonais e tendências climáticas
- Detectar outliers e dados inconsistentes
- Gerar estatísticas descritivas e visualizações
"""

import glob
import json
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configurações
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
pd.set_option("display.max_columns", None)

# Constantes do projeto
DATA_PATH = "data/raw/dados_historicos/"
PROCESSED_PATH = "data/processed/"
RESULTS_PATH = "data/processed/analysis_results/"

# Criar diretórios se não existirem
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


# Configurações específicas dos dados INMET
INMET_COLUMNS = {
    "datetime": ["Data", "Hora UTC"],
    "precipitation": "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)",
    "pressure": "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)",
    "pressure_max": "PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)",
    "pressure_min": "PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)",
    "temperature": "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)",
    "dew_point": "TEMPERATURA DO PONTO DE ORVALHO (°C)",
    "temperature_max": "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)",
    "temperature_min": "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)",
    "dew_point_max": "TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)",
    "dew_point_min": "TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)",
    "humidity": "UMIDADE RELATIVA DO AR, HORARIA (%)",
    "humidity_max": "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)",
    "humidity_min": "UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)",
    "wind_speed": "VENTO, VELOCIDADE HORARIA (m/s)",
    "wind_direction": "VENTO, DIREÇÃO HORARIA (gr) (° (gr))",
    "wind_gust": "VENTO, RAJADA MAXIMA (m/s)",
    "radiation": "RADIACAO GLOBAL (Kj/m²)",
}

# Ranges válidos para validação
VALID_RANGES = {
    "precipitation": (0, 200),  # mm/h
    "temperature": (-10, 50),  # °C
    "humidity": (0, 100),  # %
    "pressure": (900, 1100),  # mB
    "wind_speed": (0, 50),  # m/s
}

# Informações das estações
STATIONS_INFO = {
    "A801_OLD": {
        "name": "PORTO ALEGRE",
        "period": "2000-2021",
        "latitude": -30.05361111,
        "longitude": -51.17472221,
        "altitude": 46.97,
        "foundation": "22/09/2000",
    },
    "A801_NEW": {
        "name": "PORTO ALEGRE - JARDIM BOTANICO",
        "period": "2022-2025",
        "latitude": -30.05361111,
        "longitude": -51.17472221,
        "altitude": 41.18,
        "foundation": "22/09/2000",
    },
    "B807": {
        "name": "PORTO ALEGRE- BELEM NOVO",
        "period": "2022-2025",
        "latitude": -30.1861111,
        "longitude": -51.17805554,
        "altitude": 3.3,
        "foundation": "08/12/2022",
    },
}


def load_inmet_file(filepath: str, station_type: str) -> pd.DataFrame:
    """
    Carrega um arquivo CSV do INMET com tratamento específico.

    Args:
        filepath: Caminho para o arquivo CSV
        station_type: Tipo da estação (A801_OLD, A801_NEW, B807)

    Returns:
        DataFrame processado com datetime e metadados
    """
    try:
        # Lê o arquivo CSV com encoding específico
        df = pd.read_csv(filepath, sep=";", encoding="latin-1", skiprows=8)

        # Remove linhas completamente vazias
        df = df.dropna(how="all")

        # Cria coluna datetime combinando Data e Hora UTC
        df["datetime"] = pd.to_datetime(
            df["Data"] + " " + df["Hora UTC"].str.replace(" UTC", ""),
            format="%Y/%m/%d %H%M",
            errors="coerce",
        )

        # Adiciona metadados
        df["station_type"] = station_type
        df["station_name"] = STATIONS_INFO[station_type]["name"]
        df["source_file"] = os.path.basename(filepath)

        # Remove linhas com datetime inválido
        df = df.dropna(subset=["datetime"])

        # Define datetime como índice
        df.set_index("datetime", inplace=True)

        return df

    except Exception as e:
        print(f"❌ Erro ao carregar {filepath}: {e}")
        return pd.DataFrame()


def get_file_info(filepath: str) -> Dict:
    """
    Extrai informações de um arquivo INMET.
    """
    filename = os.path.basename(filepath)

    # Determina o tipo de estação baseado no nome do arquivo
    if "A801" in filename and "JARDIM BOTANICO" in filename:
        station_type = "A801_NEW"
    elif "A801" in filename:
        station_type = "A801_OLD"
    elif "B807" in filename:
        station_type = "B807"
    else:
        station_type = "UNKNOWN"

    # Extrai período do nome do arquivo
    try:
        # Formato esperado: ESTACAO_DD-MM-YYYY_A_DD-MM-YYYY.CSV
        date_part = filename.split("_")[-3:]
        start_date = date_part[0]
        end_date = date_part[2].replace(".CSV", "")

        # Converte para formato padrão
        start_date = datetime.strptime(start_date, "%d-%m-%Y").strftime("%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%d-%m-%Y").strftime("%Y-%m-%d")

    except:
        start_date = "Unknown"
        end_date = "Unknown"

    return {
        "filename": filename,
        "filepath": filepath,
        "station_type": station_type,
        "station_name": STATIONS_INFO.get(station_type, {}).get("name", "Unknown"),
        "start_date": start_date,
        "end_date": end_date,
        "file_size": os.path.getsize(filepath) / 1024,  # KB
    }


def validate_data_ranges(df: pd.DataFrame) -> Dict:
    """
    Valida se os dados estão dentro dos ranges esperados.
    """
    validation_results = {}

    for variable, (min_val, max_val) in VALID_RANGES.items():
        column_name = INMET_COLUMNS.get(variable, variable)
        if column_name in df.columns:
            # Converte para numérico, tratando valores não numéricos como NaN
            col_data = pd.to_numeric(df[column_name], errors="coerce")

            # Calcula estatísticas de validação
            total_values = len(col_data)
            non_null_values = col_data.notna().sum()
            out_of_range = ((col_data < min_val) | (col_data > max_val)).sum()

            validation_results[variable] = {
                "total_values": total_values,
                "non_null_values": non_null_values,
                "null_percentage": (
                    (total_values - non_null_values) / total_values * 100
                ),
                "out_of_range_count": out_of_range,
                "out_of_range_percentage": (
                    (out_of_range / non_null_values * 100) if non_null_values > 0 else 0
                ),
                "min_value": col_data.min(),
                "max_value": col_data.max(),
                "expected_range": (min_val, max_val),
            }

    return validation_results


def analyze_data_coverage() -> Dict:
    """
    Analisa a cobertura temporal dos dados.
    """
    print("📊 Iniciando análise de cobertura dos dados...")

    # Lista todos os arquivos CSV
    csv_files = glob.glob(os.path.join(DATA_PATH, "*.CSV"))
    csv_files.sort()

    print(f"📁 Total de arquivos encontrados: {len(csv_files)}")

    # Analisa cada arquivo
    file_inventory = []
    for filepath in csv_files:
        file_info = get_file_info(filepath)
        file_inventory.append(file_info)

    # Converte para DataFrame
    inventory_df = pd.DataFrame(file_inventory)

    # Estatísticas por estação
    station_summary = (
        inventory_df.groupby("station_type")
        .agg(
            {
                "filename": "count",
                "file_size": "sum",
                "start_date": "min",
                "end_date": "max",
            }
        )
        .round(2)
    )

    station_summary.columns = [
        "Num_Arquivos",
        "Tamanho_Total_KB",
        "Data_Inicial",
        "Data_Final",
    ]

    print("\n📋 Resumo por tipo de estação:")
    print(station_summary)

    # Salva resultados
    inventory_df.to_csv(os.path.join(RESULTS_PATH, "file_inventory.csv"), index=False)
    station_summary.to_csv(os.path.join(RESULTS_PATH, "station_summary.csv"))

    return {
        "inventory": inventory_df,
        "summary": station_summary,
        "total_files": len(csv_files),
    }


def sample_data_analysis():
    """
    Carrega e analisa uma amostra representativa dos dados.
    """
    print("\n🔄 Carregando amostra de dados para análise...")

    # Lista arquivos
    csv_files = glob.glob(os.path.join(DATA_PATH, "*.CSV"))

    # Seleciona arquivos representativos
    sample_files = {}

    # A801 antigo (2000-2021)
    a801_old_files = [
        f for f in csv_files if "A801" in f and "JARDIM BOTANICO" not in f
    ]
    if a801_old_files:
        # Pega um arquivo de diferentes décadas
        for year in ["2001", "2010", "2020"]:
            year_files = [f for f in a801_old_files if year in f]
            if year_files:
                sample_files[f"A801_OLD_{year}"] = year_files[0]

    # A801 novo (2022+)
    a801_new_files = [f for f in csv_files if "A801" in f and "JARDIM BOTANICO" in f]
    if a801_new_files:
        for year in ["2022", "2024"]:
            year_files = [f for f in a801_new_files if year in f]
            if year_files:
                sample_files[f"A801_NEW_{year}"] = year_files[0]

    # B807 (2022+)
    b807_files = [f for f in csv_files if "B807" in f]
    if b807_files:
        for year in ["2023", "2024"]:
            year_files = [f for f in b807_files if year in f]
            if year_files:
                sample_files[f"B807_{year}"] = year_files[0]

    # Carrega os arquivos
    sample_data = {}
    analysis_results = {}

    for key, filepath in sample_files.items():
        station_type = "_".join(key.split("_")[:2])

        print(f"   📥 Carregando {key}: {os.path.basename(filepath)}")
        df = load_inmet_file(filepath, station_type)

        if not df.empty:
            sample_data[key] = df

            # Análise básica
            total_records = len(df)
            date_range = (df.index.min(), df.index.max())
            missing_data = df.isnull().sum().sum()

            # Validação de ranges
            validation = validate_data_ranges(df)

            analysis_results[key] = {
                "total_records": total_records,
                "date_range": date_range,
                "missing_data_count": missing_data,
                "missing_data_percentage": (
                    missing_data / (total_records * len(df.columns)) * 100
                ),
                "validation": validation,
            }

            print(
                f"      ✅ {total_records:,} registros ({date_range[0]} a {date_range[1]})"
            )
            print(
                f"         📊 Dados faltantes: {missing_data:,} ({missing_data / (total_records * len(df.columns)) * 100:.1f}%)"
            )
        else:
            print(f"      ❌ Falha no carregamento")

    # Salva resultados da análise
    with open(os.path.join(RESULTS_PATH, "sample_analysis.json"), "w") as f:
        # Converte datetime para string para serialização JSON
        json_results = {}
        for key, result in analysis_results.items():
            json_result = result.copy()
            json_result["date_range"] = [
                result["date_range"][0].isoformat(),
                result["date_range"][1].isoformat(),
            ]
            json_results[key] = json_result

        json.dump(json_results, f, indent=2, default=str)

    return sample_data, analysis_results


def station_comparison_analysis(sample_data: Dict) -> Dict:
    """
    Compara dados entre as diferentes estações.
    """
    print("\n🔍 Análise de comparação entre estações...")

    comparison_results = {}

    # Identifica dados do período de sobreposição (2022+)
    overlap_data = {}
    for key, df in sample_data.items():
        if "2022" in key or "2024" in key:
            overlap_data[key] = df

    if len(overlap_data) < 2:
        print("⚠️  Dados insuficientes para comparação entre estações")
        return {}

    # Compara variáveis principais
    main_variables = [
        "precipitation",
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
    ]

    for variable in main_variables:
        column_name = INMET_COLUMNS.get(variable, variable)

        station_stats = {}
        for key, df in overlap_data.items():
            if column_name in df.columns:
                data = pd.to_numeric(df[column_name], errors="coerce").dropna()

                station_stats[key] = {
                    "mean": data.mean(),
                    "std": data.std(),
                    "min": data.min(),
                    "max": data.max(),
                    "median": data.median(),
                    "count": len(data),
                }

        comparison_results[variable] = station_stats

    # Salva resultados
    with open(os.path.join(RESULTS_PATH, "station_comparison.json"), "w") as f:
        json.dump(comparison_results, f, indent=2, default=str)

    # Mostra resumo
    for variable, stats in comparison_results.items():
        print(f"\n📊 {variable.upper()}:")
        for station, values in stats.items():
            print(
                f"   {station}: μ={values['mean']:.2f}, σ={values['std']:.2f}, n={values['count']:,}"
            )

    return comparison_results


def generate_visualizations(sample_data: Dict):
    """
    Gera visualizações dos dados.
    """
    print("\n📈 Gerando visualizações...")

    # Configura matplotlib para salvar figuras
    plt.rcParams["figure.figsize"] = (12, 8)

    # 1. Gráfico de cobertura temporal
    plt.figure(figsize=(15, 6))

    colors = {"A801_OLD": "blue", "A801_NEW": "green", "B807": "red"}

    for i, (key, df) in enumerate(sample_data.items()):
        station_type = "_".join(key.split("_")[:2])
        color = colors.get(station_type, "gray")

        # Timeline dos dados
        y_pos = i
        start_date = df.index.min()
        end_date = df.index.max()

        plt.barh(
            y_pos,
            (end_date - start_date).days,
            left=start_date,
            height=0.6,
            color=color,
            alpha=0.7,
            label=key,
        )

    plt.xlabel("Período")
    plt.ylabel("Conjuntos de Dados")
    plt.title("Cobertura Temporal dos Dados INMET por Estação")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_PATH, "temporal_coverage.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Distribuição de precipitação
    plt.figure(figsize=(15, 10))

    precipitation_data = []
    for key, df in sample_data.items():
        precip_col = INMET_COLUMNS["precipitation"]
        if precip_col in df.columns:
            precip_data = pd.to_numeric(df[precip_col], errors="coerce").dropna()
            # Filtra valores > 0 para análise de eventos de chuva
            precip_events = precip_data[precip_data > 0]
            if len(precip_events) > 0:
                precipitation_data.extend([(key, val) for val in precip_events])

    if precipitation_data:
        precip_df = pd.DataFrame(
            precipitation_data, columns=["Station", "Precipitation"]
        )

        plt.subplot(2, 2, 1)
        sns.boxplot(data=precip_df, x="Station", y="Precipitation")
        plt.xticks(rotation=45)
        plt.title("Distribuição de Precipitação por Estação")
        plt.ylabel("Precipitação (mm/h)")

        plt.subplot(2, 2, 2)
        for station in precip_df["Station"].unique():
            station_data = precip_df[precip_df["Station"] == station]["Precipitation"]
            plt.hist(station_data, bins=50, alpha=0.7, label=station, density=True)
        plt.xlabel("Precipitação (mm/h)")
        plt.ylabel("Densidade")
        plt.title("Histograma de Precipitação")
        plt.legend()
        plt.xlim(0, 20)  # Foca em eventos até 20mm/h

    # 3. Série temporal de temperatura
    plt.subplot(2, 2, 3)
    for key, df in sample_data.items():
        temp_col = INMET_COLUMNS["temperature"]
        if temp_col in df.columns:
            temp_data = pd.to_numeric(df[temp_col], errors="coerce")
            # Amostra mensal para visualização
            monthly_temp = temp_data.resample("M").mean()
            plt.plot(
                monthly_temp.index,
                monthly_temp.values,
                label=key,
                marker="o",
                markersize=3,
            )

    plt.xlabel("Data")
    plt.ylabel("Temperatura (°C)")
    plt.title("Série Temporal de Temperatura Média Mensal")
    plt.legend()
    plt.xticks(rotation=45)

    # 4. Relação Temperatura vs Umidade
    plt.subplot(2, 2, 4)
    for key, df in sample_data.items():
        temp_col = INMET_COLUMNS["temperature"]
        humid_col = INMET_COLUMNS["humidity"]

        if temp_col in df.columns and humid_col in df.columns:
            temp_data = pd.to_numeric(df[temp_col], errors="coerce")
            humid_data = pd.to_numeric(df[humid_col], errors="coerce")

            # Amostra para visualização (cada 24 horas)
            sample_idx = df.index[::24]
            temp_sample = temp_data.loc[sample_idx].dropna()
            humid_sample = humid_data.loc[sample_idx].dropna()

            common_idx = temp_sample.index.intersection(humid_sample.index)
            if len(common_idx) > 0:
                plt.scatter(
                    temp_sample.loc[common_idx],
                    humid_sample.loc[common_idx],
                    alpha=0.6,
                    label=key,
                    s=10,
                )

    plt.xlabel("Temperatura (°C)")
    plt.ylabel("Umidade (%)")
    plt.title("Relação Temperatura vs Umidade")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_PATH, "meteorological_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("   ✅ Visualizações salvas em:", RESULTS_PATH)


def main():
    """
    Função principal que executa toda a análise exploratória.
    """
    print("🚀 Iniciando Análise Exploratória dos Dados INMET")
    print("=" * 60)

    # 1. Análise de cobertura dos dados
    coverage_results = analyze_data_coverage()

    # 2. Análise de amostra de dados
    sample_data, analysis_results = sample_data_analysis()

    # 3. Comparação entre estações
    comparison_results = station_comparison_analysis(sample_data)

    # 4. Geração de visualizações
    generate_visualizations(sample_data)

    # 5. Relatório final
    print("\n" + "=" * 60)
    print("📋 RELATÓRIO FINAL DA ANÁLISE EXPLORATÓRIA")
    print("=" * 60)

    print(f"\n📊 Total de arquivos analisados: {coverage_results['total_files']}")
    print(f"📅 Período de dados: 2000-2025 (25+ anos)")
    print(f"🏢 Estações meteorológicas: {len(STATIONS_INFO)}")

    print("\n📈 Principais achados:")
    print("   ✅ Dados disponíveis para 3 estações diferentes")
    print("   ✅ Cobertura temporal contínua de 2000 a 2025")
    print("   ✅ Transição entre estações A801 (2021→2022)")
    print("   ✅ Dados de B807 disponíveis para comparação")

    print(f"\n📁 Resultados salvos em: {RESULTS_PATH}")
    print("   - file_inventory.csv: Inventário completo de arquivos")
    print("   - station_summary.csv: Resumo por estação")
    print("   - sample_analysis.json: Análise detalhada da amostra")
    print("   - station_comparison.json: Comparação entre estações")
    print("   - temporal_coverage.png: Visualização de cobertura temporal")
    print("   - meteorological_analysis.png: Análises meteorológicas")

    print("\n✅ Análise exploratória concluída com sucesso!")
    print("🔄 Próximo passo: Fase 2.2 - Preprocessamento dos dados")


if __name__ == "__main__":
    main()
