#!/usr/bin/env python3
"""
Sistema de Alertas de Cheias - Rio Guaíba
Fase 2.2: Preprocessamento de Dados (Versão Corrigida)

Script para preprocessamento completo dos dados meteorológicos INMET (2000-2025)
Versão corrigida que trata adequadamente o formato específico dos arquivos INMET

Funcionalidades:
- Carregamento robusto de arquivos INMET com metadados
- Padronização de formatos de data e timestamps
- Tratamento de valores missing/nulos
- Normalização e scaling de features
- Feature engineering (variáveis derivadas)
- Unificação de dados entre estações

Autor: Sistema IA
Data: 2025-01-03
"""

import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Configuração de warnings
warnings.filterwarnings("ignore")

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data/processed/preprocessing_fixed.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configurações para o preprocessamento"""

    # Caminhos
    raw_data_path: str = "data/raw/dados_historicos"
    processed_data_path: str = "data/processed"

    # Configurações de imputação
    max_gap_hours: int = 6  # Máximo gap para interpolação
    imputation_method: str = "knn"  # 'mean', 'median', 'knn', 'interpolate'

    # Configurações de normalização
    scaling_method: str = "standard"  # 'standard', 'minmax', 'robust'

    # Configurações de feature engineering
    create_derived_features: bool = True
    temporal_aggregations: List[str] = None

    # Configurações de validação
    min_data_coverage: float = 0.7  # Mínimo 70% de dados válidos

    def __post_init__(self):
        if self.temporal_aggregations is None:
            self.temporal_aggregations = ["3H", "6H", "12H", "24H"]


class INMETDataLoader:
    """Classe especializada para carregar dados INMET"""

    @staticmethod
    def load_inmet_file(file_path: Path) -> Tuple[Dict, pd.DataFrame]:
        """Carrega arquivo INMET com metadados e dados"""
        logger.info(f"Carregando arquivo INMET: {file_path.name}")

        # Ler metadados (primeiras 8 linhas)
        metadata = {}
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                for i in range(8):
                    line = f.readline().strip()
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip().replace(";", "")
        except Exception as e:
            logger.warning(f"Erro ao ler metadados: {e}")

        # Ler dados (a partir da linha 9)
        try:
            # Tentar diferentes encodings
            encodings = ["latin-1", "utf-8", "iso-8859-1", "cp1252"]
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(
                        file_path,
                        sep=";",
                        encoding=encoding,
                        skiprows=8,  # Pular metadados
                        low_memory=False,
                        na_values=["", " ", "null", "NULL", "-", "--", "---", ","],
                    )
                    logger.info(f"✅ Carregado com encoding: {encoding}")
                    break
                except Exception as e:
                    logger.debug(f"Falha com encoding {encoding}: {str(e)[:100]}")
                    continue

            if df is None:
                raise ValueError(
                    "Não foi possível carregar o arquivo com nenhum encoding"
                )

            # Limpar dados
            df = df.dropna(how="all")  # Remover linhas completamente vazias
            df = df.loc[
                :, ~df.columns.str.contains("^Unnamed")
            ]  # Remover colunas unnamed

            logger.info(f"📊 Dados carregados: {df.shape}")
            return metadata, df

        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados: {e}")
            raise


class DataPreprocessorFixed:
    """Classe principal para preprocessamento de dados meteorológicos (versão corrigida)"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.scalers = {}
        self.imputers = {}
        self.feature_columns = []
        self.preprocessing_stats = {}
        self.loader = INMETDataLoader()

        # Criar diretórios necessários
        os.makedirs(self.config.processed_data_path, exist_ok=True)
        os.makedirs(f"{self.config.processed_data_path}/unified", exist_ok=True)
        os.makedirs(f"{self.config.processed_data_path}/features", exist_ok=True)

    def load_raw_data(self) -> Dict[str, Dict]:
        """Carrega todos os dados brutos com metadados"""
        logger.info("🔄 Carregando dados brutos INMET...")

        data_files = {}
        raw_path = Path(self.config.raw_data_path)

        for file_path in raw_path.glob("*.CSV"):
            file_key = file_path.stem

            try:
                metadata, df = self.loader.load_inmet_file(file_path)

                if df is not None and not df.empty:
                    data_files[file_key] = {"metadata": metadata, "data": df}
                    logger.info(f"✅ {file_key}: {df.shape}")
                else:
                    logger.warning(f"⚠️ Arquivo vazio: {file_key}")

            except Exception as e:
                logger.error(f"❌ Falha ao carregar {file_key}: {e}")
                continue

        logger.info(f"✅ Total de arquivos carregados: {len(data_files)}")
        return data_files

    def standardize_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Padroniza formatos de data e timestamps para dados INMET"""
        logger.info("🕐 Padronizando timestamps INMET...")

        df_clean = df.copy()

        # Identificar colunas de data/hora (formato INMET específico)
        date_col = None
        time_col = None

        for col in df_clean.columns:
            col_clean = col.strip()
            if "DATA" in col_clean.upper() and (
                "YYYY" in col_clean.upper() or col_clean == "Data"
            ):
                date_col = col
            elif "HORA" in col_clean.upper() and (
                "UTC" in col_clean.upper() or col_clean == "Hora UTC"
            ):
                time_col = col

        if date_col and time_col:
            logger.info(f"Colunas identificadas - Data: {date_col}, Hora: {time_col}")

            # Criar timestamp combinado (sempre timezone-naive)
            try:
                # Limpar os dados primeiro
                date_values = df_clean[date_col].astype(str).str.strip()
                time_values = df_clean[time_col].astype(str).str.strip()

                # Remover " UTC" do tempo se presente
                time_values = time_values.str.replace(" UTC", "", regex=False)

                # Normalizar formato do tempo (HHMM -> HH:MM)
                time_normalized = time_values.apply(
                    lambda x: f"{x[:2]}:{x[2:]}" if len(x) == 4 and x.isdigit() else x
                )

                # Normalizar formato da data (YYYY/MM/DD -> YYYY-MM-DD)
                date_normalized = date_values.str.replace("/", "-", regex=False)

                timestamp_combined = pd.to_datetime(
                    date_normalized + " " + time_normalized,
                    format="%Y-%m-%d %H:%M",
                    errors="coerce",
                )

                # Garantir que seja timezone-naive
                if timestamp_combined.dt.tz is not None:
                    timestamp_combined = timestamp_combined.dt.tz_localize(None)

                df_clean["timestamp"] = timestamp_combined
                success_count = df_clean["timestamp"].notna().sum()
                success_rate = (success_count / len(df_clean)) * 100

                logger.info(
                    f"✅ Timestamps criados: {success_count}/{len(df_clean)} ({success_rate:.1f}%)"
                )

            except Exception as e:
                logger.warning(f"⚠️ Erro ao criar timestamps: {str(e)}")
                # Fallback: usar apenas data como timestamp
                df_clean["timestamp"] = pd.to_datetime(
                    df_clean[date_col], errors="coerce"
                )
                # Garantir que seja timezone-naive
                if df_clean["timestamp"].dt.tz is not None:
                    df_clean["timestamp"] = df_clean["timestamp"].dt.tz_localize(None)
        else:
            logger.warning(
                f"⚠️ Colunas de data/hora não encontradas. Procurando por 'DATA' e 'HORA'"
            )
            logger.warning(f"Colunas disponíveis: {list(df_clean.columns)}")

        return df_clean

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Padroniza nomes de colunas para dados INMET"""
        logger.info("📝 Padronizando nomes de colunas INMET...")

        df_clean = df.copy()

        # Primeiro, limpar valores especiais de missing data comuns nos dados INMET
        missing_codes = [
            "-9999",
            "-999",
            "999999",
            "null",
            "NULL",
            "",
            " ",
            "---",
            "--",
            "-",
        ]
        for col in df_clean.columns:
            if col not in [
                "timestamp",
                "Data",
                "Hora UTC",
                "DATA (YYYY-MM-DD)",
                "HORA (UTC)",
            ]:
                # Converter códigos de missing para NaN
                df_clean[col] = df_clean[col].replace(missing_codes, np.nan)

                # Tentar converter para numérico se possível
                df_clean[col] = pd.to_numeric(df_clean[col], errors="ignore")

        # Mapeamento específico para colunas INMET
        column_mapping = {
            "precipitação total, horário (mm)": "precipitacao_mm",
            "precipitação total, horário (mm)": "precipitacao_mm",
            "pressao atmosferica ao nivel da estacao, horaria (mb)": "pressao_mb",
            "pressão atmosférica ao nível da estação, horária (mb)": "pressao_mb",
            "temperatura do ar - bulbo seco, horaria (°c)": "temperatura_c",
            "temperatura do ar - bulbo seco, horaria (°c)": "temperatura_c",
            "umidade relativa do ar, horaria (%)": "umidade_relativa",
            "velocidade do vento (m/s)": "vento_velocidade_ms",
            "vento, velocidade horaria (m/s)": "vento_velocidade_ms",
            "direção do vento (graus)": "vento_direcao_graus",
            "vento, direção horaria (gr) (° (gr))": "vento_direcao_graus",
            "radiacao global (kj/m²)": "radiacao_global_kjm2",
            "radiacao global (kj/m²)": "radiacao_global_kjm2",
            "temperatura do ponto de orvalho (°c)": "temperatura_orvalho_c",
            "temperatura do ponto de orvalho (°c)": "temperatura_orvalho_c",
            "vento, rajada maxima (m/s)": "vento_rajada_ms",
        }

        # Aplicar mapeamento (case insensitive)
        new_columns = {}
        for col in df_clean.columns:
            col_clean = col.lower().strip()

            # Buscar mapeamento direto
            if col_clean in column_mapping:
                new_columns[col] = column_mapping[col_clean]
            else:
                # Buscar por palavras-chave
                new_name = col_clean
                if "precipita" in col_clean and "mm" in col_clean:
                    new_name = "precipitacao_mm"
                elif "pressao" in col_clean and "mb" in col_clean:
                    new_name = "pressao_mb"
                elif "temperatura" in col_clean and "bulbo" in col_clean:
                    new_name = "temperatura_c"
                elif "umidade" in col_clean and "%" in col_clean:
                    new_name = "umidade_relativa"
                elif "vento" in col_clean and "velocidade" in col_clean:
                    new_name = "vento_velocidade_ms"
                elif "vento" in col_clean and "dire" in col_clean:
                    new_name = "vento_direcao_graus"
                elif "radiacao" in col_clean:
                    new_name = "radiacao_global_kjm2"
                elif "orvalho" in col_clean:
                    new_name = "temperatura_orvalho_c"
                elif "rajada" in col_clean:
                    new_name = "vento_rajada_ms"

                # Limpar nome
                new_name = (
                    new_name.replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("°", "")
                    .replace("²", "2")
                    .replace("/", "_")
                    .replace("-", "_")
                    .replace(",", "")
                    .replace("°", "")
                )

                new_columns[col] = new_name

        # Renomear colunas
        df_clean = df_clean.rename(columns=new_columns)

        # Log das mudanças
        changes = [(old, new) for old, new in new_columns.items() if old != new]
        logger.info(f"✅ {len(changes)} colunas renomeadas")

        return df_clean

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trata valores missing com estratégias específicas para dados meteorológicos"""
        logger.info("🔧 Tratando valores missing...")

        df_clean = df.copy()

        # Identificar colunas numéricas
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()

        # Remover colunas de timestamp se presentes
        numeric_columns = [
            col for col in numeric_columns if "timestamp" not in col.lower()
        ]

        # Estatísticas de missing values
        missing_stats = {}
        for col in numeric_columns:
            missing_count = df_clean[col].isna().sum()
            missing_pct = (missing_count / len(df_clean)) * 100

            # Convert pandas objects to Python scalars
            if isinstance(missing_count, (pd.Series, np.ndarray)):
                count_val = (
                    missing_count.iloc[0]
                    if isinstance(missing_count, pd.Series)
                    else missing_count.item()
                )
            else:
                count_val = int(missing_count)

            if isinstance(missing_pct, (pd.Series, np.ndarray)):
                pct_val = (
                    missing_pct.iloc[0]
                    if isinstance(missing_pct, pd.Series)
                    else missing_pct.item()
                )
            else:
                pct_val = float(missing_pct)

            missing_stats[col] = {"count": count_val, "percentage": pct_val}

        # Contar colunas com missing values
        cols_with_missing = sum(1 for v in missing_stats.values() if v["count"] > 0)
        logger.info(f"Colunas com missing values: {cols_with_missing}")

        # Verificar se timestamp existe
        has_timestamp = "timestamp" in df_clean.columns

        # Estratégias específicas por tipo de variável meteorológica
        for col in numeric_columns:
            missing_pct = missing_stats[col]["percentage"]

            if missing_pct == 0:
                continue

            logger.info(f"Tratando {col}: {missing_pct:.1f}% missing")

            if missing_pct > 80:
                # Muitos valores missing - considerar remoção
                logger.warning(
                    f"⚠️ {col} tem {missing_pct:.1f}% missing - mantendo para análise"
                )
                continue

            # Estratégias específicas por tipo de variável
            if "precipitacao" in col.lower() or "precipita" in col.lower():
                # Precipitação: missing geralmente significa 0
                df_clean[col].fillna(0, inplace=True)
                logger.info(f"✅ {col}: missing preenchido com 0 (sem chuva)")

            elif has_timestamp and missing_pct < 30:
                # Interpolação temporal para outras variáveis
                try:
                    df_clean = df_clean.sort_values("timestamp")
                    df_clean[col] = df_clean[col].interpolate(
                        method="time", limit=self.config.max_gap_hours
                    )
                    logger.info(f"✅ {col}: interpolação temporal aplicada")
                except Exception as e:
                    logger.warning(f"⚠️ Erro na interpolação de {col}: {e}")

            # Imputação para valores restantes
            remaining_missing = df_clean[col].isna().sum()
            if isinstance(remaining_missing, (pd.Series, np.ndarray)):
                remaining_count = (
                    remaining_missing.iloc[0]
                    if isinstance(remaining_missing, pd.Series)
                    else remaining_missing.item()
                )
            else:
                remaining_count = remaining_missing

            if remaining_count > 0:
                if self.config.imputation_method == "mean":
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif self.config.imputation_method == "median":
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)

                logger.info(
                    f"✅ {col}: imputação {self.config.imputation_method} aplicada"
                )

        # Salvar estatísticas
        self.preprocessing_stats["missing_values"] = missing_stats

        return df_clean

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features derivadas para melhorar a capacidade preditiva"""
        logger.info("🔬 Criando features derivadas...")

        df_enhanced = df.copy()

        # Garantir que os dados numéricos estejam corretos antes de criar features
        numeric_columns = [
            "temperatura_c",
            "umidade_relativa",
            "precipitacao_mm",
            "pressao_hpa",
            "vento_velocidade_ms",
            "radiacao_kjm2",
            "pressao_mb",
            "vento_direcao_graus",
            "temperatura_ponto_orvalho_c",
            "temp_max_c",
            "temp_min_c",
            "umidade_max",
            "umidade_min",
        ]

        for col in numeric_columns:
            if col in df_enhanced.columns:
                # Converter strings para NaN e depois para numérico
                df_enhanced[col] = pd.to_numeric(df_enhanced[col], errors="coerce")

        # Verificar se temos colunas essenciais para features derivadas
        has_temperature = (
            "temperatura_c" in df_enhanced.columns
            and df_enhanced["temperatura_c"].notna().any()
        )
        has_pressure = (
            "pressao_mb" in df_enhanced.columns
            and df_enhanced["pressao_mb"].notna().any()
        )
        has_humidity = (
            "umidade_relativa" in df_enhanced.columns
            and df_enhanced["umidade_relativa"].notna().any()
        )
        has_wind = (
            "vento_velocidade_ms" in df_enhanced.columns
            and df_enhanced["vento_velocidade_ms"].notna().any()
        )
        has_precipitation = (
            "precipitacao_mm" in df_enhanced.columns
            and df_enhanced["precipitacao_mm"].notna().any()
        )

        # Features temporais
        if "timestamp" in df_enhanced.columns:
            df_enhanced["hora"] = df_enhanced["timestamp"].dt.hour
            df_enhanced["dia_semana"] = df_enhanced["timestamp"].dt.dayofweek
            df_enhanced["mes"] = df_enhanced["timestamp"].dt.month
            df_enhanced["ano"] = df_enhanced["timestamp"].dt.year

            # Estações do ano (hemisfério sul)
            df_enhanced["estacao"] = df_enhanced["mes"].map(
                {
                    12: "verao",
                    1: "verao",
                    2: "verao",
                    3: "outono",
                    4: "outono",
                    5: "outono",
                    6: "inverno",
                    7: "inverno",
                    8: "inverno",
                    9: "primavera",
                    10: "primavera",
                    11: "primavera",
                }
            )

            # Componentes cíclicas
            df_enhanced["hora_sin"] = np.sin(2 * np.pi * df_enhanced["hora"] / 24)
            df_enhanced["hora_cos"] = np.cos(2 * np.pi * df_enhanced["hora"] / 24)
            df_enhanced["mes_sin"] = np.sin(2 * np.pi * df_enhanced["mes"] / 12)
            df_enhanced["mes_cos"] = np.cos(2 * np.pi * df_enhanced["mes"] / 12)

        logger.info("✅ Features temporais criadas")

        # Features meteorológicas derivadas (somente se dados numéricos estão disponíveis)
        if (
            "temperatura_c" in df_enhanced.columns
            and df_enhanced["temperatura_c"].notna().any()
        ):
            # Índice de conforto térmico
            df_enhanced["indice_calor"] = df_enhanced["temperatura_c"] + (
                df_enhanced["umidade_relativa"] / 100
            ) * (df_enhanced["temperatura_c"] - 14.5)
            logger.info("✅ Índice de calor criado")

        if (
            "vento_velocidade_ms" in df_enhanced.columns
            and "vento_direcao_graus" in df_enhanced.columns
        ):
            # Componentes do vento (u, v)
            wind_rad = np.radians(df_enhanced["vento_direcao_graus"])
            df_enhanced["vento_u"] = -df_enhanced["vento_velocidade_ms"] * np.sin(
                wind_rad
            )
            df_enhanced["vento_v"] = -df_enhanced["vento_velocidade_ms"] * np.cos(
                wind_rad
            )
            logger.info("✅ Componentes do vento criadas")

        # Features de eventos de precipitação
        if "precipitacao_mm" in df_enhanced.columns:
            # Classificação de intensidade de chuva
            df_enhanced["chuva_fraca"] = (df_enhanced["precipitacao_mm"] > 0) & (
                df_enhanced["precipitacao_mm"] <= 2.5
            )
            df_enhanced["chuva_moderada"] = (df_enhanced["precipitacao_mm"] > 2.5) & (
                df_enhanced["precipitacao_mm"] <= 10
            )
            df_enhanced["chuva_forte"] = (df_enhanced["precipitacao_mm"] > 10) & (
                df_enhanced["precipitacao_mm"] <= 50
            )
            df_enhanced["chuva_muito_forte"] = df_enhanced["precipitacao_mm"] > 50

            # Evento de chuva (binário)
            df_enhanced["evento_chuva"] = df_enhanced["precipitacao_mm"] > 0

            logger.info("✅ Features de eventos de chuva criadas")

        # Agregações temporais (rolling windows)
        if "timestamp" in df_enhanced.columns:
            df_enhanced = df_enhanced.sort_values("timestamp")

            # Rolling statistics para precipitação
            if "precipitacao_mm" in df_enhanced.columns:
                for hours in [3, 6, 12, 24]:
                    df_enhanced[f"precipitacao_sum_{hours}h"] = (
                        df_enhanced["precipitacao_mm"]
                        .rolling(window=hours, min_periods=1)
                        .sum()
                    )
                    df_enhanced[f"precipitacao_max_{hours}h"] = (
                        df_enhanced["precipitacao_mm"]
                        .rolling(window=hours, min_periods=1)
                        .max()
                    )

            # Rolling statistics para temperatura
            if "temperatura_c" in df_enhanced.columns:
                for hours in [6, 12, 24]:
                    df_enhanced[f"temperatura_mean_{hours}h"] = (
                        df_enhanced["temperatura_c"]
                        .rolling(window=hours, min_periods=1)
                        .mean()
                    )
                    df_enhanced[f"temperatura_std_{hours}h"] = (
                        df_enhanced["temperatura_c"]
                        .rolling(window=hours, min_periods=1)
                        .std()
                    )

            # Tendência de pressão
            if "pressao_mb" in df_enhanced.columns:
                df_enhanced["pressao_trend_3h"] = (
                    df_enhanced["pressao_mb"]
                    .diff()
                    .rolling(window=3, min_periods=1)
                    .mean()
                )
                df_enhanced["pressao_trend_6h"] = (
                    df_enhanced["pressao_mb"]
                    .diff()
                    .rolling(window=6, min_periods=1)
                    .mean()
                )

            logger.info("✅ Agregações temporais criadas")

        logger.info(
            f"📊 Features criadas: {df_enhanced.shape[1] - df.shape[1]} novas colunas"
        )
        return df_enhanced

    def unify_datasets(self, data_files: Dict[str, Dict]) -> pd.DataFrame:
        """Unifica datasets de diferentes estações e períodos"""
        logger.info("🔗 Unificando datasets...")

        unified_data = []

        for file_key, file_data in data_files.items():
            logger.info(f"Processando: {file_key}")

            metadata = file_data["metadata"]
            df = file_data["data"]

            # Aplicar pipeline de preprocessamento
            df_processed = self.standardize_datetime(df)
            df_processed = self.standardize_columns(df_processed)
            df_processed = self.handle_missing_values(df_processed)

            # Adicionar metadados como colunas
            df_processed["source_file"] = file_key
            df_processed["station_code"] = metadata.get("CODIGO (WMO)", "UNKNOWN")
            df_processed["station_name"] = metadata.get("ESTACAO", "UNKNOWN")
            df_processed["latitude"] = metadata.get("LATITUDE", None)
            df_processed["longitude"] = metadata.get("LONGITUDE", None)
            df_processed["altitude"] = metadata.get("ALTITUDE", None)

            # Identificar estação baseada no código e nome
            if "A801" in file_key:
                if "JARDIM" in file_key.upper():
                    df_processed["station"] = "A801_JARDIM_BOTANICO"
                else:
                    df_processed["station"] = "A801_PORTO_ALEGRE"
            elif "B807" in file_key:
                df_processed["station"] = "B807_BELEM_NOVO"
            else:
                df_processed["station"] = "UNKNOWN"

            # Validar cobertura de dados
            if "timestamp" in df_processed.columns:
                valid_rows = df_processed.dropna(subset=["timestamp"]).shape[0]
                coverage = (
                    valid_rows / df_processed.shape[0]
                    if df_processed.shape[0] > 0
                    else 0
                )

                if coverage >= self.config.min_data_coverage:
                    unified_data.append(df_processed)
                    logger.info(f"✅ {file_key}: {coverage:.1%} cobertura - incluído")
                else:
                    logger.warning(f"⚠️ {file_key}: {coverage:.1%} cobertura - excluído")
            else:
                logger.warning(f"⚠️ {file_key}: sem timestamps válidos - excluído")

        if not unified_data:
            raise ValueError("Nenhum dataset válido encontrado para unificação")

        # Unificar dados com validação de sobreposição
        logger.info(f"📊 Unificando {len(unified_data)} datasets processados...")

        # Reset indices to avoid conflicts during concatenation
        for i, df in enumerate(unified_data):
            df_clean = df.reset_index(drop=True)

            # Ensure no duplicate columns
            if df_clean.columns.duplicated().any():
                logger.warning(f"⚠️ Dataset {i} tem colunas duplicadas, removendo...")
                df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]

            # Ensure all datasets have consistent columns
            if i == 0:
                expected_columns = df_clean.columns.tolist()
            else:
                if not df_clean.columns.equals(expected_columns):
                    logger.warning(
                        f"⚠️ Dataset {i} tem colunas diferentes, alinhando..."
                    )
                    # Reorder columns to match expected order
                    missing_cols = set(expected_columns) - set(df_clean.columns)
                    extra_cols = set(df_clean.columns) - set(expected_columns)

                    if missing_cols:
                        logger.warning(f"Colunas faltantes: {missing_cols}")
                        for col in missing_cols:
                            df_clean[col] = None

                    if extra_cols:
                        logger.warning(f"Colunas extras removidas: {extra_cols}")
                        df_clean = df_clean.drop(columns=list(extra_cols))

                    # Reorder to match expected columns
                    df_clean = df_clean[expected_columns]

            unified_data[i] = df_clean

        # Concatenate all datasets
        unified_df = pd.concat(unified_data, ignore_index=True, sort=False)

        # Handle potential timestamp duplicates by setting timestamp as index and removing duplicates
        if "timestamp" in unified_df.columns:
            # Sort by timestamp first
            # Ensure all timestamps are timezone-naive for consistent comparison
            if pd.api.types.is_datetime64_any_dtype(unified_df["timestamp"]):
                # Convert timezone-aware timestamps to naive (UTC)
                unified_df["timestamp"] = pd.to_datetime(
                    unified_df["timestamp"]
                ).dt.tz_localize(None)

            unified_df = unified_df.sort_values("timestamp")

            # Remove duplicate timestamps (keep first occurrence)
            initial_rows = len(unified_df)
            unified_df = unified_df.drop_duplicates(subset=["timestamp"], keep="first")
            removed_duplicates = initial_rows - len(unified_df)

            if removed_duplicates > 0:
                logger.warning(
                    f"⚠️ Removidas {removed_duplicates} linhas com timestamps duplicados"
                )

        logger.info(f"✅ Dataset unificado criado: {unified_df.shape}")
        logger.info(
            f"📅 Período: {unified_df['timestamp'].min()} → {unified_df['timestamp'].max()}"
        )

        return unified_df

    def create_temporal_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Cria splits temporais para treino/validação/teste"""
        logger.info("✂️ Criando splits temporais...")

        if "timestamp" not in df.columns:
            raise ValueError("Coluna 'timestamp' necessária para splits temporais")

        df_sorted = df.sort_values("timestamp")

        # Definir pontos de corte (preservando sazonalidade)
        total_rows = len(df_sorted)
        train_end = int(total_rows * 0.7)  # 70% treino
        val_end = int(total_rows * 0.85)  # 15% validação, 15% teste

        splits = {
            "train": df_sorted.iloc[:train_end].copy(),
            "validation": df_sorted.iloc[train_end:val_end].copy(),
            "test": df_sorted.iloc[val_end:].copy(),
        }

        # Log das informações dos splits
        for split_name, split_df in splits.items():
            logger.info(f"{split_name.upper()}: {split_df.shape[0]} registros")
            if "timestamp" in split_df.columns:
                logger.info(
                    f"  Período: {split_df['timestamp'].min()} a {split_df['timestamp'].max()}"
                )

        return splits

    def generate_preprocessing_report(
        self, original_data: Dict[str, Dict], processed_data: pd.DataFrame
    ) -> Dict:
        """Gera relatório detalhado do preprocessamento"""
        logger.info("📊 Gerando relatório de preprocessamento...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "original_files": len(original_data),
            "total_original_rows": sum(
                data["data"].shape[0] for data in original_data.values()
            ),
            "processed_rows": processed_data.shape[0],
            "processed_columns": processed_data.shape[1],
            "stations": (
                processed_data["station"].unique().tolist()
                if "station" in processed_data.columns
                else []
            ),
            "date_range": {
                "start": (
                    processed_data["timestamp"].min().isoformat()
                    if "timestamp" in processed_data.columns
                    else None
                ),
                "end": (
                    processed_data["timestamp"].max().isoformat()
                    if "timestamp" in processed_data.columns
                    else None
                ),
            },
            "preprocessing_stats": self.preprocessing_stats,
            "feature_columns": self.feature_columns,
            "data_quality": {},
            "station_metadata": {},
        }

        # Estatísticas de qualidade dos dados
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if processed_data[col].notna().sum() > 0:
                report["data_quality"][col] = {
                    "missing_pct": processed_data[col].isna().sum()
                    / len(processed_data)
                    * 100,
                    "mean": float(processed_data[col].mean()),
                    "std": float(processed_data[col].std()),
                    "min": float(processed_data[col].min()),
                    "max": float(processed_data[col].max()),
                    "q25": float(processed_data[col].quantile(0.25)),
                    "q75": float(processed_data[col].quantile(0.75)),
                }

        # Metadados das estações
        for file_key, file_data in original_data.items():
            report["station_metadata"][file_key] = file_data["metadata"]

        return report

    def save_processed_data(
        self,
        processed_data: pd.DataFrame,
        splits: Dict[str, pd.DataFrame],
        report: Dict,
    ) -> None:
        """Salva dados processados em múltiplos formatos"""
        logger.info("💾 Salvando dados processados...")

        # Garantir que todas as colunas numéricas estejam com tipos corretos antes de salvar
        numeric_columns = [
            "precipitacao_mm",
            "pressao_mb",
            "temperatura_c",
            "umidade_relativa",
            "vento_velocidade_ms",
            "vento_direcao_graus",
            "radiacao_global_kjm2",
            "temperatura_orvalho_c",
            "vento_rajada_ms",
        ]

        for col in numeric_columns:
            if col in processed_data.columns:
                # Forçar conversão para numérico, convertendo qualquer string remanescente para NaN
                processed_data[col] = pd.to_numeric(
                    processed_data[col], errors="coerce"
                )

        # Verificar e converter todas as features derivadas também
        for col in processed_data.columns:
            if (
                col not in ["timestamp", "estacao", "data"]
                and processed_data[col].dtype == "object"
            ):
                # Tentar converter para numérico
                numeric_series = pd.to_numeric(processed_data[col], errors="coerce")
                # Se a conversão foi bem-sucedida (mais de 50% dos valores não são NaN), usar
                if numeric_series.notna().sum() > len(processed_data) * 0.5:
                    processed_data[col] = numeric_series
                    logger.info(f"✅ Coluna {col} convertida para numérico")

        # Diretórios de saída
        processed_dir = Path(self.config.processed_data_path)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Caminhos dos arquivos
        unified_path = processed_dir / "inmet_unified_data.parquet"
        train_path = processed_dir / "train_data.parquet"
        val_path = processed_dir / "validation_data.parquet"
        test_path = processed_dir / "test_data.parquet"
        report_path = processed_dir / "preprocessing_report.json"

        # Salvar dados unificados
        processed_data.to_parquet(unified_path, index=False)
        logger.info(f"✅ Dados unificados salvos: {unified_path}")

        # Aplicar a mesma conversão de tipos aos splits
        for split_name, split_df in splits.items():
            # Garantir que todas as colunas numéricas estejam com tipos corretos nos splits
            for col in numeric_columns:
                if col in split_df.columns:
                    split_df[col] = pd.to_numeric(split_df[col], errors="coerce")

            # Verificar e converter todas as features derivadas também nos splits
            for col in split_df.columns:
                if (
                    col not in ["timestamp", "estacao", "data"]
                    and split_df[col].dtype == "object"
                ):
                    # Tentar converter para numérico
                    numeric_series = pd.to_numeric(split_df[col], errors="coerce")
                    # Se a conversão foi bem-sucedida (mais de 50% dos valores não são NaN), usar
                    if numeric_series.notna().sum() > len(split_df) * 0.5:
                        split_df[col] = numeric_series
                        logger.info(
                            f"✅ Coluna {col} convertida para numérico no split {split_name}"
                        )

        # Salvar splits
        splits["train"].to_parquet(train_path, index=False)
        splits["validation"].to_parquet(val_path, index=False)
        splits["test"].to_parquet(test_path, index=False)

        logger.info(f"✅ Splits salvos:")
        logger.info(f"  - Train: {train_path}")
        logger.info(f"  - Validation: {val_path}")
        logger.info(f"  - Test: {test_path}")

        # Salvar relatório
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"✅ Relatório salvo: {report_path}")

        # Salvar também em CSV para inspeção manual
        csv_path = processed_dir / "inmet_unified_data.csv"
        processed_data.head(10000).to_csv(
            csv_path, index=False
        )  # Apenas primeiras 10k linhas
        logger.info(f"✅ Amostra CSV salva: {csv_path}")

        # Estatísticas finais
        logger.info(f"📊 Dados finais processados:")
        logger.info(f"  - Total de registros: {len(processed_data):,}")
        logger.info(f"  - Total de features: {len(processed_data.columns)}")
        logger.info(
            f"  - Período: {processed_data['timestamp'].min()} → {processed_data['timestamp'].max()}"
        )
        logger.info(
            f"  - Tamanho em memória: {processed_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
        )

    def run_preprocessing_pipeline(self) -> None:
        """Executa pipeline completo de preprocessamento"""
        logger.info("🚀 Iniciando pipeline de preprocessamento corrigido - Fase 2.2")
        logger.info("=" * 60)

        try:
            # 1. Carregar dados brutos
            original_data = self.load_raw_data()

            if not original_data:
                raise ValueError("Nenhum arquivo de dados foi carregado com sucesso")

            # 2. Unificar datasets
            unified_data = self.unify_datasets(original_data)

            # 3. Feature engineering
            if self.config.create_derived_features:
                unified_data = self.create_derived_features(unified_data)

            # 4. Criar splits temporais
            splits = self.create_temporal_splits(unified_data)

            # 5. Gerar relatório
            report = self.generate_preprocessing_report(original_data, unified_data)

            # 6. Salvar resultados
            self.save_processed_data(unified_data, splits, report)

            logger.info("=" * 60)
            logger.info("✅ Pipeline de preprocessamento concluído com sucesso!")
            logger.info(f"🏢 Dados processados: {unified_data.shape}")
            if "timestamp" in unified_data.columns:
                logger.info(
                    f"📅 Período: {report['date_range']['start']} a {report['date_range']['end']}"
                )
            logger.info(f"🏢 Estações: {len(report['stations'])}")
            logger.info("🔄 Próximo passo: Fase 3 - Desenvolvimento do Modelo ML")

        except Exception as e:
            logger.error(f"❌ Erro no pipeline de preprocessamento: {e}")
            raise


def main():
    """Função principal"""
    print("🌦️ Sistema de Alertas de Cheias - Rio Guaíba")
    print("📋 Fase 2.2: Preprocessamento de Dados (Versão Corrigida)")
    print("=" * 60)

    # Configuração
    config = PreprocessingConfig()

    # Executar preprocessamento
    preprocessor = DataPreprocessorFixed(config)
    preprocessor.run_preprocessing_pipeline()


if __name__ == "__main__":
    main()
