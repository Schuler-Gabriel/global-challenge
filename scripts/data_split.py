#!/usr/bin/env python3
"""
Script de Split Temporal para Dados Meteorológicos INMET
========================================================

Este script implementa estratégias avançadas de divisão temporal dos dados
meteorológicos para treino, validação e teste, específico para séries temporais
e modelos de machine learning.

Features:
- Split estratificado por década para preservar representatividade
- Preservação de padrões sazonais e cíclicos
- Validação walk-forward para séries temporais
- Análise de distribuição temporal
- Exportação em formatos prontos para ML

Autor: Sistema de Alertas de Cheias - Rio Guaíba
Data: 2025
"""

import json
import logging
import os
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data_split.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class SplitStatistics:
    """Estatísticas de cada split de dados"""

    split_name: str
    start_date: str
    end_date: str
    total_records: int
    percentage: float
    years_covered: List[int]
    seasons_distribution: Dict[str, int]
    missing_data_percentage: float


@dataclass
class DataSplitReport:
    """Relatório completo do split de dados"""

    split_strategy: str
    split_timestamp: str
    total_records: int
    total_years: int
    date_range: Tuple[str, str]
    train_stats: SplitStatistics
    validation_stats: SplitStatistics
    test_stats: SplitStatistics
    seasonal_balance: Dict[str, float]
    decade_distribution: Dict[str, Dict[str, int]]
    output_files: List[str]


class TemporalDataSplitter:
    """
    Classe principal para split temporal de dados meteorológicos
    """

    def __init__(self, output_dir: str = "data/processed/splits"):
        """
        Inicializa o splitter temporal

        Args:
            output_dir: Diretório para salvar splits
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_consolidated_data(self, data_dir: str) -> pd.DataFrame:
        """
        Carrega e consolida todos os dados meteorológicos

        Args:
            data_dir: Diretório com arquivos CSV

        Returns:
            DataFrame consolidado com todos os dados
        """
        data_path = Path(data_dir)
        csv_files = list(data_path.glob("*.CSV"))

        if not csv_files:
            logger.error(f"Nenhum arquivo CSV encontrado em {data_dir}")
            return pd.DataFrame()

        logger.info(f"Consolidando {len(csv_files)} arquivos...")

        all_dataframes = []

        for csv_file in csv_files:
            try:
                # Lê dados pulando cabeçalho INMET
                df = pd.read_csv(
                    csv_file,
                    sep=";",
                    encoding="latin-1",
                    skiprows=8,
                    na_values=["", " ", "null", "NULL", "-", "--"],
                )

                # Processa datetime
                if "Data" in df.columns and "Hora UTC" in df.columns:
                    df["datetime"] = pd.to_datetime(
                        df["Data"] + " " + df["Hora UTC"].str.replace(" UTC", ""),
                        format="%Y/%m/%d %H%M",
                        errors="coerce",
                    )
                    df = df.dropna(subset=["datetime"])
                    df = df.set_index("datetime").sort_index()

                    # Remove colunas de data/hora originais
                    df = df.drop(["Data", "Hora UTC"], axis=1, errors="ignore")

                    all_dataframes.append(df)
                    logger.info(f"Processado {csv_file.name}: {len(df)} registros")

            except Exception as e:
                logger.error(f"Erro ao processar {csv_file}: {e}")
                continue

        if not all_dataframes:
            logger.error("Nenhum arquivo foi processado com sucesso")
            return pd.DataFrame()

        # Consolida todos os dados
        consolidated_df = pd.concat(all_dataframes, axis=0)
        consolidated_df = consolidated_df.sort_index()

        # Remove duplicatas (mesmo timestamp)
        consolidated_df = consolidated_df[
            ~consolidated_df.index.duplicated(keep="first")
        ]

        logger.info(
            f"Dados consolidados: {len(consolidated_df)} registros de {consolidated_df.index.min()} até {consolidated_df.index.max()}"
        )

        return consolidated_df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features temporais aos dados

        Args:
            df: DataFrame com dados temporais

        Returns:
            DataFrame com features temporais adicionadas
        """
        df_with_features = df.copy()

        # Features temporais básicas
        df_with_features["year"] = df_with_features.index.year
        df_with_features["month"] = df_with_features.index.month
        df_with_features["day"] = df_with_features.index.day
        df_with_features["hour"] = df_with_features.index.hour
        df_with_features["day_of_year"] = df_with_features.index.dayofyear
        df_with_features["week_of_year"] = df_with_features.index.isocalendar().week

        # Sazonalidade
        df_with_features["season"] = df_with_features["month"].map(
            {
                12: "Summer",
                1: "Summer",
                2: "Summer",  # Verão (Hemisfério Sul)
                3: "Autumn",
                4: "Autumn",
                5: "Autumn",  # Outono
                6: "Winter",
                7: "Winter",
                8: "Winter",  # Inverno
                9: "Spring",
                10: "Spring",
                11: "Spring",  # Primavera
            }
        )

        # Década
        df_with_features["decade"] = (df_with_features["year"] // 10) * 10

        # Features cíclicas (sin/cos para preservar continuidade)
        df_with_features["month_sin"] = np.sin(
            2 * np.pi * df_with_features["month"] / 12
        )
        df_with_features["month_cos"] = np.cos(
            2 * np.pi * df_with_features["month"] / 12
        )
        df_with_features["hour_sin"] = np.sin(2 * np.pi * df_with_features["hour"] / 24)
        df_with_features["hour_cos"] = np.cos(2 * np.pi * df_with_features["hour"] / 24)
        df_with_features["day_year_sin"] = np.sin(
            2 * np.pi * df_with_features["day_of_year"] / 365.25
        )
        df_with_features["day_year_cos"] = np.cos(
            2 * np.pi * df_with_features["day_of_year"] / 365.25
        )

        logger.info(
            f"Features temporais adicionadas: {df_with_features.shape[1]} colunas totais"
        )
        return df_with_features

    def calculate_split_statistics(
        self, df: pd.DataFrame, split_name: str
    ) -> SplitStatistics:
        """
        Calcula estatísticas para um split de dados

        Args:
            df: DataFrame do split
            split_name: Nome do split

        Returns:
            Estatísticas do split
        """
        if df.empty:
            return SplitStatistics(
                split_name=split_name,
                start_date="",
                end_date="",
                total_records=0,
                percentage=0.0,
                years_covered=[],
                seasons_distribution={},
                missing_data_percentage=0.0,
            )

        # Informações básicas
        start_date = df.index.min().strftime("%Y-%m-%d %H:%M")
        end_date = df.index.max().strftime("%Y-%m-%d %H:%M")
        total_records = len(df)

        # Anos cobertos
        years_covered = sorted(df.index.year.unique().tolist())

        # Distribuição sazonal
        if "season" in df.columns:
            seasons_dist = df["season"].value_counts().to_dict()
        else:
            seasons_dist = {}

        # Dados faltantes
        missing_percentage = (
            df.isnull().sum().sum() / (len(df) * len(df.columns))
        ) * 100

        return SplitStatistics(
            split_name=split_name,
            start_date=start_date,
            end_date=end_date,
            total_records=total_records,
            percentage=0.0,  # Será calculado posteriormente
            years_covered=years_covered,
            seasons_distribution=seasons_dist,
            missing_data_percentage=missing_percentage,
        )

    def temporal_split_by_years(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split temporal baseado em anos completos

        Args:
            df: DataFrame com dados
            train_ratio: Proporção para treino
            val_ratio: Proporção para validação
            test_ratio: Proporção para teste

        Returns:
            Tupla (train_df, val_df, test_df)
        """
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Verifica se as proporções somam 1
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("As proporções devem somar 1.0")

        # Anos únicos ordenados
        unique_years = sorted(df.index.year.unique())
        total_years = len(unique_years)

        # Calcula pontos de corte
        train_years = int(total_years * train_ratio)
        val_years = int(total_years * val_ratio)

        # Define anos para cada split
        train_year_list = unique_years[:train_years]
        val_year_list = unique_years[train_years : train_years + val_years]
        test_year_list = unique_years[train_years + val_years :]

        # Filtra dados por ano
        train_df = df[df.index.year.isin(train_year_list)]
        val_df = df[df.index.year.isin(val_year_list)]
        test_df = df[df.index.year.isin(test_year_list)]

        logger.info(f"Split temporal por anos:")
        logger.info(
            f"  Treino: {train_year_list[0]}-{train_year_list[-1]} ({len(train_df)} registros)"
        )
        logger.info(
            f"  Validação: {val_year_list[0]}-{val_year_list[-1]} ({len(val_df)} registros)"
        )
        logger.info(
            f"  Teste: {test_year_list[0]}-{test_year_list[-1]} ({len(test_df)} registros)"
        )

        return train_df, val_df, test_df

    def stratified_decade_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split estratificado por década para preservar representatividade temporal

        Args:
            df: DataFrame com dados
            train_ratio: Proporção para treino
            val_ratio: Proporção para validação
            test_ratio: Proporção para teste

        Returns:
            Tupla (train_df, val_df, test_df)
        """
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        train_dfs = []
        val_dfs = []
        test_dfs = []

        # Para cada década, faz split proporcional
        for decade in sorted(df["decade"].unique()):
            decade_data = df[df["decade"] == decade]

            if len(decade_data) == 0:
                continue

            # Split temporal dentro da década
            decade_years = sorted(decade_data.index.year.unique())
            total_decade_years = len(decade_years)

            if total_decade_years < 3:
                # Se tem poucos anos, põe tudo no treino
                train_dfs.append(decade_data)
                continue

            # Calcula pontos de corte para a década
            train_years_count = max(1, int(total_decade_years * train_ratio))
            val_years_count = max(1, int(total_decade_years * val_ratio))

            train_years = decade_years[:train_years_count]
            val_years = decade_years[
                train_years_count : train_years_count + val_years_count
            ]
            test_years = decade_years[train_years_count + val_years_count :]

            # Filtra dados da década
            if train_years:
                train_dfs.append(decade_data[decade_data.index.year.isin(train_years)])
            if val_years:
                val_dfs.append(decade_data[decade_data.index.year.isin(val_years)])
            if test_years:
                test_dfs.append(decade_data[decade_data.index.year.isin(test_years)])

        # Consolida splits
        train_df = (
            pd.concat(train_dfs, axis=0).sort_index() if train_dfs else pd.DataFrame()
        )
        val_df = pd.concat(val_dfs, axis=0).sort_index() if val_dfs else pd.DataFrame()
        test_df = (
            pd.concat(test_dfs, axis=0).sort_index() if test_dfs else pd.DataFrame()
        )

        logger.info(f"Split estratificado por década:")
        logger.info(f"  Treino: {len(train_df)} registros")
        logger.info(f"  Validação: {len(val_df)} registros")
        logger.info(f"  Teste: {len(test_df)} registros")

        return train_df, val_df, test_df

    def walk_forward_validation_splits(
        self, df: pd.DataFrame, n_splits: int = 5
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Cria splits para validação walk-forward

        Args:
            df: DataFrame com dados
            n_splits: Número de splits walk-forward

        Returns:
            Lista de tuplas (train, validation) para cada split
        """
        if df.empty:
            return []

        # Usar TimeSeriesSplit do scikit-learn
        tscv = TimeSeriesSplit(n_splits=n_splits)

        splits = []
        df_array = df.reset_index()

        for fold, (train_idx, val_idx) in enumerate(tscv.split(df_array)):
            train_data = df_array.iloc[train_idx].set_index("datetime")
            val_data = df_array.iloc[val_idx].set_index("datetime")

            splits.append((train_data, val_data))

            logger.info(
                f"Walk-forward fold {fold + 1}: "
                f"train {len(train_data)} registros, "
                f"val {len(val_data)} registros"
            )

        return splits

    def analyze_seasonal_balance(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Analisa balanço sazonal entre splits

        Args:
            train_df: DataFrame de treino
            val_df: DataFrame de validação
            test_df: DataFrame de teste

        Returns:
            Dicionário com scores de balanço sazonal
        """
        seasonal_balance = {}

        for split_name, split_df in [
            ("train", train_df),
            ("validation", val_df),
            ("test", test_df),
        ]:
            if "season" in split_df.columns and not split_df.empty:
                season_counts = split_df["season"].value_counts(normalize=True)

                # Score de balanceamento (entropia normalizada)
                entropy = -sum(p * np.log2(p) for p in season_counts.values if p > 0)
                max_entropy = np.log2(4)  # 4 estações
                balance_score = entropy / max_entropy if max_entropy > 0 else 0

                seasonal_balance[split_name] = balance_score
            else:
                seasonal_balance[split_name] = 0.0

        return seasonal_balance

    def analyze_decade_distribution(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Dict[str, Dict[str, int]]:
        """
        Analisa distribuição por década em cada split

        Args:
            train_df: DataFrame de treino
            val_df: DataFrame de validação
            test_df: DataFrame de teste

        Returns:
            Dicionário com distribuição por década
        """
        decade_dist = {}

        for split_name, split_df in [
            ("train", train_df),
            ("validation", val_df),
            ("test", test_df),
        ]:
            if "decade" in split_df.columns and not split_df.empty:
                decade_counts = split_df["decade"].value_counts().to_dict()
                decade_dist[split_name] = {str(k): v for k, v in decade_counts.items()}
            else:
                decade_dist[split_name] = {}

        return decade_dist

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        strategy_name: str,
    ) -> List[str]:
        """
        Salva splits em arquivos

        Args:
            train_df: DataFrame de treino
            val_df: DataFrame de validação
            test_df: DataFrame de teste
            strategy_name: Nome da estratégia de split

        Returns:
            Lista de arquivos salvos
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = []

        # Salva cada split
        for split_name, split_df in [
            ("train", train_df),
            ("validation", val_df),
            ("test", test_df),
        ]:
            if not split_df.empty:
                filename = f"{strategy_name}_{split_name}_{timestamp}.csv"
                filepath = self.output_dir / filename

                split_df.to_csv(filepath, index=True)
                output_files.append(str(filepath))

                logger.info(
                    f"Split {split_name} salvo: {filepath} ({len(split_df)} registros)"
                )

        return output_files

    def create_splits(
        self, data_dir: str, strategy: str = "temporal_years", **kwargs
    ) -> DataSplitReport:
        """
        Cria splits dos dados usando estratégia especificada

        Args:
            data_dir: Diretório com dados CSV
            strategy: Estratégia de split ("temporal_years", "stratified_decade", "walk_forward")
            **kwargs: Parâmetros adicionais para estratégia

        Returns:
            Relatório do split
        """
        logger.info(
            f"=== INICIANDO SPLIT DE DADOS - ESTRATÉGIA: {strategy.upper()} ==="
        )

        # Carrega dados consolidados
        df = self.load_consolidated_data(data_dir)
        if df.empty:
            logger.error("Não foi possível carregar dados")
            return DataSplitReport(
                split_strategy=strategy,
                split_timestamp=datetime.now().isoformat(),
                total_records=0,
                total_years=0,
                date_range=("", ""),
                train_stats=SplitStatistics("train", "", "", 0, 0.0, [], {}, 0.0),
                validation_stats=SplitStatistics(
                    "validation", "", "", 0, 0.0, [], {}, 0.0
                ),
                test_stats=SplitStatistics("test", "", "", 0, 0.0, [], {}, 0.0),
                seasonal_balance={},
                decade_distribution={},
                output_files=[],
            )

        # Adiciona features temporais
        df = self.add_temporal_features(df)

        # Executa estratégia de split
        if strategy == "temporal_years":
            train_df, val_df, test_df = self.temporal_split_by_years(df, **kwargs)
        elif strategy == "stratified_decade":
            train_df, val_df, test_df = self.stratified_decade_split(df, **kwargs)
        else:
            raise ValueError(f"Estratégia de split não suportada: {strategy}")

        # Calcula estatísticas
        total_records = len(df)
        train_stats = self.calculate_split_statistics(train_df, "train")
        val_stats = self.calculate_split_statistics(val_df, "validation")
        test_stats = self.calculate_split_statistics(test_df, "test")

        # Calcula percentuais
        train_stats.percentage = (train_stats.total_records / total_records) * 100
        val_stats.percentage = (val_stats.total_records / total_records) * 100
        test_stats.percentage = (test_stats.total_records / total_records) * 100

        # Analisa balanço sazonal e distribuição por década
        seasonal_balance = self.analyze_seasonal_balance(train_df, val_df, test_df)
        decade_distribution = self.analyze_decade_distribution(
            train_df, val_df, test_df
        )

        # Salva splits
        output_files = self.save_splits(train_df, val_df, test_df, strategy)

        # Cria relatório
        report = DataSplitReport(
            split_strategy=strategy,
            split_timestamp=datetime.now().isoformat(),
            total_records=total_records,
            total_years=len(df.index.year.unique()),
            date_range=(
                df.index.min().strftime("%Y-%m-%d %H:%M"),
                df.index.max().strftime("%Y-%m-%d %H:%M"),
            ),
            train_stats=train_stats,
            validation_stats=val_stats,
            test_stats=test_stats,
            seasonal_balance=seasonal_balance,
            decade_distribution=decade_distribution,
            output_files=output_files,
        )

        # Salva relatório
        self.save_split_report(report)

        logger.info("=== SPLIT CONCLUÍDO ===")
        return report

    def save_split_report(self, report: DataSplitReport) -> None:
        """
        Salva relatório de split em arquivo JSON

        Args:
            report: Relatório do split
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"split_report_{report.split_strategy}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Relatório de split salvo: {filepath}")

    def print_split_summary(self, report: DataSplitReport) -> None:
        """
        Imprime resumo do split

        Args:
            report: Relatório do split
        """
        print("\n" + "=" * 70)
        print(f"    RELATÓRIO DE SPLIT - {report.split_strategy.upper()}")
        print("=" * 70)

        print(f"\n📊 ESTATÍSTICAS GERAIS:")
        print(f"   Total de registros: {report.total_records:,}")
        print(f"   Período total: {report.date_range[0]} até {report.date_range[1]}")
        print(f"   Anos cobertos: {report.total_years}")

        print(f"\n📈 DISTRIBUIÇÃO DOS SPLITS:")
        for stats in [report.train_stats, report.validation_stats, report.test_stats]:
            print(f"   {stats.split_name.upper()}:")
            print(f"     Registros: {stats.total_records:,} ({stats.percentage:.1f}%)")
            print(f"     Período: {stats.start_date} até {stats.end_date}")
            print(
                f"     Anos: {len(stats.years_covered)} ({min(stats.years_covered) if stats.years_covered else 'N/A'}-{max(stats.years_covered) if stats.years_covered else 'N/A'})"
            )
            print(f"     Dados faltantes: {stats.missing_data_percentage:.1f}%")

        print(f"\n🌱 BALANÇO SAZONAL:")
        for split_name, balance in report.seasonal_balance.items():
            print(
                f"   {split_name.capitalize()}: {balance:.3f} (0=desbalanceado, 1=perfeito)"
            )

        print(f"\n📅 DISTRIBUIÇÃO POR DÉCADA:")
        for split_name, decade_dist in report.decade_distribution.items():
            if decade_dist:
                decades_str = ", ".join(
                    [f"{decade}s: {count}" for decade, count in decade_dist.items()]
                )
                print(f"   {split_name.capitalize()}: {decades_str}")

        print(f"\n📁 ARQUIVOS GERADOS:")
        for filepath in report.output_files:
            print(f"   {filepath}")

        print("=" * 70)


def main():
    """Função principal para execução do script"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Split temporal de dados meteorológicos INMET"
    )
    parser.add_argument("data_dir", help="Diretório com arquivos CSV de dados")
    parser.add_argument(
        "--strategy",
        choices=["temporal_years", "stratified_decade"],
        default="temporal_years",
        help="Estratégia de split",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proporção para treino (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proporção para validação (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proporção para teste (default: 0.15)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/splits",
        help="Diretório para salvar splits (default: data/processed/splits)",
    )

    args = parser.parse_args()

    try:
        splitter = TemporalDataSplitter(args.output_dir)

        report = splitter.create_splits(
            data_dir=args.data_dir,
            strategy=args.strategy,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

        splitter.print_split_summary(report)

        return 0
    except Exception as e:
        logger.error(f"Erro durante split: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
