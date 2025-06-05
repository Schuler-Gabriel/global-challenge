#!/usr/bin/env python3
"""
Script de Validação de Dados Meteorológicos INMET
================================================

Este script realiza validação avançada de consistência dos dados meteorológicos
do INMET, incluindo verificação de ranges válidos, detecção de anomalias
estatísticas e geração de relatórios de qualidade detalhados.

Features:
- Verificação de ranges válidos por variável meteorológica
- Detecção de anomalias estatísticas usando Z-score e IQR
- Análise de consistência temporal
- Detecção de gaps e dados faltantes
- Relatório detalhado de qualidade dos dados

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
from scipy import stats

warnings.filterwarnings("ignore")

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data_validation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Resultado de validação para uma variável"""

    variable_name: str
    total_records: int
    valid_records: int
    invalid_records: int
    missing_records: int
    out_of_range_count: int
    outlier_count: int
    validation_rate: float
    outlier_percentage: float
    issues: List[str]


@dataclass
class QualityMetrics:
    """Métricas de qualidade dos dados"""

    completeness: float  # % de dados não nulos
    validity: float  # % de dados dentro dos ranges válidos
    consistency: float  # % de consistência temporal
    accuracy: float  # % sem outliers extremos
    overall_score: float  # Score geral de qualidade


@dataclass
class DataValidationReport:
    """Relatório completo de validação dos dados"""

    file_name: str
    validation_timestamp: str
    total_records: int
    date_range: Tuple[str, str]
    variable_results: List[ValidationResult]
    quality_metrics: QualityMetrics
    temporal_gaps: List[Dict[str, Any]]
    anomaly_periods: List[Dict[str, Any]]
    summary_statistics: Dict[str, Dict[str, float]]


class WeatherDataValidator:
    """
    Classe principal para validação de dados meteorológicos
    """

    # Ranges válidos para variáveis meteorológicas (Porto Alegre)
    VALID_RANGES = {
        "precipitation": (0, 200),  # mm/h - máximo observado historicamente
        "temperature": (-10, 50),  # °C - extremos possíveis RS
        "dew_point": (-15, 35),  # °C - ponto de orvalho
        "humidity": (0, 100),  # % - umidade relativa
        "pressure": (900, 1100),  # mB - pressão atmosférica
        "wind_speed": (0, 50),  # m/s - velocidade do vento
        "wind_direction": (0, 360),  # graus - direção do vento
        "wind_gust": (0, 80),  # m/s - rajada máxima
        "radiation": (0, 4000),  # Kj/m² - radiação global
    }

    # Mapeamento de colunas dos dados INMET para nomes padronizados
    COLUMN_MAPPING = {
        "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": "precipitation",
        "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": "temperature",
        "TEMPERATURA DO PONTO DE ORVALHO (°C)": "dew_point",
        "UMIDADE RELATIVA DO AR, HORARIA (%)": "humidity",
        "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)": "pressure",
        "VENTO, VELOCIDADE HORARIA (m/s)": "wind_speed",
        "VENTO, DIREÇÃO HORARIA (gr) (° (gr))": "wind_direction",
        "VENTO, RAJADA MAXIMA (m/s)": "wind_gust",
        "RADIACAO GLOBAL (Kj/m²)": "radiation",
        "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)": "temp_max",
        "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)": "temp_min",
        "PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)": "pressure_max",
        "PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)": "pressure_min",
    }

    def __init__(self, output_dir: str = "data/validation"):
        """
        Inicializa o validador

        Args:
            output_dir: Diretório para salvar relatórios de validação
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_csv_data(self, filepath: Path) -> Optional[pd.DataFrame]:
        """
        Carrega dados de arquivo CSV INMET

        Args:
            filepath: Caminho para arquivo CSV

        Returns:
            DataFrame com dados carregados ou None se erro
        """
        try:
            # Lê dados pulando cabeçalho INMET
            df = pd.read_csv(
                filepath,
                sep=";",
                encoding="latin-1",
                skiprows=8,
                na_values=["", " ", "null", "NULL", "-", "--"],
            )

            # Combina data e hora
            if "Data" in df.columns and "Hora UTC" in df.columns:
                # Tenta diferentes formatos de data
                try:
                    # Formato principal: YYYY/MM/DD
                    df["datetime"] = pd.to_datetime(
                        df["Data"] + " " + df["Hora UTC"].str.replace(" UTC", ""),
                        format="%Y/%m/%d %H%M",
                        errors="coerce",
                    )
                except (ValueError, TypeError):
                    try:
                        # Formato alternativo: DD/MM/YYYY
                        df["datetime"] = pd.to_datetime(
                            df["Data"] + " " + df["Hora UTC"].str.replace(" UTC", ""),
                            format="%d/%m/%Y %H%M",
                            errors="coerce",
                        )
                    except (ValueError, TypeError):
                        # Último recurso: parsing automático
                        df["datetime"] = pd.to_datetime(
                            df["Data"]
                            + " "
                            + df["Hora UTC"].astype(str).str.replace(" UTC", ""),
                            errors="coerce",
                        )

                # Remove registros com datetime inválido
                initial_count = len(df)
                df = df.dropna(subset=["datetime"])

                # Verifica se conseguiu processar pelo menos alguns registros
                if len(df) == 0:
                    logger.warning(
                        f"Nenhum datetime válido encontrado em {filepath.name}"
                    )
                    return None

                # Verifica se perdeu muitos registros
                if len(df) < initial_count * 0.5:
                    logger.warning(
                        f"Muitos registros com datetime inválido em {filepath.name}: {initial_count - len(df)} perdidos"
                    )

                # Define datetime como índice
                df = df.set_index("datetime").sort_index()

                # Verifica se o índice é realmente datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    logger.warning(f"Índice não é DatetimeIndex em {filepath.name}")
                    return None

            logger.info(f"Arquivo carregado: {len(df)} registros de {filepath.name}")
            return df

        except Exception as e:
            logger.error(f"Erro ao carregar {filepath}: {e}")
            return None

    def validate_range(self, series: pd.Series, variable_name: str) -> ValidationResult:
        """
        Valida range de valores para uma variável

        Args:
            series: Série de dados para validar
            variable_name: Nome da variável

        Returns:
            Resultado da validação
        """
        if variable_name not in self.VALID_RANGES:
            logger.warning(f"Range não definido para {variable_name}")
            return ValidationResult(
                variable_name=variable_name,
                total_records=len(series),
                valid_records=0,
                invalid_records=0,
                missing_records=len(series),
                out_of_range_count=0,
                outlier_count=0,
                validation_rate=0.0,
                outlier_percentage=0.0,
                issues=[f"Range não definido para {variable_name}"],
            )

        min_val, max_val = self.VALID_RANGES[variable_name]

        # Converte série para numérico, forçando erros a NaN
        numeric_series = pd.to_numeric(series, errors="coerce")

        # Estatísticas básicas
        total_records = len(numeric_series)
        missing_records = numeric_series.isna().sum()
        non_missing = numeric_series.dropna()

        # Validação de range
        if len(non_missing) > 0:
            out_of_range = ((non_missing < min_val) | (non_missing > max_val)).sum()
            valid_records = len(non_missing) - out_of_range

            # Detecção de outliers usando Z-score
            if len(non_missing) > 1:  # Precisa de pelo menos 2 valores para Z-score
                z_scores = np.abs(stats.zscore(non_missing))
                outlier_count = (z_scores > 3).sum()  # Z-score > 3 considera outlier
            else:
                outlier_count = 0

            validation_rate = (valid_records / total_records) * 100
            outlier_percentage = (
                (outlier_count / len(non_missing)) * 100 if len(non_missing) > 0 else 0
            )
        else:
            out_of_range = 0
            valid_records = 0
            outlier_count = 0
            validation_rate = 0.0
            outlier_percentage = 0.0

        # Identifica problemas
        issues = []
        if missing_records > 0:
            issues.append(
                f"{missing_records} registros faltantes ({missing_records/total_records*100:.1f}%)"
            )
        if out_of_range > 0:
            issues.append(
                f"{out_of_range} valores fora do range [{min_val}, {max_val}]"
            )
        if outlier_count > 0:
            issues.append(f"{outlier_count} outliers detectados (Z-score > 3)")

        return ValidationResult(
            variable_name=variable_name,
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=out_of_range,
            missing_records=missing_records,
            out_of_range_count=out_of_range,
            outlier_count=outlier_count,
            validation_rate=validation_rate,
            outlier_percentage=outlier_percentage,
            issues=issues,
        )

    def detect_temporal_gaps(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detecta gaps temporais nos dados

        Args:
            df: DataFrame com dados temporais

        Returns:
            Lista de gaps encontrados
        """
        if df.empty or df.index.name != "datetime":
            return []

        try:
            gaps = []
            expected_freq = pd.Timedelta(hours=1)  # Dados horários esperados

            time_diffs = df.index.to_series().diff()
            large_gaps = time_diffs[time_diffs > expected_freq * 2]  # Gaps > 2 horas

            for gap_start in large_gaps.index:
                gap_duration = large_gaps[gap_start]
                previous_time = gap_start - gap_duration

                # Verifica se os timestamps são válidos
                if pd.isna(gap_start) or pd.isna(previous_time):
                    continue

                try:
                    gaps.append(
                        {
                            "start_time": previous_time.strftime("%Y-%m-%d %H:%M"),
                            "end_time": gap_start.strftime("%Y-%m-%d %H:%M"),
                            "duration_hours": gap_duration.total_seconds() / 3600,
                            "missing_records": int(gap_duration.total_seconds() / 3600)
                            - 1,
                        }
                    )
                except (AttributeError, ValueError) as e:
                    logger.warning(f"Erro ao processar gap temporal: {e}")
                    continue

            return gaps
        except Exception as e:
            logger.warning(f"Erro na detecção de gaps temporais: {e}")
            return []

    def detect_anomaly_periods(
        self, df: pd.DataFrame, variable: str = "precipitation"
    ) -> List[Dict[str, Any]]:
        """
        Detecta períodos com anomalias meteorológicas

        Args:
            df: DataFrame com dados
            variable: Variável para análise de anomalias

        Returns:
            Lista de períodos anômalos
        """
        anomalies = []

        if variable not in df.columns:
            return anomalies

        try:
            # Converte para numérico e remove NaN
            series = pd.to_numeric(df[variable], errors="coerce").dropna()
            if len(series) < 24:  # Precisa de pelo menos 24 horas de dados
                return anomalies

            # Detecta períodos de chuva extrema (> 50mm/h por mais de 3 horas consecutivas)
            if variable == "precipitation":
                extreme_rain = series > 50
                consecutive_groups = (extreme_rain != extreme_rain.shift()).cumsum()

                for group_id in consecutive_groups[extreme_rain].unique():
                    group_data = series[consecutive_groups == group_id]
                    if len(group_data) >= 3:  # 3+ horas consecutivas
                        try:
                            # Verifica se os índices são timestamps válidos
                            start_time = group_data.index[0]
                            end_time = group_data.index[-1]

                            if pd.isna(start_time) or pd.isna(end_time):
                                continue

                            anomalies.append(
                                {
                                    "type": "extreme_precipitation",
                                    "start_time": start_time.strftime("%Y-%m-%d %H:%M"),
                                    "end_time": end_time.strftime("%Y-%m-%d %H:%M"),
                                    "duration_hours": len(group_data),
                                    "max_value": float(group_data.max()),
                                    "total_precipitation": float(group_data.sum()),
                                }
                            )
                        except (AttributeError, ValueError) as e:
                            logger.warning(f"Erro ao processar anomalia: {e}")
                            continue

            return anomalies
        except Exception as e:
            logger.warning(f"Erro na detecção de anomalias: {e}")
            return []

    def calculate_quality_metrics(
        self, validation_results: List[ValidationResult], df: pd.DataFrame
    ) -> QualityMetrics:
        """
        Calcula métricas de qualidade dos dados

        Args:
            validation_results: Resultados de validação
            df: DataFrame original

        Returns:
            Métricas de qualidade calculadas
        """
        if not validation_results:
            return QualityMetrics(0, 0, 0, 0, 0)

        # Completeness: % de dados não nulos
        total_cells = sum(r.total_records for r in validation_results)
        total_missing = sum(r.missing_records for r in validation_results)
        completeness = (
            ((total_cells - total_missing) / total_cells * 100)
            if total_cells > 0
            else 0
        )

        # Validity: % de dados dentro dos ranges válidos
        total_valid = sum(r.valid_records for r in validation_results)
        total_non_missing = total_cells - total_missing
        validity = (
            (total_valid / total_non_missing * 100) if total_non_missing > 0 else 0
        )

        # Consistency: baseado em gaps temporais
        consistency = 0.0
        try:
            if (
                not df.empty
                and isinstance(df.index, pd.DatetimeIndex)
                and len(df.index) > 1
            ):
                # Calcula range temporal esperado
                start_time = df.index.min()
                end_time = df.index.max()

                # Verifica se são timestamps válidos
                if not pd.isna(start_time) and not pd.isna(end_time):
                    expected_records = len(
                        pd.date_range(start=start_time, end=end_time, freq="H")
                    )
                    actual_records = len(df)

                    # Limita consistency a no máximo 100%
                    consistency = (
                        min(100.0, (actual_records / expected_records * 100))
                        if expected_records > 0
                        else 0
                    )
        except Exception as e:
            logger.warning(f"Erro ao calcular consistency: {e}")
            consistency = 0.0

        # Accuracy: % sem outliers extremos
        total_outliers = sum(r.outlier_count for r in validation_results)
        accuracy = (
            ((total_non_missing - total_outliers) / total_non_missing * 100)
            if total_non_missing > 0
            else 0
        )

        # Score geral (média ponderada) - limita a 100
        overall_score = min(
            100.0,
            (completeness * 0.3 + validity * 0.3 + consistency * 0.2 + accuracy * 0.2),
        )

        return QualityMetrics(
            completeness=completeness,
            validity=validity,
            consistency=consistency,
            accuracy=accuracy,
            overall_score=overall_score,
        )

    def calculate_summary_statistics(
        self, df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcula estatísticas resumo para variáveis numéricas

        Args:
            df: DataFrame com dados

        Returns:
            Dicionário com estatísticas por variável
        """
        stats_dict = {}

        # Seleciona apenas colunas numéricas mapeadas
        numeric_cols = []
        for original_col, mapped_name in self.COLUMN_MAPPING.items():
            if original_col in df.columns and mapped_name in self.VALID_RANGES:
                numeric_cols.append(original_col)

        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) > 0:
                stats_dict[self.COLUMN_MAPPING[col]] = {
                    "count": int(len(series)),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "median": float(series.median()),
                    "q25": float(series.quantile(0.25)),
                    "q75": float(series.quantile(0.75)),
                }

        return stats_dict

    def validate_file(self, filepath: Path) -> DataValidationReport:
        """
        Valida um arquivo CSV completo

        Args:
            filepath: Caminho para arquivo CSV

        Returns:
            Relatório de validação
        """
        logger.info(f"Iniciando validação de {filepath.name}")

        # Carrega dados
        df = self.load_csv_data(filepath)
        if df is None:
            return DataValidationReport(
                file_name=filepath.name,
                validation_timestamp=datetime.now().isoformat(),
                total_records=0,
                date_range=("", ""),
                variable_results=[],
                quality_metrics=QualityMetrics(0, 0, 0, 0, 0),
                temporal_gaps=[],
                anomaly_periods=[],
                summary_statistics={},
            )

        # Valida cada variável
        validation_results = []
        for original_col, mapped_name in self.COLUMN_MAPPING.items():
            if original_col in df.columns and mapped_name in self.VALID_RANGES:
                series = pd.to_numeric(df[original_col], errors="coerce")
                result = self.validate_range(series, mapped_name)
                validation_results.append(result)

        # Detecta gaps temporais
        temporal_gaps = self.detect_temporal_gaps(df)

        # Detecta anomalias
        anomaly_periods = []
        if "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)" in df.columns:
            precip_anomalies = self.detect_anomaly_periods(
                df.rename(
                    columns={"PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": "precipitation"}
                ),
                "precipitation",
            )
            anomaly_periods.extend(precip_anomalies)

        # Calcula métricas de qualidade
        quality_metrics = self.calculate_quality_metrics(validation_results, df)

        # Calcula estatísticas resumo
        summary_stats = self.calculate_summary_statistics(df)

        # Range de datas
        try:
            date_range = (
                (
                    df.index.min().strftime("%Y-%m-%d %H:%M")
                    if not df.empty and not pd.isna(df.index.min())
                    else ""
                ),
                (
                    df.index.max().strftime("%Y-%m-%d %H:%M")
                    if not df.empty and not pd.isna(df.index.max())
                    else ""
                ),
            )
        except (AttributeError, ValueError) as e:
            logger.warning(f"Erro ao calcular range de datas: {e}")
            date_range = ("", "")

        report = DataValidationReport(
            file_name=filepath.name,
            validation_timestamp=datetime.now().isoformat(),
            total_records=len(df),
            date_range=date_range,
            variable_results=validation_results,
            quality_metrics=quality_metrics,
            temporal_gaps=temporal_gaps,
            anomaly_periods=anomaly_periods,
            summary_statistics=summary_stats,
        )

        logger.info(
            f"Validação concluída. Score de qualidade: {quality_metrics.overall_score:.1f}"
        )
        return report

    def save_validation_report(self, report: DataValidationReport) -> None:
        """
        Salva relatório de validação em arquivo JSON

        Args:
            report: Relatório de validação
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_{report.file_name.replace('.CSV', '')}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Relatório salvo: {filepath}")

    def print_validation_summary(self, report: DataValidationReport) -> None:
        """
        Imprime resumo da validação

        Args:
            report: Relatório de validação
        """
        print("\n" + "=" * 70)
        print(f"    RELATÓRIO DE VALIDAÇÃO - {report.file_name}")
        print("=" * 70)

        print(f"\n📊 MÉTRICAS DE QUALIDADE:")
        print(f"   Score Geral: {report.quality_metrics.overall_score:.1f}/100")
        print(f"   Completeness: {report.quality_metrics.completeness:.1f}%")
        print(f"   Validity: {report.quality_metrics.validity:.1f}%")
        print(f"   Consistency: {report.quality_metrics.consistency:.1f}%")
        print(f"   Accuracy: {report.quality_metrics.accuracy:.1f}%")

        print(f"\n📈 ESTATÍSTICAS GERAIS:")
        print(f"   Total de registros: {report.total_records:,}")
        print(f"   Período: {report.date_range[0]} até {report.date_range[1]}")
        print(f"   Gaps temporais: {len(report.temporal_gaps)}")
        print(f"   Anomalias detectadas: {len(report.anomaly_periods)}")

        print(f"\n🔍 VALIDAÇÃO POR VARIÁVEL:")
        for result in report.variable_results:
            status = (
                "✅"
                if result.validation_rate > 90
                else "⚠️" if result.validation_rate > 70 else "❌"
            )
            print(
                f"   {status} {result.variable_name}: {result.validation_rate:.1f}% válidos"
            )
            if result.issues:
                for issue in result.issues:
                    print(f"      - {issue}")

        if report.temporal_gaps:
            print(f"\n⏰ GAPS TEMPORAIS:")
            for gap in report.temporal_gaps[:5]:  # Mostra apenas os 5 primeiros
                print(
                    f"   {gap['start_time']} → {gap['end_time']} ({gap['duration_hours']:.1f}h)"
                )

        if report.anomaly_periods:
            print(f"\n🌧️ ANOMALIAS DETECTADAS:")
            for anomaly in report.anomaly_periods:
                if anomaly["type"] == "extreme_precipitation":
                    print(
                        f"   Chuva extrema: {anomaly['start_time']} ({anomaly['duration_hours']}h, {anomaly['max_value']:.1f}mm/h)"
                    )

        print("=" * 70)


def validate_directory(
    input_path: str, output_dir: str = "data/validation"
) -> List[DataValidationReport]:
    """
    Valida todos os arquivos CSV em um diretório ou um arquivo específico

    Args:
        input_path: Diretório com arquivos CSV ou arquivo específico
        output_dir: Diretório para salvar relatórios

    Returns:
        Lista de relatórios de validação
    """
    validator = WeatherDataValidator(output_dir)
    path = Path(input_path)

    if not path.exists():
        logger.error(f"Caminho não encontrado: {input_path}")
        return []

    # Se é um arquivo, valida apenas ele
    if path.is_file() and path.suffix.upper() == ".CSV":
        csv_files = [path]
    # Se é um diretório, busca todos os CSVs
    elif path.is_dir():
        csv_files = list(path.glob("*.CSV"))
    else:
        logger.error(
            f"Caminho inválido (deve ser diretório ou arquivo CSV): {input_path}"
        )
        return []

    logger.info(f"Encontrados {len(csv_files)} arquivos para validação")

    reports = []
    for csv_file in csv_files:
        try:
            report = validator.validate_file(csv_file)
            validator.save_validation_report(report)
            validator.print_validation_summary(report)
            reports.append(report)
        except Exception as e:
            logger.error(f"Erro ao validar {csv_file}: {e}")
            continue

    return reports


def main():
    """Função principal para execução do script"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validação de dados meteorológicos INMET"
    )
    parser.add_argument(
        "input_path",
        help="Diretório com arquivos CSV ou arquivo CSV específico para validar",
    )
    parser.add_argument(
        "--output-dir",
        default="data/validation",
        help="Diretório para salvar relatórios (default: data/validation)",
    )

    args = parser.parse_args()

    try:
        reports = validate_directory(args.input_path, args.output_dir)

        if reports:
            avg_quality = sum(r.quality_metrics.overall_score for r in reports) / len(
                reports
            )
            print(f"\n🎯 RESUMO GERAL:")
            print(f"   Arquivos validados: {len(reports)}")
            print(f"   Score médio de qualidade: {avg_quality:.1f}/100")

        return 0
    except Exception as e:
        logger.error(f"Erro durante validação: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
