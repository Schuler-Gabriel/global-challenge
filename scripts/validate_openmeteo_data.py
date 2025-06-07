#!/usr/bin/env python3
"""
Script de Validação de Dados Open-Meteo Historical Forecast
==========================================================

Este script realiza validação avançada dos dados JSON do Open-Meteo,
verificando completude, consistência e qualidade antes de executar
a engenharia de características atmosféricas (Fase 2.2).

Features:
- Validação de estrutura JSON dos chunks
- Verificação de ranges válidos por variável meteorológica
- Análise de consistência temporal e gaps
- Detecção de anomalias estatísticas
- Verificação específica de variáveis de pressão atmosférica
- Relatório detalhado de qualidade dos dados

Autor: Sistema de Alertas de Cheias - Rio Guaíba
Data: 2025-01-13
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/openmeteo_validation.log"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class OpenMeteoValidationResult:
    """Resultado de validação para uma variável do Open-Meteo"""
    variable_name: str
    total_records: int
    valid_records: int
    invalid_records: int
    missing_records: int
    out_of_range_count: int
    outlier_count: int
    validation_rate: float
    outlier_percentage: float
    min_value: float
    max_value: float
    mean_value: float
    std_value: float
    issues: List[str]


@dataclass
class ChunkValidationResult:
    """Resultado de validação para um chunk específico"""
    chunk_file: str
    chunk_size_kb: float
    valid_structure: bool
    date_range: Tuple[str, str]
    total_timestamps: int
    missing_timestamps: int
    variable_results: List[OpenMeteoValidationResult]
    data_quality_score: float
    issues: List[str]


@dataclass
class OpenMeteoDataReport:
    """Relatório completo de validação dos dados Open-Meteo"""
    validation_timestamp: str
    total_chunks: int
    valid_chunks: int
    invalid_chunks: int
    total_records: int
    total_date_range: Tuple[str, str]
    chunk_results: List[ChunkValidationResult]
    consolidated_variable_stats: Dict[str, Dict[str, float]]
    atmospheric_pressure_analysis: Dict[str, Any]
    temporal_gaps: List[Dict[str, Any]]
    overall_quality_score: float
    summary_statistics: Dict[str, Any]


class OpenMeteoDataValidator:
    """
    Classe principal para validação de dados Open-Meteo Historical Forecast
    """

    # Ranges válidos para variáveis meteorológicas Open-Meteo (Porto Alegre/RS)
    VALID_RANGES = {
        # Variáveis de superfície
        "temperature_2m": (-15, 50),  # °C - temperatura 2m
        "relative_humidity_2m": (0, 100),  # % - umidade relativa 2m
        "dewpoint_2m": (-20, 40),  # °C - ponto de orvalho 2m
        "apparent_temperature": (-20, 55),  # °C - temperatura aparente
        "precipitation_probability": (0, 100),  # % - probabilidade precipitação
        "precipitation": (0, 200),  # mm - precipitação
        "rain": (0, 200),  # mm - chuva
        "showers": (0, 100),  # mm - aguaceiros
        "pressure_msl": (950, 1050),  # hPa - pressão ao nível do mar
        "surface_pressure": (900, 1040),  # hPa - pressão superficial
        "cloudcover": (0, 100),  # % - cobertura de nuvens
        "cloudcover_low": (0, 100),  # % - nuvens baixas
        "cloudcover_mid": (0, 100),  # % - nuvens médias
        "cloudcover_high": (0, 100),  # % - nuvens altas
        "windspeed_10m": (0, 80),  # km/h - velocidade vento 10m
        "winddirection_10m": (0, 360),  # ° - direção vento 10m
        "windgusts_10m": (0, 150),  # km/h - rajadas vento 10m
        "cape": (0, 8000),  # J/kg - energia potencial convectiva
        "lifted_index": (-15, 15),  # índice de levantamento
        "vapour_pressure_deficit": (0, 10),  # kPa - déficit pressão vapor
        "soil_temperature_0cm": (-10, 60),  # °C - temperatura solo 0cm
        "soil_moisture_0_1cm": (0, 1),  # m³/m³ - umidade solo 0-1cm
        
        # Variáveis de níveis de pressão atmosférica (críticas para análise sinótica)
        "temperature_1000hPa": (-20, 50),  # °C
        "relative_humidity_1000hPa": (0, 100),  # %
        "wind_speed_1000hPa": (0, 200),  # km/h
        "wind_direction_1000hPa": (0, 360),  # °
        "geopotential_height_1000hPa": (0, 300),  # m
        
        "temperature_850hPa": (-30, 40),  # °C
        "relative_humidity_850hPa": (0, 100),  # %
        "wind_speed_850hPa": (0, 250),  # km/h
        "wind_direction_850hPa": (0, 360),  # °
        "geopotential_height_850hPa": (1200, 1800),  # m
        
        "temperature_700hPa": (-40, 30),  # °C
        "relative_humidity_700hPa": (0, 100),  # %
        "wind_speed_700hPa": (0, 300),  # km/h
        "wind_direction_700hPa": (0, 360),  # °
        "geopotential_height_700hPa": (2800, 3400),  # m
        
        "temperature_500hPa": (-60, 10),  # °C
        "relative_humidity_500hPa": (0, 100),  # %
        "wind_speed_500hPa": (0, 350),  # km/h
        "wind_direction_500hPa": (0, 360),  # °
        "geopotential_height_500hPa": (5200, 6200),  # m
        
        "temperature_300hPa": (-80, -20),  # °C
        "relative_humidity_300hPa": (0, 100),  # %
        "wind_speed_300hPa": (0, 400),  # km/h
        "wind_direction_300hPa": (0, 360),  # °
        "geopotential_height_300hPa": (8500, 10000),  # m
    }

    # Variáveis críticas para análise atmosférica
    CRITICAL_ATMOSPHERIC_VARS = [
        "temperature_850hPa", "temperature_500hPa",
        "geopotential_height_850hPa", "geopotential_height_500hPa",
        "wind_speed_850hPa", "wind_speed_500hPa",
        "wind_direction_850hPa", "wind_direction_500hPa",
        "relative_humidity_850hPa", "relative_humidity_500hPa"
    ]

    def __init__(self, data_path: Optional[Path] = None, output_dir: str = "data/validation"):
        """
        Inicializa o validador Open-Meteo

        Args:
            data_path: Caminho para dados Open-Meteo
            output_dir: Diretório para salvar relatórios de validação
        """
        self.data_path = data_path or Path("data/raw/Open-Meteo Historical Forecast")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_chunk_data(self, chunk_file: Path) -> Optional[Dict]:
        """
        Carrega dados de um chunk JSON do Open-Meteo

        Args:
            chunk_file: Caminho para arquivo JSON do chunk

        Returns:
            Dados JSON carregados ou None se erro
        """
        try:
            with open(chunk_file, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Chunk carregado: {chunk_file.name} ({chunk_file.stat().st_size / 1024:.1f} KB)")
            return data
        
        except Exception as e:
            logger.error(f"Erro ao carregar chunk {chunk_file.name}: {e}")
            return None

    def validate_chunk_structure(self, chunk_data: Dict, chunk_file: str) -> Tuple[bool, List[str]]:
        """
        Valida a estrutura básica de um chunk Open-Meteo

        Args:
            chunk_data: Dados JSON do chunk
            chunk_file: Nome do arquivo do chunk

        Returns:
            Tuple (é_válido, lista_de_problemas)
        """
        issues = []
        
        # Verificar chaves obrigatórias
        required_keys = ['latitude', 'longitude', 'hourly_units', 'hourly']
        for key in required_keys:
            if key not in chunk_data:
                issues.append(f"Chave obrigatória ausente: {key}")
        
        if 'hourly' in chunk_data:
            hourly_data = chunk_data['hourly']
            
            # Verificar se há dados temporais
            if 'time' not in hourly_data:
                issues.append("Campo 'time' ausente nos dados horários")
            
            # Verificar se há variáveis atmosféricas críticas
            missing_critical = []
            for var in self.CRITICAL_ATMOSPHERIC_VARS:
                if var not in hourly_data:
                    missing_critical.append(var)
            
            if missing_critical:
                issues.append(f"Variáveis atmosféricas críticas ausentes: {missing_critical}")
            
            # Verificar consistência no tamanho dos arrays
            if 'time' in hourly_data:
                time_length = len(hourly_data['time'])
                for var, values in hourly_data.items():
                    if var != 'time' and len(values) != time_length:
                        issues.append(f"Inconsistência no tamanho do array para {var}: {len(values)} vs {time_length}")
        
        return len(issues) == 0, issues

    def validate_variable_data(self, variable_name: str, data: List) -> OpenMeteoValidationResult:
        """
        Valida os dados de uma variável específica

        Args:
            variable_name: Nome da variável
            data: Lista de valores da variável

        Returns:
            Resultado da validação
        """
        # Converter para numpy array para análise
        data_array = np.array(data, dtype=float)
        
        # Contar valores válidos/inválidos
        valid_mask = ~np.isnan(data_array)
        total_records = len(data_array)
        valid_records = np.sum(valid_mask)
        missing_records = total_records - valid_records
        
        # Análise apenas dos valores válidos
        valid_data = data_array[valid_mask]
        
        if len(valid_data) == 0:
            return OpenMeteoValidationResult(
                variable_name=variable_name,
                total_records=total_records,
                valid_records=0,
                invalid_records=0,
                missing_records=missing_records,
                out_of_range_count=0,
                outlier_count=0,
                validation_rate=0.0,
                outlier_percentage=0.0,
                min_value=0.0,
                max_value=0.0,
                mean_value=0.0,
                std_value=0.0,
                issues=["Nenhum valor válido encontrado"]
            )
        
        # Verificar ranges válidos
        if variable_name in self.VALID_RANGES:
            min_valid, max_valid = self.VALID_RANGES[variable_name]
            out_of_range_mask = (valid_data < min_valid) | (valid_data > max_valid)
            out_of_range_count = np.sum(out_of_range_mask)
        else:
            out_of_range_count = 0
        
        # Detectar outliers usando Z-score
        if len(valid_data) > 3:
            z_scores = np.abs(stats.zscore(valid_data))
            outlier_mask = z_scores > 3
            outlier_count = np.sum(outlier_mask)
        else:
            outlier_count = 0
        
        # Calcular estatísticas
        min_value = float(np.min(valid_data))
        max_value = float(np.max(valid_data))
        mean_value = float(np.mean(valid_data))
        std_value = float(np.std(valid_data))
        
        # Taxa de validação
        validation_rate = valid_records / total_records * 100
        outlier_percentage = outlier_count / valid_records * 100 if valid_records > 0 else 0
        
        # Identificar problemas
        issues = []
        if missing_records > total_records * 0.1:  # >10% valores faltantes
            issues.append(f"Muitos valores faltantes: {missing_records}/{total_records}")
        
        if out_of_range_count > 0:
            issues.append(f"Valores fora do range válido: {out_of_range_count}")
        
        if outlier_percentage > 5:  # >5% outliers
            issues.append(f"Muitos outliers: {outlier_percentage:.1f}%")
        
        if variable_name in self.CRITICAL_ATMOSPHERIC_VARS and validation_rate < 90:
            issues.append(f"Variável crítica com baixa qualidade: {validation_rate:.1f}%")
        
        return OpenMeteoValidationResult(
            variable_name=variable_name,
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=out_of_range_count,
            missing_records=missing_records,
            out_of_range_count=out_of_range_count,
            outlier_count=outlier_count,
            validation_rate=validation_rate,
            outlier_percentage=outlier_percentage,
            min_value=min_value,
            max_value=max_value,
            mean_value=mean_value,
            std_value=std_value,
            issues=issues
        )

    def detect_temporal_gaps(self, time_data: List[str]) -> List[Dict[str, Any]]:
        """
        Detecta gaps temporais nos dados

        Args:
            time_data: Lista de timestamps

        Returns:
            Lista de gaps encontrados
        """
        gaps = []
        
        # Converter timestamps para datetime
        timestamps = [datetime.fromisoformat(t) for t in time_data]
        timestamps.sort()
        
        expected_interval = timedelta(hours=1)  # Open-Meteo dados horários
        
        for i in range(1, len(timestamps)):
            actual_interval = timestamps[i] - timestamps[i-1]
            
            if actual_interval > expected_interval:
                gap_hours = actual_interval.total_seconds() / 3600
                gaps.append({
                    "start_time": timestamps[i-1].isoformat(),
                    "end_time": timestamps[i].isoformat(),
                    "gap_hours": gap_hours,
                    "missing_records": int(gap_hours) - 1
                })
        
        return gaps

    def validate_chunk(self, chunk_file: Path) -> ChunkValidationResult:
        """
        Valida um chunk completo

        Args:
            chunk_file: Caminho para arquivo do chunk

        Returns:
            Resultado da validação do chunk
        """
        chunk_data = self.load_chunk_data(chunk_file)
        chunk_size_kb = chunk_file.stat().st_size / 1024
        
        if chunk_data is None:
            return ChunkValidationResult(
                chunk_file=chunk_file.name,
                chunk_size_kb=chunk_size_kb,
                valid_structure=False,
                date_range=("", ""),
                total_timestamps=0,
                missing_timestamps=0,
                variable_results=[],
                data_quality_score=0.0,
                issues=["Falha ao carregar chunk"]
            )
        
        # Validar estrutura
        valid_structure, structure_issues = self.validate_chunk_structure(chunk_data, chunk_file.name)
        
        issues = structure_issues.copy()
        variable_results = []
        
        if valid_structure:
            hourly_data = chunk_data['hourly']
            
            # Analisar dados temporais
            time_data = hourly_data['time']
            total_timestamps = len(time_data)
            date_range = (time_data[0], time_data[-1]) if time_data else ("", "")
            
            # Detectar gaps temporais
            temporal_gaps = self.detect_temporal_gaps(time_data)
            missing_timestamps = sum(gap['missing_records'] for gap in temporal_gaps)
            
            if temporal_gaps:
                issues.append(f"Encontrados {len(temporal_gaps)} gaps temporais")
            
            # Validar cada variável
            for variable_name, variable_data in hourly_data.items():
                if variable_name != 'time':
                    result = self.validate_variable_data(variable_name, variable_data)
                    variable_results.append(result)
        else:
            date_range = ("", "")
            total_timestamps = 0
            missing_timestamps = 0
        
        # Calcular score de qualidade dos dados
        if variable_results:
            critical_var_scores = []
            for result in variable_results:
                if result.variable_name in self.CRITICAL_ATMOSPHERIC_VARS:
                    critical_var_scores.append(result.validation_rate)
            
            if critical_var_scores:
                data_quality_score = np.mean(critical_var_scores)
            else:
                all_scores = [r.validation_rate for r in variable_results]
                data_quality_score = np.mean(all_scores) if all_scores else 0.0
        else:
            data_quality_score = 0.0
        
        return ChunkValidationResult(
            chunk_file=chunk_file.name,
            chunk_size_kb=chunk_size_kb,
            valid_structure=valid_structure,
            date_range=date_range,
            total_timestamps=total_timestamps,
            missing_timestamps=missing_timestamps,
            variable_results=variable_results,
            data_quality_score=data_quality_score,
            issues=issues
        )

    def analyze_atmospheric_pressure_levels(self, chunk_results: List[ChunkValidationResult]) -> Dict[str, Any]:
        """
        Análise específica das variáveis de níveis de pressão atmosférica

        Args:
            chunk_results: Resultados de validação dos chunks

        Returns:
            Análise das variáveis de pressão atmosférica
        """
        pressure_levels = ['1000hPa', '850hPa', '700hPa', '500hPa', '300hPa']
        analysis = {
            'pressure_level_completeness': {},
            'temperature_gradient_analysis': {},
            'wind_consistency_analysis': {},
            'issues': []
        }
        
        for level in pressure_levels:
            temp_var = f'temperature_{level}'
            height_var = f'geopotential_height_{level}'
            wind_speed_var = f'wind_speed_{level}'
            wind_dir_var = f'wind_direction_{level}'
            
            # Coletar estatísticas de todas as variáveis deste nível
            level_stats = {}
            for var in [temp_var, height_var, wind_speed_var, wind_dir_var]:
                all_validation_rates = []
                for chunk_result in chunk_results:
                    for var_result in chunk_result.variable_results:
                        if var_result.variable_name == var:
                            all_validation_rates.append(var_result.validation_rate)
                
                if all_validation_rates:
                    level_stats[var] = {
                        'mean_validation_rate': np.mean(all_validation_rates),
                        'min_validation_rate': np.min(all_validation_rates),
                        'chunks_with_data': len(all_validation_rates)
                    }
            
            analysis['pressure_level_completeness'][level] = level_stats
            
            # Verificar se nível tem dados suficientes
            if level_stats:
                temp_quality = level_stats.get(temp_var, {}).get('mean_validation_rate', 0)
                height_quality = level_stats.get(height_var, {}).get('mean_validation_rate', 0)
                
                if temp_quality < 80 or height_quality < 80:
                    analysis['issues'].append(
                        f"Nível {level} com qualidade insuficiente - Temp: {temp_quality:.1f}%, Height: {height_quality:.1f}%"
                    )
        
        return analysis

    def validate_all_chunks(self) -> OpenMeteoDataReport:
        """
        Valida todos os chunks do Open-Meteo

        Returns:
            Relatório completo de validação
        """
        logger.info(f"Iniciando validação dos dados Open-Meteo em: {self.data_path}")
        
        # Encontrar todos os chunks
        chunk_files = list(self.data_path.glob("chunk_*.json"))
        chunk_files.sort()
        
        if not chunk_files:
            logger.error("Nenhum chunk encontrado!")
            return OpenMeteoDataReport(
                validation_timestamp=datetime.now().isoformat(),
                total_chunks=0,
                valid_chunks=0,
                invalid_chunks=0,
                total_records=0,
                total_date_range=("", ""),
                chunk_results=[],
                consolidated_variable_stats={},
                atmospheric_pressure_analysis={},
                temporal_gaps=[],
                overall_quality_score=0.0,
                summary_statistics={}
            )
        
        logger.info(f"Encontrados {len(chunk_files)} chunks para validação")
        
        # Validar cada chunk
        chunk_results = []
        for chunk_file in chunk_files:
            logger.info(f"Validando: {chunk_file.name}")
            result = self.validate_chunk(chunk_file)
            chunk_results.append(result)
        
        # Consolidar resultados
        valid_chunks = sum(1 for r in chunk_results if r.valid_structure)
        invalid_chunks = len(chunk_results) - valid_chunks
        total_records = sum(r.total_timestamps for r in chunk_results)
        
        # Range temporal total
        all_start_dates = [r.date_range[0] for r in chunk_results if r.date_range[0]]
        all_end_dates = [r.date_range[1] for r in chunk_results if r.date_range[1]]
        
        if all_start_dates and all_end_dates:
            total_date_range = (min(all_start_dates), max(all_end_dates))
        else:
            total_date_range = ("", "")
        
        # Consolidar estatísticas por variável
        consolidated_stats = {}
        for chunk_result in chunk_results:
            for var_result in chunk_result.variable_results:
                var_name = var_result.variable_name
                if var_name not in consolidated_stats:
                    consolidated_stats[var_name] = {
                        'validation_rates': [],
                        'means': [],
                        'stds': [],
                        'total_records': 0,
                        'total_valid': 0,
                        'total_missing': 0
                    }
                
                stats = consolidated_stats[var_name]
                stats['validation_rates'].append(var_result.validation_rate)
                stats['means'].append(var_result.mean_value)
                stats['stds'].append(var_result.std_value)
                stats['total_records'] += var_result.total_records
                stats['total_valid'] += var_result.valid_records
                stats['total_missing'] += var_result.missing_records
        
        # Calcular estatísticas finais
        final_stats = {}
        for var_name, stats in consolidated_stats.items():
            final_stats[var_name] = {
                'mean_validation_rate': np.mean(stats['validation_rates']),
                'overall_completeness': stats['total_valid'] / stats['total_records'] * 100 if stats['total_records'] > 0 else 0,
                'mean_value': np.mean(stats['means']),
                'mean_std': np.mean(stats['stds']),
                'total_records': stats['total_records'],
                'total_missing': stats['total_missing']
            }
        
        # Análise específica de pressão atmosférica
        pressure_analysis = self.analyze_atmospheric_pressure_levels(chunk_results)
        
        # Detectar gaps temporais globais
        all_temporal_gaps = []
        for chunk_result in chunk_results:
            if chunk_result.missing_timestamps > 0:
                all_temporal_gaps.append({
                    'chunk': chunk_result.chunk_file,
                    'missing_timestamps': chunk_result.missing_timestamps,
                    'date_range': chunk_result.date_range
                })
        
        # Score geral de qualidade
        if chunk_results:
            overall_quality_score = np.mean([r.data_quality_score for r in chunk_results])
        else:
            overall_quality_score = 0.0
        
        # Estatísticas sumárias
        summary_stats = {
            'total_size_mb': sum(r.chunk_size_kb for r in chunk_results) / 1024,
            'avg_chunk_size_kb': np.mean([r.chunk_size_kb for r in chunk_results]) if chunk_results else 0,
            'critical_vars_avg_quality': np.mean([
                final_stats[var]['mean_validation_rate'] 
                for var in self.CRITICAL_ATMOSPHERIC_VARS 
                if var in final_stats
            ]) if final_stats else 0,
            'total_missing_records': sum(r.missing_timestamps for r in chunk_results)
        }
        
        report = OpenMeteoDataReport(
            validation_timestamp=datetime.now().isoformat(),
            total_chunks=len(chunk_results),
            valid_chunks=valid_chunks,
            invalid_chunks=invalid_chunks,
            total_records=total_records,
            total_date_range=total_date_range,
            chunk_results=chunk_results,
            consolidated_variable_stats=final_stats,
            atmospheric_pressure_analysis=pressure_analysis,
            temporal_gaps=all_temporal_gaps,
            overall_quality_score=overall_quality_score,
            summary_statistics=summary_stats
        )
        
        logger.info(f"Validação concluída. Score geral: {overall_quality_score:.1f}%")
        
        return report

    def save_validation_report(self, report: OpenMeteoDataReport) -> None:
        """
        Salva o relatório de validação em arquivo JSON

        Args:
            report: Relatório de validação
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"openmeteo_validation_report_{timestamp}.json"
        
        # Converter dataclasses para dict
        report_dict = asdict(report)
        
        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
        
        logger.info(f"Relatório salvo em: {output_file}")

    def print_validation_summary(self, report: OpenMeteoDataReport) -> None:
        """
        Imprime resumo da validação

        Args:
            report: Relatório de validação
        """
        print("\n" + "="*80)
        print("RELATÓRIO DE VALIDAÇÃO - DADOS OPEN-METEO HISTORICAL FORECAST")
        print("="*80)
        
        print(f"\n📊 RESUMO GERAL:")
        print(f"   • Validação realizada em: {report.validation_timestamp}")
        print(f"   • Total de chunks: {report.total_chunks}")
        print(f"   • Chunks válidos: {report.valid_chunks}")
        print(f"   • Chunks inválidos: {report.invalid_chunks}")
        print(f"   • Total de registros: {report.total_records:,}")
        print(f"   • Range temporal: {report.total_date_range[0]} até {report.total_date_range[1]}")
        print(f"   • Score geral de qualidade: {report.overall_quality_score:.1f}%")
        
        print(f"\n📁 ESTATÍSTICAS DOS ARQUIVOS:")
        print(f"   • Tamanho total: {report.summary_statistics['total_size_mb']:.1f} MB")
        print(f"   • Tamanho médio por chunk: {report.summary_statistics['avg_chunk_size_kb']:.1f} KB")
        print(f"   • Registros faltantes: {report.summary_statistics['total_missing_records']:,}")
        
        print(f"\n🌡️ VARIÁVEIS ATMOSFÉRICAS CRÍTICAS:")
        print(f"   • Qualidade média: {report.summary_statistics['critical_vars_avg_quality']:.1f}%")
        
        # Top 5 variáveis com melhor qualidade
        sorted_vars = sorted(
            report.consolidated_variable_stats.items(),
            key=lambda x: x[1]['mean_validation_rate'],
            reverse=True
        )
        
        print(f"\n✅ TOP 5 VARIÁVEIS COM MELHOR QUALIDADE:")
        for i, (var, stats) in enumerate(sorted_vars[:5]):
            print(f"   {i+1}. {var}: {stats['mean_validation_rate']:.1f}%")
        
        # Variáveis com problemas
        problematic_vars = [
            (var, stats) for var, stats in sorted_vars
            if stats['mean_validation_rate'] < 90
        ]
        
        if problematic_vars:
            print(f"\n⚠️ VARIÁVEIS COM QUALIDADE < 90%:")
            for var, stats in problematic_vars[:5]:
                print(f"   • {var}: {stats['mean_validation_rate']:.1f}%")
        
        # Análise de pressão atmosférica
        pressure_issues = report.atmospheric_pressure_analysis.get('issues', [])
        if pressure_issues:
            print(f"\n🌪️ PROBLEMAS EM NÍVEIS DE PRESSÃO:")
            for issue in pressure_issues[:3]:
                print(f"   • {issue}")
        
        # Gaps temporais
        if report.temporal_gaps:
            print(f"\n⏰ GAPS TEMPORAIS DETECTADOS:")
            print(f"   • Total de chunks com gaps: {len(report.temporal_gaps)}")
            total_missing = sum(gap['missing_timestamps'] for gap in report.temporal_gaps)
            print(f"   • Total de registros faltantes: {total_missing:,}")
        
        # Recomendações
        print(f"\n💡 RECOMENDAÇÕES:")
        if report.overall_quality_score >= 95:
            print("   ✅ Dados em excelente qualidade - proceder com Fase 2.2")
        elif report.overall_quality_score >= 85:
            print("   ⚠️ Dados em boa qualidade - proceder com cautela na Fase 2.2")
        elif report.overall_quality_score >= 70:
            print("   🔴 Dados com qualidade moderada - revisar problemas antes da Fase 2.2")
        else:
            print("   🚫 Dados com qualidade insuficiente - não proceder com Fase 2.2")
        
        print("="*80)


def main():
    """Função principal para validação dos dados Open-Meteo"""
    validator = OpenMeteoDataValidator()
    
    # Executar validação
    report = validator.validate_all_chunks()
    
    # Salvar relatório
    validator.save_validation_report(report)
    
    # Imprimir resumo
    validator.print_validation_summary(report)
    
    return report.overall_quality_score >= 85  # Retorna True se qualidade suficiente


if __name__ == "__main__":
    main() 