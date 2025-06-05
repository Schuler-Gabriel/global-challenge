#!/usr/bin/env python3
"""
Script de Setup e Organização de Dados Meteorológicos INMET
=========================================================

Este script consolida e organiza automaticamente os dados meteorológicos 
históricos do INMET (2000-2025) para o projeto de alertas de cheias.

Features:
- Consolidação automática de CSVs por ano
- Validação de integridade dos dados
- Detecção de arquivos corrompidos
- Geração de metadados e relatórios

Autor: Sistema de Alertas de Cheias - Rio Guaíba
Data: 2025
"""

import os
import re
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib
import json
from dataclasses import dataclass, asdict


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    """Metadados de arquivo CSV INMET"""
    filename: str
    station_code: str
    station_name: str
    year: int
    start_date: str
    end_date: str
    file_size: int
    num_records: int
    md5_hash: str
    is_valid: bool
    error_message: Optional[str] = None


@dataclass
class DataQualityReport:
    """Relatório de qualidade dos dados"""
    total_files: int
    valid_files: int
    corrupted_files: int
    total_records: int
    date_coverage: Dict[str, List[int]]
    missing_data_percentage: float
    file_metadata: List[FileMetadata]


class INMETDataSetup:
    """
    Classe principal para setup e organização dos dados INMET
    """
    
    # Configurações dos dados INMET
    RAW_DATA_PATH = Path("data/raw/dados_historicos")
    PROCESSED_DATA_PATH = Path("data/processed")
    METADATA_PATH = Path("data/metadata")
    
    # Padrões de arquivo
    FILE_PATTERN = re.compile(
        r'INMET_S_RS_([AB]\d{3})_(.+?)_(\d{2}-\d{2}-\d{4})_A_(\d{2}-\d{2}-\d{4})\.CSV'
    )
    
    # Colunas principais esperadas nos dados INMET
    EXPECTED_COLUMNS = {
        'datetime': ['Data', 'Hora UTC'],
        'precipitation': 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)',
        'pressure': 'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)',
        'pressure_max': 'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)',
        'pressure_min': 'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)',
        'radiation': 'RADIACAO GLOBAL (Kj/m²)',
        'temperature': 'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)',
        'dew_point': 'TEMPERATURA DO PONTO DE ORVALHO (°C)',
        'temp_max': 'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)',
        'temp_min': 'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)',
        'dew_point_max': 'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)',
        'dew_point_min': 'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)',
        'humidity_max': 'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)',
        'humidity_min': 'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)',
        'humidity': 'UMIDADE RELATIVA DO AR, HORARIA (%)',
        'wind_direction': 'VENTO, DIREÇÃO HORARIA (gr) (° (gr))',
        'wind_gust': 'VENTO, RAJADA MAXIMA (m/s)',
        'wind_speed': 'VENTO, VELOCIDADE HORARIA (m/s)'
    }
    
    def __init__(self):
        """Inicializa o setup de dados"""
        self.setup_directories()
        
    def setup_directories(self) -> None:
        """Cria diretórios necessários se não existirem"""
        self.PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        self.METADATA_PATH.mkdir(parents=True, exist_ok=True)
        logger.info("Diretórios de trabalho configurados")
        
    def scan_data_files(self) -> List[Path]:
        """
        Escaneia diretório de dados brutos e retorna lista de arquivos CSV
        
        Returns:
            Lista de caminhos para arquivos CSV encontrados
        """
        if not self.RAW_DATA_PATH.exists():
            logger.error(f"Diretório de dados não encontrado: {self.RAW_DATA_PATH}")
            return []
            
        csv_files = list(self.RAW_DATA_PATH.glob("*.CSV"))
        logger.info(f"Encontrados {len(csv_files)} arquivos CSV")
        return csv_files
    
    def parse_filename(self, filepath: Path) -> Optional[Dict[str, str]]:
        """
        Extrai metadados do nome do arquivo INMET
        
        Args:
            filepath: Caminho para arquivo CSV
            
        Returns:
            Dicionário com metadados extraídos ou None se inválido
        """
        match = self.FILE_PATTERN.match(filepath.name)
        if not match:
            logger.warning(f"Nome de arquivo inválido: {filepath.name}")
            return None
            
        station_code, station_name, start_date, end_date = match.groups()
        
        # Converte datas para formato padrão
        try:
            start_dt = datetime.strptime(start_date, "%d-%m-%Y")
            end_dt = datetime.strptime(end_date, "%d-%m-%Y")
            year = start_dt.year
        except ValueError as e:
            logger.error(f"Erro ao processar datas em {filepath.name}: {e}")
            return None
            
        return {
            'station_code': station_code,
            'station_name': station_name.strip(),
            'year': year,
            'start_date': start_dt.strftime("%Y-%m-%d"),
            'end_date': end_dt.strftime("%Y-%m-%d")
        }
    
    def calculate_file_hash(self, filepath: Path) -> str:
        """
        Calcula hash MD5 do arquivo para verificação de integridade
        
        Args:
            filepath: Caminho para arquivo
            
        Returns:
            Hash MD5 em formato hexadecimal
        """
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Erro ao calcular hash de {filepath}: {e}")
            return ""
    
    def validate_csv_structure(self, filepath: Path) -> Tuple[bool, Optional[str], int]:
        """
        Valida estrutura básica do arquivo CSV INMET
        
        Args:
            filepath: Caminho para arquivo CSV
            
        Returns:
            Tupla (is_valid, error_message, num_records)
        """
        try:
            # Lê apenas as primeiras linhas para verificar estrutura
            df_sample = pd.read_csv(
                filepath, 
                sep=';', 
                encoding='latin-1',
                nrows=10,
                skiprows=8  # Pula cabeçalho INMET
            )
            
            # Verifica se tem as colunas principais
            expected_cols = ['Data', 'Hora UTC']
            missing_cols = [col for col in expected_cols if col not in df_sample.columns]
            
            if missing_cols:
                return False, f"Colunas faltando: {missing_cols}", 0
            
            # Conta total de registros (aproximado)
            with open(filepath, 'r', encoding='latin-1') as f:
                num_lines = sum(1 for line in f)
            num_records = max(0, num_lines - 9)  # Subtrai cabeçalho
            
            return True, None, num_records
            
        except Exception as e:
            return False, f"Erro ao ler arquivo: {str(e)}", 0
    
    def process_file_metadata(self, filepath: Path) -> FileMetadata:
        """
        Processa metadados completos de um arquivo
        
        Args:
            filepath: Caminho para arquivo CSV
            
        Returns:
            Objeto FileMetadata com informações do arquivo
        """
        logger.info(f"Processando metadados: {filepath.name}")
        
        # Parse do nome do arquivo
        file_info = self.parse_filename(filepath)
        if not file_info:
            return FileMetadata(
                filename=filepath.name,
                station_code="UNKNOWN",
                station_name="UNKNOWN",
                year=0,
                start_date="",
                end_date="",
                file_size=0,
                num_records=0,
                md5_hash="",
                is_valid=False,
                error_message="Nome de arquivo inválido"
            )
        
        # Informações básicas do arquivo
        file_size = filepath.stat().st_size
        md5_hash = self.calculate_file_hash(filepath)
        
        # Validação da estrutura
        is_valid, error_message, num_records = self.validate_csv_structure(filepath)
        
        return FileMetadata(
            filename=filepath.name,
            station_code=file_info['station_code'],
            station_name=file_info['station_name'],
            year=file_info['year'],
            start_date=file_info['start_date'],
            end_date=file_info['end_date'],
            file_size=file_size,
            num_records=num_records,
            md5_hash=md5_hash,
            is_valid=is_valid,
            error_message=error_message
        )
    
    def consolidate_by_year(self, file_metadata_list: List[FileMetadata]) -> Dict[int, List[FileMetadata]]:
        """
        Consolida arquivos por ano
        
        Args:
            file_metadata_list: Lista de metadados de arquivos
            
        Returns:
            Dicionário agrupado por ano
        """
        consolidated = {}
        
        for metadata in file_metadata_list:
            if metadata.is_valid and metadata.year > 0:
                if metadata.year not in consolidated:
                    consolidated[metadata.year] = []
                consolidated[metadata.year].append(metadata)
        
        # Ordena por ano
        return dict(sorted(consolidated.items()))
    
    def generate_coverage_report(self, file_metadata_list: List[FileMetadata]) -> Dict[str, List[int]]:
        """
        Gera relatório de cobertura temporal por estação
        
        Args:
            file_metadata_list: Lista de metadados de arquivos
            
        Returns:
            Dicionário com cobertura por estação
        """
        coverage = {}
        
        for metadata in file_metadata_list:
            if metadata.is_valid:
                station_key = f"{metadata.station_code}_{metadata.station_name}"
                if station_key not in coverage:
                    coverage[station_key] = []
                coverage[station_key].append(metadata.year)
        
        # Remove duplicatas e ordena
        for station in coverage:
            coverage[station] = sorted(list(set(coverage[station])))
        
        return coverage
    
    def run_setup(self) -> DataQualityReport:
        """
        Executa processo completo de setup dos dados
        
        Returns:
            Relatório de qualidade dos dados
        """
        logger.info("=== INICIANDO SETUP DE DADOS INMET ===")
        
        # Escaneia arquivos
        csv_files = self.scan_data_files()
        if not csv_files:
            logger.error("Nenhum arquivo CSV encontrado")
            return DataQualityReport(
                total_files=0,
                valid_files=0,
                corrupted_files=0,
                total_records=0,
                date_coverage={},
                missing_data_percentage=0.0,
                file_metadata=[]
            )
        
        # Processa metadados de cada arquivo
        logger.info("Processando metadados dos arquivos...")
        file_metadata_list = []
        
        for csv_file in csv_files:
            try:
                metadata = self.process_file_metadata(csv_file)
                file_metadata_list.append(metadata)
            except Exception as e:
                logger.error(f"Erro ao processar {csv_file}: {e}")
                continue
        
        # Gera estatísticas
        valid_files = [m for m in file_metadata_list if m.is_valid]
        corrupted_files = [m for m in file_metadata_list if not m.is_valid]
        total_records = sum(m.num_records for m in valid_files)
        
        # Consolida por ano
        consolidated = self.consolidate_by_year(valid_files)
        
        # Gera relatório de cobertura
        coverage = self.generate_coverage_report(valid_files)
        
        # Calcula percentual de dados faltantes (estimativa baseada em arquivos corrompidos)
        missing_percentage = (len(corrupted_files) / len(file_metadata_list) * 100) if file_metadata_list else 0
        
        # Cria relatório final
        report = DataQualityReport(
            total_files=len(file_metadata_list),
            valid_files=len(valid_files),
            corrupted_files=len(corrupted_files),
            total_records=total_records,
            date_coverage=coverage,
            missing_data_percentage=missing_percentage,
            file_metadata=file_metadata_list
        )
        
        # Salva metadados
        self.save_metadata(report, consolidated)
        
        # Log do resultado
        logger.info("=== SETUP CONCLUÍDO ===")
        logger.info(f"Total de arquivos: {report.total_files}")
        logger.info(f"Arquivos válidos: {report.valid_files}")
        logger.info(f"Arquivos corrompidos: {report.corrupted_files}")
        logger.info(f"Total de registros: {report.total_records:,}")
        logger.info(f"Dados faltantes: {report.missing_data_percentage:.1f}%")
        
        return report
    
    def save_metadata(self, report: DataQualityReport, consolidated: Dict[int, List[FileMetadata]]) -> None:
        """
        Salva metadados e relatórios em arquivos JSON
        
        Args:
            report: Relatório de qualidade
            consolidated: Dados consolidados por ano
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salva relatório completo
        report_file = self.METADATA_PATH / f"data_quality_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False, default=str)
        
        # Salva consolidação por ano
        consolidated_file = self.METADATA_PATH / f"consolidated_by_year_{timestamp}.json"
        consolidated_dict = {
            str(year): [asdict(metadata) for metadata in metadata_list]
            for year, metadata_list in consolidated.items()
        }
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_dict, f, indent=2, ensure_ascii=False, default=str)
        
        # Salva resumo simples
        summary_file = self.METADATA_PATH / "data_summary.json"
        summary = {
            'last_updated': timestamp,
            'total_files': report.total_files,
            'valid_files': report.valid_files,
            'corrupted_files': report.corrupted_files,
            'total_records': report.total_records,
            'years_available': sorted(consolidated.keys()),
            'stations': list(report.date_coverage.keys()),
            'missing_data_percentage': report.missing_data_percentage
        }
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metadados salvos em: {self.METADATA_PATH}")
    
    def print_summary(self, report: DataQualityReport) -> None:
        """
        Imprime resumo detalhado do setup
        
        Args:
            report: Relatório de qualidade dos dados
        """
        print("\n" + "="*60)
        print("         RELATÓRIO DE SETUP - DADOS INMET")
        print("="*60)
        
        print(f"\n📊 ESTATÍSTICAS GERAIS:")
        print(f"   Total de arquivos encontrados: {report.total_files}")
        print(f"   Arquivos válidos: {report.valid_files}")
        print(f"   Arquivos corrompidos: {report.corrupted_files}")
        print(f"   Total de registros: {report.total_records:,}")
        print(f"   Dados faltantes: {report.missing_data_percentage:.1f}%")
        
        print(f"\n🏢 COBERTURA POR ESTAÇÃO:")
        for station, years in report.date_coverage.items():
            print(f"   {station}: {min(years)}-{max(years)} ({len(years)} anos)")
        
        if report.corrupted_files > 0:
            print(f"\n❌ ARQUIVOS COM PROBLEMAS:")
            for metadata in report.file_metadata:
                if not metadata.is_valid:
                    print(f"   {metadata.filename}: {metadata.error_message}")
        
        print(f"\n✅ Setup concluído com sucesso!")
        print("="*60)


def main():
    """Função principal para execução do script"""
    setup = INMETDataSetup()
    
    try:
        report = setup.run_setup()
        setup.print_summary(report)
        return 0
    except Exception as e:
        logger.error(f"Erro durante setup: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 