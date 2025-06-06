#!/usr/bin/env python3
"""
Script para Consolidar Chunks do Open-Meteo Historical Forecast
Sistema de Alertas de Cheias - Rio Guaíba

Consolida todos os chunks JSON em um único arquivo pandas DataFrame
para facilitar o feature engineering atmosférico.

Author: Sistema de Previsão Meteorológica
Date: 2025-06-06
"""

import json
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_chunk(chunk_path: Path) -> pd.DataFrame:
    """
    Carrega um chunk JSON e converte para DataFrame.
    
    Args:
        chunk_path: Caminho para o arquivo chunk
        
    Returns:
        DataFrame com dados do chunk
    """
    logger.info(f"Carregando chunk: {chunk_path.name}")
    
    with open(chunk_path, 'r') as f:
        data = json.load(f)
    
    # Extrair dados hourly
    hourly_data = data['hourly']
    
    # Converter para DataFrame
    df = pd.DataFrame(hourly_data)
    
    # Converter coluna time para datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Adicionar metadados
    df['latitude'] = data['latitude']
    df['longitude'] = data['longitude']
    df['elevation'] = data['elevation']
    
    logger.info(f"Chunk carregado: {len(df)} registros de {df['time'].min()} a {df['time'].max()}")
    
    return df

def consolidate_chunks(data_dir: Path) -> pd.DataFrame:
    """
    Consolida todos os chunks em um único DataFrame.
    
    Args:
        data_dir: Diretório com os chunks
        
    Returns:
        DataFrame consolidado
    """
    logger.info("Iniciando consolidação de chunks...")
    
    # Encontrar todos os chunks
    chunk_files = list(data_dir.glob("chunk_*.json"))
    chunk_files.sort()
    
    if not chunk_files:
        raise ValueError(f"Nenhum chunk encontrado em {data_dir}")
    
    logger.info(f"Encontrados {len(chunk_files)} chunks para consolidar")
    
    # Carregar todos os chunks
    dataframes = []
    
    for chunk_file in chunk_files:
        try:
            df = load_chunk(chunk_file)
            dataframes.append(df)
        except Exception as e:
            logger.error(f"Erro ao carregar chunk {chunk_file.name}: {e}")
            continue
    
    if not dataframes:
        raise ValueError("Nenhum chunk válido foi carregado")
    
    # Concatenar todos os DataFrames
    logger.info("Concatenando DataFrames...")
    consolidated_df = pd.concat(dataframes, ignore_index=True)
    
    # Remover duplicatas por timestamp
    logger.info("Removendo duplicatas...")
    consolidated_df = consolidated_df.drop_duplicates(subset=['time'], keep='first')
    
    # Ordenar por tempo
    consolidated_df = consolidated_df.sort_values('time').reset_index(drop=True)
    
    logger.info(f"Consolidação concluída: {len(consolidated_df)} registros únicos")
    logger.info(f"Período: {consolidated_df['time'].min()} a {consolidated_df['time'].max()}")
    
    return consolidated_df

def save_consolidated_data(df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    """
    Salva os dados consolidados em diferentes formatos.
    
    Args:
        df: DataFrame consolidado
        output_dir: Diretório de saída
        
    Returns:
        Dict com caminhos dos arquivos salvos
    """
    output_dir.mkdir(exist_ok=True)
    
    saved_files = {}
    
    # 1. Salvar como Parquet (mais eficiente)
    parquet_path = output_dir / "openmeteo_historical_forecast_consolidated.parquet"
    logger.info(f"Salvando como Parquet: {parquet_path}")
    df.to_parquet(parquet_path, index=False)
    saved_files['parquet'] = parquet_path
    
    # 2. Salvar como CSV (compatibilidade)
    csv_path = output_dir / "openmeteo_historical_forecast_consolidated.csv"
    logger.info(f"Salvando como CSV: {csv_path}")
    df.to_csv(csv_path, index=False)
    saved_files['csv'] = csv_path
    
    # 3. Salvar metadados
    metadata = {
        'total_records': len(df),
        'start_date': df['time'].min().isoformat(),
        'end_date': df['time'].max().isoformat(),
        'variables': list(df.columns),
        'total_variables': len(df.columns),
        'pressure_levels': ['1000hPa', '850hPa', '700hPa', '500hPa', '300hPa'],
        'surface_variables': [col for col in df.columns if not any(level in col for level in ['1000hPa', '850hPa', '700hPa', '500hPa', '300hPa'])],
        'consolidation_date': datetime.now().isoformat()
    }
    
    metadata_path = output_dir / "consolidation_metadata.json"
    logger.info(f"Salvando metadados: {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    saved_files['metadata'] = metadata_path
    
    return saved_files

def main():
    """Função principal para consolidação dos chunks."""
    logger.info("=== Iniciando Consolidação de Chunks Open-Meteo ===")
    
    # Configuração de caminhos
    data_dir = Path("data/raw/Open-Meteo Historical Forecast")
    output_dir = Path("data/processed")
    
    try:
        # Consolidar chunks
        consolidated_df = consolidate_chunks(data_dir)
        
        # Salvar dados consolidados
        saved_files = save_consolidated_data(consolidated_df, output_dir)
        
        # Relatório final
        logger.info("=== Consolidação Concluída ===")
        logger.info(f"Total de registros: {len(consolidated_df):,}")
        logger.info(f"Total de variáveis: {len(consolidated_df.columns)}")
        logger.info(f"Período: {consolidated_df['time'].min()} a {consolidated_df['time'].max()}")
        
        # Verificar variáveis de pressão
        pressure_vars = [col for col in consolidated_df.columns if any(level in col for level in ['850hPa', '500hPa'])]
        logger.info(f"Variáveis de pressão encontradas: {len(pressure_vars)}")
        
        # Arquivos salvos
        logger.info("Arquivos salvos:")
        for format_type, file_path in saved_files.items():
            logger.info(f"  {format_type}: {file_path}")
        
        # Verificação de qualidade básica
        missing_data = consolidated_df.isnull().sum()
        if missing_data.any():
            logger.warning("Dados faltantes encontrados:")
            for col, missing_count in missing_data[missing_data > 0].items():
                logger.warning(f"  {col}: {missing_count} valores faltantes")
        else:
            logger.info("✅ Nenhum dado faltante encontrado")
        
        logger.info("Consolidação concluída com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante consolidação: {e}")
        raise

if __name__ == "__main__":
    main() 