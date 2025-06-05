#!/usr/bin/env python3
"""
Script para coletar dados históricos da Open-Meteo (2000-2025)

Execução:
    python scripts/collect_open_meteo_data.py
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Adicionar o diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from app.features.forecast.infra.open_meteo_client import collect_historical_data_2000_2025


def setup_logging():
    """Configura logging detalhado"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/open_meteo_collection.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_directories():
    """Cria diretórios necessários"""
    directories = [
        "data/raw/open_meteo",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Diretório criado: {directory}")


async def main():
    """Função principal"""
    print("🌤️  Iniciando coleta de dados históricos da Open-Meteo")
    print("📅 Período: 2000-2025")
    print("📍 Local: Porto Alegre, RS")
    print("=" * 50)
    
    # Configurar ambiente
    create_directories()
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("Iniciando coleta de dados Open-Meteo")
    
    try:
        # Coletar dados
        hourly_df, daily_df = await collect_historical_data_2000_2025()
        
        # Estatísticas finais
        if hourly_df is not None:
            print(f"\n✅ Coleta concluída com sucesso!")
            print(f"📊 Dados horários: {len(hourly_df):,} registros")
            print(f"📊 Período horário: {hourly_df.index.min()} até {hourly_df.index.max()}")
            print(f"📊 Variáveis horárias: {len(hourly_df.columns)}")
            
        if daily_df is not None:
            print(f"📊 Dados diários: {len(daily_df):,} registros")
            print(f"📊 Período diário: {daily_df.index.min()} até {daily_df.index.max()}")
            print(f"📊 Variáveis diárias: {len(daily_df.columns)}")
            
        print(f"\n💾 Arquivos salvos em: data/raw/open_meteo/")
        print("🎯 Dados prontos para treinamento do modelo!")
        
    except Exception as e:
        logger.error(f"Erro durante a coleta: {e}")
        print(f"\n❌ Erro durante a coleta: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Verificar dependências
    try:
        import httpx
        import pandas as pd
    except ImportError as e:
        print(f"❌ Dependência faltando: {e}")
        print("💡 Execute: pip install httpx pandas")
        sys.exit(1)
    
    # Executar coleta
    asyncio.run(main()) 