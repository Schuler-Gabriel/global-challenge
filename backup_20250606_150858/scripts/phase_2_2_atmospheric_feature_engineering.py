#!/usr/bin/env python3
"""
Fase 2.2: Engenharia de Características Atmosféricas
Sistema de Alertas de Cheias - Rio Guaíba

Este script implementa a consolidação dos dados fragmentados do Open-Meteo 
e a engenharia de características atmosféricas para detecção de frentes frias
e vórtices, visando melhorar a precisão das previsões (meta: 82-87%).

Funcionalidades:
- Consolidação de chunks JSON do Open-Meteo Historical Forecast
- Engenharia de características atmosféricas (850hPa/500hPa)
- Detecção de padrões meteorológicos (frentes frias, vórtices)
- Geração de variáveis derivadas para modelos LSTM
- Organização por ano ou arquivo único conforme especificado

Author: Sistema de Previsão Meteorológica
Date: 2025-01-13
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# Add app to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.core.exceptions import DataValidationError
from app.core.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/atmospheric_feature_engineering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AtmosphericFeatureEngineer:
    """
    Processador de dados atmosféricos do Open-Meteo com foco em 
    características de pressão para detecção de frentes frias e vórtices.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Inicializa o processador de características atmosféricas.
        
        Args:
            data_path: Caminho para dados brutos do Open-Meteo
        """
        self.settings = get_settings()
        self.data_path = data_path or Path("data/raw/Open-Meteo Historical Forecast")
        self.processed_path = Path("data/processed")
        self.processed_path.mkdir(exist_ok=True)
        
        # Configurações para detecção de padrões atmosféricos
        self.pressure_levels = ['850hPa', '500hPa', '1000hPa', '700hPa', '300hPa']
        self.atmospheric_vars = [
            'temperature_{}', 'relative_humidity_{}', 'wind_speed_{}', 
            'wind_direction_{}', 'geopotential_height_{}'
        ]
        
        logger.info(f"AtmosphericFeatureEngineer inicializado: {self.data_path}")
    
    def consolidate_forecast_chunks(self, output_format: str = "yearly") -> Dict[str, Path]:
        """
        Consolida os chunks JSON do Open-Meteo Historical Forecast.
        
        Args:
            output_format: "yearly" para arquivos por ano ou "single" para arquivo único
            
        Returns:
            Dict com paths dos arquivos consolidados
        """
        logger.info(f"Iniciando consolidação de chunks - formato: {output_format}")
        
        # Listar e ordenar chunks
        chunk_files = list(self.data_path.glob("chunk_*.json"))
        chunk_files.sort()
        
        if not chunk_files:
            raise DataValidationError("Nenhum chunk encontrado no diretório de dados")
        
        logger.info(f"Encontrados {len(chunk_files)} chunks para processar")
        
        # Carregar e consolidar dados
        all_data = []
        
        for chunk_file in chunk_files:
            logger.info(f"Processando chunk: {chunk_file.name}")
            
            try:
                with open(chunk_file, 'r') as f:
                    chunk_data = json.load(f)
                
                # Validar estrutura do chunk
                if not self._validate_chunk_structure(chunk_data):
                    logger.warning(f"Estrutura inválida no chunk: {chunk_file.name}")
                    continue
                
                # Converter para DataFrame para facilitar processamento
                df = self._chunk_to_dataframe(chunk_data)
                all_data.append(df)
                
            except Exception as e:
                logger.error(f"Erro ao processar chunk {chunk_file.name}: {e}")
                continue
        
        # Concatenar todos os dados
        if not all_data:
            raise DataValidationError("Nenhum chunk válido encontrado")
        
        consolidated_df = pd.concat(all_data, ignore_index=True)
        consolidated_df = consolidated_df.sort_values('time').reset_index(drop=True)
        
        logger.info(f"Dados consolidados: {len(consolidated_df)} registros de {consolidated_df['time'].min()} a {consolidated_df['time'].max()}")
        
        # Salvar conforme formato especificado
        output_files = {}
        
        if output_format == "yearly":
            output_files = self._save_yearly_files(consolidated_df)
        else:
            output_files = self._save_single_file(consolidated_df)
        
        return output_files
    
    def engineer_atmospheric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica engenharia de características atmosféricas para detecção
        de padrões meteorológicos importantes.
        
        Args:
            df: DataFrame com dados atmosféricos consolidados
            
        Returns:
            DataFrame com características atmosféricas derivadas
        """
        logger.info("Iniciando engenharia de características atmosféricas")
        
        df = df.copy()
        
        # 1. Gradientes de pressão atmosférica
        df = self._calculate_pressure_gradients(df)
        
        # 2. Indicadores de instabilidade atmosférica  
        df = self._calculate_stability_indices(df)
        
        # 3. Detecção de frentes frias
        df = self._detect_cold_fronts(df)
        
        # 4. Detecção de vórtices atmosféricos
        df = self._detect_atmospheric_vortices(df)
        
        # 5. Características de vento em diferentes níveis
        df = self._calculate_wind_features(df)
        
        # 6. Índices de convergência/divergência
        df = self._calculate_convergence_indices(df)
        
        # 7. Características temporais (tendências)
        df = self._calculate_temporal_features(df)
        
        logger.info(f"Engenharia de características concluída. Total de colunas: {len(df.columns)}")
        
        return df
    
    def _validate_chunk_structure(self, chunk_data: Dict) -> bool:
        """Valida se o chunk tem a estrutura esperada do Open-Meteo"""
        required_keys = ['latitude', 'longitude', 'hourly_units', 'hourly']
        
        if not all(key in chunk_data for key in required_keys):
            return False
        
        hourly_data = chunk_data['hourly']
        if 'time' not in hourly_data:
            return False
        
        # Verificar se há dados de pressão atmosférica
        pressure_vars = [f'temperature_{level.lower()}' for level in self.pressure_levels]
        has_pressure_data = any(var in hourly_data for var in pressure_vars)
        
        return has_pressure_data
    
    def _chunk_to_dataframe(self, chunk_data: Dict) -> pd.DataFrame:
        """Converte dados JSON do chunk para DataFrame"""
        hourly_data = chunk_data['hourly']
        
        # Criar DataFrame base
        df = pd.DataFrame()
        df['time'] = pd.to_datetime(hourly_data['time'])
        
        # Adicionar coordenadas
        df['latitude'] = chunk_data['latitude']
        df['longitude'] = chunk_data['longitude']
        
        # Adicionar todas as variáveis meteorológicas disponíveis
        for var_name, values in hourly_data.items():
            if var_name != 'time':
                df[var_name] = values
        
        return df
    
    def _save_yearly_files(self, df: pd.DataFrame) -> Dict[str, Path]:
        """Salva dados consolidados em arquivos separados por ano"""
        output_files = {}
        
        df['year'] = df['time'].dt.year
        years = df['year'].unique()
        
        for year in sorted(years):
            year_data = df[df['year'] == year].drop('year', axis=1)
            
            output_path = self.processed_path / f"openmeteo_forecast_{year}.parquet"
            year_data.to_parquet(output_path, index=False)
            
            output_files[str(year)] = output_path
            logger.info(f"Salvo arquivo para {year}: {len(year_data)} registros")
        
        return output_files
    
    def _save_single_file(self, df: pd.DataFrame) -> Dict[str, Path]:
        """Salva todos os dados em um único arquivo"""
        output_path = self.processed_path / "openmeteo_forecast_consolidated.parquet"
        df.to_parquet(output_path, index=False)
        
        logger.info(f"Salvo arquivo consolidado: {len(df)} registros")
        return {"consolidated": output_path}
    
    def _calculate_pressure_gradients(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula gradientes de pressão entre níveis atmosféricos"""
        logger.info("Calculando gradientes de pressão atmosférica")
        
        # Gradientes verticais entre níveis de pressão
        pressure_pairs = [
            ('1000hPa', '850hPa'),
            ('850hPa', '700hPa'), 
            ('700hPa', '500hPa'),
            ('500hPa', '300hPa')
        ]
        
        for level1, level2 in pressure_pairs:
            temp1_col = f'temperature_{level1.lower()}'
            temp2_col = f'temperature_{level2.lower()}'
            
            if temp1_col in df.columns and temp2_col in df.columns:
                # Gradiente térmico vertical
                df[f'temp_gradient_{level1}_{level2}'] = (
                    df[temp2_col] - df[temp1_col]
                )
                
                # Gradiente de altura geopotencial
                height1_col = f'geopotential_height_{level1.lower()}'
                height2_col = f'geopotential_height_{level2.lower()}'
                
                if height1_col in df.columns and height2_col in df.columns:
                    df[f'height_gradient_{level1}_{level2}'] = (
                        df[height2_col] - df[height1_col]
                    )
        
        # Gradiente horizontal de pressão (aproximação)
        if 'pressure_msl' in df.columns:
            df['pressure_gradient_temporal'] = df['pressure_msl'].diff()
            df['pressure_tendency_3h'] = df['pressure_msl'].diff(periods=3)
        
        return df
    
    def _calculate_stability_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula índices de estabilidade atmosférica"""
        logger.info("Calculando índices de estabilidade atmosférica")
        
        # Lifted Index (simplificado)
        temp_850 = f'temperature_850hpa'
        temp_500 = f'temperature_500hpa'
        
        if temp_850 in df.columns and temp_500 in df.columns:
            # Aproximação do Lifted Index
            df['lifted_index'] = df[temp_500] - (df[temp_850] - 9.8 * 3.5)  # Aproximação
            
        # Índice de instabilidade baseado em gradiente térmico
        if 'temp_gradient_850hPa_500hPa' in df.columns:
            df['thermal_instability'] = -df['temp_gradient_850hPa_500hPa']
        
        # CAPE simplificado usando dados disponíveis
        if 'cape' in df.columns:
            df['cape_normalized'] = df['cape'] / 1000.0  # Normalizar
            df['cape_categorical'] = pd.cut(
                df['cape'], 
                bins=[0, 1000, 2500, 4000, float('inf')],
                labels=['stable', 'moderate', 'strong', 'extreme']
            )
        
        return df
    
    def _detect_cold_fronts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta passagem de frentes frias usando indicadores atmosféricos"""
        logger.info("Detectando frentes frias")
        
        # Critérios para detecção de frente fria:
        # 1. Queda brusca de temperatura
        # 2. Mudança de direção do vento
        # 3. Aumento da pressão atmosférica
        # 4. Mudança na umidade
        
        # 1. Detecção por queda de temperatura
        temp_drop_threshold = -2.0  # °C em 3 horas
        if 'temperature_2m' in df.columns:
            temp_change_3h = df['temperature_2m'].diff(periods=3)
            df['temp_drop_3h'] = temp_change_3h < temp_drop_threshold
        
        # 2. Mudança significativa na direção do vento
        if 'winddirection_10m' in df.columns:
            wind_dir_change = df['winddirection_10m'].diff().abs()
            # Lidar com mudança de 360° para 0°
            wind_dir_change = np.minimum(wind_dir_change, 360 - wind_dir_change)
            df['wind_direction_change'] = wind_dir_change
            df['significant_wind_change'] = wind_dir_change > 45  # Mudança > 45°
        
        # 3. Tendência de pressão
        if 'pressure_msl' in df.columns:
            pressure_change_3h = df['pressure_msl'].diff(periods=3)
            df['pressure_rise_3h'] = pressure_change_3h > 2.0  # hPa
        
        # 4. Índice combinado de frente fria
        front_indicators = []
        if 'temp_drop_3h' in df.columns:
            front_indicators.append('temp_drop_3h')
        if 'significant_wind_change' in df.columns:
            front_indicators.append('significant_wind_change')
        if 'pressure_rise_3h' in df.columns:
            front_indicators.append('pressure_rise_3h')
        
        if front_indicators:
            df['cold_front_probability'] = (
                df[front_indicators].sum(axis=1) / len(front_indicators)
            )
            df['cold_front_detected'] = df['cold_front_probability'] > 0.6
        
        return df
    
    def _detect_atmospheric_vortices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta vórtices atmosféricos usando dados de vento em diferentes níveis"""
        logger.info("Detectando vórtices atmosféricos")
        
        # Detectar vorticidade relativa usando velocidade e direção do vento
        for level in ['10m', '850hpa', '500hpa']:
            wind_speed_col = f'wind_speed_{level}' if level != '10m' else f'windspeed_{level}'
            wind_dir_col = f'wind_direction_{level}' if level != '10m' else f'winddirection_{level}'
            
            if wind_speed_col in df.columns and wind_dir_col in df.columns:
                # Componentes u e v do vento
                wind_rad = np.radians(df[wind_dir_col])
                df[f'wind_u_{level}'] = -df[wind_speed_col] * np.sin(wind_rad)
                df[f'wind_v_{level}'] = -df[wind_speed_col] * np.cos(wind_rad)
                
                # Aproximação da vorticidade (diferença temporal)
                du_dt = df[f'wind_u_{level}'].diff()
                dv_dt = df[f'wind_v_{level}'].diff()
                df[f'vorticity_approx_{level}'] = dv_dt - du_dt
                
                # Detecção de vórtice (alta vorticidade)
                vorticity_threshold = df[f'vorticity_approx_{level}'].std() * 2
                df[f'vortex_detected_{level}'] = (
                    df[f'vorticity_approx_{level}'].abs() > vorticity_threshold
                )
        
        # Índice combinado de vórtice
        vortex_cols = [col for col in df.columns if col.startswith('vortex_detected_')]
        if vortex_cols:
            df['vortex_probability'] = df[vortex_cols].sum(axis=1) / len(vortex_cols)
            df['atmospheric_vortex'] = df['vortex_probability'] > 0.5
        
        return df
    
    def _calculate_wind_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula características do vento em diferentes níveis"""
        logger.info("Calculando características do vento")
        
        # Wind shear entre níveis
        wind_pairs = [
            ('10m', '850hpa'),
            ('850hpa', '500hpa')
        ]
        
        for level1, level2 in wind_pairs:
            speed1 = f'windspeed_{level1}' if level1 == '10m' else f'wind_speed_{level1}'
            speed2 = f'wind_speed_{level2}'
            
            if speed1 in df.columns and speed2 in df.columns:
                df[f'wind_shear_{level1}_{level2}'] = df[speed2] - df[speed1]
        
        # Rajadas de vento
        if 'windgusts_10m' in df.columns and 'windspeed_10m' in df.columns:
            df['gust_factor'] = df['windgusts_10m'] / (df['windspeed_10m'] + 0.1)
            df['strong_gusts'] = df['windgusts_10m'] > 15  # m/s
        
        # Persistência da direção do vento
        if 'winddirection_10m' in df.columns:
            # Desvio padrão da direção do vento nas últimas 6 horas
            df['wind_direction_std_6h'] = (
                df['winddirection_10m'].rolling(window=6, min_periods=3).std()
            )
            df['wind_direction_stable'] = df['wind_direction_std_6h'] < 30
        
        return df
    
    def _calculate_convergence_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula índices de convergência/divergência atmosférica"""
        logger.info("Calculando índices de convergência")
        
        # Convergência de umidade (aproximação)
        if 'relative_humidity_2m' in df.columns:
            rh_change = df['relative_humidity_2m'].diff()
            df['humidity_convergence'] = rh_change > 5  # Aumento > 5%
        
        # Convergência de temperatura em diferentes níveis
        temp_levels = ['850hpa', '700hpa', '500hpa']
        for level in temp_levels:
            temp_col = f'temperature_{level}'
            if temp_col in df.columns:
                temp_change = df[temp_col].diff()
                df[f'temp_convergence_{level}'] = temp_change.abs() > 2
        
        return df
    
    def _calculate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula características temporais e tendências"""
        logger.info("Calculando características temporais")
        
        # Tendências de 6h, 12h, 24h para variáveis principais
        key_vars = ['temperature_2m', 'pressure_msl', 'relative_humidity_2m']
        windows = [6, 12, 24]
        
        for var in key_vars:
            if var in df.columns:
                for window in windows:
                    # Tendência linear
                    df[f'{var}_trend_{window}h'] = (
                        df[var] - df[var].shift(window)
                    )
                    
                    # Variabilidade
                    df[f'{var}_std_{window}h'] = (
                        df[var].rolling(window=window, min_periods=window//2).std()
                    )
        
        # Ciclo diurno
        df['hour'] = df['time'].dt.hour
        df['day_of_year'] = df['time'].dt.dayofyear
        
        # Sazonalidade
        df['season'] = df['time'].dt.month % 12 // 3 + 1  # 1=spring, 2=summer, 3=fall, 4=winter
        
        return df
    
    def process_complete_dataset(
        self, 
        consolidation_format: str = "yearly",
        save_features: bool = True
    ) -> Dict[str, Any]:
        """
        Processa o dataset completo: consolida chunks e aplica engenharia de características.
        
        Args:
            consolidation_format: "yearly" ou "single"
            save_features: Se deve salvar características processadas
            
        Returns:
            Dict com informações do processamento
        """
        logger.info("Iniciando processamento completo do dataset")
        
        results = {
            'consolidated_files': {},
            'feature_files': {},
            'statistics': {},
            'processing_time': None
        }
        
        start_time = datetime.now()
        
        try:
            # Etapa 1: Consolidar chunks
            consolidated_files = self.consolidate_forecast_chunks(consolidation_format)
            results['consolidated_files'] = {str(k): str(v) for k, v in consolidated_files.items()}
            
            # Etapa 2: Aplicar engenharia de características em cada arquivo
            for file_key, file_path in consolidated_files.items():
                logger.info(f"Processando características para: {file_key}")
                
                # Carregar dados consolidados
                df = pd.read_parquet(file_path)
                
                # Aplicar engenharia de características
                df_features = self.engineer_atmospheric_features(df)
                
                # Calcular estatísticas
                stats = self._calculate_dataset_statistics(df_features)
                results['statistics'][file_key] = stats
                
                # Salvar arquivo com características se solicitado
                if save_features:
                    if consolidation_format == "yearly":
                        features_path = self.processed_path / f"openmeteo_features_{file_key}.parquet"
                    else:
                        features_path = self.processed_path / "openmeteo_features_consolidated.parquet"
                    
                    df_features.to_parquet(features_path, index=False)
                    results['feature_files'][file_key] = str(features_path)
                    
                    logger.info(f"Características salvas: {features_path}")
            
            # Tempo total de processamento
            processing_time = datetime.now() - start_time
            results['processing_time'] = str(processing_time)
            
            logger.info(f"Processamento completo finalizado em {processing_time}")
            
            return results
            
        except Exception as e:
            logger.error(f"Erro durante processamento: {e}")
            raise
    
    def _calculate_dataset_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula estatísticas do dataset processado"""
        stats = {
            'total_records': len(df),
            'date_range': {
                'start': df['time'].min().isoformat(),
                'end': df['time'].max().isoformat()
            },
            'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'feature_counts': {
                'atmospheric_features': len([c for c in df.columns if any(x in c for x in ['gradient', 'index', 'front', 'vortex'])]),
                'wind_features': len([c for c in df.columns if 'wind' in c]),
                'pressure_features': len([c for c in df.columns if 'pressure' in c]),
                'total_features': len(df.columns)
            }
        }
        
        return stats


def main():
    """Função principal para execução do script"""
    logger.info("=== Iniciando Fase 2.2: Engenharia de Características Atmosféricas ===")
    
    # Verificar se o diretório de dados existe
    data_path = Path("data/raw/Open-Meteo Historical Forecast")
    if not data_path.exists():
        logger.error(f"Diretório de dados não encontrado: {data_path}")
        sys.exit(1)
    
    try:
        # Inicializar processador
        processor = AtmosphericFeatureEngineer(data_path)
        
        # Processar dataset completo
        results = processor.process_complete_dataset(
            consolidation_format="yearly",  # Mudar para "single" se preferir arquivo único
            save_features=True
        )
        
        # Exibir resultados
        logger.info("=== RESULTADOS DO PROCESSAMENTO ===")
        logger.info(f"Tempo de processamento: {results['processing_time']}")
        logger.info(f"Arquivos consolidados: {len(results['consolidated_files'])}")
        logger.info(f"Arquivos de características: {len(results['feature_files'])}")
        
        for file_key, stats in results['statistics'].items():
            logger.info(f"\n--- Estatísticas para {file_key} ---")
            logger.info(f"Total de registros: {stats['total_records']:,}")
            logger.info(f"Período: {stats['date_range']['start']} a {stats['date_range']['end']}")
            logger.info(f"Dados faltantes: {stats['missing_data_percentage']:.2f}%")
            logger.info(f"Total de características: {stats['feature_counts']['total_features']}")
            logger.info(f"Características atmosféricas: {stats['feature_counts']['atmospheric_features']}")
        
        logger.info("=== Fase 2.2 concluída com sucesso! ===")
        
    except Exception as e:
        logger.error(f"Erro na execução: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 