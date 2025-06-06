"""
Hybrid Data Loader - Phase 3.1

Carregador de dados especializado para o modelo híbrido:
- Open-Meteo Historical Forecast (149 features atmosféricas)
- Open-Meteo Historical Weather (25 features de superfície)
- Integração e preprocessamento coordenado
- Cache e otimização para inferência

Usado pelos componentes AtmosphericLSTMComponent e SurfaceLSTMComponent.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class HybridDataLoader:
    """
    Carregador de dados para modelo híbrido Phase 3.1
    
    Gerencia:
    - Dados atmosféricos Open-Meteo Historical Forecast (JSON chunks)
    - Dados de superfície Open-Meteo Historical Weather (CSV files)
    - Sincronização temporal entre fontes
    - Cache para performance
    """

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or Path("data/raw")
        
        # Caminhos específicos
        self.atmospheric_path = self.data_path / "Open-Meteo Historical Forecast"
        self.surface_path = self.data_path / "Open-Meteo Historical Weather"
        
        # Cache interno
        self._atmospheric_cache: Dict[str, pd.DataFrame] = {}
        self._surface_cache: Dict[str, pd.DataFrame] = {}
        
        # Configurações
        self.cache_enabled = True
        self.max_cache_size = 100  # MB
        
        logger.info(f"HybridDataLoader inicializado")
        logger.info(f"  Atmospheric: {self.atmospheric_path}")
        logger.info(f"  Surface: {self.surface_path}")

    def load_atmospheric_data(
        self,
        start_date: datetime,
        end_date: datetime,
        variables: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """
        Carrega dados atmosféricos Open-Meteo Historical Forecast
        
        Args:
            start_date: Data início
            end_date: Data fim  
            variables: Variáveis específicas (opcional)
            
        Returns:
            Dict: Dados atmosféricos com arrays por variável
        """
        try:
            logger.debug(f"Carregando dados atmosféricos: {start_date} até {end_date}")
            
            # Encontrar chunks relevantes
            chunk_files = self._find_atmospheric_chunks(start_date, end_date)
            
            if not chunk_files:
                raise ValueError(f"Nenhum chunk atmosférico encontrado para período {start_date} - {end_date}")
            
            # Carregar e combinar chunks
            all_data = []
            for chunk_file in chunk_files:
                chunk_data = self._load_atmospheric_chunk(chunk_file)
                if chunk_data is not None and len(chunk_data) > 0:
                    all_data.append(chunk_data)
            
            if not all_data:
                raise ValueError("Nenhum dado atmosférico válido carregado")
            
            # Combinar todos os chunks
            df = pd.concat(all_data, ignore_index=True)
            
            # Filtrar por período exato
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            
            # Ordenar por timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Converter para formato de arrays
            atmospheric_data = self._dataframe_to_atmospheric_dict(df, variables)
            
            logger.info(f"Dados atmosféricos carregados: {len(df)} registros, {len(atmospheric_data)} variáveis")
            
            return atmospheric_data
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados atmosféricos: {e}")
            raise ValueError(f"Falha ao carregar dados atmosféricos: {str(e)}")

    def load_surface_data(
        self,
        start_date: datetime,
        end_date: datetime,
        variables: Optional[List[str]] = None
    ) -> List[Dict[str, float]]:
        """
        Carrega dados de superfície Open-Meteo Historical Weather
        
        Args:
            start_date: Data início
            end_date: Data fim
            variables: Variáveis específicas (opcional)
            
        Returns:
            List[Dict]: Lista de registros com dados de superfície
        """
        try:
            logger.debug(f"Carregando dados de superfície: {start_date} até {end_date}")
            
            # Encontrar arquivos CSV relevantes
            csv_files = self._find_surface_csvs(start_date, end_date)
            
            if not csv_files:
                raise ValueError(f"Nenhum arquivo de superfície encontrado para período {start_date} - {end_date}")
            
            # Carregar e combinar CSVs
            all_data = []
            for csv_file in csv_files:
                csv_data = self._load_surface_csv(csv_file)
                if csv_data is not None and len(csv_data) > 0:
                    all_data.append(csv_data)
            
            if not all_data:
                raise ValueError("Nenhum dado de superfície válido carregado")
            
            # Combinar dados
            df = pd.concat(all_data, ignore_index=True)
            
            # Filtrar por período
            df = df[(df['date'] >= start_date.date()) & (df['date'] <= end_date.date())]
            
            # Ordenar por data
            df = df.sort_values('date').reset_index(drop=True)
            
            # Converter para lista de dicts
            surface_data = self._dataframe_to_surface_list(df, variables)
            
            logger.info(f"Dados de superfície carregados: {len(surface_data)} registros")
            
            return surface_data
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados de superfície: {e}")
            raise ValueError(f"Falha ao carregar dados de superfície: {str(e)}")

    def load_combined_data(
        self,
        start_date: datetime,
        end_date: datetime,
        atmospheric_vars: Optional[List[str]] = None,
        surface_vars: Optional[List[str]] = None
    ) -> Tuple[Dict[str, List[float]], List[Dict[str, float]]]:
        """
        Carrega dados atmosféricos e de superfície sincronizados
        
        Args:
            start_date: Data início
            end_date: Data fim
            atmospheric_vars: Variáveis atmosféricas específicas
            surface_vars: Variáveis de superfície específicas
            
        Returns:
            Tuple: (dados_atmosféricos, dados_superfície)
        """
        try:
            logger.debug(f"Carregando dados combinados: {start_date} até {end_date}")
            
            # Carregar em paralelo
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit tasks
                atmospheric_future = executor.submit(
                    self.load_atmospheric_data, start_date, end_date, atmospheric_vars
                )
                surface_future = executor.submit(
                    self.load_surface_data, start_date, end_date, surface_vars
                )
                
                # Aguardar resultados
                atmospheric_data = atmospheric_future.result()
                surface_data = surface_future.result()
            
            logger.info(f"Dados combinados carregados com sucesso")
            
            return atmospheric_data, surface_data
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados combinados: {e}")
            raise ValueError(f"Falha ao carregar dados combinados: {str(e)}")

    def _find_atmospheric_chunks(self, start_date: datetime, end_date: datetime) -> List[Path]:
        """Encontra chunks atmosféricos relevantes para o período"""
        try:
            chunk_files = []
            
            if not self.atmospheric_path.exists():
                logger.warning(f"Diretório atmosférico não existe: {self.atmospheric_path}")
                return chunk_files
            
            # Padrão: chunk_YYYYMMDD_to_YYYYMMDD.json
            for json_file in self.atmospheric_path.glob("chunk_*.json"):
                try:
                    # Extrair datas do nome do arquivo
                    filename = json_file.stem
                    if "chunk_" not in filename or "_to_" not in filename:
                        continue
                    
                    parts = filename.replace("chunk_", "").split("_to_")
                    if len(parts) != 2:
                        continue
                    
                    chunk_start_str, chunk_end_str = parts
                    chunk_start = datetime.strptime(chunk_start_str, "%Y%m%d")
                    chunk_end = datetime.strptime(chunk_end_str, "%Y%m%d")
                    
                    # Verificar sobreposição com período solicitado
                    if (chunk_start <= end_date and chunk_end >= start_date):
                        chunk_files.append(json_file)
                        
                except (ValueError, IndexError) as e:
                    logger.warning(f"Erro ao processar nome do chunk {json_file.name}: {e}")
                    continue
            
            # Ordenar por data
            chunk_files.sort()
            
            logger.debug(f"Chunks atmosféricos encontrados: {len(chunk_files)}")
            return chunk_files
            
        except Exception as e:
            logger.error(f"Erro ao encontrar chunks atmosféricos: {e}")
            return []

    def _find_surface_csvs(self, start_date: datetime, end_date: datetime) -> List[Path]:
        """Encontra CSVs de superfície relevantes para o período"""
        try:
            csv_files = []
            
            if not self.surface_path.exists():
                logger.warning(f"Diretório de superfície não existe: {self.surface_path}")
                return csv_files
            
            # Padrão: open_meteo_hourly_YYYY.csv
            for year in range(start_date.year, end_date.year + 1):
                csv_file = self.surface_path / f"open_meteo_hourly_{year}.csv"
                if csv_file.exists():
                    csv_files.append(csv_file)
            
            logger.debug(f"CSVs de superfície encontrados: {len(csv_files)}")
            return csv_files
            
        except Exception as e:
            logger.error(f"Erro ao encontrar CSVs de superfície: {e}")
            return []

    def _load_atmospheric_chunk(self, chunk_file: Path) -> Optional[pd.DataFrame]:
        """Carrega um chunk atmosférico JSON"""
        try:
            cache_key = f"atm_{chunk_file.name}"
            
            # Verificar cache
            if self.cache_enabled and cache_key in self._atmospheric_cache:
                logger.debug(f"Cache hit: {chunk_file.name}")
                return self._atmospheric_cache[cache_key]
            
            logger.debug(f"Carregando chunk: {chunk_file.name}")
            
            with open(chunk_file, 'r') as f:
                data = json.load(f)
            
            # Parse JSON para DataFrame
            df = self._parse_atmospheric_json(data)
            
            # Cache se habilitado
            if self.cache_enabled and df is not None:
                self._atmospheric_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao carregar chunk {chunk_file.name}: {e}")
            return None

    def _load_surface_csv(self, csv_file: Path) -> Optional[pd.DataFrame]:
        """Carrega um CSV de superfície"""
        try:
            cache_key = f"surf_{csv_file.name}"
            
            # Verificar cache
            if self.cache_enabled and cache_key in self._surface_cache:
                logger.debug(f"Cache hit: {csv_file.name}")
                return self._surface_cache[cache_key]
            
            logger.debug(f"Carregando CSV: {csv_file.name}")
            
            # Carregar CSV
            df = pd.read_csv(csv_file)
            
            # Converter coluna de data
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Cache se habilitado
            if self.cache_enabled and df is not None:
                self._surface_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao carregar CSV {csv_file.name}: {e}")
            return None

    def _parse_atmospheric_json(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Parse JSON atmosférico para DataFrame"""
        try:
            hourly_data = data.get('hourly', {})
            
            # Extrair timestamps
            timestamps = hourly_data.get('time', [])
            if not timestamps:
                logger.warning("Timestamps não encontrados no JSON atmosférico")
                return None
            
            # Converter timestamps
            timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
            
            # Construir DataFrame
            df_data = {'timestamp': timestamps}
            
            # Surface variables
            surface_mapping = {
                'temperature_2m': 'temperature_2m',
                'relativehumidity_2m': 'relative_humidity_2m',
                'dewpoint_2m': 'dewpoint_2m',
                'apparent_temperature': 'apparent_temperature',
                'precipitation_probability': 'precipitation_probability',
                'precipitation': 'precipitation',
                'rain': 'rain',
                'showers': 'showers',
                'pressure_msl': 'pressure_msl',
                'surface_pressure': 'surface_pressure',
                'cloudcover': 'cloudcover',
                'cloudcover_low': 'cloudcover_low',
                'cloudcover_mid': 'cloudcover_mid',
                'cloudcover_high': 'cloudcover_high',
                'windspeed_10m': 'windspeed_10m',
                'winddirection_10m': 'winddirection_10m',
                'windgusts_10m': 'windgusts_10m',
                'cape': 'cape',
                'lifted_index': 'lifted_index',
                'vapour_pressure_deficit': 'vapour_pressure_deficit',
                'soil_temperature_0cm': 'soil_temperature_0cm'
            }
            
            for json_key, feature_name in surface_mapping.items():
                if json_key in hourly_data:
                    df_data[feature_name] = hourly_data[json_key]
                else:
                    df_data[feature_name] = [0.0] * len(timestamps)
            
            # Pressure level variables
            pressure_levels = [1000, 850, 700, 500, 300]
            pressure_vars = ['temperature', 'relativehumidity', 'windspeed', 'winddirection', 'geopotential_height']
            
            for level in pressure_levels:
                for var in pressure_vars:
                    json_key = f"{var}_{level}hPa"
                    feature_name = f"{var.replace('relativehumidity', 'relative_humidity').replace('windspeed', 'wind_speed').replace('winddirection', 'wind_direction')}_{level}hPa"
                    
                    if json_key in hourly_data:
                        df_data[feature_name] = hourly_data[json_key]
                    else:
                        df_data[feature_name] = [0.0] * len(timestamps)
            
            df = pd.DataFrame(df_data)
            
            logger.debug(f"JSON atmosférico processado: {len(df)} registros")
            return df
            
        except Exception as e:
            logger.error(f"Erro no parse JSON atmosférico: {e}")
            return None

    def _dataframe_to_atmospheric_dict(
        self, 
        df: pd.DataFrame, 
        variables: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """Converte DataFrame para dict de arrays atmosféricos"""
        try:
            result = {}
            
            # Usar todas as colunas exceto timestamp se variáveis não especificadas
            if variables is None:
                variables = [col for col in df.columns if col != 'timestamp']
            
            for var in variables:
                if var in df.columns:
                    # Converter para lista de floats, tratando NaN
                    values = df[var].fillna(0.0).astype(float).tolist()
                    result[var] = values
                else:
                    logger.warning(f"Variável atmosférica não encontrada: {var}")
                    result[var] = [0.0] * len(df)
            
            return result
            
        except Exception as e:
            logger.error(f"Erro ao converter DataFrame atmosférico: {e}")
            return {}

    def _dataframe_to_surface_list(
        self, 
        df: pd.DataFrame, 
        variables: Optional[List[str]] = None
    ) -> List[Dict[str, float]]:
        """Converte DataFrame para lista de dicts de superfície"""
        try:
            result = []
            
            # Usar todas as colunas exceto date se variáveis não especificadas
            if variables is None:
                variables = [col for col in df.columns if col != 'date']
            
            for _, row in df.iterrows():
                record = {}
                for var in variables:
                    if var in df.columns:
                        value = row[var]
                        # Tratar NaN e converter para float
                        if pd.isna(value):
                            record[var] = 0.0
                        else:
                            record[var] = float(value)
                    else:
                        record[var] = 0.0
                result.append(record)
            
            return result
            
        except Exception as e:
            logger.error(f"Erro ao converter DataFrame de superfície: {e}")
            return []

    def get_latest_data(
        self, 
        hours_back: int = 72,
        atmospheric_vars: Optional[List[str]] = None,
        surface_vars: Optional[List[str]] = None
    ) -> Tuple[Dict[str, List[float]], List[Dict[str, float]]]:
        """
        Carrega dados mais recentes disponíveis
        
        Args:
            hours_back: Horas para trás a partir de agora
            atmospheric_vars: Variáveis atmosféricas específicas
            surface_vars: Variáveis de superfície específicas
            
        Returns:
            Tuple: (dados_atmosféricos, dados_superfície)
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=hours_back)
            
            logger.info(f"Carregando dados mais recentes: últimas {hours_back} horas")
            
            return self.load_combined_data(
                start_date, end_date, atmospheric_vars, surface_vars
            )
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados mais recentes: {e}")
            raise ValueError(f"Falha ao carregar dados recentes: {str(e)}")

    def validate_data_availability(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Valida disponibilidade de dados para período
        
        Args:
            start_date: Data início
            end_date: Data fim
            
        Returns:
            Dict: Relatório de disponibilidade
        """
        try:
            report = {
                "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                "atmospheric": {"available": False, "chunks": 0, "coverage": 0.0},
                "surface": {"available": False, "files": 0, "coverage": 0.0},
                "issues": []
            }
            
            # Verificar dados atmosféricos
            atm_chunks = self._find_atmospheric_chunks(start_date, end_date)
            if atm_chunks:
                report["atmospheric"]["available"] = True
                report["atmospheric"]["chunks"] = len(atm_chunks)
                # Estimativa simplificada de cobertura
                report["atmospheric"]["coverage"] = min(1.0, len(atm_chunks) * 0.25)
            else:
                report["issues"].append("Nenhum chunk atmosférico disponível")
            
            # Verificar dados de superfície
            surf_csvs = self._find_surface_csvs(start_date, end_date)
            if surf_csvs:
                report["surface"]["available"] = True
                report["surface"]["files"] = len(surf_csvs)
                # Estimativa de cobertura baseada nos anos
                years_needed = end_date.year - start_date.year + 1
                report["surface"]["coverage"] = min(1.0, len(surf_csvs) / years_needed)
            else:
                report["issues"].append("Nenhum arquivo de superfície disponível")
            
            logger.info(f"Validação concluída: {len(report['issues'])} problemas encontrados")
            
            return report
            
        except Exception as e:
            logger.error(f"Erro na validação de disponibilidade: {e}")
            return {"error": str(e)}

    def clear_cache(self):
        """Limpa cache interno"""
        self._atmospheric_cache.clear()
        self._surface_cache.clear()
        logger.info("Cache limpo")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache"""
        return {
            "atmospheric_entries": len(self._atmospheric_cache),
            "surface_entries": len(self._surface_cache),
            "cache_enabled": self.cache_enabled,
            "max_cache_size_mb": self.max_cache_size
        } 