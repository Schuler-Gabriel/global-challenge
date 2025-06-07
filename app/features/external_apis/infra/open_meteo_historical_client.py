"""
Open-Meteo Historical API Client for Hybrid Strategy

Cliente para integração com a API Open-Meteo Historical para dados meteorológicos
históricos de 2000-2024, complementando a estratégia híbrida.
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta, date
from typing import Dict, Optional, List, Union, Any
from dataclasses import dataclass
import pandas as pd
import logging

from app.core.config import get_settings
from app.core.exceptions import ExternalApiException, DataValidationException


logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class OpenMeteoHistoricalConfig:
    """Configuração para cliente Open-Meteo Historical"""
    base_url: str = "https://archive-api.open-meteo.com/v1/archive"
    coordinates: tuple = (-30.0331, -51.2300)  # Porto Alegre
    timezone: str = "America/Sao_Paulo"
    timeout: int = 30  # Dados históricos podem demorar mais
    max_retries: int = 3
    chunk_size_days: int = 365  # 1 ano por requisição para evitar timeouts
    cache_ttl: int = 86400  # 24 horas (dados históricos mudam pouco)


class OpenMeteoHistoricalClient:
    """
    Cliente para dados meteorológicos históricos via Open-Meteo Archive API
    
    Implementa coleta de dados históricos de superfície (25 variáveis) 
    para extensão temporal da estratégia híbrida de 2000 até 2024.
    """

    def __init__(self, config: Optional[OpenMeteoHistoricalConfig] = None):
        self.config = config or OpenMeteoHistoricalConfig()
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def get_historical_data(
        self, 
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime]
    ) -> Dict[str, Any]:
        """
        Busca dados meteorológicos históricos para período especificado
        
        Args:
            start_date: Data de início (YYYY-MM-DD)
            end_date: Data de fim (YYYY-MM-DD)
            
        Returns:
            Dict: Dados meteorológicos históricos processados
            
        Raises:
            ExternalApiException: Erro na API Open-Meteo
            DataValidationException: Dados inválidos recebidos
        """
        
        # Converte datas para formato correto
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()
            
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()

        # Valida período
        if start_date >= end_date:
            raise DataValidationException("Data de início deve ser anterior à data de fim")
            
        if start_date < date(2000, 1, 1):
            raise DataValidationException("Dados históricos disponíveis apenas a partir de 2000")
            
        if end_date > date.today():
            end_date = date.today()

        # Divide requisições em chunks para períodos longos
        all_data = []
        current_start = start_date
        
        while current_start < end_date:
            chunk_end = min(
                current_start + timedelta(days=self.config.chunk_size_days),
                end_date
            )
            
            chunk_data = await self._get_data_chunk(current_start, chunk_end)
            if chunk_data:
                all_data.append(chunk_data)
                
            current_start = chunk_end + timedelta(days=1)

        # Combina todos os chunks
        return self._combine_chunks(all_data, start_date, end_date)

    async def _get_data_chunk(self, start_date: date, end_date: date) -> Optional[Dict]:
        """
        Busca dados para um chunk específico
        
        Args:
            start_date: Data de início do chunk
            end_date: Data de fim do chunk
            
        Returns:
            Dict: Dados do chunk ou None se falhou
        """
        
        params = {
            'latitude': self.config.coordinates[0],
            'longitude': self.config.coordinates[1],
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'timezone': self.config.timezone,
            # 25 variáveis de superfície para estratégia híbrida
            'daily': [
                # Temperatura
                'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
                
                # Precipitação
                'precipitation_sum', 'rain_sum', 'snowfall_sum',
                
                # Pressão atmosférica
                'pressure_msl_mean',
                
                # Vento
                'wind_speed_10m_max', 'wind_speed_10m_mean',
                'wind_direction_10m_dominant', 'wind_gusts_10m_max',
                
                # Umidade
                'relative_humidity_2m_max', 'relative_humidity_2m_min', 'relative_humidity_2m_mean',
                
                # Radiação solar
                'shortwave_radiation_sum', 'direct_radiation_sum', 'diffuse_radiation_sum',
                
                # Evapotranspiração
                'et0_fao_evapotranspiration',
                
                # Condições atmosféricas
                'weather_code_most_frequent',
                'cloud_cover_mean',
                
                # Temperatura aparente
                'apparent_temperature_max', 'apparent_temperature_min', 'apparent_temperature_mean',
                
                # Horas de sol
                'sunshine_duration',
                
                # Ponto de orvalho
                'dewpoint_2m_mean'
            ]
        }

        try:
            if not self.session:
                async with aiohttp.ClientSession() as session:
                    return await self._make_chunk_request(session, params, start_date, end_date)
            else:
                return await self._make_chunk_request(self.session, params, start_date, end_date)
                
        except Exception as e:
            logger.error(f"Erro ao buscar chunk {start_date}-{end_date}: {str(e)}")
            return None

    async def _make_chunk_request(
        self, 
        session: aiohttp.ClientSession, 
        params: Dict,
        start_date: date,
        end_date: date
    ) -> Dict:
        """Executa requisição para um chunk específico"""
        
        for attempt in range(self.config.max_retries):
            try:
                async with session.get(self.config.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_chunk_data(data, start_date, end_date)
                    else:
                        error_text = await response.text()
                        if response.status == 400 and "date range" in error_text.lower():
                            logger.warning(f"Período inválido {start_date}-{end_date}: {error_text}")
                            return None
                        raise ExternalApiException(
                            f"Open-Meteo Historical API error: {response.status} - {error_text}"
                        )
                        
            except asyncio.TimeoutError:
                if attempt == self.config.max_retries - 1:
                    raise ExternalApiException(
                        f"Open-Meteo Historical API timeout para período {start_date}-{end_date}"
                    )
                await asyncio.sleep(2 ** attempt)
                
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries - 1:
                    raise ExternalApiException(
                        f"Erro de conexão Open-Meteo Historical: {str(e)}"
                    )
                await asyncio.sleep(2 ** attempt)

    def _process_chunk_data(self, raw_data: Dict, start_date: date, end_date: date) -> Dict:
        """
        Processa dados brutos de um chunk
        
        Args:
            raw_data: Dados JSON da API
            start_date: Data de início do chunk
            end_date: Data de fim do chunk
            
        Returns:
            Dict: Dados processados do chunk
        """
        
        try:
            daily = raw_data.get('daily', {})
            
            if not daily or 'time' not in daily:
                logger.warning(f"Dados diários ausentes para período {start_date}-{end_date}")
                return {
                    'period': {'start': start_date.isoformat(), 'end': end_date.isoformat()},
                    'data': {},
                    'record_count': 0,
                    'data_quality': {'overall_score': 0.0, 'issues': ['no_data']}
                }

            # Converte dados para DataFrame para facilitar processamento
            df_data = {}
            times = daily['time']
            
            for variable, values in daily.items():
                if variable != 'time' and values:
                    df_data[variable] = values

            # Análise de qualidade dos dados
            quality_assessment = self._assess_chunk_quality(df_data, times)

            # Estatísticas do período
            period_stats = self._calculate_period_statistics(df_data)

            processed_chunk = {
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat(),
                    'days_requested': (end_date - start_date).days + 1,
                    'days_received': len(times)
                },
                'location': {
                    'latitude': raw_data.get('latitude'),
                    'longitude': raw_data.get('longitude'),
                    'elevation': raw_data.get('elevation'),
                    'timezone': raw_data.get('timezone')
                },
                'data': df_data,
                'times': times,
                'record_count': len(times),
                'statistics': period_stats,
                'data_quality': quality_assessment,
                'api_source': 'open_meteo_historical',
                'processing_info': {
                    'processed_at': datetime.now().isoformat(),
                    'variables_count': len(df_data),
                    'chunk_start': start_date.isoformat(),
                    'chunk_end': end_date.isoformat()
                }
            }

            return processed_chunk
            
        except Exception as e:
            raise DataValidationException(
                f"Erro ao processar chunk {start_date}-{end_date}: {str(e)}"
            )

    def _assess_chunk_quality(self, data: Dict, times: List[str]) -> Dict[str, Any]:
        """
        Avalia qualidade dos dados de um chunk
        
        Args:
            data: Dados do chunk
            times: Lista de timestamps
            
        Returns:
            Dict: Avaliação de qualidade
        """
        
        quality = {
            'overall_score': 1.0,
            'completeness': 0.0,
            'missing_days': 0,
            'missing_variables': [],
            'data_gaps': [],
            'issues': []
        }
        
        try:
            if not data or not times:
                quality['overall_score'] = 0.0
                quality['issues'].append('no_data')
                return quality

            total_days = len(times)
            
            # Avalia completude dos dados
            complete_records = 0
            for i, time_str in enumerate(times):
                record_complete = True
                for var, values in data.items():
                    if i >= len(values) or values[i] is None:
                        record_complete = False
                        break
                if record_complete:
                    complete_records += 1

            quality['completeness'] = complete_records / total_days if total_days > 0 else 0
            quality['missing_days'] = total_days - complete_records

            # Verifica variáveis essenciais
            essential_vars = [
                'temperature_2m_mean', 'precipitation_sum', 
                'pressure_msl_mean', 'relative_humidity_2m_mean'
            ]
            
            for var in essential_vars:
                if var not in data or not any(v is not None for v in data[var]):
                    quality['missing_variables'].append(var)
                    quality['overall_score'] -= 0.2

            # Detecta gaps significativos nos dados
            for var, values in data.items():
                gaps = self._detect_data_gaps(values, var)
                if gaps:
                    quality['data_gaps'].extend(gaps)

            # Penaliza baixa completude
            if quality['completeness'] < 0.8:
                quality['overall_score'] -= 0.3
                quality['issues'].append('low_completeness')

            if quality['missing_days'] > total_days * 0.1:
                quality['issues'].append('many_missing_days')

            # Assegura que score não seja negativo
            quality['overall_score'] = max(0.0, quality['overall_score'])

        except Exception as e:
            logger.warning(f"Erro ao avaliar qualidade do chunk: {str(e)}")
            quality['overall_score'] = 0.5
            quality['issues'].append('quality_assessment_error')

        return quality

    def _detect_data_gaps(self, values: List, variable_name: str) -> List[Dict]:
        """
        Detecta gaps nos dados de uma variável
        
        Args:
            values: Lista de valores da variável
            variable_name: Nome da variável
            
        Returns:
            List[Dict]: Lista de gaps detectados
        """
        
        gaps = []
        
        try:
            gap_start = None
            for i, value in enumerate(values):
                if value is None:
                    if gap_start is None:
                        gap_start = i
                else:
                    if gap_start is not None:
                        gap_length = i - gap_start
                        if gap_length >= 7:  # Gaps de 7+ dias são significativos
                            gaps.append({
                                'variable': variable_name,
                                'start_index': gap_start,
                                'end_index': i - 1,
                                'length_days': gap_length
                            })
                        gap_start = None
            
            # Verifica se terminou com gap
            if gap_start is not None:
                gap_length = len(values) - gap_start
                if gap_length >= 7:
                    gaps.append({
                        'variable': variable_name,
                        'start_index': gap_start,
                        'end_index': len(values) - 1,
                        'length_days': gap_length
                    })

        except Exception as e:
            logger.warning(f"Erro ao detectar gaps em {variable_name}: {str(e)}")

        return gaps

    def _calculate_period_statistics(self, data: Dict) -> Dict[str, Any]:
        """
        Calcula estatísticas do período para cada variável
        
        Args:
            data: Dados do período
            
        Returns:
            Dict: Estatísticas calculadas
        """
        
        stats = {}
        
        try:
            for variable, values in data.items():
                clean_values = [v for v in values if v is not None]
                
                if not clean_values:
                    stats[variable] = {
                        'count': 0,
                        'mean': None,
                        'min': None,
                        'max': None,
                        'std': None
                    }
                    continue
                
                import statistics
                
                stats[variable] = {
                    'count': len(clean_values),
                    'mean': statistics.mean(clean_values),
                    'min': min(clean_values),
                    'max': max(clean_values),
                    'std': statistics.stdev(clean_values) if len(clean_values) > 1 else 0.0
                }
                
                # Estatísticas específicas por tipo de variável
                if 'precipitation' in variable or 'rain' in variable:
                    stats[variable]['total'] = sum(clean_values)
                    stats[variable]['days_with_precipitation'] = sum(1 for v in clean_values if v > 0.1)
                    
        except Exception as e:
            logger.warning(f"Erro ao calcular estatísticas: {str(e)}")

        return stats

    def _combine_chunks(self, chunks: List[Dict], start_date: date, end_date: date) -> Dict[str, Any]:
        """
        Combina múltiplos chunks em um dataset unificado
        
        Args:
            chunks: Lista de chunks processados
            start_date: Data de início total
            end_date: Data de fim total
            
        Returns:
            Dict: Dataset combinado
        """
        
        if not chunks:
            return self._create_empty_dataset(start_date, end_date)

        try:
            # Combina todos os dados
            combined_data = {}
            combined_times = []
            total_records = 0
            
            # Primeira passada: identifica todas as variáveis
            all_variables = set()
            for chunk in chunks:
                if 'data' in chunk:
                    all_variables.update(chunk['data'].keys())

            # Segunda passada: combina os dados
            for chunk in chunks:
                if 'times' in chunk and 'data' in chunk:
                    chunk_times = chunk['times']
                    chunk_data = chunk['data']
                    
                    combined_times.extend(chunk_times)
                    total_records += len(chunk_times)
                    
                    for variable in all_variables:
                        if variable not in combined_data:
                            combined_data[variable] = []
                        
                        if variable in chunk_data:
                            combined_data[variable].extend(chunk_data[variable])
                        else:
                            # Preenche com None para manter consistência
                            combined_data[variable].extend([None] * len(chunk_times))

            # Calcula estatísticas combinadas
            combined_stats = self._calculate_period_statistics(combined_data)

            # Avalia qualidade combinada
            combined_quality = self._assess_combined_quality(chunks, total_records)

            combined_dataset = {
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat(),
                    'total_days': (end_date - start_date).days + 1,
                    'chunks_processed': len(chunks)
                },
                'location': chunks[0].get('location', {}) if chunks else {},
                'data': combined_data,
                'times': combined_times,
                'record_count': total_records,
                'statistics': combined_stats,
                'data_quality': combined_quality,
                'chunks_info': [
                    {
                        'period': chunk.get('period', {}),
                        'record_count': chunk.get('record_count', 0),
                        'quality_score': chunk.get('data_quality', {}).get('overall_score', 0.0)
                    }
                    for chunk in chunks
                ],
                'api_source': 'open_meteo_historical_combined',
                'processing_info': {
                    'processed_at': datetime.now().isoformat(),
                    'variables_count': len(combined_data),
                    'chunks_combined': len(chunks),
                    'total_period_days': (end_date - start_date).days + 1
                }
            }

            return combined_dataset

        except Exception as e:
            logger.error(f"Erro ao combinar chunks: {str(e)}")
            return self._create_empty_dataset(start_date, end_date, error=str(e))

    def _assess_combined_quality(self, chunks: List[Dict], total_records: int) -> Dict[str, Any]:
        """
        Avalia qualidade dos dados combinados
        
        Args:
            chunks: Lista de chunks
            total_records: Total de registros
            
        Returns:
            Dict: Avaliação de qualidade combinada
        """
        
        quality = {
            'overall_score': 0.0,
            'chunk_scores': [],
            'successful_chunks': 0,
            'failed_chunks': 0,
            'total_completeness': 0.0,
            'issues': []
        }
        
        try:
            chunk_scores = []
            successful_chunks = 0
            
            for chunk in chunks:
                chunk_quality = chunk.get('data_quality', {})
                score = chunk_quality.get('overall_score', 0.0)
                chunk_scores.append(score)
                
                if score > 0.5:
                    successful_chunks += 1
                else:
                    quality['issues'].extend(chunk_quality.get('issues', []))

            quality['chunk_scores'] = chunk_scores
            quality['successful_chunks'] = successful_chunks
            quality['failed_chunks'] = len(chunks) - successful_chunks

            # Score geral é a média ponderada dos chunks
            if chunk_scores:
                quality['overall_score'] = sum(chunk_scores) / len(chunk_scores)
            
            # Penaliza se muitos chunks falharam
            if quality['failed_chunks'] > len(chunks) * 0.2:
                quality['overall_score'] *= 0.7
                quality['issues'].append('many_failed_chunks')

            # Calcula completude total
            if total_records > 0:
                # Assumindo que cada chunk representa período proporcional
                quality['total_completeness'] = successful_chunks / len(chunks) if chunks else 0

        except Exception as e:
            logger.warning(f"Erro ao avaliar qualidade combinada: {str(e)}")
            quality['overall_score'] = 0.3
            quality['issues'].append('quality_assessment_error')

        return quality

    def _create_empty_dataset(self, start_date: date, end_date: date, error: str = None) -> Dict[str, Any]:
        """
        Cria dataset vazio para casos de falha
        
        Args:
            start_date: Data de início
            end_date: Data de fim
            error: Mensagem de erro opcional
            
        Returns:
            Dict: Dataset vazio
        """
        
        return {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'total_days': (end_date - start_date).days + 1,
                'chunks_processed': 0
            },
            'location': {
                'latitude': self.config.coordinates[0],
                'longitude': self.config.coordinates[1],
                'elevation': None,
                'timezone': self.config.timezone
            },
            'data': {},
            'times': [],
            'record_count': 0,
            'statistics': {},
            'data_quality': {
                'overall_score': 0.0,
                'successful_chunks': 0,
                'failed_chunks': 1,
                'issues': ['no_data'] + ([f'error: {error}'] if error else [])
            },
            'chunks_info': [],
            'api_source': 'open_meteo_historical_empty',
            'processing_info': {
                'processed_at': datetime.now().isoformat(),
                'variables_count': 0,
                'chunks_combined': 0,
                'error': error
            }
        }

    async def get_yearly_data(self, year: int) -> Dict[str, Any]:
        """
        Busca dados de um ano específico
        
        Args:
            year: Ano desejado (2000-2024)
            
        Returns:
            Dict: Dados meteorológicos do ano
        """
        
        if year < 2000 or year > 2024:
            raise DataValidationException(f"Ano {year} fora do intervalo suportado (2000-2024)")

        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        
        # Ajusta para não ir além de hoje
        if end_date > date.today():
            end_date = date.today()

        return await self.get_historical_data(start_date, end_date)

    async def get_recent_years(self, years_back: int = 5) -> Dict[str, Any]:
        """
        Busca dados dos últimos N anos
        
        Args:
            years_back: Número de anos para trás (padrão: 5)
            
        Returns:
            Dict: Dados meteorológicos dos últimos anos
        """
        
        end_date = date.today()
        start_date = date(end_date.year - years_back, 1, 1)
        
        return await self.get_historical_data(start_date, end_date)


# Funções de conveniência
async def get_historical_porto_alegre(
    start_date: Union[str, date], 
    end_date: Union[str, date]
) -> Dict[str, Any]:
    """
    Função de conveniência para obter dados históricos de Porto Alegre
    
    Args:
        start_date: Data de início
        end_date: Data de fim
        
    Returns:
        Dict: Dados meteorológicos históricos
    """
    
    async with OpenMeteoHistoricalClient() as client:
        return await client.get_historical_data(start_date, end_date)


async def get_year_data_porto_alegre(year: int) -> Dict[str, Any]:
    """
    Função de conveniência para obter dados de um ano específico
    
    Args:
        year: Ano desejado
        
    Returns:
        Dict: Dados meteorológicos do ano
    """
    
    async with OpenMeteoHistoricalClient() as client:
        return await client.get_yearly_data(year)


async def test_historical_connection() -> bool:
    """
    Testa conexão com API Open-Meteo Historical
    
    Returns:
        bool: True se conexão bem-sucedida
    """
    
    try:
        # Testa com uma semana recente
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=7)
        
        async with OpenMeteoHistoricalClient() as client:
            data = await client.get_historical_data(start_date, end_date)
            return data['data_quality']['overall_score'] > 0.3
    except Exception as e:
        logger.error(f"Teste de conexão Open-Meteo Historical falhou: {str(e)}")
        return False