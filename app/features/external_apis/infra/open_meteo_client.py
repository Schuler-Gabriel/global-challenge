"""
Open-Meteo API Client for Hybrid Strategy

Cliente para integração com as APIs Open-Meteo para dados meteorológicos
em tempo real e históricos, implementando a estratégia híbrida.
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Union, Any
from dataclasses import dataclass
import logging

from app.core.config import get_settings
from app.core.exceptions import ExternalApiException, DataValidationException


logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class OpenMeteoConfig:
    """Configuração para cliente Open-Meteo"""
    forecast_base_url: str = "https://api.open-meteo.com/v1/forecast"
    historical_base_url: str = "https://archive-api.open-meteo.com/v1/archive"
    coordinates: tuple = (-30.0331, -51.2300)  # Porto Alegre
    timezone: str = "America/Sao_Paulo"
    timeout: int = 10
    max_retries: int = 3
    cache_ttl: int = 3600  # 1 hora


class OpenMeteoCurrentWeatherClient:
    """
    Cliente para dados meteorológicos em tempo real via Open-Meteo API
    
    Implementa a estratégia híbrida coletando dados de superfície e 
    níveis de pressão para análise sinótica em tempo real.
    """

    def __init__(self, config: Optional[OpenMeteoConfig] = None):
        self.config = config or OpenMeteoConfig()
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

    async def get_current_conditions(self) -> Dict[str, Any]:
        """
        Busca condições meteorológicas atuais das últimas 24h
        
        Returns:
            Dict: Dados meteorológicos processados com análise sinótica
            
        Raises:
            ExternalApiException: Erro na API Open-Meteo
            DataValidationException: Dados inválidos recebidos
        """
        
        params = {
            'latitude': self.config.coordinates[0],
            'longitude': self.config.coordinates[1],
            'timezone': self.config.timezone,
            'current': [
                'temperature_2m', 'relative_humidity_2m', 'precipitation',
                'pressure_msl', 'wind_speed_10m', 'wind_direction_10m',
                'weather_code', 'cloud_cover'
            ],
            'hourly': [
                'temperature_2m', 'relative_humidity_2m', 'precipitation',
                'pressure_msl', 'wind_speed_10m', 'wind_direction_10m',
                'cloud_cover', 'visibility'
            ],
            'past_days': 1,  # Últimas 24h
            'forecast_days': 1,  # Próximas 24h para contexto
            # Dados sinóticos em tempo real
            'pressure_level': [850, 500],
            'pressure_level_variables': [
                'temperature', 'wind_speed', 'wind_direction',
                'geopotential_height', 'relative_humidity'
            ]
        }

        try:
            if not self.session:
                async with aiohttp.ClientSession() as session:
                    return await self._make_request(session, params)
            else:
                return await self._make_request(self.session, params)
                
        except Exception as e:
            logger.error(f"Erro ao buscar dados Open-Meteo: {str(e)}")
            # Fallback para dados históricos se API falhar
            return await self._get_fallback_data()

    async def _make_request(self, session: aiohttp.ClientSession, params: Dict) -> Dict:
        """Executa requisição para API Open-Meteo"""
        
        for attempt in range(self.config.max_retries):
            try:
                async with session.get(self.config.forecast_base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_weather_data(data)
                    else:
                        error_text = await response.text()
                        raise ExternalApiException(
                            f"Open-Meteo API error: {response.status} - {error_text}"
                        )
                        
            except asyncio.TimeoutError:
                if attempt == self.config.max_retries - 1:
                    raise ExternalApiException("Open-Meteo API timeout após múltiplas tentativas")
                await asyncio.sleep(2 ** attempt)  # Backoff exponencial
                
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries - 1:
                    raise ExternalApiException(f"Erro de conexão Open-Meteo: {str(e)}")
                await asyncio.sleep(2 ** attempt)

    def _process_weather_data(self, raw_data: Dict) -> Dict[str, Any]:
        """
        Processa dados brutos da Open-Meteo para formato interno
        
        Args:
            raw_data: Dados JSON da API Open-Meteo
            
        Returns:
            Dict: Dados processados com análise sinótica
        """
        
        try:
            current = raw_data.get('current', {})
            hourly = raw_data.get('hourly', {})

            # Extrai dados das últimas 24h
            last_24h_data = self._extract_last_24h(hourly)

            # Processa dados de níveis de pressão
            synoptic_data = self._process_pressure_levels(raw_data)

            # Análise de qualidade dos dados
            data_quality = self._assess_data_quality(current, hourly)

            processed_data = {
                'timestamp': current.get('time'),
                'location': {
                    'latitude': raw_data.get('latitude'),
                    'longitude': raw_data.get('longitude'),
                    'elevation': raw_data.get('elevation'),
                    'timezone': raw_data.get('timezone')
                },
                'current_conditions': {
                    'temperature': current.get('temperature_2m'),
                    'humidity': current.get('relative_humidity_2m'),
                    'precipitation': current.get('precipitation', 0.0),
                    'pressure': current.get('pressure_msl'),
                    'wind_speed': current.get('wind_speed_10m'),
                    'wind_direction': current.get('wind_direction_10m'),
                    'weather_code': current.get('weather_code'),
                    'cloud_cover': current.get('cloud_cover')
                },
                'last_24h_trends': last_24h_data,
                'synoptic_analysis': synoptic_data,
                'data_quality': data_quality,
                'api_source': 'open_meteo_forecast',
                'processing_info': {
                    'processed_at': datetime.now().isoformat(),
                    'variables_count': len(current) + len(synoptic_data),
                    'has_pressure_data': bool(synoptic_data)
                }
            }
            
            # Validação dos dados processados
            self._validate_processed_data(processed_data)
            
            return processed_data
            
        except Exception as e:
            raise DataValidationException(f"Erro ao processar dados Open-Meteo: {str(e)}")

    def _extract_last_24h(self, hourly: Dict) -> Dict[str, Any]:
        """Extrai e analisa dados das últimas 24h"""
        
        if not hourly or 'time' not in hourly:
            return {}
            
        try:
            # Pega últimas 24 entradas (últimas 24h)
            times = hourly.get('time', [])
            last_24h_count = min(24, len(times))
            
            trends = {
                'temperature_trend': self._calculate_trend(
                    hourly.get('temperature_2m', [])[-last_24h_count:]
                ),
                'pressure_trend': self._calculate_trend(
                    hourly.get('pressure_msl', [])[-last_24h_count:]
                ),
                'precipitation_total': sum([
                    p for p in hourly.get('precipitation', [])[-last_24h_count:] if p is not None
                ]),
                'wind_speed_max': max([
                    w for w in hourly.get('wind_speed_10m', [])[-last_24h_count:] if w is not None
                ] or [0]),
                'cloud_cover_avg': sum([
                    c for c in hourly.get('cloud_cover', [])[-last_24h_count:] if c is not None
                ]) / last_24h_count if last_24h_count > 0 else 0
            }
            
            return trends
            
        except Exception as e:
            logger.warning(f"Erro ao extrair tendências 24h: {str(e)}")
            return {}

    def _process_pressure_levels(self, data: Dict) -> Dict[str, Any]:
        """
        Processa dados de níveis de pressão para análise sinótica
        
        Args:
            data: Dados completos da API
            
        Returns:
            Dict: Análise sinótica dos níveis de pressão
        """
        
        synoptic = {}

        try:
            # Análise 850hPa (frentes frias)
            pressure_850_key = [k for k in data.keys() if '850' in k and 'pressure_level' in k]
            if pressure_850_key:
                level_850 = data[pressure_850_key[0]]
                temp_850 = level_850.get('temperature', [])
                wind_850 = level_850.get('wind_speed', [])
                
                synoptic['850hPa'] = {
                    'temperature': temp_850[-1] if temp_850 else None,
                    'wind_speed': wind_850[-1] if wind_850 else None,
                    'frontal_indicator': self._detect_frontal_activity(temp_850),
                    'level_description': 'Análise de frentes frias'
                }

            # Análise 500hPa (vórtices)
            pressure_500_key = [k for k in data.keys() if '500' in k and 'pressure_level' in k]
            if pressure_500_key:
                level_500 = data[pressure_500_key[0]]
                height_500 = level_500.get('geopotential_height', [])
                wind_500 = level_500.get('wind_speed', [])
                
                synoptic['500hPa'] = {
                    'geopotential': height_500[-1] if height_500 else None,
                    'wind_speed': wind_500[-1] if wind_500 else None,
                    'vorticity_indicator': self._detect_vortex_activity(height_500),
                    'level_description': 'Análise de vórtices e padrões sinóticos'
                }

            # Análise combinada
            if '850hPa' in synoptic and '500hPa' in synoptic:
                synoptic['combined_analysis'] = {
                    'atmospheric_stability': self._assess_atmospheric_stability(synoptic),
                    'weather_pattern': self._classify_weather_pattern(synoptic),
                    'risk_level': self._calculate_synoptic_risk(synoptic)
                }

        except Exception as e:
            logger.warning(f"Erro ao processar níveis de pressão: {str(e)}")

        return synoptic

    def _detect_frontal_activity(self, temp_850: List[float]) -> str:
        """
        Detecta atividade frontal baseada em temperatura 850hPa
        
        Args:
            temp_850: Série temporal de temperatura em 850hPa
            
        Returns:
            str: Indicador de atividade frontal
        """
        
        if len(temp_850) < 6:
            return "insufficient_data"

        try:
            # Gradiente de temperatura nas últimas 6h
            recent_gradient = temp_850[-1] - temp_850[-6]

            if recent_gradient < -3:
                return "cold_front_approaching"
            elif recent_gradient > 3:
                return "warm_front_approaching"
            elif abs(recent_gradient) > 1.5:
                return "frontal_activity_detected"
            else:
                return "stable"
                
        except Exception:
            return "analysis_error"

    def _detect_vortex_activity(self, height_500: List[float]) -> str:
        """
        Detecta atividade de vórtices baseada em altura geopotencial 500hPa
        
        Args:
            height_500: Série temporal de altura geopotencial
            
        Returns:
            str: Indicador de atividade de vórtices
        """
        
        if len(height_500) < 4:
            return "insufficient_data"

        try:
            # Análise de variabilidade da altura geopotencial
            recent_values = height_500[-4:]
            variability = max(recent_values) - min(recent_values)
            
            if variability > 50:
                return "high_vorticity_detected"
            elif variability > 25:
                return "moderate_vorticity"
            else:
                return "low_vorticity"
                
        except Exception:
            return "analysis_error"

    def _assess_atmospheric_stability(self, synoptic: Dict) -> str:
        """Avalia estabilidade atmosférica baseada em dados sinóticos"""
        
        try:
            frontal_activity = synoptic.get('850hPa', {}).get('frontal_indicator', 'stable')
            vortex_activity = synoptic.get('500hPa', {}).get('vorticity_indicator', 'low_vorticity')
            
            if 'approaching' in frontal_activity or 'high' in vortex_activity:
                return "unstable"
            elif 'detected' in frontal_activity or 'moderate' in vortex_activity:
                return "moderately_unstable"
            else:
                return "stable"
                
        except Exception:
            return "unknown"

    def _classify_weather_pattern(self, synoptic: Dict) -> str:
        """Classifica padrão meteorológico baseado em análise sinótica"""
        
        try:
            frontal = synoptic.get('850hPa', {}).get('frontal_indicator', '')
            vortex = synoptic.get('500hPa', {}).get('vorticity_indicator', '')
            
            if 'cold_front' in frontal:
                return "cold_front_passage"
            elif 'warm_front' in frontal:
                return "warm_front_passage"
            elif 'high_vorticity' in vortex:
                return "vortex_system"
            elif 'frontal_activity' in frontal:
                return "frontal_system"
            else:
                return "stable_pattern"
                
        except Exception:
            return "unknown_pattern"

    def _calculate_synoptic_risk(self, synoptic: Dict) -> str:
        """Calcula nível de risco baseado em análise sinótica"""
        
        try:
            stability = synoptic.get('combined_analysis', {}).get('atmospheric_stability', 'stable')
            pattern = synoptic.get('combined_analysis', {}).get('weather_pattern', 'stable_pattern')
            
            high_risk_patterns = ['cold_front_passage', 'vortex_system']
            unstable_conditions = ['unstable', 'moderately_unstable']
            
            if any(p in pattern for p in high_risk_patterns) and stability in unstable_conditions:
                return "high"
            elif any(p in pattern for p in high_risk_patterns) or stability in unstable_conditions:
                return "moderate"
            else:
                return "low"
                
        except Exception:
            return "unknown"

    def _calculate_trend(self, values: List[float]) -> str:
        """Calcula tendência de uma série temporal"""
        
        if len(values) < 2:
            return "insufficient_data"
        
        try:
            # Remove valores None
            clean_values = [v for v in values if v is not None]
            if len(clean_values) < 2:
                return "insufficient_data"
            
            # Calcula tendência simples
            first_half = sum(clean_values[:len(clean_values)//2])
            second_half = sum(clean_values[len(clean_values)//2:])
            
            diff = second_half - first_half
            
            if diff > 0.5:
                return "increasing"
            elif diff < -0.5:
                return "decreasing"
            else:
                return "stable"
                
        except Exception:
            return "analysis_error"

    def _assess_data_quality(self, current: Dict, hourly: Dict) -> Dict[str, Any]:
        """
        Avalia qualidade dos dados recebidos
        
        Returns:
            Dict: Métricas de qualidade dos dados
        """
        
        quality = {
            'overall_score': 1.0,
            'current_data_complete': bool(current),
            'hourly_data_available': bool(hourly),
            'missing_fields': [],
            'data_freshness': 'unknown'
        }
        
        try:
            # Verifica campos obrigatórios
            required_current = ['temperature_2m', 'pressure_msl', 'time']
            missing_current = [field for field in required_current if field not in current]
            
            if missing_current:
                quality['missing_fields'].extend(missing_current)
                quality['overall_score'] -= 0.3
            
            # Verifica frescor dos dados
            if 'time' in current:
                data_time = datetime.fromisoformat(current['time'].replace('Z', '+00:00'))
                age_minutes = (datetime.now().astimezone() - data_time).total_seconds() / 60
                
                if age_minutes < 60:
                    quality['data_freshness'] = 'fresh'
                elif age_minutes < 180:
                    quality['data_freshness'] = 'recent'
                    quality['overall_score'] -= 0.1
                else:
                    quality['data_freshness'] = 'stale'
                    quality['overall_score'] -= 0.2
            
            # Verifica dados horáros
            if hourly and 'time' in hourly:
                hourly_count = len(hourly['time'])
                if hourly_count < 24:
                    quality['overall_score'] -= 0.1
                
        except Exception as e:
            logger.warning(f"Erro ao avaliar qualidade dos dados: {str(e)}")
            quality['overall_score'] = 0.5
            
        return quality

    def _validate_processed_data(self, data: Dict) -> None:
        """
        Valida dados processados
        
        Raises:
            DataValidationException: Se dados estão inválidos
        """
        
        required_sections = ['current_conditions', 'location', 'data_quality']
        
        for section in required_sections:
            if section not in data:
                raise DataValidationException(f"Seção obrigatória ausente: {section}")
        
        # Valida condições atuais
        current = data['current_conditions']
        if current.get('temperature') is None or current.get('pressure') is None:
            raise DataValidationException("Dados meteorológicos básicos ausentes")
        
        # Valida qualidade mínima
        quality_score = data['data_quality'].get('overall_score', 0)
        if quality_score < 0.3:
            raise DataValidationException(f"Qualidade dos dados muito baixa: {quality_score}")

    async def _get_fallback_data(self) -> Dict[str, Any]:
        """
        Retorna dados de fallback quando a API principal falha
        
        Returns:
            Dict: Dados básicos de fallback
        """
        
        logger.warning("Usando dados de fallback - API Open-Meteo indisponível")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'location': {
                'latitude': self.config.coordinates[0],
                'longitude': self.config.coordinates[1],
                'elevation': 46.0,
                'timezone': self.config.timezone
            },
            'current_conditions': {
                'temperature': None,
                'humidity': None,
                'precipitation': 0.0,
                'pressure': None,
                'wind_speed': None,
                'wind_direction': None,
                'weather_code': None,
                'cloud_cover': None
            },
            'last_24h_trends': {},
            'synoptic_analysis': {},
            'data_quality': {
                'overall_score': 0.0,
                'current_data_complete': False,
                'hourly_data_available': False,
                'missing_fields': ['all'],
                'data_freshness': 'unavailable'
            },
            'api_source': 'fallback',
            'processing_info': {
                'processed_at': datetime.now().isoformat(),
                'variables_count': 0,
                'has_pressure_data': False,
                'fallback_reason': 'api_unavailable'
            }
        }


# Funções de conveniência
async def get_porto_alegre_weather() -> Dict[str, Any]:
    """
    Função de conveniência para obter dados meteorológicos de Porto Alegre
    
    Returns:
        Dict: Dados meteorológicos atuais com análise sinótica
    """
    
    async with OpenMeteoCurrentWeatherClient() as client:
        return await client.get_current_conditions()


async def test_openmeteo_connection() -> bool:
    """
    Testa conexão com API Open-Meteo
    
    Returns:
        bool: True se conexão bem-sucedida
    """
    
    try:
        async with OpenMeteoCurrentWeatherClient() as client:
            data = await client.get_current_conditions()
            return data['data_quality']['overall_score'] > 0.5
    except Exception as e:
        logger.error(f"Teste de conexão Open-Meteo falhou: {str(e)}")
        return False