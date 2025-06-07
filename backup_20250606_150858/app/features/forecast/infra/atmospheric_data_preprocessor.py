"""
Atmospheric Data Preprocessor - Phase 3.1

Preprocessador especializado para dados atmosféricos Open-Meteo Historical Forecast:
- 149 variáveis atmosféricas (surface + pressure levels + derived)
- Análise sinóptica (850hPa frontal, 500hPa vórtex) 
- Gradientes atmosféricos e wind shear
- Engenharia de features meteorológicas avançadas

Usado pelo AtmosphericLSTMComponent para preparar dados de entrada.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AtmosphericDataPreprocessor:
    """
    Preprocessador de dados atmosféricos para o modelo híbrido Phase 3.1
    
    Funcionalidades:
    - Parsing de dados Open-Meteo Historical Forecast JSON
    - Cálculo de features sinópticas derivadas
    - Normalização e padronização
    - Preenchimento de dados faltantes
    - Validação de qualidade dos dados
    """

    def __init__(self):
        self.feature_columns = self._define_atmospheric_features()
        self.sequence_length = 72  # 3 dias para padrões sinópticos
        
        # Configurações de qualidade
        self.max_missing_ratio = 0.3  # Máximo 30% de dados faltantes
        self.outlier_threshold = 3.0  # Z-score para detecção de outliers
        
        logger.info(f"Preprocessador atmosférico: {len(self.feature_columns)} features")

    def _define_atmospheric_features(self) -> List[str]:
        """Define as 149 features atmosféricas do Open-Meteo Historical Forecast"""
        
        # Surface variables (21 features)
        surface_vars = [
            "temperature_2m", "relative_humidity_2m", "dewpoint_2m",
            "apparent_temperature", "precipitation_probability", "precipitation",
            "rain", "showers", "pressure_msl", "surface_pressure", 
            "cloudcover", "cloudcover_low", "cloudcover_mid", "cloudcover_high",
            "windspeed_10m", "winddirection_10m", "windgusts_10m",
            "cape", "lifted_index", "vapour_pressure_deficit",
            "soil_temperature_0cm"
        ]
        
        # Pressure levels: 1000, 850, 700, 500, 300 hPa (5 x 25 = 125 features)
        pressure_levels = ["1000hPa", "850hPa", "700hPa", "500hPa", "300hPa"]
        pressure_vars = [
            "temperature", "relative_humidity", "wind_speed", 
            "wind_direction", "geopotential_height"
        ]
        
        features = surface_vars.copy()
        for level in pressure_levels:
            for var in pressure_vars:
                features.append(f"{var}_{level}")
        
        # Derived synoptic features (10)
        synoptic_features = [
            "wind_shear_850_500", "wind_shear_1000_850",      # Wind shear
            "temp_gradient_850_500", "temp_gradient_surface_850",  # Temperature gradients  
            "frontal_strength_850", "temperature_advection_850",    # Frontal analysis
            "vorticity_500", "divergence_500",                # Vortex analysis
            "atmospheric_instability", "moisture_flux"        # Thermodynamic features
        ]
        
        features.extend(synoptic_features)
        return features

    def load_atmospheric_data_from_json(
        self, 
        json_files: List[Path],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Carrega dados atmosféricos de arquivos JSON Open-Meteo Historical Forecast
        
        Args:
            json_files: Lista de arquivos JSON chunks
            start_date: Data início para filtrar (opcional)
            end_date: Data fim para filtrar (opcional)
            
        Returns:
            pd.DataFrame: Dados atmosféricos combinados com timestamp index
        """
        try:
            all_data = []
            
            for json_file in json_files:
                logger.debug(f"Carregando {json_file.name}")
                
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Converter para DataFrame
                df_chunk = self._parse_open_meteo_json(data)
                if len(df_chunk) > 0:
                    all_data.append(df_chunk)
            
            if not all_data:
                raise ValueError("Nenhum dado atmosférico válido encontrado")
            
            # Combinar todos os chunks
            df = pd.concat(all_data, ignore_index=True)
            
            # Ordenar por timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Filtrar por datas se especificado
            if start_date or end_date:
                df = self._filter_by_date_range(df, start_date, end_date)
            
            # Remover duplicatas por timestamp
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
            
            logger.info(f"Dados atmosféricos carregados: {len(df)} registros")
            logger.info(f"Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados atmosféricos: {e}")
            raise ValueError(f"Falha ao carregar dados atmosféricos: {str(e)}")

    def _parse_open_meteo_json(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Parse de um chunk JSON do Open-Meteo Historical Forecast"""
        try:
            hourly_data = data.get('hourly', {})
            
            # Extrair timestamps
            timestamps = hourly_data.get('time', [])
            if not timestamps:
                logger.warning("Timestamps não encontrados no JSON")
                return pd.DataFrame()
            
            # Converter timestamps para datetime
            timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
            
            # Construir DataFrame com todas as variáveis disponíveis
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
            
            # Validar dados básicos
            if len(df) == 0:
                logger.warning("DataFrame vazio após parsing")
                return df
            
            # Log estatísticas básicas
            non_zero_cols = sum(1 for col in df.columns if col != 'timestamp' and df[col].sum() != 0)
            logger.debug(f"Chunk processado: {len(df)} registros, {non_zero_cols} colunas com dados")
            
            return df
            
        except Exception as e:
            logger.error(f"Erro no parsing JSON: {e}")
            return pd.DataFrame()

    def _filter_by_date_range(
        self, 
        df: pd.DataFrame, 
        start_date: Optional[datetime], 
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Filtra DataFrame por intervalo de datas"""
        if start_date:
            df = df[df['timestamp'] >= start_date]
        if end_date:
            df = df[df['timestamp'] <= end_date]
        
        logger.debug(f"Filtrado por data: {len(df)} registros restantes")
        return df

    def calculate_synoptic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula features derivadas para análise sinóptica
        
        Args:
            df: DataFrame com dados atmosféricos base
            
        Returns:
            pd.DataFrame: DataFrame com features sinópticas adicionadas
        """
        try:
            logger.debug("Calculando features sinópticas...")
            
            # Wind shear calculations
            if all(col in df.columns for col in ["wind_speed_850hPa", "wind_speed_500hPa"]):
                df["wind_shear_850_500"] = df["wind_speed_850hPa"] - df["wind_speed_500hPa"]
            else:
                df["wind_shear_850_500"] = 0.0
            
            if all(col in df.columns for col in ["wind_speed_1000hPa", "wind_speed_850hPa"]):
                df["wind_shear_1000_850"] = df["wind_speed_1000hPa"] - df["wind_speed_850hPa"]
            else:
                df["wind_shear_1000_850"] = 0.0
            
            # Temperature gradients
            if all(col in df.columns for col in ["temperature_850hPa", "temperature_500hPa"]):
                df["temp_gradient_850_500"] = df["temperature_850hPa"] - df["temperature_500hPa"]
            else:
                df["temp_gradient_850_500"] = 0.0
            
            if all(col in df.columns for col in ["temperature_2m", "temperature_850hPa"]):
                df["temp_gradient_surface_850"] = df["temperature_2m"] - df["temperature_850hPa"]
            else:
                df["temp_gradient_surface_850"] = 0.0
            
            # Frontal analysis (850hPa) - based on temperature changes
            if "temperature_850hPa" in df.columns:
                # Temporal gradient as proxy for frontal strength
                df["frontal_strength_850"] = df["temperature_850hPa"].diff().fillna(0)
                
                # Temperature advection proxy
                df["temperature_advection_850"] = (
                    df["temperature_850hPa"].rolling(window=3, center=True).mean() - df["temperature_850hPa"]
                ).fillna(0)
            else:
                df["frontal_strength_850"] = 0.0
                df["temperature_advection_850"] = 0.0
            
            # Vortex analysis (500hPa) - simplified vorticity and divergence
            if "wind_direction_500hPa" in df.columns:
                # Vorticity proxy: rate of change in wind direction
                df["vorticity_500"] = df["wind_direction_500hPa"].diff().fillna(0)
                # Handle wind direction wrap-around (0-360 degrees)
                df.loc[df["vorticity_500"] > 180, "vorticity_500"] -= 360
                df.loc[df["vorticity_500"] < -180, "vorticity_500"] += 360
            else:
                df["vorticity_500"] = 0.0
            
            if "wind_speed_500hPa" in df.columns:
                # Divergence proxy: rate of change in wind speed
                df["divergence_500"] = df["wind_speed_500hPa"].diff().fillna(0)
            else:
                df["divergence_500"] = 0.0
            
            # Atmospheric instability
            if all(col in df.columns for col in ["cape", "lifted_index"]):
                # Instability index combining CAPE and Lifted Index
                df["atmospheric_instability"] = df["cape"] / (df["lifted_index"].abs() + 1e-6)
                # Cap extreme values
                df["atmospheric_instability"] = df["atmospheric_instability"].clip(-1000, 1000)
            else:
                df["atmospheric_instability"] = 0.0
            
            # Moisture flux (850hPa level)
            if all(col in df.columns for col in ["relative_humidity_850hPa", "wind_speed_850hPa"]):
                df["moisture_flux"] = (df["relative_humidity_850hPa"] * df["wind_speed_850hPa"]) / 100.0
            else:
                df["moisture_flux"] = 0.0
            
            logger.debug("✓ Features sinópticas calculadas")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao calcular features sinópticas: {e}")
            # Adicionar features zeradas em caso de erro
            for feature in ["wind_shear_850_500", "wind_shear_1000_850", "temp_gradient_850_500", 
                           "temp_gradient_surface_850", "frontal_strength_850", "temperature_advection_850",
                           "vorticity_500", "divergence_500", "atmospheric_instability", "moisture_flux"]:
                if feature not in df.columns:
                    df[feature] = 0.0
            return df

    def validate_data_quality(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Valida qualidade dos dados atmosféricos
        
        Args:
            df: DataFrame com dados atmosféricos
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (dados_validados, relatório_qualidade)
        """
        try:
            logger.debug("Validando qualidade dos dados...")
            
            quality_report = {
                "total_records": len(df),
                "missing_data": {},
                "outliers": {},
                "data_issues": []
            }
            
            # Verificar dados faltantes
            for col in self.feature_columns:
                if col in df.columns:
                    missing_count = df[col].isna().sum()
                    missing_ratio = missing_count / len(df)
                    quality_report["missing_data"][col] = {
                        "count": int(missing_count),
                        "ratio": float(missing_ratio)
                    }
                    
                    if missing_ratio > self.max_missing_ratio:
                        quality_report["data_issues"].append(
                            f"Coluna {col}: {missing_ratio:.1%} dados faltantes (limite: {self.max_missing_ratio:.1%})"
                        )
                else:
                    quality_report["missing_data"][col] = {"count": len(df), "ratio": 1.0}
                    quality_report["data_issues"].append(f"Coluna {col} não encontrada nos dados")
            
            # Detectar outliers para variáveis críticas
            critical_vars = ["temperature_2m", "pressure_msl", "windspeed_10m", "precipitation"]
            
            for var in critical_vars:
                if var in df.columns and df[var].dtype in [np.float64, np.int64]:
                    mean_val = df[var].mean()
                    std_val = df[var].std()
                    
                    if std_val > 0:
                        z_scores = np.abs((df[var] - mean_val) / std_val)
                        outlier_count = (z_scores > self.outlier_threshold).sum()
                        outlier_ratio = outlier_count / len(df)
                        
                        quality_report["outliers"][var] = {
                            "count": int(outlier_count),
                            "ratio": float(outlier_ratio),
                            "threshold": self.outlier_threshold
                        }
                        
                        if outlier_ratio > 0.05:  # >5% outliers
                            quality_report["data_issues"].append(
                                f"Coluna {var}: {outlier_ratio:.1%} outliers (Z-score > {self.outlier_threshold})"
                            )
            
            # Verificar continuidade temporal
            if 'timestamp' in df.columns:
                df_sorted = df.sort_values('timestamp')
                time_diffs = df_sorted['timestamp'].diff().dt.total_seconds() / 3600  # horas
                
                # Detectar gaps maiores que 2 horas
                large_gaps = time_diffs[time_diffs > 2].count()
                if large_gaps > 0:
                    quality_report["data_issues"].append(
                        f"Encontrados {large_gaps} gaps temporais > 2 horas"
                    )
            
            logger.info(f"Validação concluída: {len(quality_report['data_issues'])} problemas encontrados")
            
            return df, quality_report
            
        except Exception as e:
            logger.error(f"Erro na validação de qualidade: {e}")
            return df, {"error": str(e)}

    def preprocess_for_model(
        self, 
        df: pd.DataFrame,
        target_sequence_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Preprocessa dados para entrada no modelo LSTM
        
        Args:
            df: DataFrame com dados atmosféricos
            target_sequence_length: Comprimento da sequência (padrão: self.sequence_length)
            
        Returns:
            np.ndarray: Array preprocessado shape (1, sequence_length, features)
        """
        try:
            sequence_length = target_sequence_length or self.sequence_length
            
            logger.debug(f"Preprocessando para modelo: {len(df)} -> {sequence_length} timesteps")
            
            # Calcular features sinópticas se ainda não foram calculadas
            if "wind_shear_850_500" not in df.columns:
                df = self.calculate_synoptic_features(df)
            
            # Padding se necessário
            if len(df) < sequence_length:
                df = self._pad_sequence(df, sequence_length)
            
            # Selecionar últimas sequence_length horas
            df = df.tail(sequence_length).copy()
            
            # Garantir que todas as features estão presentes
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
                    logger.warning(f"Feature {col} faltante, preenchida com 0")
            
            # Extrair features na ordem correta
            features = df[self.feature_columns].values
            
            # Tratar valores infinitos e NaN
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Reshape para formato LSTM: (1, sequence_length, n_features)
            features = features.reshape(1, sequence_length, len(self.feature_columns))
            
            logger.debug(f"Preprocessamento concluído: shape={features.shape}")
            
            return features
            
        except Exception as e:
            logger.error(f"Erro no preprocessamento: {e}")
            raise ValueError(f"Falha no preprocessamento: {str(e)}")

    def _pad_sequence(self, df: pd.DataFrame, target_length: int) -> pd.DataFrame:
        """Preenche sequência repetindo últimos valores disponíveis"""
        if len(df) == 0:
            raise ValueError("DataFrame vazio para padding")
        
        current_length = len(df)
        padding_needed = target_length - current_length
        
        if padding_needed > 0:
            logger.warning(f"Padding sequência: {current_length} -> {target_length}")
            
            # Repetir última linha
            last_row = df.iloc[-1:].copy()
            
            # Criar DataFrame com repetições da última linha
            padding_data = []
            for i in range(padding_needed):
                row_copy = last_row.copy()
                # Ajustar timestamp se presente
                if 'timestamp' in row_copy.columns:
                    new_timestamp = last_row['timestamp'].iloc[0] + timedelta(hours=i+1)
                    row_copy['timestamp'] = new_timestamp
                padding_data.append(row_copy)
            
            if padding_data:
                padding_df = pd.concat(padding_data, ignore_index=True)
                df = pd.concat([df, padding_df], ignore_index=True)
        
        return df

    def get_feature_importance_weights(self) -> Dict[str, float]:
        """
        Retorna pesos de importância para diferentes grupos de features
        
        Returns:
            Dict: Mapeamento feature -> peso de importância
        """
        # Pesos baseados na relevância meteorológica
        weights = {}
        
        # Surface variables - alta importância para precipitação
        surface_high_importance = ["precipitation", "temperature_2m", "relative_humidity_2m", 
                                  "pressure_msl", "windspeed_10m"]
        for feature in surface_high_importance:
            weights[feature] = 1.0
        
        # Pressure level variables - importância média a alta
        for level in ["850hPa", "700hPa", "500hPa"]:
            for var in ["temperature", "relative_humidity", "wind_speed"]:
                feature = f"{var}_{level}"
                weights[feature] = 0.8
        
        # Synoptic features - alta importância para previsão
        synoptic_features = ["wind_shear_850_500", "temp_gradient_850_500", 
                           "frontal_strength_850", "atmospheric_instability"]
        for feature in synoptic_features:
            weights[feature] = 0.9
        
        # Features menos críticas
        for feature in self.feature_columns:
            if feature not in weights:
                weights[feature] = 0.5
        
        return weights

    def export_preprocessed_data(
        self, 
        df: pd.DataFrame, 
        output_path: Path,
        include_metadata: bool = True
    ) -> bool:
        """
        Exporta dados preprocessados para arquivo
        
        Args:
            df: DataFrame com dados preprocessados
            output_path: Caminho para salvar
            include_metadata: Se deve incluir metadados
            
        Returns:
            bool: True se exportação foi bem-sucedida
        """
        try:
            logger.info(f"Exportando dados preprocessados para {output_path}")
            
            # Salvar dados principais
            df.to_parquet(output_path, index=False, compression='snappy')
            
            # Salvar metadados se solicitado
            if include_metadata:
                metadata = {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_records": len(df),
                    "features_count": len(self.feature_columns),
                    "feature_columns": self.feature_columns,
                    "sequence_length": self.sequence_length,
                    "data_period": {
                        "start": df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
                        "end": df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None
                    }
                }
                
                metadata_path = output_path.with_suffix('.metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"✓ Dados exportados: {len(df)} registros")
            return True
            
        except Exception as e:
            logger.error(f"Erro na exportação: {e}")
            return False 