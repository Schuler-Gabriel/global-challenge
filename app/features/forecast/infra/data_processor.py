"""
Data Processor - Infrastructure Layer

Este módulo é responsável pelo processamento de dados meteorológicos
para uso no modelo LSTM, incluindo normalização, formatação de sequências
e transformações específicas.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

# Importações condicionais para permitir teste sem dependências opcionais
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..domain.entities import WeatherData


class DataProcessor:
    """
    Processador de dados meteorológicos para modelo LSTM
    
    Responsabilidades:
    - Formatar dados para entrada do modelo
    - Normalizar features com StandardScaler
    - Criar sequências temporais
    - Validar integridade dos dados
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        """
        Inicializa o processador de dados
        
        Args:
            data_dir: Diretório para artefatos de preprocessing
        """
        self.data_dir = data_dir
        self.feature_scaler = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Mapeamento entre campos de WeatherData e colunas de entrada
        self.feature_mapping = {
            'precipitation': 'precipitation',
            'pressure': 'pressure',
            'temperature': 'temperature',
            'dew_point': 'dew_point',
            'humidity': 'humidity',
            'wind_speed': 'wind_speed',
            'wind_direction': 'wind_direction',
            'radiation': 'radiation',
            'pressure_max': 'pressure_max',
            'pressure_min': 'pressure_min',
            'temperature_max': 'temperature_max',
            'temperature_min': 'temperature_min',
            'humidity_max': 'humidity_max',
            'humidity_min': 'humidity_min',
            'dew_point_max': 'dew_point_max',
            'dew_point_min': 'dew_point_min'
        }
        
        # Colunas de feature na ordem esperada pelo modelo
        self.feature_columns = list(self.feature_mapping.values())
        
        # Verificar disponibilidade das dependências
        if not PANDAS_AVAILABLE:
            self.logger.warning("Pandas não disponível. Funcionalidade limitada.")
        
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Scikit-learn não disponível. Normalização desabilitada.")
        
        # Tentar carregar scalers salvos
        self._load_scalers()
    
    def _load_scalers(self) -> bool:
        """
        Carrega scalers salvos em disco
        
        Returns:
            bool: True se carregou com sucesso
        """
        if not SKLEARN_AVAILABLE:
            return False
        
        scaler_path = os.path.join(self.data_dir, "feature_scaler.joblib")
        
        if os.path.exists(scaler_path):
            try:
                self.feature_scaler = joblib.load(scaler_path)
                self.logger.info(f"Scaler carregado de {scaler_path}")
                return True
            except Exception as e:
                self.logger.warning(f"Erro ao carregar scaler: {str(e)}")
        
        self.logger.info("Criando novo scaler")
        self.feature_scaler = StandardScaler()
        return False
    
    def save_scalers(self) -> bool:
        """
        Salva scalers em disco
        
        Returns:
            bool: True se salvou com sucesso
        """
        if not SKLEARN_AVAILABLE or self.feature_scaler is None:
            return False
        
        # Garantir que diretório existe
        os.makedirs(self.data_dir, exist_ok=True)
        
        scaler_path = os.path.join(self.data_dir, "feature_scaler.joblib")
        
        try:
            joblib.dump(self.feature_scaler, scaler_path)
            self.logger.info(f"Scaler salvo em {scaler_path}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao salvar scaler: {str(e)}")
            return False
    
    def process_weather_data(
        self, 
        weather_data: List[WeatherData], 
        sequence_length: int = 24
    ) -> np.ndarray:
        """
        Processa dados meteorológicos para entrada do modelo
        
        Args:
            weather_data: Lista de dados meteorológicos
            sequence_length: Tamanho da sequência temporal
            
        Returns:
            np.ndarray: Array formatado para entrada do modelo
            
        Raises:
            ValueError: Se dados são insuficientes ou inválidos
        """
        if len(weather_data) < sequence_length:
            raise ValueError(
                f"Dados insuficientes: {len(weather_data)} < {sequence_length}"
            )
        
        # Garantir que os dados estão em ordem cronológica
        weather_data = sorted(weather_data, key=lambda x: x.timestamp)
        
        # Usar os dados mais recentes
        recent_data = weather_data[-sequence_length:]
        
        # Converter para formato matricial
        if PANDAS_AVAILABLE:
            # Usar pandas para processamento mais robusto
            return self._process_with_pandas(recent_data, sequence_length)
        else:
            # Fallback para processamento direto com numpy
            return self._process_with_numpy(recent_data, sequence_length)
    
    def _process_with_pandas(
        self, 
        weather_data: List[WeatherData], 
        sequence_length: int
    ) -> np.ndarray:
        """Processa dados usando pandas"""
        # Converter para DataFrame
        data_dict = []
        for wd in weather_data:
            row = {}
            for attr, col in self.feature_mapping.items():
                # Tratar valores None
                value = getattr(wd, attr, None)
                row[col] = 0.0 if value is None else float(value)
            data_dict.append(row)
        
        df = pd.DataFrame(data_dict)
        
        # Verificar se todas as colunas estão presentes
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        # Selecionar colunas na ordem correta
        features = df[self.feature_columns].values
        
        # Normalizar se scaler estiver disponível
        if SKLEARN_AVAILABLE and self.feature_scaler is not None:
            if hasattr(self.feature_scaler, 'mean_'):
                features = self.feature_scaler.transform(features)
            else:
                self.logger.warning("Scaler não treinado, usando dados não normalizados")
        
        # Reshape para formato do LSTM: (1, sequence_length, features)
        return features.reshape(1, sequence_length, len(self.feature_columns))
    
    def _process_with_numpy(
        self, 
        weather_data: List[WeatherData], 
        sequence_length: int
    ) -> np.ndarray:
        """Processa dados diretamente com numpy (fallback)"""
        # Inicializar matriz vazia
        feature_count = len(self.feature_columns)
        features = np.zeros((sequence_length, feature_count))
        
        # Preencher com valores
        for i, wd in enumerate(weather_data):
            for j, attr in enumerate(self.feature_mapping.keys()):
                value = getattr(wd, attr, None)
                features[i, j] = 0.0 if value is None else float(value)
        
        # Normalizar se possível
        if SKLEARN_AVAILABLE and self.feature_scaler is not None:
            if hasattr(self.feature_scaler, 'mean_'):
                features = self.feature_scaler.transform(features)
        
        # Reshape para formato do LSTM: (1, sequence_length, features)
        return features.reshape(1, sequence_length, feature_count)
    
    def validate_data_quality(self, weather_data: List[WeatherData]) -> Dict[str, Any]:
        """
        Valida qualidade dos dados meteorológicos
        
        Args:
            weather_data: Lista de dados meteorológicos
            
        Returns:
            Dict: Relatório de qualidade dos dados
        """
        report = {
            "total_records": len(weather_data),
            "missing_values": {},
            "out_of_range_values": {},
            "timestamp_gaps": [],
            "quality_score": 1.0
        }
        
        if not weather_data:
            report["quality_score"] = 0.0
            return report
        
        # Verificar valores ausentes
        for attr in self.feature_mapping.keys():
            missing = sum(1 for wd in weather_data if getattr(wd, attr, None) is None)
            if missing > 0:
                report["missing_values"][attr] = missing
        
        # Verificar valores fora do range esperado
        valid_ranges = {
            'precipitation': (0, 200),    # mm/h
            'temperature': (-10, 50),     # °C
            'humidity': (0, 100),         # %
            'pressure': (900, 1100),      # mB
            'wind_speed': (0, 50)         # m/s
        }
        
        for attr, (min_val, max_val) in valid_ranges.items():
            out_of_range = sum(
                1 for wd in weather_data 
                if hasattr(wd, attr) and 
                getattr(wd, attr) is not None and 
                (getattr(wd, attr) < min_val or getattr(wd, attr) > max_val)
            )
            if out_of_range > 0:
                report["out_of_range_values"][attr] = out_of_range
        
        # Verificar gaps temporais (se dados ordenados)
        sorted_data = sorted(weather_data, key=lambda x: x.timestamp)
        for i in range(1, len(sorted_data)):
            gap = (sorted_data[i].timestamp - sorted_data[i-1].timestamp).total_seconds() / 3600
            if gap > 1.0:  # Gap maior que 1 hora
                report["timestamp_gaps"].append({
                    "from": sorted_data[i-1].timestamp.isoformat(),
                    "to": sorted_data[i].timestamp.isoformat(),
                    "hours": round(gap, 1)
                })
        
        # Calcular score de qualidade
        penalties = 0.0
        
        # Penalidade por valores ausentes
        missing_ratio = sum(report["missing_values"].values()) / (len(weather_data) * len(self.feature_mapping))
        penalties += missing_ratio * 0.4
        
        # Penalidade por valores fora do range
        out_of_range_ratio = sum(report["out_of_range_values"].values()) / (len(weather_data) * len(valid_ranges))
        penalties += out_of_range_ratio * 0.3
        
        # Penalidade por gaps temporais
        gap_ratio = len(report["timestamp_gaps"]) / max(1, len(weather_data) - 1)
        penalties += gap_ratio * 0.3
        
        report["quality_score"] = max(0.0, 1.0 - penalties)
        
        return report
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retorna importância relativa das features (se disponível)
        
        Returns:
            Dict: Mapeamento de feature para importância
        """
        # Em um sistema real, isso viria do modelo ou análise estatística
        # Aqui, usamos valores aproximados para demonstração
        return {
            "precipitation": 0.35,
            "pressure": 0.15,
            "temperature": 0.12,
            "humidity": 0.10,
            "wind_speed": 0.08,
            "wind_direction": 0.05,
            "dew_point": 0.08,
            "radiation": 0.07
        }
    
    def fit_scaler(self, weather_data: List[WeatherData]) -> bool:
        """
        Treina o scaler com os dados fornecidos
        
        Args:
            weather_data: Dados para treinar o scaler
            
        Returns:
            bool: True se treinou com sucesso
        """
        if not SKLEARN_AVAILABLE:
            return False
        
        if not weather_data:
            return False
        
        try:
            # Processar dados para formato matricial
            data_dict = []
            for wd in weather_data:
                row = {}
                for attr, col in self.feature_mapping.items():
                    value = getattr(wd, attr, None)
                    row[col] = 0.0 if value is None else float(value)
                data_dict.append(row)
            
            if PANDAS_AVAILABLE:
                df = pd.DataFrame(data_dict)
                for col in self.feature_columns:
                    if col not in df.columns:
                        df[col] = 0.0
                features = df[self.feature_columns].values
            else:
                # Fallback para numpy
                features = np.zeros((len(weather_data), len(self.feature_columns)))
                for i, wd in enumerate(weather_data):
                    for j, attr in enumerate(self.feature_mapping.keys()):
                        value = getattr(wd, attr, None)
                        features[i, j] = 0.0 if value is None else float(value)
            
            # Treinar scaler
            self.feature_scaler = StandardScaler()
            self.feature_scaler.fit(features)
            
            # Salvar scaler
            self.save_scalers()
            
            self.logger.info(f"Scaler treinado com {len(weather_data)} registros")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao treinar scaler: {str(e)}")
            return False
    
    def get_processor_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o processador
        
        Returns:
            Dict: Informações de configuração
        """
        info = {
            "features": self.feature_columns,
            "scaler_type": "StandardScaler" if SKLEARN_AVAILABLE else "None",
            "scaler_fitted": False,
            "pandas_available": PANDAS_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE
        }
        
        if SKLEARN_AVAILABLE and self.feature_scaler is not None:
            info["scaler_fitted"] = hasattr(self.feature_scaler, 'mean_')
            if info["scaler_fitted"]:
                info["feature_means"] = self.feature_scaler.mean_.tolist()
                info["feature_scales"] = self.feature_scaler.scale_.tolist()
        
        return info 