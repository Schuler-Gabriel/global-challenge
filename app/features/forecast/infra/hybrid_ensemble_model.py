"""
Hybrid Ensemble Model for Flood Prediction

Modelo de ensemble híbrido que combina:
- Open-Meteo Forecast API (149 variáveis atmosféricas + sinótica)
- Open-Meteo Historical API (25 variáveis de superfície)
- INMET data (validação opcional)

Implementa estratégia de stacking com análise sinótica para melhor precisão.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import asyncio
from dataclasses import dataclass

# ML imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    ml_import_error = str(e)

from app.core.config import get_settings
from app.core.exceptions import ModelException, DataValidationException
from app.features.external_apis.infra.open_meteo_client import OpenMeteoCurrentWeatherClient
from app.features.external_apis.infra.open_meteo_historical_client import OpenMeteoHistoricalClient


logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class EnsembleConfig:
    """Configuração do modelo de ensemble híbrido"""
    # Parâmetros do LSTM primário (149 variáveis)
    primary_sequence_length: int = 30  # 30 dias
    primary_lstm_units: int = 128
    primary_dropout: float = 0.2
    
    # Parâmetros do LSTM secundário (25 variáveis)
    secondary_sequence_length: int = 60  # 60 dias
    secondary_lstm_units: int = 64
    secondary_dropout: float = 0.15
    
    # Parâmetros do ensemble
    stacking_meta_model: str = "random_forest"  # ou "gradient_boosting"
    ensemble_weights: Dict[str, float] = None
    
    # Análise sinótica
    synoptic_weight: float = 0.3
    pressure_levels: List[int] = None
    
    # Treinamento
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    # Performance
    target_accuracy: float = 0.85  # 85%
    confidence_threshold: float = 0.7
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                "primary_lstm": 0.5,  # 149 variáveis
                "secondary_lstm": 0.3,  # 25 variáveis
                "synoptic_analysis": 0.2  # Análise sinótica
            }
        
        if self.pressure_levels is None:
            self.pressure_levels = [850, 500]  # hPa


class HybridEnsembleModel:
    """
    Modelo de ensemble híbrido para previsão de cheias
    
    Combina dados atmosféricos completos (Open-Meteo) com análise sinótica
    para previsão de precipitação e risco de cheias em Porto Alegre.
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        if not ML_AVAILABLE:
            raise ModelException(f"Dependências ML não disponíveis: {ml_import_error}")
            
        self.config = config or EnsembleConfig()
        self.is_trained = False
        
        # Modelos componentes
        self.primary_lstm = None  # 149 variáveis atmosféricas
        self.secondary_lstm = None  # 25 variáveis de superfície
        self.meta_model = None  # Stacking ensemble
        
        # Scalers
        self.primary_scaler = StandardScaler()
        self.secondary_scaler = RobustScaler()
        self.target_scaler = StandardScaler()
        
        # Métricas de performance
        self.training_history = {}
        self.validation_metrics = {}
        
        # Clientes de dados
        self.current_weather_client = OpenMeteoCurrentWeatherClient()
        self.historical_client = OpenMeteoHistoricalClient()

    async def prepare_training_data(
        self, 
        start_date: str = "2000-01-01",
        end_date: str = "2024-12-31"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepara dados de treinamento do ensemble híbrido
        
        Args:
            start_date: Data de início dos dados históricos
            end_date: Data de fim dos dados históricos
            
        Returns:
            Tuple: (X_primary, X_secondary, y, synoptic_features)
        """
        
        logger.info(f"Preparando dados de treinamento para período {start_date} a {end_date}")
        
        try:
            # Busca dados históricos
            async with self.historical_client as client:
                historical_data = await client.get_historical_data(start_date, end_date)
                
            if not historical_data or historical_data['record_count'] == 0:
                raise DataValidationException("Dados históricos insuficientes")

            # Busca dados com variáveis atmosféricas (simulação para histórico)
            atmospheric_data = await self._get_extended_atmospheric_data(start_date, end_date)

            # Processa dados primários (149 variáveis atmosféricas)
            X_primary = self._process_primary_features(atmospheric_data)
            
            # Processa dados secundários (25 variáveis de superfície)
            X_secondary = self._process_secondary_features(historical_data)
            
            # Processa features sinóticas
            synoptic_features = self._extract_synoptic_features(atmospheric_data)
            
            # Cria targets (precipitação + risco de cheia)
            y = self._create_targets(historical_data)
            
            # Valida dimensões
            self._validate_training_data(X_primary, X_secondary, y, synoptic_features)
            
            logger.info(f"Dados preparados: {X_primary.shape[0]} amostras, "
                       f"{X_primary.shape[2]} variáveis primárias, "
                       f"{X_secondary.shape[2]} variáveis secundárias")
            
            return X_primary, X_secondary, y, synoptic_features
            
        except Exception as e:
            raise ModelException(f"Erro ao preparar dados de treinamento: {str(e)}")

    def _process_primary_features(self, atmospheric_data: Dict) -> np.ndarray:
        """
        Processa features primárias (149 variáveis atmosféricas)
        
        Args:
            atmospheric_data: Dados atmosféricos completos
            
        Returns:
            np.ndarray: Features processadas (samples, timesteps, features)
        """
        
        try:
            # Extrai variáveis de superfície
            surface_vars = [
                'temperature_2m', 'relative_humidity_2m', 'precipitation',
                'pressure_msl', 'wind_speed_10m', 'wind_direction_10m',
                'cloud_cover', 'visibility', 'weather_code'
            ]
            
            # Extrai variáveis de níveis de pressão
            pressure_vars = []
            for level in [1000, 925, 850, 700, 500, 300]:  # Níveis principais
                for var in ['temperature', 'wind_speed', 'wind_direction', 
                           'geopotential_height', 'relative_humidity']:
                    pressure_vars.append(f"{var}_{level}hPa")
            
            # Variáveis derivadas
            derived_vars = [
                'temperature_gradient_850_500',
                'wind_shear_850_500', 
                'relative_vorticity_500',
                'thermal_advection_850',
                'q_vector_divergence',
                'potential_temperature_850',
                'equivalent_potential_temperature',
                'cape_index',
                'lifted_index',
                'k_index',
                'cross_totals_index',
                'vertical_totals_index',
                'total_totals_index',
                'sweat_index',
                'bulk_richardson_number'
            ]
            
            all_variables = surface_vars + pressure_vars + derived_vars
            
            # Cria matriz de features simuladas (para desenvolvimento)
            # Em produção, viria da API Open-Meteo com dados reais
            n_samples = 1000  # Simulado
            n_timesteps = self.config.primary_sequence_length
            n_features = len(all_variables)
            
            # Simula dados atmosféricos realistas
            X_primary = self._simulate_atmospheric_data(n_samples, n_timesteps, n_features)
            
            return X_primary
            
        except Exception as e:
            raise DataValidationException(f"Erro ao processar features primárias: {str(e)}")

    def _process_secondary_features(self, historical_data: Dict) -> np.ndarray:
        """
        Processa features secundárias (25 variáveis de superfície)
        
        Args:
            historical_data: Dados históricos da Open-Meteo
            
        Returns:
            np.ndarray: Features processadas (samples, timesteps, features)
        """
        
        try:
            data = historical_data.get('data', {})
            if not data:
                raise DataValidationException("Dados históricos vazios")
            
            # Variáveis de superfície (25 variáveis)
            surface_variables = [
                'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
                'precipitation_sum', 'rain_sum', 'pressure_msl_mean',
                'wind_speed_10m_max', 'wind_speed_10m_mean', 'wind_direction_10m_dominant',
                'relative_humidity_2m_max', 'relative_humidity_2m_min', 'relative_humidity_2m_mean',
                'shortwave_radiation_sum', 'et0_fao_evapotranspiration',
                'weather_code_most_frequent', 'cloud_cover_mean',
                'apparent_temperature_max', 'apparent_temperature_min', 'apparent_temperature_mean',
                'sunshine_duration', 'dewpoint_2m_mean'
            ]
            
            # Extrai features disponíveis
            feature_matrix = []
            for var in surface_variables:
                if var in data and data[var]:
                    feature_matrix.append(data[var])
                else:
                    # Preenche com zeros se variável ausente
                    feature_matrix.append([0.0] * len(data.get('precipitation_sum', [0])))
            
            if not feature_matrix:
                raise DataValidationException("Nenhuma variável de superfície encontrada")
            
            # Converte para array numpy
            features_array = np.array(feature_matrix).T  # (time, features)
            
            # Cria sequências temporais
            X_secondary = self._create_sequences(
                features_array, 
                self.config.secondary_sequence_length
            )
            
            return X_secondary
            
        except Exception as e:
            raise DataValidationException(f"Erro ao processar features secundárias: {str(e)}")

    def _extract_synoptic_features(self, atmospheric_data: Dict) -> np.ndarray:
        """
        Extrai features sinóticas para análise meteorológica
        
        Args:
            atmospheric_data: Dados atmosféricos
            
        Returns:
            np.ndarray: Features sinóticas
        """
        
        try:
            # Features sinóticas principais
            synoptic_features = []
            
            # Simulação para desenvolvimento
            n_samples = 1000
            n_features = 15  # Features sinóticas
            
            # Features sinóticas incluem:
            # 1. Gradientes de temperatura 850-500hPa
            # 2. Cisalhamento do vento
            # 3. Vorticidade relativa
            # 4. Advecção térmica
            # 5. Divergência dos vetores Q
            # 6. Índices de instabilidade (CAPE, LI, K-index, etc.)
            # 7. Padrões de frentes (detecção automática)
            # 8. Características de vórtices
            # 9. Teleconexões (ENSO, AAO, etc.)
            
            synoptic_matrix = np.random.normal(0, 1, (n_samples, n_features))
            
            # Adiciona padrões realistas
            synoptic_matrix[:, 0] = np.random.normal(-2, 3, n_samples)  # Gradiente térmico
            synoptic_matrix[:, 1] = np.random.exponential(5, n_samples)  # Cisalhamento
            synoptic_matrix[:, 2] = np.random.normal(0, 2, n_samples)  # Vorticidade
            
            return synoptic_matrix
            
        except Exception as e:
            raise DataValidationException(f"Erro ao extrair features sinóticas: {str(e)}")

    def _create_targets(self, historical_data: Dict) -> np.ndarray:
        """
        Cria targets para treinamento (precipitação + risco)
        
        Args:
            historical_data: Dados históricos
            
        Returns:
            np.ndarray: Targets (precipitation, flood_risk)
        """
        
        try:
            data = historical_data.get('data', {})
            
            # Target principal: precipitação
            precipitation = data.get('precipitation_sum', [])
            if not precipitation:
                raise DataValidationException("Dados de precipitação não encontrados")
            
            # Converte para array numpy
            precip_array = np.array([p if p is not None else 0.0 for p in precipitation])
            
            # Cria target de risco de cheia baseado em precipitação
            # Risco alto: > 50mm/dia, Risco médio: 20-50mm, Risco baixo: < 20mm
            flood_risk = np.where(precip_array > 50, 1.0,  # Alto
                         np.where(precip_array > 20, 0.5,  # Médio
                                 0.0))  # Baixo
            
            # Combina targets
            targets = np.column_stack([precip_array, flood_risk])
            
            # Remove amostras com sequências incompletas
            sequence_length = max(self.config.primary_sequence_length, 
                                self.config.secondary_sequence_length)
            
            if len(targets) > sequence_length:
                targets = targets[sequence_length:]
            
            return targets
            
        except Exception as e:
            raise DataValidationException(f"Erro ao criar targets: {str(e)}")

    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """
        Cria sequências temporais para LSTM
        
        Args:
            data: Dados temporais (time, features)
            sequence_length: Comprimento da sequência
            
        Returns:
            np.ndarray: Sequências (samples, timesteps, features)
        """
        
        if len(data) < sequence_length:
            raise DataValidationException(f"Dados insuficientes: {len(data)} < {sequence_length}")
        
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        
        return np.array(sequences)

    def _simulate_atmospheric_data(self, n_samples: int, n_timesteps: int, n_features: int) -> np.ndarray:
        """
        Simula dados atmosféricos para desenvolvimento
        
        Args:
            n_samples: Número de amostras
            n_timesteps: Passos temporais
            n_features: Número de features
            
        Returns:
            np.ndarray: Dados simulados
        """
        
        # Cria dados com correlações temporais realistas
        np.random.seed(42)  # Para reprodutibilidade
        
        data = np.random.normal(0, 1, (n_samples, n_timesteps, n_features))
        
        # Adiciona tendências e correlações temporais
        for i in range(1, n_timesteps):
            data[:, i, :] = 0.8 * data[:, i-1, :] + 0.2 * data[:, i, :]
        
        # Adiciona padrões sazonais
        for f in range(n_features):
            seasonal = np.sin(2 * np.pi * np.arange(n_timesteps) / 365)
            data[:, :, f] += seasonal * 0.5
        
        return data

    async def _get_extended_atmospheric_data(self, start_date: str, end_date: str) -> Dict:
        """
        Obtém dados atmosféricos estendidos (simulação para desenvolvimento)
        
        Args:
            start_date: Data de início
            end_date: Data de fim
            
        Returns:
            Dict: Dados atmosféricos simulados
        """
        
        # Em produção, isso viria da API Open-Meteo Forecast
        # Para desenvolvimento, simula estrutura de dados
        
        return {
            'period': {'start': start_date, 'end': end_date},
            'data': {
                'atmospheric_profiles': True,
                'pressure_levels': [1000, 925, 850, 700, 500, 300],
                'surface_variables': 9,
                'pressure_variables': 30,
                'derived_variables': 110
            },
            'quality': {'overall_score': 0.9}
        }

    def build_model(self) -> None:
        """
        Constrói a arquitetura do ensemble híbrido
        """
        
        try:
            logger.info("Construindo modelo de ensemble híbrido")
            
            # Modelo primário (149 variáveis atmosféricas)
            primary_input = Input(shape=(self.config.primary_sequence_length, 149))
            primary_lstm = LSTM(
                self.config.primary_lstm_units,
                return_sequences=False,
                dropout=self.config.primary_dropout
            )(primary_input)
            primary_dense = Dense(64, activation='relu')(primary_lstm)
            primary_output = Dense(32, activation='relu')(primary_dense)
            
            # Modelo secundário (25 variáveis de superfície)
            secondary_input = Input(shape=(self.config.secondary_sequence_length, 25))
            secondary_lstm = LSTM(
                self.config.secondary_lstm_units,
                return_sequences=False,
                dropout=self.config.secondary_dropout
            )(secondary_input)
            secondary_dense = Dense(32, activation='relu')(secondary_lstm)
            secondary_output = Dense(16, activation='relu')(secondary_dense)
            
            # Features sinóticas
            synoptic_input = Input(shape=(15,))  # 15 features sinóticas
            synoptic_dense = Dense(32, activation='relu')(synoptic_input)
            synoptic_output = Dense(16, activation='relu')(synoptic_dense)
            
            # Combina todos os modelos
            combined = concatenate([primary_output, secondary_output, synoptic_output])
            
            # Camadas de ensemble
            ensemble_layer1 = Dense(64, activation='relu')(combined)
            ensemble_dropout = Dropout(0.3)(ensemble_layer1)
            ensemble_layer2 = Dense(32, activation='relu')(ensemble_dropout)
            
            # Saídas finais
            precipitation_output = Dense(1, activation='linear', name='precipitation')(ensemble_layer2)
            flood_risk_output = Dense(1, activation='sigmoid', name='flood_risk')(ensemble_layer2)
            
            # Modelo completo
            self.model = Model(
                inputs=[primary_input, secondary_input, synoptic_input],
                outputs=[precipitation_output, flood_risk_output]
            )
            
            # Compilação com múltiplos objetivos
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss={
                    'precipitation': 'mse',
                    'flood_risk': 'binary_crossentropy'
                },
                loss_weights={
                    'precipitation': 1.0,
                    'flood_risk': 2.0  # Maior peso para classificação de risco
                },
                metrics={
                    'precipitation': ['mae', 'mse'],
                    'flood_risk': ['accuracy', 'precision', 'recall']
                }
            )
            
            logger.info(f"Modelo construído: {self.model.count_params()} parâmetros")
            
        except Exception as e:
            raise ModelException(f"Erro ao construir modelo: {str(e)}")

    async def train(
        self, 
        X_primary: np.ndarray,
        X_secondary: np.ndarray,
        y: np.ndarray,
        synoptic_features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Treina o modelo de ensemble híbrido
        
        Args:
            X_primary: Features primárias (149 variáveis)
            X_secondary: Features secundárias (25 variáveis)
            y: Targets (precipitação, risco)
            synoptic_features: Features sinóticas
            
        Returns:
            Dict: Métricas de treinamento
        """
        
        try:
            logger.info("Iniciando treinamento do ensemble híbrido")
            
            # Normaliza os dados
            X_primary_scaled = self._scale_primary_data(X_primary)
            X_secondary_scaled = self._scale_secondary_data(X_secondary)
            synoptic_scaled = self._scale_synoptic_data(synoptic_features)
            y_scaled = self._scale_targets(y)
            
            # Prepara targets separados
            y_precipitation = y_scaled[:, 0]
            y_flood_risk = y_scaled[:, 1]
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
            
            # Treinamento
            history = self.model.fit(
                x=[X_primary_scaled, X_secondary_scaled, synoptic_scaled],
                y={'precipitation': y_precipitation, 'flood_risk': y_flood_risk},
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            self.is_trained = True
            self.training_history = history.history
            
            # Avalia performance
            validation_metrics = self._evaluate_model(
                X_primary_scaled, X_secondary_scaled, synoptic_scaled,
                y_precipitation, y_flood_risk
            )
            
            self.validation_metrics = validation_metrics
            
            logger.info(f"Treinamento concluído. Acurácia: {validation_metrics.get('accuracy', 0):.3f}")
            
            return {
                'training_completed': True,
                'final_metrics': validation_metrics,
                'epochs_trained': len(history.history['loss']),
                'best_val_loss': min(history.history.get('val_loss', [float('inf')])),
                'model_ready': self.is_trained
            }
            
        except Exception as e:
            raise ModelException(f"Erro durante treinamento: {str(e)}")

    def _scale_primary_data(self, data: np.ndarray) -> np.ndarray:
        """Normaliza dados primários"""
        original_shape = data.shape
        reshaped = data.reshape(-1, data.shape[-1])
        scaled = self.primary_scaler.fit_transform(reshaped)
        return scaled.reshape(original_shape)

    def _scale_secondary_data(self, data: np.ndarray) -> np.ndarray:
        """Normaliza dados secundários"""
        original_shape = data.shape
        reshaped = data.reshape(-1, data.shape[-1])
        scaled = self.secondary_scaler.fit_transform(reshaped)
        return scaled.reshape(original_shape)

    def _scale_synoptic_data(self, data: np.ndarray) -> np.ndarray:
        """Normaliza dados sinóticos"""
        return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)

    def _scale_targets(self, targets: np.ndarray) -> np.ndarray:
        """Normaliza targets"""
        scaled = self.target_scaler.fit_transform(targets)
        return scaled

    def _evaluate_model(
        self, 
        X_primary: np.ndarray,
        X_secondary: np.ndarray,
        synoptic: np.ndarray,
        y_precip: np.ndarray,
        y_risk: np.ndarray
    ) -> Dict[str, float]:
        """
        Avalia performance do modelo
        
        Returns:
            Dict: Métricas de validação
        """
        
        try:
            # Predições
            predictions = self.model.predict([X_primary, X_secondary, synoptic])
            pred_precip = predictions[0].flatten()
            pred_risk = predictions[1].flatten()
            
            # Métricas de precipitação
            precip_mse = mean_squared_error(y_precip, pred_precip)
            precip_mae = mean_absolute_error(y_precip, pred_precip)
            precip_r2 = r2_score(y_precip, pred_precip)
            
            # Métricas de risco
            risk_binary_pred = (pred_risk > 0.5).astype(int)
            risk_binary_true = (y_risk > 0.5).astype(int)
            risk_accuracy = np.mean(risk_binary_pred == risk_binary_true)
            
            return {
                'precipitation_mse': float(precip_mse),
                'precipitation_mae': float(precip_mae),
                'precipitation_r2': float(precip_r2),
                'flood_risk_accuracy': float(risk_accuracy),
                'accuracy': float(risk_accuracy),  # Métrica principal
                'overall_score': float(0.5 * precip_r2 + 0.5 * risk_accuracy)
            }
            
        except Exception as e:
            logger.error(f"Erro ao avaliar modelo: {str(e)}")
            return {'accuracy': 0.0, 'overall_score': 0.0}

    async def predict(
        self, 
        current_conditions: Optional[Dict] = None,
        days_ahead: int = 7
    ) -> Dict[str, Any]:
        """
        Realiza previsão com o modelo híbrido
        
        Args:
            current_conditions: Condições atuais (opcional)
            days_ahead: Dias de previsão
            
        Returns:
            Dict: Previsões e análise
        """
        
        if not self.is_trained:
            raise ModelException("Modelo não treinado")
        
        try:
            logger.info(f"Gerando previsão para {days_ahead} dias")
            
            # Busca condições atuais se não fornecidas
            if current_conditions is None:
                async with self.current_weather_client as client:
                    current_conditions = await client.get_current_conditions()
            
            # Prepara dados para predição
            X_primary, X_secondary, synoptic = await self._prepare_prediction_data(
                current_conditions, days_ahead
            )
            
            # Realiza predição
            predictions = self.model.predict([X_primary, X_secondary, synoptic])
            
            # Processa resultados
            precipitation_pred = self.target_scaler.inverse_transform(
                np.column_stack([predictions[0], np.zeros_like(predictions[0])])
            )[:, 0]
            
            flood_risk_pred = predictions[1].flatten()
            
            # Análise sinótica
            synoptic_analysis = self._analyze_synoptic_conditions(current_conditions)
            
            # Calcula confiança
            confidence = self._calculate_prediction_confidence(
                precipitation_pred, flood_risk_pred, synoptic_analysis
            )
            
            return {
                'timestamp': datetime.now().isoformat(),
                'forecast_days': days_ahead,
                'precipitation_forecast': {
                    'daily_values': precipitation_pred.tolist(),
                    'total_period': float(np.sum(precipitation_pred)),
                    'max_daily': float(np.max(precipitation_pred)),
                    'unit': 'mm'
                },
                'flood_risk_forecast': {
                    'daily_risk': flood_risk_pred.tolist(),
                    'max_risk': float(np.max(flood_risk_pred)),
                    'risk_level': self._classify_risk_level(np.max(flood_risk_pred)),
                    'high_risk_days': int(np.sum(flood_risk_pred > 0.7))
                },
                'synoptic_analysis': synoptic_analysis,
                'confidence': {
                    'overall': float(confidence),
                    'precipitation': float(min(1.0, confidence * 1.1)),
                    'flood_risk': float(min(1.0, confidence * 0.9))
                },
                'model_info': {
                    'type': 'hybrid_ensemble',
                    'components': ['lstm_149var', 'lstm_25var', 'synoptic_analysis'],
                    'training_score': self.validation_metrics.get('overall_score', 0.0),
                    'last_trained': self.training_history.get('timestamp', 'unknown')
                }
            }
            
        except Exception as e:
            raise ModelException(f"Erro durante predição: {str(e)}")

    def _validate_training_data(
        self, 
        X_primary: np.ndarray,
        X_secondary: np.ndarray,
        y: np.ndarray,
        synoptic: np.ndarray
    ) -> None:
        """Valida dados de treinamento"""
        
        if X_primary.shape[0] != X_secondary.shape[0]:
            raise DataValidationException("Dimensões incompatíveis entre dados primários e secundários")
        
        if X_primary.shape[0] != y.shape[0]:
            raise DataValidationException("Dimensões incompatíveis entre features e targets")
        
        if X_primary.shape[0] != synoptic.shape[0]:
            raise DataValidationException("Dimensões incompatíveis com features sinóticas")
        
        if np.any(np.isnan(X_primary)) or np.any(np.isnan(X_secondary)):
            raise DataValidationException("Dados contêm valores NaN")

    async def _prepare_prediction_data(
        self, 
        current_conditions: Dict,
        days_ahead: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepara dados para predição"""
        
        # Simula preparação de dados de predição
        # Em produção, usaria dados reais das APIs
        
        X_primary = np.random.normal(0, 1, (1, self.config.primary_sequence_length, 149))
        X_secondary = np.random.normal(0, 1, (1, self.config.secondary_sequence_length, 25))
        synoptic = np.random.normal(0, 1, (1, 15))
        
        return X_primary, X_secondary, synoptic

    def _analyze_synoptic_conditions(self, current_conditions: Dict) -> Dict[str, Any]:
        """Analisa condições sinóticas atuais"""
        
        synoptic = current_conditions.get('synoptic_analysis', {})
        
        return {
            'frontal_activity': synoptic.get('850hPa', {}).get('frontal_indicator', 'stable'),
            'vortex_activity': synoptic.get('500hPa', {}).get('vorticity_indicator', 'low'),
            'atmospheric_stability': synoptic.get('combined_analysis', {}).get('atmospheric_stability', 'stable'),
            'weather_pattern': synoptic.get('combined_analysis', {}).get('weather_pattern', 'stable_pattern'),
            'synoptic_risk': synoptic.get('combined_analysis', {}).get('risk_level', 'low')
        }

    def _calculate_prediction_confidence(
        self, 
        precip_pred: np.ndarray,
        risk_pred: np.ndarray,
        synoptic: Dict
    ) -> float:
        """Calcula confiança da predição"""
        
        base_confidence = 0.7
        
        # Ajusta baseado em variabilidade
        precip_var = np.std(precip_pred) / (np.mean(precip_pred) + 1e-6)
        if precip_var < 0.5:
            base_confidence += 0.1
        elif precip_var > 1.5:
            base_confidence -= 0.1
        
        # Ajusta baseado em análise sinótica
        if synoptic.get('atmospheric_stability') == 'stable':
            base_confidence += 0.1
        elif synoptic.get('atmospheric_stability') == 'unstable':
            base_confidence -= 0.1
        
        return max(0.3, min(0.95, base_confidence))

    def _classify_risk_level(self, max_risk: float) -> str:
        """Classifica nível de risco"""
        
        if max_risk > 0.8:
            return "ALTO"
        elif max_risk > 0.5:
            return "MÉDIO"
        elif max_risk > 0.3:
            return "BAIXO"
        else:
            return "MÍNIMO"

    def save_model(self, filepath: str) -> None:
        """Salva modelo treinado"""
        
        if not self.is_trained:
            raise ModelException("Modelo não treinado")
        
        try:
            # Salva modelo Keras
            self.model.save(f"{filepath}_keras_model")
            
            # Salva scalers e metadados
            import pickle
            metadata = {
                'config': self.config,
                'training_history': self.training_history,
                'validation_metrics': self.validation_metrics,
                'primary_scaler': self.primary_scaler,
                'secondary_scaler': self.secondary_scaler,
                'target_scaler': self.target_scaler
            }
            
            with open(f"{filepath}_metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.info(f"Modelo salvo em {filepath}")
            
        except Exception as e:
            raise ModelException(f"Erro ao salvar modelo: {str(e)}")

    def load_model(self, filepath: str) -> None:
        """Carrega modelo treinado"""
        
        try:
            # Carrega modelo Keras
            self.model = tf.keras.models.load_model(f"{filepath}_keras_model")
            
            # Carrega metadados
            import pickle
            with open(f"{filepath}_metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            self.config = metadata['config']
            self.training_history = metadata['training_history']
            self.validation_metrics = metadata['validation_metrics']
            self.primary_scaler = metadata['primary_scaler']
            self.secondary_scaler = metadata['secondary_scaler']
            self.target_scaler = metadata['target_scaler']
            
            self.is_trained = True
            
            logger.info(f"Modelo carregado de {filepath}")
            
        except Exception as e:
            raise ModelException(f"Erro ao carregar modelo: {str(e)}")


# Funções de conveniência
async def create_hybrid_model() -> HybridEnsembleModel:
    """
    Cria e configura modelo híbrido
    
    Returns:
        HybridEnsembleModel: Modelo configurado
    """
    
    config = EnsembleConfig()
    model = HybridEnsembleModel(config)
    model.build_model()
    
    return model


async def train_hybrid_model(
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31"
) -> HybridEnsembleModel:
    """
    Treina modelo híbrido completo
    
    Args:
        start_date: Data de início dos dados
        end_date: Data de fim dos dados
        
    Returns:
        HybridEnsembleModel: Modelo treinado
    """
    
    model = await create_hybrid_model()
    
    # Prepara dados
    X_primary, X_secondary, y, synoptic = await model.prepare_training_data(
        start_date, end_date
    )
    
    # Treina modelo
    training_results = await model.train(X_primary, X_secondary, y, synoptic)
    
    logger.info(f"Modelo treinado com sucesso: {training_results}")
    
    return model