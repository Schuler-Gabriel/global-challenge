"""
Forecast Model - Infrastructure Layer

Este módulo implementa o wrapper do modelo LSTM para previsão meteorológica.
Encapsula a funcionalidade de inferência, processamento de entrada e saída,
e interpretação dos resultados.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

# Importações condicionais para permitir teste sem TensorFlow instalado
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class ForecastModel:
    """
    Encapsula o modelo LSTM para previsão meteorológica
    
    Responsabilidades:
    - Executar inferência com modelo TensorFlow
    - Processar dados de entrada e saída
    - Interpretar resultados da previsão
    - Calcular scores de confiança
    """
    
    def __init__(self, model=None, model_version: str = "unknown"):
        """
        Inicializa o modelo de previsão
        
        Args:
            model: Modelo TensorFlow carregado (opcional)
            model_version: Versão do modelo
        """
        self.model = model
        self.model_version = model_version
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Configurações padrão
        self.sequence_length = 24  # 24 horas de histórico
        self.forecast_horizon = 24  # 24 horas de previsão à frente
        self.features_count = 16   # Número de features
        
        # Verificar disponibilidade do TensorFlow
        if not TENSORFLOW_AVAILABLE and model is not None:
            self.logger.warning(
                "TensorFlow não disponível. ForecastModel funcionará em modo de compatibilidade."
            )
    
    def predict(
        self, 
        input_data: np.ndarray, 
        return_confidence: bool = True
    ) -> Tuple[float, Optional[float]]:
        """
        Realiza previsão de precipitação
        
        Args:
            input_data: Dados de entrada (shape: [1, sequence_length, features_count])
            return_confidence: Se deve calcular score de confiança
            
        Returns:
            Tuple[float, Optional[float]]: (precipitação prevista, score de confiança)
            
        Raises:
            RuntimeError: Se modelo não inicializado ou TensorFlow indisponível
        """
        if self.model is None:
            raise RuntimeError("Modelo não inicializado")
        
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow não disponível para inferência")
        
        start_time = datetime.now()
        
        # Validar shape dos dados de entrada
        if len(input_data.shape) != 3:
            raise ValueError(f"Formato de entrada inválido: {input_data.shape}. Esperado: [1, {self.sequence_length}, {self.features_count}]")
            
        # Executar inferência
        self.logger.debug(f"Executando inferência com modelo {self.model_version}")
        
        try:
            prediction = self.model.predict(input_data, verbose=0)
            
            # Registrar tempo de inferência
            inference_time = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.debug(f"Inferência concluída em {inference_time:.2f}ms")
            
            # Extrair valor de precipitação (assumindo que é o primeiro output)
            precipitation_value = float(prediction[0][0])
            
            # Calcular score de confiança se solicitado
            confidence_score = None
            if return_confidence:
                confidence_score = self._calculate_confidence_score(prediction, input_data)
            
            return precipitation_value, confidence_score
        
        except Exception as e:
            self.logger.error(f"Erro durante inferência: {str(e)}")
            raise RuntimeError(f"Falha na inferência: {str(e)}")
    
    def _calculate_confidence_score(
        self, 
        prediction: np.ndarray, 
        input_data: np.ndarray
    ) -> float:
        """
        Calcula score de confiança para a previsão
        
        Args:
            prediction: Saída do modelo
            input_data: Entrada do modelo
            
        Returns:
            float: Score de confiança (0.0 - 1.0)
        """
        # Nota: Esta é uma implementação simplificada.
        # Em um sistema real, poderia incorporar:
        # - Análise estatística de volatilidade
        # - Comparação com distribuição histórica
        # - Meta-modelo para estimativa de incerteza
        
        # Implementação básica: score baseado na estabilidade dos dados de entrada
        # Instabilidade = maior variância nos dados de entrada
        
        try:
            # Calcular variância média das features
            feature_variances = np.var(input_data[0], axis=0)
            mean_variance = np.mean(feature_variances)
            
            # Converter para score (inversamente proporcional à variância)
            # Quanto maior a variância, menor a confiança
            raw_score = 1.0 / (1.0 + mean_variance)
            
            # Normalizar para range 0.5-1.0
            # Mesmo com alta variância, garantimos confiança mínima de 0.5
            confidence = 0.5 + (raw_score * 0.5)
            
            return float(confidence)
        
        except Exception as e:
            self.logger.warning(f"Erro ao calcular score de confiança: {str(e)}")
            return 0.75  # Valor padrão moderado
    
    def set_model(self, model, model_version: str = "unknown") -> None:
        """
        Define o modelo TensorFlow a ser usado
        
        Args:
            model: Modelo TensorFlow
            model_version: Versão do modelo
        """
        self.model = model
        self.model_version = model_version
        
        # Tentar inferir configurações do modelo
        if TENSORFLOW_AVAILABLE and model is not None:
            try:
                # Inferir sequence_length e features_count da forma de entrada
                input_shape = model.input_shape
                if input_shape and len(input_shape) >= 3:
                    self.sequence_length = input_shape[1]
                    self.features_count = input_shape[2]
                    self.logger.info(
                        f"Configurações do modelo inferidas: sequence_length={self.sequence_length}, "
                        f"features_count={self.features_count}"
                    )
            except Exception as e:
                self.logger.warning(f"Não foi possível inferir configurações do modelo: {str(e)}")
    
    def predict_next_hours(
        self, 
        input_data: np.ndarray, 
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Realiza previsão para as próximas horas
        
        Args:
            input_data: Dados de entrada
            hours: Número de horas a prever
            
        Returns:
            List[Dict]: Lista de previsões horárias
        """
        if hours <= 0:
            raise ValueError("Número de horas deve ser positivo")
        
        predictions = []
        current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        for hour in range(hours):
            # Para hora atual, fazer previsão direta
            if hour == 0:
                precipitation, confidence = self.predict(input_data)
            else:
                # Para horas futuras, confiança diminui linearmente
                # Em um sistema real, faria previsões recorrentes
                precipitation, base_confidence = self.predict(input_data)
                confidence = max(0.5, base_confidence * (1.0 - (hour / (2 * hours))))
            
            # Criar entrada na lista de previsões
            prediction_time = current_time + timedelta(hours=hour)
            predictions.append({
                "timestamp": prediction_time.isoformat(),
                "hour": hour,
                "precipitation_mm": round(precipitation, 2),
                "confidence_score": round(confidence, 2)
            })
        
        return predictions
    
    def is_precipitation_expected(
        self, 
        predictions: List[Dict[str, Any]], 
        threshold: float = 0.1
    ) -> bool:
        """
        Verifica se há previsão de precipitação nas próximas horas
        
        Args:
            predictions: Lista de previsões horárias
            threshold: Limite mínimo para considerar precipitação (mm/h)
            
        Returns:
            bool: True se chuva é esperada
        """
        for prediction in predictions:
            if prediction["precipitation_mm"] >= threshold:
                return True
        return False
    
    def get_max_precipitation(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Obtém a previsão com maior precipitação
        
        Args:
            predictions: Lista de previsões horárias
            
        Returns:
            Dict: Previsão com maior precipitação
        """
        if not predictions:
            return {}
        
        max_prediction = max(
            predictions, 
            key=lambda p: p["precipitation_mm"]
        )
        
        return max_prediction
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtém informações sobre o modelo carregado
        
        Returns:
            Dict: Informações do modelo
        """
        info = {
            "model_version": self.model_version,
            "sequence_length": self.sequence_length,
            "features_count": self.features_count,
            "forecast_horizon": self.forecast_horizon,
            "is_loaded": self.model is not None
        }
        
        # Adicionar informações do TensorFlow se disponível
        if TENSORFLOW_AVAILABLE and self.model is not None:
            try:
                info["input_shape"] = self.model.input_shape
                info["output_shape"] = self.model.output_shape
                info["layers_count"] = len(self.model.layers)
            except Exception:
                pass
        
        return info 