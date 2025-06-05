import unittest
import asyncio
import os
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from app.features.forecast.domain.entities import Forecast, PrecipitationLevel
from app.features.forecast.domain.repositories import ForecastQuery
from app.features.forecast.infra.repositories import FileForecastRepository


class TestFileForecastRepository(unittest.TestCase):
    """Testes para FileForecastRepository"""
    
    def setUp(self):
        """Configuração para cada teste"""
        # Usar diretório temporário para testes
        self.test_data_dir = "test_data"
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Criar repositório
        self.repository = FileForecastRepository(data_dir=self.test_data_dir)
        
        # Criar dados de teste
        self.test_forecasts = [
            Forecast(
                id=str(uuid.uuid4()),
                timestamp=datetime.now() - timedelta(hours=i),
                precipitation_mm=i * 1.5,
                confidence_score=0.8 - (i * 0.01),
                forecast_horizon_hours=24,
                input_sequence_length=24,
                model_version="v1.0.0",
                inference_time_ms=150 + i
            )
            for i in range(10)  # 10 previsões
        ]
        
        # Salvar dados de teste
        self._save_test_data()
    
    def tearDown(self):
        """Limpeza após cada teste"""
        # Remover arquivo de teste
        data_file = Path(self.test_data_dir) / "forecasts.json"
        if data_file.exists():
            data_file.unlink()
        
        # Remover diretório de teste
        try:
            os.rmdir(self.test_data_dir)
        except:
            pass
    
    def _save_test_data(self):
        """Salvar dados de teste diretamente no arquivo"""
        data_file = Path(self.test_data_dir) / "forecasts.json"
        
        # Converter para formato serializável
        data_list = []
        for item in self.test_forecasts:
            item_dict = item.__dict__.copy()
            item_dict["timestamp"] = item_dict["timestamp"].isoformat()
            data_list.append(item_dict)
        
        # Salvar para arquivo
        with open(data_file, "w") as f:
            json.dump(data_list, f, indent=2)
    
    def test_get_latest_forecast(self):
        """Testar obtenção da previsão mais recente"""
        # Executar método assíncrono
        result = asyncio.run(self.repository.get_latest_forecast())
        
        # Verificar resultado
        self.assertIsNotNone(result)
        
        # Verificar que é a mais recente
        self.assertEqual(result.timestamp, self.test_forecasts[0].timestamp)
    
    def test_get_forecast_by_id(self):
        """Testar obtenção de previsão por ID"""
        # Pegar ID de uma previsão existente
        test_forecast = self.test_forecasts[3]
        
        # Executar método assíncrono
        result = asyncio.run(self.repository.get_forecast_by_id(test_forecast.id))
        
        # Verificar resultado
        self.assertIsNotNone(result)
        self.assertEqual(result.id, test_forecast.id)
    
    def test_get_forecasts_by_period(self):
        """Testar obtenção de previsões por período"""
        # Definir período
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=5)
        
        # Executar método assíncrono
        result = asyncio.run(self.repository.get_forecasts_by_period(start_date, end_date))
        
        # Verificar resultado
        self.assertGreater(len(result), 0)
        
        # Verificar que todos os registros estão no período
        for item in result:
            self.assertGreaterEqual(item.timestamp, start_date)
            self.assertLessEqual(item.timestamp, end_date)
    
    def test_get_forecasts_by_query(self):
        """Testar obtenção de previsões usando objeto de consulta"""
        # Criar query
        query = ForecastQuery(
            start_date=datetime.now() - timedelta(hours=8),
            end_date=datetime.now(),
            model_version="v1.0.0",
            min_confidence=0.75,
            order_by="timestamp",
            order_direction="desc",
            limit=3
        )
        
        # Executar método assíncrono
        result = asyncio.run(self.repository.get_forecasts_by_query(query))
        
        # Verificar resultado
        self.assertLessEqual(len(result), 3)
        self.assertGreater(len(result), 0)
        
        # Verificar que todos os registros correspondem à query
        for item in result:
            self.assertEqual(item.model_version, query.model_version)
            self.assertGreaterEqual(item.timestamp, query.start_date)
            self.assertLessEqual(item.timestamp, query.end_date)
            self.assertGreaterEqual(item.confidence_score, query.min_confidence)
        
        # Verificar ordenação decrescente
        for i in range(1, len(result)):
            self.assertGreaterEqual(getattr(result[i-1], query.order_by), 
                                   getattr(result[i], query.order_by))
    
    def test_save_forecast(self):
        """Testar salvamento de previsão"""
        # Criar nova previsão
        new_forecast = Forecast(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            precipitation_mm=25.5,
            confidence_score=0.95,
            forecast_horizon_hours=24,
            input_sequence_length=24,
            model_version="v1.0.0",
            inference_time_ms=120
        )
        
        # Executar método assíncrono
        success = asyncio.run(self.repository.save_forecast(new_forecast))
        
        # Verificar resultado
        self.assertTrue(success)
        
        # Verificar que a previsão foi salva
        result = asyncio.run(self.repository.get_forecast_by_id(new_forecast.id))
        self.assertIsNotNone(result)
        self.assertEqual(result.id, new_forecast.id)
        self.assertEqual(result.precipitation_mm, new_forecast.precipitation_mm)
    
    def test_delete_forecast(self):
        """Testar remoção de previsão"""
        # Pegar ID de uma previsão existente
        test_forecast = self.test_forecasts[2]
        
        # Executar método assíncrono
        success = asyncio.run(self.repository.delete_forecast(test_forecast.id))
        
        # Verificar resultado
        self.assertTrue(success)
        
        # Verificar que a previsão foi removida
        result = asyncio.run(self.repository.get_forecast_by_id(test_forecast.id))
        self.assertIsNone(result)
    
    def test_precipitation_level(self):
        """Testar cálculo do nível de precipitação"""
        levels = [
            (0.0, PrecipitationLevel.NONE),
            (0.1, PrecipitationLevel.LIGHT),
            (2.5, PrecipitationLevel.MODERATE),
            (15.0, PrecipitationLevel.HEAVY),
            (60.0, PrecipitationLevel.EXTREME)
        ]
        
        # Criar previsão com diferentes níveis e testar
        for value, expected_level in levels:
            forecast = Forecast(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                precipitation_mm=value,
                confidence_score=0.9,
                forecast_horizon_hours=24,
                input_sequence_length=24,
                model_version="v1.0.0",
                inference_time_ms=120
            )
            
            # Verificar nível
            self.assertEqual(forecast.get_precipitation_level(), expected_level)


if __name__ == "__main__":
    unittest.main() 