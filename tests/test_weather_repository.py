import unittest
import asyncio
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

from app.features.forecast.domain.entities import WeatherData
from app.features.forecast.domain.repositories import WeatherDataQuery
from app.features.forecast.infra.repositories import FileWeatherDataRepository


class TestFileWeatherDataRepository(unittest.TestCase):
    """Testes para FileWeatherDataRepository"""
    
    def setUp(self):
        """Configuração para cada teste"""
        # Usar diretório temporário para testes
        self.test_data_dir = "test_data"
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Criar repositório
        self.repository = FileWeatherDataRepository(data_dir=self.test_data_dir)
        
        # Criar dados de teste
        self.test_data = [
            WeatherData(
                timestamp=datetime.now() - timedelta(hours=i),
                precipitation=0.5 * i,
                temperature=25.0 - (i * 0.5),
                pressure=1010.0 + i,
                humidity=60.0 + i,
                wind_speed=10.0 - (i * 0.2),
                dew_point=15.0,
                wind_direction=180.0,
                station_id="TEST001"
            )
            for i in range(1, 25)  # 24 registros para 24 horas
        ]
        
        # Salvar dados de teste
        self._save_test_data()
    
    def tearDown(self):
        """Limpeza após cada teste"""
        # Remover arquivo de teste
        data_file = Path(self.test_data_dir) / "weather_data.json"
        if data_file.exists():
            data_file.unlink()
        
        # Remover diretório de teste
        try:
            os.rmdir(self.test_data_dir)
        except:
            pass
    
    def _save_test_data(self):
        """Salvar dados de teste diretamente no arquivo"""
        data_file = Path(self.test_data_dir) / "weather_data.json"
        
        # Converter para formato serializável
        data_list = []
        for item in self.test_data:
            item_dict = item.__dict__.copy()
            item_dict["timestamp"] = item_dict["timestamp"].isoformat()
            data_list.append(item_dict)
        
        # Salvar para arquivo
        with open(data_file, "w") as f:
            json.dump(data_list, f, indent=2)
    
    def test_get_latest_data(self):
        """Testar obtenção dos dados mais recentes"""
        # Executar método assíncrono
        result = asyncio.run(self.repository.get_latest_data(count=10))
        
        # Verificar resultado
        self.assertEqual(len(result), 10)
        
        # Verificar ordenação cronológica (mais antigo primeiro)
        for i in range(1, len(result)):
            self.assertLessEqual(result[i-1].timestamp, result[i].timestamp)
    
    def test_get_data_by_period(self):
        """Testar obtenção de dados por período"""
        # Definir período
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=12)
        
        # Executar método assíncrono
        result = asyncio.run(self.repository.get_data_by_period(start_date, end_date))
        
        # Verificar resultado
        self.assertGreater(len(result), 0)
        
        # Verificar que todos os registros estão no período
        for item in result:
            self.assertGreaterEqual(item.timestamp, start_date)
            self.assertLessEqual(item.timestamp, end_date)
    
    def test_get_data_by_query(self):
        """Testar obtenção de dados usando objeto de consulta"""
        # Criar query
        query = WeatherDataQuery(
            station_id="TEST001",
            start_date=datetime.now() - timedelta(hours=24),
            end_date=datetime.now(),
            order_by="timestamp",
            order_direction="asc",
            limit=5
        )
        
        # Executar método assíncrono
        result = asyncio.run(self.repository.get_data_by_query(query))
        
        # Verificar resultado
        self.assertLessEqual(len(result), 5)
        self.assertGreater(len(result), 0)
        
        # Verificar que todos os registros correspondem à query
        for item in result:
            self.assertEqual(item.station_id, query.station_id)
            self.assertGreaterEqual(item.timestamp, query.start_date)
            self.assertLessEqual(item.timestamp, query.end_date)
        
        # Verificar ordenação
        for i in range(1, len(result)):
            self.assertLessEqual(getattr(result[i-1], query.order_by), 
                               getattr(result[i], query.order_by))


if __name__ == "__main__":
    unittest.main() 