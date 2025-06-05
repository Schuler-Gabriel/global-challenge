import unittest
import sys
import os

# Garantir que o app está no path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.features.forecast.dependencies import (
    get_weather_data_repository,
    get_forecast_repository,
    get_model_repository,
    get_cache_repository,
    get_forecast_service,
    get_model_validation_service,
    get_generate_forecast_usecase
)

class TestDependencies(unittest.TestCase):
    """Testes para injeção de dependências"""
    
    def test_repositories_dependencies(self):
        """Testar criação de repositories"""
        # Obter instances
        weather_repo = get_weather_data_repository()
        forecast_repo = get_forecast_repository()
        model_repo = get_model_repository()
        cache_repo = get_cache_repository()
        
        # Verificar tipos
        self.assertIsNotNone(weather_repo)
        self.assertIsNotNone(forecast_repo)
        self.assertIsNotNone(model_repo)
        self.assertIsNotNone(cache_repo)
        
        # Verificar singleton (mesma instância)
        self.assertIs(weather_repo, get_weather_data_repository())
        self.assertIs(forecast_repo, get_forecast_repository())
        self.assertIs(model_repo, get_model_repository())
        self.assertIs(cache_repo, get_cache_repository())
    
    def test_services_dependencies(self):
        """Testar criação de serviços"""
        # Obter instances
        forecast_service = get_forecast_service()
        model_validation_service = get_model_validation_service()
        
        # Verificar tipos
        self.assertIsNotNone(forecast_service)
        self.assertIsNotNone(model_validation_service)
        
        # Verificar singleton (mesma instância)
        self.assertIs(forecast_service, get_forecast_service())
        self.assertIs(model_validation_service, get_model_validation_service())
    
    def test_usecases_dependencies(self):
        """Testar criação de use cases"""
        # Obter instance
        generate_forecast_usecase = get_generate_forecast_usecase()
        
        # Verificar tipo
        self.assertIsNotNone(generate_forecast_usecase)
        
        # Verificar singleton (mesma instância)
        self.assertIs(generate_forecast_usecase, get_generate_forecast_usecase())
        
        # Verificar injeção de dependências
        self.assertIsNotNone(generate_forecast_usecase.weather_data_repository)
        self.assertIsNotNone(generate_forecast_usecase.forecast_repository)
        self.assertIsNotNone(generate_forecast_usecase.model_repository)
        self.assertIsNotNone(generate_forecast_usecase.cache_repository)
        self.assertIsNotNone(generate_forecast_usecase.forecast_service)


if __name__ == "__main__":
    unittest.main() 