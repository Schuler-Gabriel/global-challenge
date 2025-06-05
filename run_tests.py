#!/usr/bin/env python
"""
Script para executar todos os testes unitários do sistema.
"""

import unittest
import sys
import os

def run_all_tests():
    """Executa todos os testes disponíveis"""
    # Adicionar diretório atual ao path do Python
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Descobrir e executar todos os testes
    test_suite = unittest.defaultTestLoader.discover('tests', pattern='test_*.py')
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Retornar código de saída baseado no resultado
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    # Executar testes e sair com código apropriado
    sys.exit(run_all_tests()) 