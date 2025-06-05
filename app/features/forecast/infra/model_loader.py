"""
Model Loader - Infrastructure Layer

Este módulo implementa o carregamento de modelos TensorFlow para inferência.
É responsável por gerenciar o ciclo de vida dos modelos, carregamento eficiente
e metadados associados.
"""

import os
import json
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import glob

# Importações condicionais para permitir teste sem TensorFlow instalado
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class ModelLoader:
    """
    Responsável por carregar e gerenciar modelos TensorFlow
    
    Características:
    - Carregamento de modelos salvos em formato SavedModel
    - Suporte a diferentes versões de modelo
    - Cache de modelos carregados para performance
    - Metadados de modelo (data, métricas, parâmetros)
    """
    
    def __init__(self, models_dir: str):
        """
        Inicializa o loader de modelos
        
        Args:
            models_dir: Diretório onde os modelos estão armazenados
        """
        self.models_dir = models_dir
        self.models_cache: Dict[str, Any] = {}  # Versão -> Modelo carregado
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Verificar disponibilidade do TensorFlow
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning(
                "TensorFlow não disponível. ModelLoader funcionará em modo de compatibilidade."
            )
    
    def load_model(self, model_version: str) -> Any:
        """
        Carrega um modelo específico
        
        Args:
            model_version: Versão do modelo a carregar
            
        Returns:
            Model: Modelo TensorFlow carregado
            
        Raises:
            FileNotFoundError: Se modelo não encontrado
            RuntimeError: Se TensorFlow não disponível
        """
        # Verificar se modelo já está em cache
        if model_version in self.models_cache:
            self.logger.info(f"Usando modelo em cache: {model_version}")
            return self.models_cache[model_version]
        
        # Verificar disponibilidade do TensorFlow
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError(
                "TensorFlow não disponível. Impossível carregar modelo."
            )
        
        # Construir caminho do modelo
        model_path = os.path.join(self.models_dir, model_version)
        
        # Verificar se existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        # Carregar o modelo
        self.logger.info(f"Carregando modelo: {model_version} de {model_path}")
        start_time = datetime.now()
        
        try:
            model = tf.keras.models.load_model(model_path)
            
            # Registrar tempo de carregamento
            load_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Modelo carregado em {load_time:.2f}s: {model_version}")
            
            # Armazenar em cache
            self.models_cache[model_version] = model
            
            return model
        
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo {model_version}: {str(e)}")
            raise RuntimeError(f"Falha ao carregar modelo: {str(e)}")
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Lista todos os modelos disponíveis com metadados
        
        Returns:
            Dict: Mapeamento de versão -> metadados
        """
        models = {}
        
        # Procurar diretórios de modelo
        for model_dir in glob.glob(os.path.join(self.models_dir, "*")):
            if os.path.isdir(model_dir):
                version = os.path.basename(model_dir)
                metadata = self.get_model_metadata(version)
                models[version] = metadata
        
        return models
    
    def get_model_metadata(self, model_version: str) -> Dict[str, Any]:
        """
        Obtém metadados de um modelo específico
        
        Args:
            model_version: Versão do modelo
            
        Returns:
            Dict: Metadados do modelo
        """
        # Construir caminho do arquivo de metadados
        metadata_path = os.path.join(self.models_dir, model_version, "metadata.json")
        
        # Valores padrão
        metadata = {
            "version": model_version,
            "created_at": None,
            "input_shape": None,
            "output_shape": None,
            "features": None,
            "performance": {}
        }
        
        # Tentar carregar metadados
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    loaded_metadata = json.load(f)
                    metadata.update(loaded_metadata)
            except Exception as e:
                self.logger.warning(f"Erro ao carregar metadados para {model_version}: {str(e)}")
        
        # Se modelo está em cache, adicionar informações extras
        if TENSORFLOW_AVAILABLE and model_version in self.models_cache:
            model = self.models_cache[model_version]
            try:
                metadata["input_shape"] = model.input_shape
                metadata["output_shape"] = model.output_shape
            except Exception:
                pass
        
        return metadata
    
    def save_model_metadata(self, model_version: str, metadata: Dict[str, Any]) -> bool:
        """
        Salva metadados de um modelo
        
        Args:
            model_version: Versão do modelo
            metadata: Metadados a salvar
            
        Returns:
            bool: True se salvou com sucesso
        """
        # Construir caminho do arquivo de metadados
        model_dir = os.path.join(self.models_dir, model_version)
        metadata_path = os.path.join(model_dir, "metadata.json")
        
        # Verificar se diretório existe
        if not os.path.exists(model_dir):
            try:
                os.makedirs(model_dir)
            except Exception as e:
                self.logger.error(f"Erro ao criar diretório para {model_version}: {str(e)}")
                return False
        
        # Salvar metadados
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Erro ao salvar metadados para {model_version}: {str(e)}")
            return False
    
    def get_latest_model_version(self) -> Optional[str]:
        """
        Obtém a versão mais recente do modelo disponível
        
        Returns:
            Optional[str]: Versão mais recente ou None
        """
        models = self.get_available_models()
        
        if not models:
            return None
        
        # Ordenar por data de criação (se disponível)
        def get_created_date(version):
            metadata = models[version]
            created_at = metadata.get("created_at")
            if created_at:
                try:
                    return datetime.fromisoformat(created_at)
                except (ValueError, TypeError):
                    pass
            return datetime.min
        
        sorted_versions = sorted(models.keys(), key=get_created_date, reverse=True)
        return sorted_versions[0] if sorted_versions else None
    
    def clear_cache(self) -> None:
        """Limpa o cache de modelos carregados"""
        self.models_cache.clear()
        
    def get_model_summary(self, model_version: str) -> Optional[str]:
        """
        Obtém o resumo do modelo (estrutura)
        
        Args:
            model_version: Versão do modelo
            
        Returns:
            Optional[str]: Resumo do modelo ou None
        """
        if not TENSORFLOW_AVAILABLE:
            return None
        
        # Carregar modelo se não estiver em cache
        if model_version not in self.models_cache:
            try:
                self.load_model(model_version)
            except Exception:
                return None
        
        # Obter resumo
        model = self.models_cache[model_version]
        
        # Capturar saída do summary()
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            model.summary()
        
        return f.getvalue() 