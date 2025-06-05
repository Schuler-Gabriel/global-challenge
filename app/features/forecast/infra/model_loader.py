"""
Model Loader Infrastructure - LSTM Model Management

Este módulo implementa o carregamento, versionamento e gerenciamento
de modelos LSTM treinados para previsão meteorológica.

Funcionalidades:
- Carregamento automático de modelos
- Versionamento e fallback
- Validação de integridade
- Cache de modelos em memória
"""

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from app.core.exceptions import ModelLoadError, ValidationError
from app.features.forecast.infra.forecast_model import WeatherLSTMModel

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Informações de uma versão do modelo"""

    name: str
    path: Path
    version: str
    training_date: datetime
    metrics: Dict[str, float]
    file_hash: str
    is_valid: bool


class ModelLoader:
    """
    Gerenciador de carregamento e versionamento de modelos LSTM

    Responsável por:
    - Descobrir modelos disponíveis
    - Validar integridade dos modelos
    - Implementar estratégias de fallback
    - Cache de modelos em memória
    """

    def __init__(self, models_path: Optional[Path] = None):
        """
        Inicializa o carregador de modelos

        Args:
            models_path: Caminho para diretório de modelos
        """
        self.models_path = models_path or Path("data/modelos_treinados")
        self.models_path.mkdir(exist_ok=True)

        # Cache de modelos carregados
        self._model_cache: Dict[str, WeatherLSTMModel] = {}
        self._available_models: List[ModelVersion] = []

        logger.info(f"ModelLoader inicializado: {self.models_path}")

    def discover_models(self) -> List[ModelVersion]:
        """
        Descobre todos os modelos disponíveis no diretório

        Returns:
            List[ModelVersion]: Lista de modelos encontrados
        """
        logger.info("Descobrindo modelos disponíveis...")

        models = []

        # Procurar por diretórios de modelos TensorFlow
        for model_dir in self.models_path.iterdir():
            if model_dir.is_dir() and (model_dir / "saved_model.pb").exists():
                try:
                    model_version = self._analyze_model(model_dir)
                    if model_version:
                        models.append(model_version)
                        logger.info(
                            f"✓ Modelo encontrado: {model_version.name} (v{model_version.version})"
                        )
                except Exception as e:
                    logger.warning(f"Erro ao analisar modelo {model_dir.name}: {e}")

        # Procurar por arquivos .h5/.keras
        for model_file in self.models_path.glob("*.h5"):
            try:
                model_version = self._analyze_h5_model(model_file)
                if model_version:
                    models.append(model_version)
                    logger.info(f"✓ Modelo H5 encontrado: {model_version.name}")
            except Exception as e:
                logger.warning(f"Erro ao analisar modelo H5 {model_file.name}: {e}")

        # Ordenar por data de treinamento (mais recente primeiro)
        models.sort(key=lambda x: x.training_date, reverse=True)

        self._available_models = models
        logger.info(f"Total de modelos encontrados: {len(models)}")

        return models

    def _analyze_model(self, model_dir: Path) -> Optional[ModelVersion]:
        """Analisa um diretório de modelo TensorFlow"""
        try:
            # Carregar metadados se disponíveis
            metadata_path = self.models_path / "model_metadata.json"
            metadata = {}

            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

            # Calcular hash do modelo
            file_hash = self._calculate_model_hash(model_dir)

            # Extrair informações
            training_date_str = metadata.get(
                "training_date", datetime.now().isoformat()
            )
            training_date = datetime.fromisoformat(
                training_date_str.replace("Z", "+00:00")
            )

            version = training_date.strftime("%Y%m%d_%H%M%S")
            metrics = metadata.get("model_metrics", {})

            # Validar modelo
            is_valid = self._validate_model_files(model_dir)

            return ModelVersion(
                name=model_dir.name,
                path=model_dir,
                version=version,
                training_date=training_date,
                metrics=metrics,
                file_hash=file_hash,
                is_valid=is_valid,
            )

        except Exception as e:
            logger.error(f"Erro ao analisar modelo {model_dir}: {e}")
            return None

    def _analyze_h5_model(self, model_file: Path) -> Optional[ModelVersion]:
        """Analisa um arquivo de modelo H5"""
        try:
            # Usar timestamp do arquivo como versão
            stat = model_file.stat()
            training_date = datetime.fromtimestamp(stat.st_mtime)
            version = training_date.strftime("%Y%m%d_%H%M%S")

            # Calcular hash
            file_hash = self._calculate_file_hash(model_file)

            return ModelVersion(
                name=model_file.stem,
                path=model_file,
                version=version,
                training_date=training_date,
                metrics={},
                file_hash=file_hash,
                is_valid=True,  # Assumir válido para arquivos H5
            )

        except Exception as e:
            logger.error(f"Erro ao analisar modelo H5 {model_file}: {e}")
            return None

    def _validate_model_files(self, model_dir: Path) -> bool:
        """Valida se os arquivos do modelo estão íntegros"""
        required_files = ["saved_model.pb"]
        variables_dir = model_dir / "variables"

        # Verificar arquivos obrigatórios
        for file_name in required_files:
            if not (model_dir / file_name).exists():
                logger.warning(f"Arquivo obrigatório não encontrado: {file_name}")
                return False

        # Verificar diretório de variáveis
        if not variables_dir.exists():
            logger.warning("Diretório 'variables' não encontrado")
            return False

        # Verificar se há arquivos de variáveis
        var_files = list(variables_dir.glob("variables.*"))
        if not var_files:
            logger.warning("Arquivos de variáveis não encontrados")
            return False

        return True

    def _calculate_model_hash(self, model_dir: Path) -> str:
        """Calcula hash MD5 dos arquivos do modelo"""
        hasher = hashlib.md5()

        # Hash do arquivo principal
        saved_model_path = model_dir / "saved_model.pb"
        if saved_model_path.exists():
            hasher.update(saved_model_path.read_bytes())

        # Hash dos arquivos de variáveis
        variables_dir = model_dir / "variables"
        if variables_dir.exists():
            for var_file in sorted(variables_dir.glob("variables.*")):
                hasher.update(var_file.read_bytes())

        return hasher.hexdigest()

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calcula hash MD5 de um arquivo"""
        hasher = hashlib.md5()
        hasher.update(file_path.read_bytes())
        return hasher.hexdigest()

    def load_best_model(self) -> WeatherLSTMModel:
        """
        Carrega o melhor modelo disponível

        Returns:
            WeatherLSTMModel: Modelo carregado

        Raises:
            ModelLoadError: Se não conseguir carregar nenhum modelo
        """
        if not self._available_models:
            self.discover_models()

        if not self._available_models:
            raise ModelLoadError("Nenhum modelo disponível")

        # Tentar carregar modelos em ordem de prioridade
        for model_version in self._available_models:
            if not model_version.is_valid:
                continue

            try:
                return self.load_model(model_version.name)
            except Exception as e:
                logger.warning(f"Falha ao carregar modelo {model_version.name}: {e}")
                continue

        raise ModelLoadError("Não foi possível carregar nenhum modelo válido")

    def load_model(self, model_name: str) -> WeatherLSTMModel:
        """
        Carrega um modelo específico

        Args:
            model_name: Nome do modelo a carregar

        Returns:
            WeatherLSTMModel: Modelo carregado

        Raises:
            ModelLoadError: Se não conseguir carregar o modelo
        """
        # Verificar cache
        if model_name in self._model_cache:
            logger.info(f"Modelo {model_name} carregado do cache")
            return self._model_cache[model_name]

        logger.info(f"Carregando modelo: {model_name}")

        try:
            # Criar instância do modelo
            model = WeatherLSTMModel(self.models_path)

            # Carregar modelo
            success = model.load_model(model_name)
            if not success:
                raise ModelLoadError(f"Falha ao carregar modelo {model_name}")

            # Validar performance se possível
            if model.metadata:
                if not model.validate_model_performance():
                    logger.warning(
                        f"Modelo {model_name} não atende aos critérios de performance"
                    )

            # Adicionar ao cache
            self._model_cache[model_name] = model

            logger.info(f"✓ Modelo {model_name} carregado com sucesso")
            return model

        except Exception as e:
            logger.error(f"Erro ao carregar modelo {model_name}: {e}")
            raise ModelLoadError(f"Falha ao carregar modelo {model_name}: {str(e)}")

    def load_model_with_fallback(
        self, preferred_model: Optional[str] = None
    ) -> WeatherLSTMModel:
        """
        Carrega modelo com estratégia de fallback

        Args:
            preferred_model: Nome do modelo preferido (opcional)

        Returns:
            WeatherLSTMModel: Modelo carregado
        """
        # Tentar carregar modelo preferido primeiro
        if preferred_model:
            try:
                return self.load_model(preferred_model)
            except Exception as e:
                logger.warning(
                    f"Falha ao carregar modelo preferido {preferred_model}: {e}"
                )

        # Fallback para o melhor modelo disponível
        return self.load_best_model()

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retorna informações sobre um modelo específico

        Args:
            model_name: Nome do modelo

        Returns:
            Dict: Informações do modelo ou None se não encontrado
        """
        for model_version in self._available_models:
            if model_version.name == model_name:
                return {
                    "name": model_version.name,
                    "version": model_version.version,
                    "training_date": model_version.training_date.isoformat(),
                    "metrics": model_version.metrics,
                    "file_hash": model_version.file_hash,
                    "is_valid": model_version.is_valid,
                    "path": str(model_version.path),
                }
        return None

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Lista todos os modelos disponíveis

        Returns:
            List[Dict]: Lista de informações dos modelos
        """
        if not self._available_models:
            self.discover_models()

        return [
            {
                "name": mv.name,
                "version": mv.version,
                "training_date": mv.training_date.isoformat(),
                "metrics": mv.metrics,
                "is_valid": mv.is_valid,
                "file_hash": mv.file_hash[:8],  # Apenas primeiros 8 caracteres
            }
            for mv in self._available_models
        ]

    def validate_model_integrity(self, model_name: str) -> bool:
        """
        Valida a integridade de um modelo

        Args:
            model_name: Nome do modelo

        Returns:
            bool: True se modelo está íntegro
        """
        model_info = self.get_model_info(model_name)
        if not model_info:
            return False

        model_path = Path(model_info["path"])

        if model_path.is_dir():
            # Modelo TensorFlow SavedModel
            return self._validate_model_files(model_path)
        elif model_path.suffix in [".h5", ".keras"]:
            # Modelo H5/Keras
            return model_path.exists()

        return False

    def cleanup_cache(self):
        """Remove modelos do cache de memória"""
        logger.info(f"Limpando cache de modelos ({len(self._model_cache)} modelos)")
        self._model_cache.clear()

    def backup_model(self, model_name: str, backup_dir: Optional[Path] = None) -> bool:
        """
        Cria backup de um modelo

        Args:
            model_name: Nome do modelo
            backup_dir: Diretório de backup (opcional)

        Returns:
            bool: True se backup foi criado com sucesso
        """
        try:
            model_info = self.get_model_info(model_name)
            if not model_info:
                logger.error(f"Modelo {model_name} não encontrado")
                return False

            source_path = Path(model_info["path"])
            backup_dir = backup_dir or (self.models_path / "backups")
            backup_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{model_name}_backup_{timestamp}"
            backup_path = backup_dir / backup_name

            if source_path.is_dir():
                shutil.copytree(source_path, backup_path)
            else:
                shutil.copy2(source_path, backup_path.with_suffix(source_path.suffix))

            logger.info(f"✓ Backup criado: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Erro ao criar backup do modelo {model_name}: {e}")
            return False

    def delete_model(self, model_name: str, create_backup: bool = True) -> bool:
        """
        Remove um modelo do sistema

        Args:
            model_name: Nome do modelo
            create_backup: Se deve criar backup antes de deletar

        Returns:
            bool: True se modelo foi removido com sucesso
        """
        try:
            if create_backup:
                self.backup_model(model_name)

            model_info = self.get_model_info(model_name)
            if not model_info:
                logger.error(f"Modelo {model_name} não encontrado")
                return False

            model_path = Path(model_info["path"])

            if model_path.is_dir():
                shutil.rmtree(model_path)
            else:
                model_path.unlink()

            # Remover do cache
            if model_name in self._model_cache:
                del self._model_cache[model_name]

            # Atualizar lista de modelos
            self._available_models = [
                mv for mv in self._available_models if mv.name != model_name
            ]

            logger.info(f"✓ Modelo {model_name} removido")
            return True

        except Exception as e:
            logger.error(f"Erro ao remover modelo {model_name}: {e}")
            return False


# Instância global do carregador de modelos
model_loader = ModelLoader()
