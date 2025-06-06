#!/usr/bin/env python3
"""
🏗️ Script de Migração para Nova Arquitetura
Reorganiza o projeto seguindo as melhores práticas ML/TensorFlow
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
import sys

class ArchitectureMigrator:
    def __init__(self, base_path="."):
        self.base_path = Path(base_path).resolve()
        self.backup_path = self.base_path / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"🏗️ MIGRAÇÃO PARA NOVA ARQUITETURA")
        print(f"📁 Base path: {self.base_path}")
        print(f"💾 Backup path: {self.backup_path}")
        
    def create_backup(self):
        """Cria backup do projeto atual"""
        print("\n💾 CRIANDO BACKUP DO PROJETO ATUAL")
        print("=" * 50)
        
        try:
            # Itens para backup
            items_to_backup = [
                "notebooks", "scripts", "data", "models", 
                "app", "configs", "tests", "docs"
            ]
            
            self.backup_path.mkdir(exist_ok=True)
            
            for item in items_to_backup:
                source = self.base_path / item
                if source.exists():
                    if source.is_dir():
                        shutil.copytree(source, self.backup_path / item, 
                                      ignore_dangling_symlinks=True)
                    else:
                        shutil.copy2(source, self.backup_path / item)
                    print(f"✅ Backup: {item}")
                else:
                    print(f"⚠️ Não encontrado: {item}")
                    
            print(f"✅ Backup criado em: {self.backup_path}")
            return True
            
        except Exception as e:
            print(f"❌ Erro no backup: {e}")
            return False
    
    def create_ml_platform_structure(self):
        """Cria estrutura da ML Platform"""
        print("\n🧠 CRIANDO ESTRUTURA ML PLATFORM")
        print("=" * 50)
        
        ml_platform = self.base_path / "ml-platform"
        
        # Estrutura de diretórios ML Platform
        ml_structure = {
            "data": ["raw", "processed", "features", "models", "validation"],
            "notebooks": [
                "01-data-collection", 
                "02-exploratory-analysis",
                "03-feature-engineering", 
                "04-model-development",
                "05-model-evaluation", 
                "06-model-validation"
            ],
            "src": {
                "data": ["collectors", "processors", "validators"],
                "features": ["atmospheric", "temporal", "transforms"],
                "models": ["base", "lstm", "ensemble", "hybrid"],
                "training": ["trainers", "optimizers"],
                "evaluation": ["metrics", "validators", "reports"],
                "utils": []
            },
            "scripts": {
                "data": [],
                "training": [],
                "evaluation": [],
                "deployment": []
            },
            "configs": ["data", "models", "training"],
            "experiments": ["runs", "results"],
            "tests": ["test_data", "test_models", "test_training"]
        }
        
        def create_structure(base, structure):
            """Recursivamente cria estrutura de diretórios"""
            for name, content in structure.items():
                current_path = base / name
                current_path.mkdir(exist_ok=True)
                
                # Criar __init__.py em pastas Python
                if name in ["src", "data", "features", "models", "training", "evaluation", "utils"]:
                    (current_path / "__init__.py").touch()
                
                if isinstance(content, dict):
                    create_structure(current_path, content)
                elif isinstance(content, list):
                    for subdir in content:
                        subdir_path = current_path / subdir
                        subdir_path.mkdir(exist_ok=True)
                        # Criar __init__.py em subpastas Python se necessário
                        if current_path.name in ["src"]:
                            (subdir_path / "__init__.py").touch()
        
        create_structure(ml_platform, ml_structure)
        print(f"✅ Estrutura ML Platform criada: {ml_platform}")
        
        return ml_platform
    
    def create_api_backend_structure(self):
        """Cria estrutura do API Backend"""
        print("\n🌐 CRIANDO ESTRUTURA API BACKEND")
        print("=" * 50)
        
        api_backend = self.base_path / "api-backend"
        
        # Estrutura API Backend
        api_structure = {
            "src": {
                "core": [],
                "api": {
                    "v1": ["weather", "alerts", "predictions"],
                    "health": []
                },
                "services": [],
                "repositories": [],
                "models": [],
                "integrations": ["openmeteo", "ml_platform", "notifications"],
                "utils": []
            },
            "tests": ["unit", "integration", "e2e"],
            "alembic": [],
            "configs": []
        }
        
        def create_api_structure(base, structure):
            for name, content in structure.items():
                current_path = base / name
                current_path.mkdir(exist_ok=True)
                
                # Criar __init__.py em pastas Python
                if name == "src" or current_path.parent.name == "src":
                    (current_path / "__init__.py").touch()
                
                if isinstance(content, dict):
                    create_api_structure(current_path, content)
                elif isinstance(content, list):
                    for subdir in content:
                        subdir_path = current_path / subdir
                        subdir_path.mkdir(exist_ok=True)
                        if current_path.parent.name == "src":
                            (subdir_path / "__init__.py").touch()
        
        create_api_structure(api_backend, api_structure)
        print(f"✅ Estrutura API Backend criada: {api_backend}")
        
        return api_backend
    
    def migrate_notebooks(self, ml_platform):
        """Migra notebooks para a nova estrutura"""
        print("\n📓 MIGRANDO NOTEBOOKS")
        print("=" * 50)
        
        current_notebooks = self.base_path / "notebooks"
        if not current_notebooks.exists():
            print("⚠️ Pasta notebooks não encontrada")
            return
        
        # Mapeamento de notebooks para novos diretórios
        notebook_mapping = {
            "0_data_pipeline.ipynb": "01-data-collection/data_pipeline.ipynb",
            "1_exploratory_analysis.ipynb": "02-exploratory-analysis/exploratory_analysis.ipynb",
            "2_model_training.ipynb": "04-model-development/model_training.ipynb",
            "3_model_evaluation.ipynb": "05-model-evaluation/model_evaluation.ipynb",
            "4_model_validation.ipynb": "06-model-validation/model_validation.ipynb",
            "5_model_architecture_experiments.ipynb": "04-model-development/architecture_experiments.ipynb",
            "6_practical_examples.ipynb": "05-model-evaluation/practical_examples.ipynb",
            "7_api_integration.ipynb": "06-model-validation/api_integration.ipynb"
        }
        
        target_notebooks = ml_platform / "notebooks"
        
        for old_name, new_path in notebook_mapping.items():
            old_file = current_notebooks / old_name
            new_file = target_notebooks / new_path
            
            if old_file.exists():
                new_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old_file, new_file)
                print(f"✅ Migrado: {old_name} → {new_path}")
            else:
                print(f"⚠️ Não encontrado: {old_name}")
        
        # Copiar README e outros arquivos
        for file in current_notebooks.glob("*.md"):
            shutil.copy2(file, target_notebooks / file.name)
            print(f"✅ Copiado: {file.name}")
    
    def migrate_scripts(self, ml_platform):
        """Migra scripts para a nova estrutura organizada"""
        print("\n📜 MIGRANDO SCRIPTS")
        print("=" * 50)
        
        current_scripts = self.base_path / "scripts"
        if not current_scripts.exists():
            print("⚠️ Pasta scripts não encontrada")
            return
        
        # Mapeamento de scripts por categoria
        script_mapping = {
            "data": [
                "collect_openmeteo_hybrid_new.py",
                "analyze_openmeteo_apis.py",
                "process_existing_data.py",
                "consolidate_openmeteo_chunks.py",
                "validate_openmeteo_data.py",
                "setup_data.py",
                "data_preprocessing_fixed.py",
                "data_split.py"
            ],
            "training": [
                "train_hybrid_lstm_model.py",
                "train_hybrid_model.py",
                "train_model.py",
                "hybrid_tensorflow_trainer.py",
                "ensemble_model_trainer.py",
                "training_pipeline.py"
            ],
            "evaluation": [
                "test_model_validation.py",
                "test_forecast_domain.py",
                "inmet_exploratory_analysis.py",
                "quality_visualization_analysis.py",
                "phase_2_3_quality_analysis.py"
            ],
            "deployment": [
                "advanced_forecast_collector.py"
            ]
        }
        
        target_scripts = ml_platform / "scripts"
        
        # Migrar scripts organizadamente
        for category, script_list in script_mapping.items():
            category_path = target_scripts / category
            category_path.mkdir(exist_ok=True)
            
            for script_name in script_list:
                old_file = current_scripts / script_name
                new_file = category_path / script_name
                
                if old_file.exists():
                    shutil.copy2(old_file, new_file)
                    print(f"✅ Migrado: {script_name} → {category}/")
                else:
                    print(f"⚠️ Não encontrado: {script_name}")
        
        # Scripts não categorizados
        remaining_scripts = []
        for script in current_scripts.glob("*.py"):
            if not any(script.name in scripts for scripts in script_mapping.values()):
                remaining_scripts.append(script.name)
        
        if remaining_scripts:
            print(f"\n📋 Scripts não categorizados: {remaining_scripts}")
            misc_path = target_scripts / "misc"
            misc_path.mkdir(exist_ok=True)
            
            for script_name in remaining_scripts:
                old_file = current_scripts / script_name
                new_file = misc_path / script_name
                shutil.copy2(old_file, new_file)
                print(f"✅ Migrado: {script_name} → misc/")
    
    def migrate_data(self, ml_platform):
        """Migra dados para a ML Platform"""
        print("\n📊 MIGRANDO DADOS")
        print("=" * 50)
        
        current_data = self.base_path / "data"
        if not current_data.exists():
            print("⚠️ Pasta data não encontrada")
            return
        
        target_data = ml_platform / "data"
        
        # Mapeamento de estrutura de dados
        data_structure = ["raw", "processed", "features", "validation"]
        
        for subdir in data_structure:
            source_path = current_data / subdir
            target_path = target_data / subdir
            
            if source_path.exists():
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(source_path, target_path)
                print(f"✅ Migrado: data/{subdir}/")
            else:
                print(f"⚠️ Não encontrado: data/{subdir}/")
        
        # Migrar pasta analysis para validation
        analysis_path = current_data / "analysis"
        if analysis_path.exists():
            target_analysis = target_data / "validation" / "analysis"
            target_analysis.parent.mkdir(exist_ok=True)
            if target_analysis.exists():
                shutil.rmtree(target_analysis)
            shutil.copytree(analysis_path, target_analysis)
            print(f"✅ Migrado: data/analysis/ → data/validation/analysis/")
    
    def migrate_existing_api(self, api_backend):
        """Migra código existente da API para nova estrutura"""
        print("\n🌐 MIGRANDO CÓDIGO DA API")
        print("=" * 50)
        
        current_app = self.base_path / "app"
        if not current_app.exists():
            print("⚠️ Pasta app não encontrada")
            return
        
        target_src = api_backend / "src"
        
        # Copiar estrutura existente para base da nova arquitetura
        if (current_app / "main.py").exists():
            shutil.copy2(current_app / "main.py", api_backend / "main.py")
            print("✅ Migrado: main.py")
        
        # Migrar módulos existentes
        for item in current_app.iterdir():
            if item.is_dir() and item.name != "__pycache__":
                target_path = target_src / item.name
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(item, target_path)
                print(f"✅ Migrado: app/{item.name}/ → src/{item.name}/")
    
    def create_shared_components(self):
        """Cria componentes compartilhados"""
        print("\n🔗 CRIANDO COMPONENTES COMPARTILHADOS")
        print("=" * 50)
        
        shared = self.base_path / "shared"
        
        shared_structure = ["schemas", "utils", "configs", "types"]
        for component in shared_structure:
            (shared / component).mkdir(parents=True, exist_ok=True)
            (shared / component / "__init__.py").touch()
        
        print(f"✅ Componentes compartilhados criados: {shared}")
        
        return shared
    
    def create_infrastructure(self):
        """Cria estrutura de infraestrutura"""
        print("\n🐳 CRIANDO ESTRUTURA DE INFRAESTRUTURA")
        print("=" * 50)
        
        infra = self.base_path / "infrastructure"
        
        infra_structure = {
            "docker": ["ml-platform", "api-backend"],
            "k8s": [],
            "terraform": [],
            "monitoring": []
        }
        
        for name, subdirs in infra_structure.items():
            component_path = infra / name
            component_path.mkdir(exist_ok=True)
            
            for subdir in subdirs:
                (component_path / subdir).mkdir(exist_ok=True)
        
        # Migrar docker existente se houver
        current_docker = self.base_path / "docker"
        if current_docker.exists():
            target_docker = infra / "docker"
            for item in current_docker.iterdir():
                if item.is_file():
                    shutil.copy2(item, target_docker / item.name)
                    print(f"✅ Migrado: docker/{item.name}")
        
        print(f"✅ Infraestrutura criada: {infra}")
        
        return infra
    
    def create_config_files(self, ml_platform, api_backend):
        """Cria arquivos de configuração para ambos os componentes"""
        print("\n⚙️ CRIANDO ARQUIVOS DE CONFIGURAÇÃO")
        print("=" * 50)
        
        # ML Platform pyproject.toml
        ml_pyproject = {
            "tool": {
                "poetry": {
                    "name": "ml-platform",
                    "version": "0.1.0",
                    "description": "ML Platform para Sistema de Alertas de Cheias",
                    "dependencies": {
                        "python": "^3.9",
                        "tensorflow": "^2.13.0",
                        "pandas": "^2.0.0",
                        "numpy": "^1.24.0",
                        "scikit-learn": "^1.3.0",
                        "matplotlib": "^3.7.0",
                        "seaborn": "^0.12.0",
                        "jupyter": "^1.0.0",
                        "plotly": "^5.15.0"
                    }
                }
            }
        }
        
        # API Backend pyproject.toml
        api_pyproject = {
            "tool": {
                "poetry": {
                    "name": "api-backend",
                    "version": "0.1.0",
                    "description": "API Backend para Sistema de Alertas de Cheias",
                    "dependencies": {
                        "python": "^3.9",
                        "fastapi": "^0.100.0",
                        "uvicorn": "^0.23.0",
                        "pydantic": "^2.0.0",
                        "sqlalchemy": "^2.0.0",
                        "alembic": "^1.11.0",
                        "redis": "^4.6.0",
                        "httpx": "^0.24.0",
                        "python-multipart": "^0.0.6"
                    }
                }
            }
        }
        
        # Salvar arquivos de configuração
        import toml
        
        with open(ml_platform / "pyproject.toml", "w") as f:
            toml.dump(ml_pyproject, f)
        print("✅ ML Platform pyproject.toml criado")
        
        with open(api_backend / "pyproject.toml", "w") as f:
            toml.dump(api_pyproject, f)
        print("✅ API Backend pyproject.toml criado")
    
    def create_makefiles(self, ml_platform, api_backend):
        """Cria Makefiles para ambos os componentes"""
        print("\n🔧 CRIANDO MAKEFILES")
        print("=" * 50)
        
        # ML Platform Makefile
        ml_makefile = """# ML Platform Makefile

.PHONY: setup install clean test train evaluate

# Setup environment
setup:
	python -m venv venv
	source venv/bin/activate && pip install -e .

# Install dependencies
install:
	pip install -e .

# Clean artifacts
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ dist/ *.egg-info/

# Run tests
test:
	python -m pytest tests/ -v

# Collect data
collect-data:
	python scripts/data/collect_openmeteo.py

# Train model
train:
	python scripts/training/train_hybrid.py

# Evaluate model
evaluate:
	python scripts/evaluation/evaluate_model.py

# Run notebooks
jupyter:
	jupyter lab notebooks/
"""
        
        # API Backend Makefile
        api_makefile = """# API Backend Makefile

.PHONY: setup install clean test run dev

# Setup environment
setup:
	python -m venv venv
	source venv/bin/activate && pip install -e .

# Install dependencies
install:
	pip install -e .

# Clean artifacts
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Run tests
test:
	python -m pytest tests/ -v

# Run development server
dev:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run production server
run:
	uvicorn main:app --host 0.0.0.0 --port 8000

# Database migrations
migrate:
	alembic upgrade head
"""
        
        with open(ml_platform / "Makefile", "w") as f:
            f.write(ml_makefile)
        print("✅ ML Platform Makefile criado")
        
        with open(api_backend / "Makefile", "w") as f:
            f.write(api_makefile)
        print("✅ API Backend Makefile criado")
    
    def create_readme_files(self, ml_platform, api_backend):
        """Cria READMEs específicos para cada componente"""
        print("\n📖 CRIANDO ARQUIVOS README")
        print("=" * 50)
        
        # ML Platform README
        ml_readme = """# 🧠 ML Platform - Sistema de Alertas de Cheias

Plataforma de Machine Learning para previsão de cheias do Rio Guaíba.

## 🚀 Quick Start

```bash
# Setup
make setup
make install

# Coleta de dados
make collect-data

# Treinamento
make train

# Avaliação
make evaluate

# Notebooks
make jupyter
```

## 📁 Estrutura

- `data/`: Dados de treinamento e validação
- `notebooks/`: Análises e experimentos
- `src/`: Código fonte ML
- `scripts/`: Scripts de processamento
- `configs/`: Configurações
- `experiments/`: Experimentos ML

## 🔬 Desenvolvimento

1. Notebooks para experimentação em `notebooks/`
2. Código modular em `src/`
3. Scripts de automação em `scripts/`
4. Configurações em `configs/`
"""
        
        # API Backend README
        api_readme = """# 🌐 API Backend - Sistema de Alertas de Cheias

API Backend para sistema de alertas meteorológicos.

## 🚀 Quick Start

```bash
# Setup
make setup
make install

# Desenvolvimento
make dev

# Produção
make run

# Migrações
make migrate
```

## 📁 Estrutura

- `src/`: Código fonte da API
- `tests/`: Testes da aplicação
- `configs/`: Configurações
- `alembic/`: Migrações de banco

## 🔧 API Endpoints

- `/weather/`: Dados meteorológicos
- `/alerts/`: Sistema de alertas
- `/predictions/`: Previsões do modelo
- `/health/`: Health checks
"""
        
        with open(ml_platform / "README.md", "w") as f:
            f.write(ml_readme)
        print("✅ ML Platform README.md criado")
        
        with open(api_backend / "README.md", "w") as f:
            f.write(api_readme)
        print("✅ API Backend README.md criado")
    
    def run_migration(self):
        """Executa migração completa"""
        print(f"\n{'='*60}")
        print("🏗️ INICIANDO MIGRAÇÃO PARA NOVA ARQUITETURA")
        print(f"{'='*60}")
        
        try:
            # 1. Criar backup
            if not self.create_backup():
                print("❌ Falha no backup - Abortando migração")
                return False
            
            # 2. Criar estruturas principais
            ml_platform = self.create_ml_platform_structure()
            api_backend = self.create_api_backend_structure()
            shared = self.create_shared_components()
            infra = self.create_infrastructure()
            
            # 3. Migrar conteúdo existente
            self.migrate_notebooks(ml_platform)
            self.migrate_scripts(ml_platform)
            self.migrate_data(ml_platform)
            self.migrate_existing_api(api_backend)
            
            # 4. Criar arquivos de configuração
            self.create_config_files(ml_platform, api_backend)
            self.create_makefiles(ml_platform, api_backend)
            self.create_readme_files(ml_platform, api_backend)
            
            print(f"\n{'='*60}")
            print("✅ MIGRAÇÃO CONCLUÍDA COM SUCESSO!")
            print(f"{'='*60}")
            print(f"📁 Backup salvo em: {self.backup_path}")
            print(f"🧠 ML Platform: {ml_platform}")
            print(f"🌐 API Backend: {api_backend}")
            print(f"🔗 Shared: {shared}")
            print(f"🐳 Infrastructure: {infra}")
            
            print(f"\n🔧 Próximos passos:")
            print(f"1. cd ml-platform && make setup")
            print(f"2. cd api-backend && make setup")
            print(f"3. Revisar e ajustar imports nos arquivos migrados")
            print(f"4. Testar funcionamento de ambos os componentes")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro durante migração: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Função principal"""
    migrator = ArchitectureMigrator()
    
    # Confirmar migração
    response = input(f"\n⚠️ Esta operação irá reorganizar todo o projeto. Continuar? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Migração cancelada pelo usuário")
        return
    
    success = migrator.run_migration()
    
    if success:
        print(f"\n🎉 Migração concluída! Nova arquitetura implementada.")
    else:
        print(f"\n💥 Migração falhou. Verifique o backup em: {migrator.backup_path}")

if __name__ == "__main__":
    main() 