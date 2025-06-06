#!/usr/bin/env python3
"""
🏗️ Demo da Nova Arquitetura
Cria a estrutura de pastas da nova arquitetura para demonstração
"""

import os
from pathlib import Path
from datetime import datetime

def create_new_architecture_demo():
    """Cria estrutura da nova arquitetura para demonstração"""
    
    print("🏗️ DEMO DA NOVA ARQUITETURA")
    print("=" * 50)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    base_path = Path(".")
    demo_path = base_path / "architecture-demo"
    
    # Limpar demo anterior se existir
    if demo_path.exists():
        import shutil
        shutil.rmtree(demo_path)
    
    print(f"📁 Criando demo em: {demo_path}")
    
    # Estrutura completa da nova arquitetura
    structure = {
        "ml-platform": {
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
        },
        "api-backend": {
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
        },
        "shared": ["schemas", "utils", "configs", "types"],
        "infrastructure": {
            "docker": ["ml-platform", "api-backend"],
            "k8s": [],
            "terraform": [],
            "monitoring": []
        },
        "docs": ["architecture", "api", "ml", "deployment"]
    }
    
    def create_structure_recursive(base, struct, level=0):
        """Cria estrutura recursivamente"""
        indent = "  " * level
        
        if isinstance(struct, dict):
            for name, content in struct.items():
                current_path = base / name
                current_path.mkdir(parents=True, exist_ok=True)
                
                # Criar __init__.py em pastas Python relevantes
                if level > 0 and name in ["src", "data", "features", "models", "training", "evaluation", "utils"]:
                    (current_path / "__init__.py").touch()
                
                print(f"{indent}📁 {name}/")
                create_structure_recursive(current_path, content, level + 1)
                
        elif isinstance(struct, list):
            for item in struct:
                item_path = base / item
                item_path.mkdir(parents=True, exist_ok=True)
                
                # Criar __init__.py em algumas pastas Python
                if base.name == "src" or base.parent.name == "src":
                    (item_path / "__init__.py").touch()
                
                print(f"{indent}📁 {item}/")
    
    # Criar estrutura
    create_structure_recursive(demo_path, structure)
    
    # Criar alguns arquivos exemplo
    create_example_files(demo_path)
    
    print(f"\n✅ DEMO CRIADO COM SUCESSO!")
    print(f"📁 Localização: {demo_path}")
    print(f"\n🔍 Para explorar:")
    print(f"tree {demo_path} -I '__pycache__'")
    
    return demo_path

def create_example_files(demo_path):
    """Cria arquivos exemplo na estrutura"""
    print(f"\n📝 CRIANDO ARQUIVOS EXEMPLO")
    print("=" * 30)
    
    # ML Platform README
    ml_readme = """# 🧠 ML Platform

Plataforma independente para Machine Learning.

## Estrutura
- `data/`: Dados próximos aos notebooks
- `notebooks/`: Organizados por fase
- `src/`: Código modular
- `scripts/`: Scripts organizados
"""
    
    # API Backend README  
    api_readme = """# 🌐 API Backend

Backend independente com Clean Architecture.

## Estrutura
- `src/api/`: Endpoints REST
- `src/services/`: Lógica de negócio
- `src/repositories/`: Acesso a dados
"""
    
    # Shared README
    shared_readme = """# 🔗 Shared Components

Componentes compartilhados entre ML Platform e API Backend.

## Componentes
- `schemas/`: Schemas de dados
- `utils/`: Utilitários comuns
- `types/`: Tipos de dados
"""
    
    # Infrastructure README
    infra_readme = """# 🐳 Infrastructure

Infraestrutura e deploy.

## Componentes
- `docker/`: Containers especializados
- `k8s/`: Kubernetes manifests
- `terraform/`: Infrastructure as Code
"""
    
    # Makefile exemplo ML Platform
    ml_makefile = """# ML Platform Commands
.PHONY: setup train evaluate

setup:
\tpython -m pip install -r requirements.txt

train:
\tpython scripts/training/train_hybrid.py

evaluate:
\tpython scripts/evaluation/evaluate_model.py
"""
    
    # Makefile exemplo API Backend
    api_makefile = """# API Backend Commands
.PHONY: setup run test

setup:
\tpython -m pip install -r requirements.txt

run:
\tuvicorn main:app --reload

test:
\tpytest tests/ -v
"""
    
    # Criar arquivos
    files_to_create = [
        (demo_path / "ml-platform" / "README.md", ml_readme),
        (demo_path / "api-backend" / "README.md", api_readme),
        (demo_path / "shared" / "README.md", shared_readme),
        (demo_path / "infrastructure" / "README.md", infra_readme),
        (demo_path / "ml-platform" / "Makefile", ml_makefile),
        (demo_path / "api-backend" / "Makefile", api_makefile),
    ]
    
    for file_path, content in files_to_create:
        with open(file_path, "w") as f:
            f.write(content)
        print(f"✅ {file_path.relative_to(demo_path)}")
    
    # Criar alguns notebooks exemplo
    notebook_examples = [
        "01-data-collection/data_pipeline.ipynb",
        "02-exploratory-analysis/exploratory_analysis.ipynb", 
        "04-model-development/model_training.ipynb",
        "05-model-evaluation/model_evaluation.ipynb"
    ]
    
    notebook_content = """{
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""
    
    for notebook in notebook_examples:
        notebook_path = demo_path / "ml-platform" / "notebooks" / notebook
        with open(notebook_path, "w") as f:
            f.write(notebook_content)
        print(f"✅ notebooks/{notebook}")

def show_comparison():
    """Mostra comparação visual das arquiteturas"""
    print(f"\n📊 COMPARAÇÃO VISUAL")
    print("=" * 50)
    
    print(f"\n❌ ESTRUTURA ATUAL (Problemática):")
    print("""
notebooks/                    # Longe dos dados
├── 0_data_pipeline.ipynb
├── 1_exploratory_analysis.ipynb
└── ... (8 notebooks misturados)

scripts/                      # Todos amontoados
├── collect_openmeteo_hybrid_new.py
├── train_hybrid_lstm_model.py
└── ... (23+ scripts sem organização)

data/                        # Longe dos notebooks
├── raw/
└── processed/

app/                         # Misturado com ML
├── core/
└── main.py
""")
    
    print(f"\n✅ NOVA ESTRUTURA (Organizada):")
    print("""
ml-platform/                 # Componente ML independente
├── data/                    # Dados próximos
├── notebooks/               # Organizados por função
│   ├── 01-data-collection/
│   ├── 02-exploratory-analysis/
│   └── 04-model-development/
├── scripts/                 # Scripts organizados por função
│   ├── data/
│   ├── training/
│   └── evaluation/
└── src/                     # Código modular

api-backend/                 # Componente API independente
├── src/                     # Clean Architecture
│   ├── api/
│   ├── services/
│   └── repositories/
└── tests/                   # Testes específicos

shared/                      # Componentes compartilhados
infrastructure/              # Deploy e infraestrutura
""")

def main():
    """Função principal"""
    print("🏗️ DEMONSTRAÇÃO DA NOVA ARQUITETURA")
    print("=" * 60)
    
    # Mostrar comparação
    show_comparison()
    
    # Criar demo
    demo_path = create_new_architecture_demo()
    
    print(f"\n🎯 BENEFÍCIOS DA NOVA ESTRUTURA:")
    print(f"✅ Notebooks próximos aos dados")
    print(f"✅ Scripts organizados por função")
    print(f"✅ ML Platform independente da API")
    print(f"✅ Clean Architecture no backend")
    print(f"✅ Deploy independente por componente")
    print(f"✅ Testes organizados")
    print(f"✅ Configurações específicas")
    
    print(f"\n🔗 PRÓXIMOS PASSOS:")
    print(f"1. Explorar estrutura: tree {demo_path}")
    print(f"2. Revisar benefícios: docs/ARCHITECTURE_COMPARISON.md")
    print(f"3. Executar migração: python scripts/migrate_to_new_architecture.py")

if __name__ == "__main__":
    main() 