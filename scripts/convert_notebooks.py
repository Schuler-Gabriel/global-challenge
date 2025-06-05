#!/usr/bin/env python3
"""
Script para conversão automática de notebooks Python para Jupyter
Uso: python scripts/convert_notebooks.py [nome_arquivo]
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def convert_notebook(python_file: str, force: bool = False):
    """
    Converte arquivo Python para notebook Jupyter
    
    Args:
        python_file: Nome do arquivo Python (sem extensão)
        force: Se True, sobrescreve notebook existente
    """
    
    # Definir caminhos
    project_root = Path(__file__).parent.parent
    python_dir = project_root / 'notebooks' / 'python'
    jupyter_dir = project_root / 'notebooks' / 'jupyter'
    
    python_path = python_dir / f"{python_file}.py"
    jupyter_path = jupyter_dir / f"{python_file}.ipynb"
    
    # Verificar se arquivo Python existe
    if not python_path.exists():
        print(f"❌ Arquivo não encontrado: {python_path}")
        return False
    
    # Verificar se notebook já existe
    if jupyter_path.exists() and not force:
        response = input(f"📝 Notebook {jupyter_path.name} já existe. Sobrescrever? (y/n): ")
        if response.lower() != 'y':
            print("Operação cancelada.")
            return False
    
    try:
        # Deletar notebook existente se houver
        if jupyter_path.exists():
            jupyter_path.unlink()
            print(f"🗑️ Notebook antigo deletado: {jupyter_path.name}")
        
        # Converter Python para Jupyter
        print(f"🔄 Convertendo {python_path.name} para notebook...")
        
        result = subprocess.run([
            'jupytext', '--to', 'notebook', str(python_path)
        ], cwd=python_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Erro na conversão: {result.stderr}")
            return False
        
        # Mover notebook para pasta jupyter
        temp_notebook = python_dir / f"{python_file}.ipynb"
        if temp_notebook.exists():
            temp_notebook.rename(jupyter_path)
            print(f"✅ Notebook criado: {jupyter_path}")
            return True
        else:
            print(f"❌ Notebook não foi gerado: {temp_notebook}")
            return False
            
    except Exception as e:
        print(f"❌ Erro durante conversão: {e}")
        return False

def convert_all_notebooks(force: bool = False):
    """
    Converte todos os arquivos Python para notebooks
    """
    
    project_root = Path(__file__).parent.parent
    python_dir = project_root / 'notebooks' / 'python'
    
    # Encontrar todos os arquivos Python
    python_files = list(python_dir.glob("*.py"))
    
    if not python_files:
        print("❌ Nenhum arquivo Python encontrado em notebooks/python/")
        return
    
    print(f"📁 Encontrados {len(python_files)} arquivos Python")
    
    success_count = 0
    
    for python_file in python_files:
        filename = python_file.stem  # Nome sem extensão
        print(f"\n{'='*50}")
        print(f"Processando: {filename}")
        
        if convert_notebook(filename, force):
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"✅ Conversão concluída: {success_count}/{len(python_files)} notebooks")

def list_notebooks():
    """
    Lista todos os notebooks disponíveis
    """
    
    project_root = Path(__file__).parent.parent
    python_dir = project_root / 'notebooks' / 'python'
    jupyter_dir = project_root / 'notebooks' / 'jupyter'
    
    print("📚 NOTEBOOKS DISPONÍVEIS:")
    print(f"{'='*60}")
    
    python_files = list(python_dir.glob("*.py"))
    
    for python_file in sorted(python_files):
        filename = python_file.stem
        jupyter_file = jupyter_dir / f"{filename}.ipynb"
        
        python_size = python_file.stat().st_size
        python_modified = python_file.stat().st_mtime
        
        status = "✅" if jupyter_file.exists() else "❌"
        
        print(f"{status} {filename}")
        print(f"   Python: {python_size:,} bytes")
        
        if jupyter_file.exists():
            jupyter_size = jupyter_file.stat().st_size
            jupyter_modified = jupyter_file.stat().st_mtime
            
            print(f"   Jupyter: {jupyter_size:,} bytes")
            
            if python_modified > jupyter_modified:
                print(f"   ⚠️ Python mais recente que Jupyter - recomenda conversão")
        else:
            print(f"   Jupyter: não existe - necessária conversão")
        
        print()

def main():
    parser = argparse.ArgumentParser(
        description="Conversor de notebooks Python para Jupyter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python scripts/convert_notebooks.py --all               # Converter todos
  python scripts/convert_notebooks.py model_training      # Converter específico
  python scripts/convert_notebooks.py --list              # Listar notebooks
  python scripts/convert_notebooks.py --all --force       # Converter todos (forçar)
        """
    )
    
    parser.add_argument(
        'notebook',
        nargs='?',
        help='Nome do notebook a converter (sem extensão)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Converter todos os notebooks'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='Listar todos os notebooks disponíveis'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Sobrescrever notebooks existentes sem perguntar'
    )
    
    args = parser.parse_args()
    
    # Verificar se jupytext está instalado
    try:
        subprocess.run(['jupytext', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ jupytext não está instalado!")
        print("Instale com: pip install jupytext")
        sys.exit(1)
    
    if args.list:
        list_notebooks()
    elif args.all:
        convert_all_notebooks(args.force)
    elif args.notebook:
        convert_notebook(args.notebook, args.force)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 