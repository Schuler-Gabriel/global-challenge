# 📚 Notebooks - Sistema de Alertas de Cheias

Este diretório contém os notebooks Jupyter para análise de dados e treinamento de modelos do projeto.

## 📁 Estrutura de Pastas

```
notebooks/
├── python/          # 🐍 Arquivos Python (.py) - FONTE PRINCIPAL
│   ├── exploratory_analysis.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_architecture_experiments.py
│   └── model_evaluation.py
└── jupyter/         # 📓 Notebooks Jupyter (.ipynb) - GERADOS
    ├── exploratory_analysis.ipynb
    ├── data_preprocessing.ipynb
    ├── model_training.ipynb
    ├── model_architecture_experiments.ipynb
    └── model_evaluation.ipynb
```

## ⚡ Workflow de Desenvolvimento

### 🔄 Regra Principal

**SEMPRE edite os arquivos Python (.py) primeiro, NUNCA os notebooks (.ipynb) diretamente.**

### 📝 Como Editar um Notebook

1. **Editar arquivo Python**:

   ```bash
   # Abrir arquivo para edição
   code notebooks/python/exploratory_analysis.py
   # ou
   vim notebooks/python/exploratory_analysis.py
   ```

2. **Converter para Jupyter**:

   ```bash
   # Converter notebook específico
   make sync-notebook NOTEBOOK=exploratory_analysis

   # Ou converter todos os notebooks
   make notebooks-convert-all
   ```

3. **Abrir no Jupyter**:
   ```bash
   # Abrir notebook específico
   make notebook-analysis
   # ou
   jupyter notebook notebooks/jupyter/exploratory_analysis.ipynb
   ```

## 📋 Notebooks Disponíveis

### 1. 🔍 Análise Exploratória (`exploratory_analysis`)

- **Arquivo**: `python/exploratory_analysis.py`
- **Jupyter**: `jupyter/exploratory_analysis.ipynb`
- **Comando**: `make notebook-analysis`
- **Objetivo**: Análise exploratória completa dos dados INMET (2000-2025)

**Conteúdo**:

- Carregamento e estrutura dos dados
- Análise de qualidade e valores missing
- Estatísticas descritivas
- Análise temporal (padrões sazonais)
- Correlações entre variáveis
- Detecção de outliers
- Análise de eventos extremos

### 2. 🧹 Preprocessamento (`data_preprocessing`)

- **Arquivo**: `python/data_preprocessing.py`
- **Jupyter**: `jupyter/data_preprocessing.ipynb`
- **Comando**: `make notebook-preprocessing`
- **Objetivo**: Limpeza e preparação dos dados para modelagem

**Conteúdo**:

- Limpeza e normalização
- Tratamento de valores missing
- Feature engineering
- Criação de variáveis derivadas
- Divisão temporal (treino/validação/teste)
- Salvamento dos dados processados

### 3. 🧠 Treinamento LSTM (`model_training`)

- **Arquivo**: `python/model_training.py`
- **Jupyter**: `jupyter/model_training.ipynb`
- **Comando**: `make notebook-training`
- **Objetivo**: Treinamento do modelo LSTM principal

**Conteúdo**:

- Configuração de arquiteturas LSTM
- Pipeline de treinamento completo
- Callbacks (EarlyStopping, ReduceLR)
- Monitoramento com TensorBoard
- Salvamento de modelos treinados

### 4. 🔬 Experimentos de Arquitetura (`model_architecture_experiments`)

- **Arquivo**: `python/model_architecture_experiments.py`
- **Jupyter**: `jupyter/model_architecture_experiments.ipynb`
- **Comando**: `make notebook-experiments`
- **Objetivo**: Experimentos sistemáticos de hiperparâmetros

**Conteúdo**:

- Grid search automatizado
- Teste de múltiplas arquiteturas
- Comparação de performance
- Análise de trade-offs
- Seleção da melhor configuração

### 5. 📊 Avaliação de Modelo (`model_evaluation`)

- **Arquivo**: `python/model_evaluation.py`
- **Jupyter**: `jupyter/model_evaluation.ipynb`
- **Comando**: `make notebook-evaluation`
- **Objetivo**: Avaliação completa do modelo treinado

**Conteúdo**:

- Métricas de regressão (MAE, RMSE, R²)
- Métricas de classificação (Accuracy, Precision, Recall)
- Análise de erros
- Performance em eventos extremos
- Relatório final de avaliação

## 🛠️ Comandos Make Disponíveis

### Comandos de Abertura

```bash
make notebook-analysis      # Abre análise exploratória
make notebook-preprocessing # Abre preprocessamento
make notebook-training      # Abre treinamento
make notebook-experiments   # Abre experimentos
make notebook-evaluation    # Abre avaliação
```

### Comandos de Conversão

```bash
make notebooks-list         # Lista status de todos os notebooks
make notebooks-convert-all  # Converte todos os Python para Jupyter
make notebooks-convert NOTEBOOK=nome  # Converte notebook específico
make notebooks-check        # Verifica se notebooks estão atualizados
```

### Comandos de Edição

```bash
make edit-notebook NOTEBOOK=nome     # Abre Python para edição
make sync-notebook NOTEBOOK=nome     # Converte após edição
```

### Aliases Curtos

```bash
make nb-list     # = notebooks-list
make nb-convert  # = notebooks-convert-all
make nb-check    # = notebooks-check
make nb-sync     # = sync-notebook
```

## 📚 Dependências

```bash
# Instalar dependências dos notebooks
make notebooks-install

# Ou manualmente:
pip install jupyter jupytext nbformat nbconvert
```

## 🚨 Troubleshooting

### Problema: Notebook não abre no Jupyter

```bash
# Verificar formato
head -5 notebooks/jupyter/nome_arquivo.ipynb

# Deve começar com: {"cells": [
# Se não, regenerar:
make sync-notebook NOTEBOOK=nome_arquivo
```

### Problema: Erro de conversão

```bash
# Verificar sintaxe do Python
python3 -m py_compile notebooks/python/nome_arquivo.py

# Verificar marcadores de célula
grep "# %%" notebooks/python/nome_arquivo.py
```

### Problema: Jupyter não reconhece

```bash
# Tentar conversão direta
cd notebooks/python/
jupytext --to ipynb nome_arquivo.py
mv nome_arquivo.ipynb ../jupyter/
```

## ✅ Vantagens desta Metodologia

1. **🔄 Controle de Versão**: Arquivos Python são mais limpos no Git
2. **⚡ Edição Eficiente**: IDEs funcionam melhor com arquivos .py
3. **🎯 Consistência**: Formato padrão sempre mantido
4. **🤖 Automação**: Pipeline de conversão padronizado
5. **💾 Backup**: Fonte única de verdade nos arquivos Python
6. **🔍 Review**: Diffs mais claros em code reviews
7. **🚀 Performance**: Arquivos Python carregam mais rápido

## 📈 Fluxo Recomendado

1. **Desenvolvimento**: Edite `notebooks/python/*.py`
2. **Conversão**: Execute `make sync-notebook NOTEBOOK=nome`
3. **Teste**: Abra com `make notebook-nome`
4. **Iteração**: Volte ao passo 1 para ajustes
5. **Commit**: Faça commit apenas dos arquivos Python

---

**💡 Dica**: Mantenha sempre os arquivos Python como fonte da verdade. Os notebooks Jupyter são apenas uma visualização dos arquivos Python!
