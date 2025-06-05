# Fase 3.1 - Arquitetura do Modelo LSTM

## Visão Geral

Esta fase implementa a arquitetura LSTM para previsão meteorológica baseada nos dados históricos do INMET (2000-2025). O objetivo é criar um modelo com precisão > 75% para previsão de chuva 24h à frente.

## Arquivos Implementados

### Notebooks

1. **`notebooks/model_training.ipynb`**

   - Notebook principal para treinamento do modelo LSTM
   - Implementa arquitetura multivariada (16+ features)
   - Configurações de callbacks (EarlyStopping, ReduceLROnPlateau, TensorBoard)
   - Avaliação completa com métricas de sucesso

2. **`notebooks/model_architecture_experiments.ipynb`**
   - Experimentos com diferentes arquiteturas LSTM
   - Testes de 1-3 camadas com diferentes números de neurônios
   - Comparação de dropout rates (0.1-0.3)
   - Grid search automatizado

### Scripts

3. **`scripts/train_model.py`**
   - Script Python para treinamento automatizado
   - Suporte a múltiplas arquiteturas predefinidas
   - Configuração via linha de comando
   - Logging estruturado e salvamento de artefatos

### Configurações

4. **`configs/model_config_examples.json`**
   - Configurações de exemplo para diferentes casos de uso
   - Desde experimentos rápidos até configurações de produção
   - Templates para grid search

## Arquiteturas Disponíveis

### 1. Simple 1-Layer

```python
lstm_units: [64]
dropout_rate: 0.2
```

- Modelo mais simples
- Menor consumo de memória
- Treinamento rápido

### 2. Simple 2-Layers (Recomendado para desenvolvimento)

```python
lstm_units: [128, 64]
dropout_rate: 0.2
```

- Bom equilíbrio entre performance e velocidade
- Recomendado para experimentos iniciais

### 3. Simple 3-Layers (Production)

```python
lstm_units: [256, 128, 64]
dropout_rate: 0.2
```

- Configuração padrão para produção
- Maior capacidade de aprendizado
- Baseado na documentação do projeto

### 4. Heavy 2-Layers

```python
lstm_units: [256, 128]
dropout_rate: 0.3
```

- Para datasets complexos
- Maior regularização
- Requer mais recursos computacionais

### 5. Light 3-Layers

```python
lstm_units: [64, 32, 16]
dropout_rate: 0.1
```

- Para ambientes com recursos limitados
- Menor dropout para compensar capacidade reduzida

## Comandos Disponíveis

### Treinamento Básico

```bash
# Configuração padrão de produção
make train-model

# Modo experimental (epochs reduzidos)
make train-experiment
```

### Testes de Arquitetura

```bash
# Testar todas as arquiteturas
make train-architectures

# Testar diferentes sequence lengths
make train-sequence-lengths

# Testar diferentes learning rates
make train-learning-rates

# Grid search completo (demorado!)
make train-full-grid
```

### Monitoramento e Análise

```bash
# TensorBoard
make tensorboard

# Informações do modelo atual
make model-info

# Comparar experimentos
make model-compare

# Validar modelo
make validate-model
```

### Notebooks

```bash
# Notebook de treinamento
make notebook-training

# Notebook de experimentos
make notebook-experiments

# Análise exploratória
make notebook-analysis
```

## Configurações Detalhadas

### Parâmetros Principais

- **`sequence_length`**: Comprimento da sequência de entrada (12, 24, 48 horas)
- **`forecast_horizon`**: Horizonte de previsão (24 horas padrão)
- **`lstm_units`**: Lista com número de neurônios por camada LSTM
- **`dropout_rate`**: Taxa de dropout para regularização
- **`learning_rate`**: Taxa de aprendizado (0.001, 0.0001, 0.01)
- **`batch_size`**: Tamanho do batch (16, 32, 64, 128)
- **`epochs`**: Número máximo de epochs
- **`patience`**: Paciência para early stopping

### Features do INMET

O modelo utiliza até 16 variáveis meteorológicas dos dados históricos:

1. Precipitação total horária (mm)
2. Pressão atmosférica (mB)
3. Temperatura do ar (°C)
4. Temperatura do ponto de orvalho (°C)
5. Umidade relativa (%)
6. Velocidade do vento (m/s)
7. Direção do vento (graus)
8. Radiação global (Kj/m²)
9. Pressão máxima/mínima
10. Temperatura máxima/mínima
11. Umidade máxima/mínima
12. Ponto de orvalho máximo/mínimo

## Critérios de Sucesso

### Métricas Alvo

- **MAE (Mean Absolute Error)**: < 2.0 mm/h
- **RMSE (Root Mean Square Error)**: < 3.0 mm/h
- **Accuracy**: > 75% para classificação de eventos de chuva
- **Tempo de inferência**: < 100ms

### Validação

- Validação temporal (não aleatória)
- Split por períodos: 70% treino, 15% validação, 15% teste
- Preservação de sazonalidade

## Callbacks Implementados

### 1. EarlyStopping

```python
EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)
```

### 2. ReduceLROnPlateau

```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7
)
```

### 3. ModelCheckpoint

```python
ModelCheckpoint(
    filepath='data/modelos_treinados/best_model.h5',
    monitor='val_loss',
    save_best_only=True
)
```

### 4. TensorBoard

```python
TensorBoard(
    log_dir='data/modelos_treinados/tensorboard_logs',
    histogram_freq=1,
    write_graph=True
)
```

## Estrutura de Saída

Após o treinamento, os seguintes artefatos são gerados:

```
data/modelos_treinados/
├── best_model.h5                 # Modelo em formato H5
├── lstm_weather_model/           # Modelo em formato SavedModel
├── feature_scaler.pkl            # Scaler para features
├── target_scaler.pkl             # Scaler para target
├── model_metadata.json           # Metadados do modelo
├── training_history.csv          # Histórico de treinamento
├── model_architecture.png        # Diagrama da arquitetura
├── training_history.png          # Gráficos de treinamento
└── tensorboard_logs/             # Logs do TensorBoard
```

### Experimentos

```
data/modelos_treinados/experiments/
├── experiment_results.csv        # Resultados de todos os experimentos
├── experiment_summary.json       # Resumo e rankings
└── architecture_comparison.png   # Visualizações comparativas
```

## Exemplos de Uso

### 1. Treinamento Rápido para Teste

```bash
python scripts/train_model.py \
    --experiment \
    --architecture simple_2_layers \
    --epochs 10 \
    --batch-size 64
```

### 2. Treinamento de Produção

```bash
python scripts/train_model.py \
    --architecture production \
    --sequence-length 24 \
    --learning-rate 0.001 \
    --epochs 100
```

### 3. Configuração Customizada

```bash
python scripts/train_model.py \
    --config configs/model_config_examples.json \
    --architecture high_accuracy
```

### 4. Treinamento com Monitoramento

```bash
# Terminal 1: Iniciar TensorBoard
make tensorboard

# Terminal 2: Treinar modelo
make train-model
```

## Próximos Passos

Após completar a Fase 3.1:

1. **Fase 3.2**: Treinamento e Validação

   - Cross-validation temporal
   - Otimização de hiperparâmetros
   - Validação com métricas específicas

2. **Fase 3.3**: Infrastructure ML

   - Integração com aplicação FastAPI
   - Pipeline de preprocessamento
   - Versionamento de modelos

3. **Fase 4**: Feature Forecast
   - Domain layer para previsões
   - Use cases de negócio
   - APIs REST

## Troubleshooting

### Problemas Comuns

1. **OutOfMemoryError**

   - Reduzir `batch_size`
   - Usar arquitetura `light_3_layers`
   - Reduzir `sequence_length`

2. **Convergência Lenta**

   - Aumentar `learning_rate`
   - Reduzir `dropout_rate`
   - Verificar normalização dos dados

3. **Overfitting**

   - Aumentar `dropout_rate`
   - Reduzir número de camadas
   - Aumentar dados de validação

4. **Underfitting**
   - Aumentar capacidade do modelo
   - Reduzir `dropout_rate`
   - Aumentar `epochs`

### Logs e Debugging

```bash
# Ver logs de treinamento
tail -f model_training.log

# Verificar TensorBoard
tensorboard --logdir=data/modelos_treinados/tensorboard_logs

# Validar modelo salvo
make validate-model

# Limpar modelos antigos
make clean-models
```

## Contribuindo

Para contribuir com melhorias na arquitetura:

1. Teste novas configurações usando o notebook de experimentos
2. Documente resultados no formato CSV
3. Atualize configurações de exemplo se necessário
4. Mantenha compatibilidade com a interface existente

## Referências

- [TensorFlow LSTM Guide](https://www.tensorflow.org/guide/keras/rnn)
- [Time Series Forecasting with LSTMs](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/)
- [Clean Architecture for ML](https://neptune.ai/blog/ml-pipeline-architecture)
- [Documentação do Projeto](../PROJETO_DOCUMENTACAO.md)
