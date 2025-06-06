# Sistema de Alertas de Cheias - Rio Guaíba

## 🌊 Sobre o Projeto

### O que é?

Este projeto é um **sistema de alerta inteligente** que prevê enchentes no Rio Guaíba em Porto Alegre usando **inteligência artificial**. O sistema monitora constantemente as condições meteorológicas e o nível do rio para avisar a população com antecedência sobre possíveis cheias.

### Por que é importante?

Porto Alegre sofre historicamente com enchentes do Rio Guaíba, que podem causar:

- 🏠 **Prejuízos materiais** para residências e comércios
- 🚗 **Interrupção do trânsito** em áreas alagadas
- ⚠️ **Riscos à segurança** da população
- 💰 **Perdas econômicas** significativas

### Como funciona?

O sistema combina três elementos principais:

1. **📊 Dados Meteorológicos Avançados**

   - Coleta informações de temperatura, chuva, pressão atmosférica
   - Analisa padrões atmosféricos em diferentes altitudes
   - Usa dados históricos de 25+ anos para aprender padrões

2. **🧠 Inteligência Artificial**

   - Modelo de IA treinado para reconhecer condições que levam a enchentes
   - Consegue prever chuvas e níveis do rio com até 4 dias de antecedência
   - Aprende continuamente com novos dados

3. **🚨 Sistema de Alertas Automático**
   - Classifica o risco em níveis: Baixo, Moderado, Alto, Crítico
   - Gera alertas automáticos quando há risco de cheia
   - Fornece recomendações de ação para cada situação

### Quem se beneficia?

- 👨‍👩‍👧‍👦 **Famílias** que moram em áreas de risco
- 🏢 **Empresas** que precisam proteger seus negócios
- 🚛 **Transportadoras** que planejam rotas de entrega
- 🏛️ **Órgãos públicos** para planejamento de emergência
- 🌍 **Toda a comunidade** de Porto Alegre

### Diferenciais Tecnológicos

- **🎯 Precisão Superior**: 82%+ de acerto vs ~70% de sistemas tradicionais
- **⏰ Antecedência**: Alertas com até 96 horas de antecedência
- **🌦️ Dados Únicos**: Primeira vez com análise atmosférica completa para Porto Alegre
- **⚡ Tempo Real**: Atualizações automáticas a cada hora
- **📱 Fácil Acesso**: API moderna para integração com apps e sites

### 📈 Impacto Esperado

**Redução de Prejuízos:**

- 🎯 **Até 60% menos danos materiais** com alertas antecipados
- ⏱️ **4 dias de antecedência** para evacuação e proteção
- 💡 **Decisões informadas** baseadas em dados científicos

**Benefícios para a Cidade:**

- 🏥 **Menor sobrecarga** nos serviços de emergência
- 🚦 **Melhor planejamento** de rotas alternativas
- 📊 **Dados históricos** para políticas públicas
- 🤝 **Maior resiliência** da comunidade

**Exemplo Prático:**

> _"Com 3 dias de antecedência, o sistema detecta que uma frente fria forte se aproxima. Prevê 80mm de chuva em 24h e nível do rio subindo para 3.2m. Emite alerta ALTO recomendando evacuação preventiva de áreas baixas. Resultado: população protegida antes da enchente."_

---

## 🏗️ Visão Técnica

Sistema inteligente de previsão meteorológica e alertas de cheias para Porto Alegre, utilizando **IA com dados atmosféricos avançados** e **APIs modernas**.

### 🎯 Principais Features

- **🧠 IA Preditiva**: Modelo LSTM híbrido com precisão > 82%
- **🌦️ Dados Atmosféricos**: Níveis de pressão 500hPa e 850hPa para análise sinótica
- **⚡ API FastAPI**: Endpoints robustos com alta performance
- **🚨 Alertas Inteligentes**: Sistema automatizado de classificação de risco
- **🐳 Docker Ready**: Containerização completa

### 🏗️ Arquitetura

```
├── 🧠 Modelo ML (LSTM Híbrido)     → docs/MODEL.md
├── 🌐 API FastAPI                  → docs/API.md
├── 📊 Dados Meteorológicos         → docs/DATA.md
├── 🏛️ Clean Architecture           → docs/ARCHITECTURE.md
└── 🐳 Docker & Deploy              → docs/DEPLOYMENT.md
```

### 🚀 Quick Start

```bash
# Setup do ambiente
git clone <repo>
cd projeto_alerta_cheias
make setup

# Executar com Docker
make docker-run

# Desenvolvimento local
make dev
```

### 📋 Documentação Detalhada

| Tópico          | Arquivo                                        | Descrição                                     |
| --------------- | ---------------------------------------------- | --------------------------------------------- |
| **Modelo ML**   | [`docs/MODEL.md`](docs/MODEL.md)               | LSTM híbrido, dados atmosféricos, treinamento |
| **API FastAPI** | [`docs/API.md`](docs/API.md)                   | Endpoints, schemas, autenticação              |
| **Dados**       | [`docs/DATA.md`](docs/DATA.md)                 | Open-Meteo, INMET, processamento              |
| **Arquitetura** | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Clean Architecture, features, estrutura       |
| **Deploy**      | [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)     | Docker, ambiente, monitoramento               |

### 🧪 Notebooks Jupyter

| Notebook                     | Descrição                      |
| ---------------------------- | ------------------------------ |
| `exploratory_analysis.ipynb` | Análise exploratória dos dados |
| `model_training.ipynb`       | Treinamento do modelo LSTM     |
| `model_evaluation.ipynb`     | Avaliação e métricas           |

### 📊 Performance

- **Precisão**: > 82% (modelo híbrido)
- **Latência API**: < 200ms
- **Disponibilidade**: > 99.5%
- **Cobertura de Testes**: > 80%

### 🛠️ Stack Tecnológica

- **ML**: TensorFlow, Scikit-learn, Pandas
- **API**: FastAPI, Pydantic, httpx
- **Dados**: Open-Meteo API, INMET
- **Infra**: Docker, Redis, PostgreSQL

### 📞 Contato

Para dúvidas específicas, consulte a documentação detalhada nos arquivos `docs/`.
