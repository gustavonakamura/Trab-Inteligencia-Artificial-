
**README DO PROJETO:**

# Trabalho de Inteligência Artificial: Flappy Bird com Regressão Logística

## Visão Geral

Este projeto implementa uma versão do jogo Flappy Bird usando Inteligência Artificial, onde um modelo de regressão logística aprende a jogar através de imitação de uma política especialista (expert policy). O trabalho explora conceitos fundamentais de aprendizado por imitação e aprendizado supervisionado em um ambiente de jogos.

## Contexto Acadêmico

Este é um trabalho prático de Inteligência Artificial que demonstra:
- **Aprendizado por Imitação (Imitation Learning)**: Coleta de dados de um agente especialista
- **Aprendizado Supervisionado**: Treinamento de um modelo de classificação binária
- **Engenharia de Features**: Extração e normalização de características do ambiente
- **Experimentação em IA**: Comparação de diferentes configurações e parâmetros

## Estrutura do Projeto

### Arquivos Principais

#### 1. `game_env.py` - Ambiente do Jogo
- **Propósito**: Implementa o ambiente Flappy Bird como uma classe `FlappyEnv`
- **Funcionalidades**:
  - Simulação completa da física do jogo (gravidade, impulso, colisões)
  - Sistema de observação com 4 features normalizadas:
    - `y_norm`: Posição vertical normalizada (0-1)
    - `vy_norm`: Velocidade vertical normalizada (-1 a 1)
    - `dist_norm`: Distância horizontal até o próximo cano (0-1)
    - `delta_gap_norm`: Diferença entre posição do pássaro e centro do gap (-1 a 1)
  - Sistema de recompensas (+0.1 por frame vivo, +1 por cano passado, -1 por colisão)
  - Renderização visual opcional com Pygame

#### 2. `expert_policy.py` - Política Especialista
- **Propósito**: Define a estratégia heurística que serve como "professor" para o modelo
- **Lógica da Heurística**:
  - Pula se estiver muito abaixo do gap (`delta_gap_norm > 0.12`)
  - Pula se estiver próximo do cano (`dist_norm < 0.25`), abaixo do centro do gap e caindo
  - Caso contrário, não pula
- **Importância**: Gera dados de treinamento de alta qualidade para o modelo

#### 3. `collect_dataset.py` - Coleta de Dados
- **Propósito**: Executa a política especialista para gerar dataset de treinamento
- **Características**:
  - Coleta pares (observação, ação) durante múltiplos episódios
  - Suporte a injeção de ruído (`epsilon`) para aumentar diversidade dos dados
  - Configuração flexível de dificuldade (tamanho do gap entre canos)
  - Salva dados em formato CSV para processamento posterior

#### 4. `train_logreg.py` - Treinamento do Modelo
- **Propósito**: Implementa regressão logística do zero usando NumPy
- **Algoritmo**:
  - Função sigmoide para ativação
  - Perda de entropia cruzada binária (Binary Cross-Entropy)
  - Gradiente descendente para otimização
  - Normalização Z-score das features
  - Divisão treino/validação para monitoramento
- **Saída**: Modelo treinado salvo como arquivo `.npy` com pesos e parâmetros de normalização

#### 5. `run_experiments.py` - Experimentação Sistemática
- **Propósito**: Conduz experimentos extensivos com diferentes configurações
- **Funcionalidades**:
  - Grid search sobre hiperparâmetros (learning rate, número de episódios, etc.)
  - Suporte a features polinomiais (grau 1 e 2)
  - Múltiplas execuções para robustez estatística
  - Identificação automática do melhor modelo
  - Logging detalhado de resultados

#### 6. Scripts de Teste e Demonstração
- **`play_best.py`**: Executa o melhor modelo encontrado nos experimentos
- **`play_with_model.py`**: Testa um modelo específico
- **`human_play.py`**: Permite jogo manual para comparação

## Metodologia Científica

### 1. Coleta de Dados
```bash
python collect_dataset.py --episodes 50 --gap 150 --epsilon 0.1
```
- **Episódios**: 50 jogos completos
- **Gap**: 150 pixels (dificuldade moderada)
- **Epsilon**: 10% de ações aleatórias para diversidade

### 2. Engenharia de Features
As 4 features escolhidas capturam aspectos essenciais:
- **Posição**: Onde o pássaro está na tela
- **Velocidade**: Tendência de movimento (subindo/descendo)
- **Proximidade**: Urgência da próxima decisão
- **Alinhamento**: Relação espacial com o objetivo (gap)

### 3. Treinamento do Modelo
```bash
python train_logreg.py --data data.csv --lr 0.1 --epochs 50
```
- **Algoritmo**: Regressão logística com gradiente descendente
- **Features**: Normalizadas para melhor convergência
- **Validação**: 20% dos dados reservados para teste

### 4. Experimentação
```bash
python run_experiments.py
```
- **Grid Search**: Combinações sistemáticas de parâmetros
- **Métricas**: Acurácia de validação e performance no jogo
- **Seleção**: Melhor modelo baseado em critérios combinados

## Resultados e Análise

### Estrutura de Saída
```
runs/
├── best_weights.npy          # Melhor modelo encontrado
├── run_1_weights.npy         # Modelos individuais
├── run_2_weights.npy
└── ...
```

### Métricas de Avaliação
1. **Acurácia de Classificação**: Percentual de ações corretas previstas
2. **Score no Jogo**: Número de canos atravessados
3. **Sobrevivência**: Tempo de vida médio no ambiente

### Insights Esperados
- **Generalização**: Capacidade do modelo de lidar com situações não vistas
- **Robustez**: Performance consistente em diferentes configurações
- **Limitações**: Comparação entre política heurística e modelo aprendido

## Instalação e Execução

### Pré-requisitos
```bash
pip install -r requirements.txt
```

### Dependências
- **Python 3.7+**
- **NumPy**: Computação numérica e álgebra linear
- **Pandas**: Manipulação de dados estruturados
- **Pygame**: Renderização gráfica do jogo

### Fluxo de Execução Completo

1. **Coleta de Dados**:
```bash
python collect_dataset.py --episodes 100 --gap 150
```

2. **Treinamento Simples**:
```bash
python train_logreg.py --data data.csv --epochs 50
```

3. **Experimentação Avançada**:
```bash
python run_experiments.py
```

4. **Teste do Melhor Modelo**:
```bash
python play_best.py --episodes 5
```

5. **Comparação com Expert**:
```bash
python human_play.py  # Para controle manual
```

## Conceitos de IA Demonstrados

### 1. Aprendizado por Imitação
- **Definição**: Aprender comportamentos observando um especialista
- **Vantagem**: Não requer definição manual de função de recompensa
- **Aplicação**: Política heurística gera dados de treinamento

### 2. Classificação Binária
- **Problema**: Decidir entre duas ações (pular ou não pular)
- **Solução**: Regressão logística com função sigmoide
- **Interpretação**: Probabilidade de cada ação

### 3. Representação de Estado
- **Desafio**: Capturar informação relevante do ambiente
- **Solução**: Features engenheiradas que capturam física e geometria do jogo
- **Normalização**: Garantir escala similar entre diferentes variáveis

### 4. Overfitting vs. Generalização
- **Problema**: Modelo pode memorizar dados de treino
- **Solução**: Validação cruzada e regularização implícita
- **Monitoramento**: Comparação entre acurácia de treino e validação

## Extensões Possíveis

### 1. Algoritmos Alternativos
- **Redes Neurais**: MLPs para capturar padrões não-lineares
- **Árvores de Decisão**: Interpretabilidade das regras aprendidas
- **SVM**: Classificação com margens máximas

### 2. Features Avançadas
- **Histórico**: Sequência de observações passadas
- **Features Visuais**: Pixels da tela como entrada
- **Engenharia Automática**: Seleção automática de features

### 3. Aprendizado por Reforço
- **Q-Learning**: Aprender diretamente das recompensas do ambiente
- **Policy Gradient**: Otimização direta da política
- **Comparação**: RL vs. Imitation Learning

### 4. Avaliação Robusta
- **Cross-Validation**: Múltiplas divisões treino/teste
- **Diferentes Dificuldades**: Gaps de tamanhos variados
- **Métricas Adicionais**: Tempo de reação, suavidade da política

## Conclusões Esperadas

Este projeto demonstra como técnicas básicas de Machine Learning podem ser aplicadas em ambientes interativos. A regressão logística, apesar de simples, é capaz de capturar padrões comportamentais complexos quando fornecida com features bem engenheiradas.

### Pontos-Chave para o Artigo:
1. **Simplicidade vs. Efetividade**: Modelos simples podem ter boa performance
2. **Importância das Features**: Representação adequada é crucial
3. **Aprendizado por Imitação**: Alternativa eficiente ao aprendizado por reforço
4. **Experimentação Sistemática**: Metodologia científica em IA aplicada
5. **Reprodutibilidade**: Código estruturado e documentado para replicação

## Referências Técnicas

- **Aprendizado por Imitação**: Russell, S. & Norvig, P. "Artificial Intelligence: A Modern Approach"
- **Regressão Logística**: James, G. et al. "An Introduction to Statistical Learning"
- **Engenharia de Features**: Zheng, A. & Casari, A. "Feature Engineering for Machine Learning"
- **Ambientes de Jogo para IA**: Bellemare, M. et al. "The Arcade Learning Environment"

---

**Autores**: [Incluir nomes dos desenvolvedores]  
**Disciplina**: Inteligência Artificial  
**Instituição**: [Nome da universidade/curso]  
**Data**: Setembro 2025

---

**INSTRUÇÕES FINAIS:**
- Mantenha rigor científico em todas as seções
- Use citações apropriadas (mesmo que hipotéticas)
- Inclua análise crítica das limitações
- Destaque as contribuições originais
- Sugira direções futuras de pesquisa
- Assegure-se de que o artigo seja informativo e bem estruturado
