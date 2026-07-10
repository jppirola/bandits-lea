# Bandits e decisões sequenciais

Este repositório organiza as simulações do trabalho de bandits e decisões sequenciais. A implementação foi separada dos notebooks de análise para manter o projeto modular, reproduzível e fácil de verificar.

O trabalho possui duas partes:

1. inferência após seleção em um experimento adaptativo com recompensas Gaussianas;
2. comparação de políticas em um bandit Poisson com recompensas de contagem.

## Partes do trabalho

### Parte 1: inferência após seleção

A primeira parte compara políticas de alocação em um bandit Gaussiano com 5 braços, sob o caso nulo em que todas as médias verdadeiras são iguais a zero.

O modelo usado é

```text
R_t | a_t = a ~ N(mu_a, sigma^2)
mu = (0, 0, 0, 0, 0)
sigma = 1
```

Ao final de cada experimento, calcula-se a média amostral de cada braço, seleciona-se o braço com maior média amostral e avalia-se a cobertura empírica de um intervalo de confiança ingênuo construído após a seleção.

Políticas implementadas:

- alocação uniforme aleatória;
- greedy;
- UCB;
- Thompson Sampling Gaussiano.

### Parte 2: recompensa acumulada em um exemplo Poisson

A segunda parte compara políticas de alocação em um bandit Poisson com 5 campanhas de divulgação. A recompensa diária é o número de novos cadastros obtidos pela campanha escolhida no dia.

O modelo usado é

```text
R_t | a_t = a ~ Poisson(lambda_a)
lambda = (18, 20, 21, 23, 27)
lambda_a ~ Gamma(alpha_0, beta_0)
```

Neste repositório, a distribuição Gamma é parametrizada por taxa. Os valores padrão são `alpha_0 = 4` e `beta_0 = 0.2`, o que gera média prévia igual a `20` e desvio padrão prévio igual a `10`.

Essa escolha mantém a prior na escala esperada do número de cadastros por dia, mas ainda permite incerteza suficiente para que os dados atualizem as taxas ao longo do experimento.

Políticas implementadas:

- alocação uniforme aleatória;
- greedy;
- Thompson Sampling Poisson-Gamma;
- UCB Bayesiano Poisson-Gamma;
- epsilon-greedy.

## Estrutura

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 00_validacao_rapida.ipynb
│   ├── 01_simulacao_e_visualizacoes.ipynb
│   └── 02_poisson_recompensa_acumulada.ipynb
├── reports/
│   ├── figures/
│   └── tables/
├── scripts/
│   ├── run_simulation.py
│   └── run_poisson_simulation.py
├── src/
│   └── bandit_inference/
│       ├── analysis.py
│       ├── analysis_poisson.py
│       ├── environment.py
│       ├── poisson_environment.py
│       ├── simulation.py
│       ├── simulation_poisson.py
│       ├── visualization.py
│       ├── visualization_poisson.py
│       └── policies/
│           ├── base.py
│           ├── epsilon_greedy.py
│           ├── greedy.py
│           ├── poisson_thompson.py
│           ├── poisson_ucb.py
│           ├── random_policy.py
│           ├── thompson.py
│           └── ucb.py
└── tests/
    ├── test_simulation.py
    └── test_poisson_simulation.py
```

## Instalação

O projeto usa `Poetry` para gerenciamento de dependências.

Na raiz do repositório, execute:

```bash
poetry install
```

Para usar os notebooks no VS Code ou no Jupyter, registre o kernel:

```bash
poetry run python -m ipykernel install --user --name bandit-inference --display-name "bandit-inference"
```

## Validação rápida

Antes de rodar os experimentos completos, é possível validar o projeto com testes e simulações pequenas:

```bash
poetry run pytest
poetry run python scripts/run_simulation.py --n-replications 100 --horizon 200
poetry run python scripts/run_poisson_simulation.py --n-replications 100 --horizon 100
```

O resultado esperado dos testes é:

```text
4 passed
```

## Execução principal da Parte 1

O comando abaixo reproduz a simulação usada no relatório para a Parte 1:

```bash
poetry run python scripts/run_simulation.py --n-replications 2000 --horizon 1000 --seed 12345
```

Saídas principais:

```text
data/processed/simulation_results.csv
reports/tables/summary_by_policy.csv
reports/figures/coverage_by_policy.png
reports/figures/selected_mean_distribution.png
reports/figures/allocation_distribution.png
reports/figures/selected_arm_frequency.png
reports/figures/selected_n_vs_mean.png
```

## Execução principal da Parte 2

O comando abaixo reproduz a simulação usada no relatório para a Parte 2:

```bash
poetry run python scripts/run_poisson_simulation.py \
  --n-replications 1000 \
  --horizon 365 \
  --seed 12345 \
  --prior-alpha 4 \
  --prior-beta 0.2 \
  --ucb-exploration 2 \
  --epsilon 0.1
```

Saídas principais:

```text
data/processed/poisson_results.csv
data/processed/poisson_trajectories.csv
reports/tables/poisson_summary_by_policy.csv
reports/figures/poisson_cumulative_reward.png
reports/figures/poisson_total_reward_boxplot.png
reports/figures/poisson_optimal_arm_proportion.png
reports/figures/poisson_allocation_distribution.png
reports/figures/poisson_lambda_estimates.png
```

## Notebooks

Os notebooks são usados para validação, análise, visualização e interpretação dos resultados. A implementação das políticas e das simulações fica em `src/`.

```text
00_validacao_rapida.ipynb
01_simulacao_e_visualizacoes.ipynb
02_poisson_recompensa_acumulada.ipynb
```

## Reprodutibilidade

As simulações usam sementes aleatórias fixadas por argumento de linha de comando. Portanto, os resultados principais do relatório podem ser reproduzidos executando os comandos das seções "Execução principal da Parte 1" e "Execução principal da Parte 2".

A estrutura do projeto segue o padrão `src/`, com testes em `tests/`, scripts de execução em `scripts/` e notebooks separados da implementação principal.

## Observação sobre os resultados

Os arquivos em `reports/figures/` e `reports/tables/` são gerados automaticamente pelos scripts e notebooks. Eles foram mantidos no repositório para facilitar a conferência dos resultados apresentados no relatório.

Os arquivos em `data/processed/` armazenam os resultados das simulações em formato tabular, permitindo reproduzir as tabelas e gráficos sem precisar rodar novamente todos os experimentos.