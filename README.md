# bandits e decisões sequenciais

este repositório organiza as simulações do trabalho de bandits e decisões sequenciais.
a ideia é manter a implementação modular, reproduzível e separada dos notebooks de análise.

## partes do trabalho

### parte 1: inferência após seleção

comparar políticas de alocação em um bandit gaussiano com 5 braços, sob o caso nulo em que todas as médias verdadeiras são iguais a zero. ao final de cada experimento, seleciona-se o braço com maior média amostral e avalia-se a cobertura empírica de um intervalo de confiança ingênuo construído após a seleção.

políticas implementadas:

- alocação uniforme aleatória;
- greedy;
- ucb;
- thompson sampling gaussiano.

### parte 2: recompensa acumulada em um exemplo poisson

comparar políticas de alocação em um bandit poisson com 5 campanhas de divulgação. a recompensa diária é o número de novos cadastros obtidos pela campanha escolhida no dia.

modelo usado:

```text
R_t | a_t = a ~ Poisson(lambda_a)
lambda = (18, 20, 21, 23, 27)
lambda_a ~ Gamma(alpha_0, beta_0)
```

neste repositório, a gamma é parametrizada por taxa. os valores padrão são `alpha_0 = 4` e `beta_0 = 0.2`, o que gera média prévia `20` e desvio padrão prévio `10`. essa escolha é fraca o suficiente para permitir aprendizado pelos dados, mas mantém a escala compatível com o número esperado de cadastros por dia.

políticas implementadas:

- alocação uniforme aleatória;
- greedy;
- thompson sampling poisson-gamma;
- ucb bayesiano poisson-gamma;
- epsilon-greedy.

## estrutura

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
```

## instalação

```bash
poetry install
poetry run python -m ipykernel install --user --name bandit-inference --display-name "bandit-inference"
```

## validação rápida

```bash
poetry run pytest
poetry run python scripts/run_simulation.py --n-replications 100 --horizon 200
poetry run python scripts/run_poisson_simulation.py --n-replications 100 --horizon 100
```

## execução principal da parte 1

```bash
poetry run python scripts/run_simulation.py --n-replications 5000 --horizon 1000 --seed 12345
```

saídas principais:

```text
data/processed/simulation_results.csv
reports/tables/summary_by_policy.csv
reports/figures/coverage_by_policy.png
reports/figures/selected_mean_distribution.png
reports/figures/allocation_distribution.png
reports/figures/selected_arm_frequency.png
reports/figures/selected_n_vs_mean.png
```

## execução principal da parte 2

```bash
poetry run python scripts/run_poisson_simulation.py \
  --n-replications 2000 \
  --horizon 365 \
  --seed 12345 \
  --prior-alpha 4 \
  --prior-beta 0.2 \
  --ucb-exploration 2 \
  --epsilon 0.1
```

saídas principais:

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

## notebooks

use os notebooks apenas para análise, visualização e redação interpretativa. a implementação das políticas fica em `src/`.

```text
00_validacao_rapida.ipynb
01_simulacao_e_visualizacoes.ipynb
02_poisson_recompensa_acumulada.ipynb
```

## observação sobre versionamento

por padrão, arquivos gerados em `data/processed/`, `reports/figures/` e `reports/tables/` ficam fora do git. quando as figuras finais forem escolhidas para o relatório, elas podem ser adicionadas explicitamente.
