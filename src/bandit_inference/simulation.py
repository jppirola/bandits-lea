"""rotinas de simulação para inferência após seleção."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm.auto import tqdm

from bandit_inference.environment import GaussianBanditEnvironment
from bandit_inference.policies import (
    GaussianThompsonSamplingPolicy,
    GreedyPolicy,
    UCBPolicy,
    UniformRandomPolicy,
)
from bandit_inference.policies.base import BanditPolicy

PolicyFactory = Callable[[np.random.Generator], BanditPolicy]


@dataclass
class ExperimentConfig:
    """guardar parâmetros principais da simulação."""

    n_arms: int = 5
    horizon: int = 1000
    n_replications: int = 2000
    sigma: float = 1.0
    seed: int = 6
    true_means: NDArray[np.float64] = field(default_factory=lambda: np.zeros(5))
    ci_z_value: float = 1.96
    ucb_exploration: float = 2.0
    thompson_prior_mean: float = 0.0
    thompson_prior_var: float = 10.0

    def __post_init__(self) -> None:
        # converter médias verdadeiras para vetor numérico
        self.true_means = np.asarray(self.true_means, dtype=float)

        # validar consistência entre número de braços e médias
        if self.true_means.size != self.n_arms:
            msg = "o número de médias verdadeiras deve ser igual ao número de braços"
            raise ValueError(msg)

        # validar horizonte mínimo para estimar todos os braços
        if self.horizon < self.n_arms:
            msg = "o horizonte deve ser pelo menos igual ao número de braços"
            raise ValueError(msg)


def build_default_policy_factories(config: ExperimentConfig) -> dict[str, PolicyFactory]:
    """criar fábricas das políticas comparadas no exercício."""
    return {
        "aleatória uniforme": lambda rng: UniformRandomPolicy(n_arms=config.n_arms, rng=rng),
        "greedy": lambda rng: GreedyPolicy(n_arms=config.n_arms, rng=rng),
        "ucb": lambda rng: UCBPolicy(
            n_arms=config.n_arms,
            exploration=config.ucb_exploration,
            rng=rng,
        ),
        "thompson sampling": lambda rng: GaussianThompsonSamplingPolicy(
            n_arms=config.n_arms,
            reward_sigma=config.sigma,
            prior_mean=config.thompson_prior_mean,
            prior_var=config.thompson_prior_var,
            rng=rng,
        ),
    }


def _spawn_rngs(seed_sequence: np.random.SeedSequence, n_children: int) -> list[np.random.Generator]:
    """gerar fluxos independentes de aleatoriedade."""
    child_sequences = seed_sequence.spawn(n_children)
    return [np.random.default_rng(child_sequence) for child_sequence in child_sequences]


def run_single_experiment(
    config: ExperimentConfig,
    policy: BanditPolicy,
    replication: int,
    policy_name: str,
    env_rng: np.random.Generator,
) -> dict[str, float | int | str]:
    """executar uma replicação para uma política específica."""
    # inicializar ambiente e estatísticas suficientes
    environment = GaussianBanditEnvironment(
        means=config.true_means,
        sigma=config.sigma,
        rng=env_rng,
    )
    policy.reset()
    counts = np.zeros(config.n_arms, dtype=int)
    reward_sums = np.zeros(config.n_arms, dtype=float)
    total_reward = 0.0

    # executar decisões sequenciais
    for t in range(1, config.horizon + 1):
        arm = policy.select_arm(t=t, counts=counts, reward_sums=reward_sums)
        reward = environment.pull(arm)

        # atualizar estatísticas observadas
        counts[arm] += 1
        reward_sums[arm] += reward
        total_reward += reward
        policy.update(arm=arm, reward=reward)

    # calcular médias amostrais finais por braço
    sample_means = reward_sums / counts

    # selecionar o braço com maior média amostral final
    selected_arm = int(np.argmax(sample_means))
    selected_n = int(counts[selected_arm])
    selected_mean = float(sample_means[selected_arm])
    selected_true_mean = float(config.true_means[selected_arm])

    # construir intervalo ingênuo tratando o braço selecionado como fixo
    half_width = config.ci_z_value * config.sigma / np.sqrt(selected_n)
    ci_lower = selected_mean - half_width
    ci_upper = selected_mean + half_width
    covered = ci_lower <= selected_true_mean <= ci_upper

    # organizar resultados agregados da replicação
    row: dict[str, float | int | str] = {
        "replication": replication,
        "policy": policy_name,
        "selected_arm": selected_arm + 1,
        "selected_n": selected_n,
        "selected_mean": selected_mean,
        "selected_true_mean": selected_true_mean,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "ci_half_width": float(half_width),
        "covered": int(covered),
        "total_reward": float(total_reward),
        "average_reward": float(total_reward / config.horizon),
    }

    # adicionar médias e alocações finais por braço
    for arm_idx in range(config.n_arms):
        arm_label = arm_idx + 1
        row[f"n_arm_{arm_label}"] = int(counts[arm_idx])
        row[f"prop_arm_{arm_label}"] = float(counts[arm_idx] / config.horizon)
        row[f"mean_arm_{arm_label}"] = float(sample_means[arm_idx])

    return row


def run_many_experiments(
    config: ExperimentConfig,
    policy_factories: dict[str, PolicyFactory] | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """executar muitas replicações para todas as políticas."""
    if policy_factories is None:
        policy_factories = build_default_policy_factories(config)

    # criar sequência mestre para reprodutibilidade
    seed_sequence = np.random.SeedSequence(config.seed)
    total_runs = config.n_replications * len(policy_factories)
    run_sequences = seed_sequence.spawn(total_runs)

    rows: list[dict[str, float | int | str]] = []
    iterator = range(config.n_replications)
    if progress:
        iterator = tqdm(iterator, desc="simular replicações")

    run_index = 0
    for replication in iterator:
        for policy_name, policy_factory in policy_factories.items():
            # criar geradores independentes para ambiente e política
            env_rng, policy_rng = _spawn_rngs(run_sequences[run_index], n_children=2)
            policy = policy_factory(policy_rng)

            row = run_single_experiment(
                config=config,
                policy=policy,
                replication=replication,
                policy_name=policy_name,
                env_rng=env_rng,
            )
            rows.append(row)
            run_index += 1

    return pd.DataFrame(rows)
