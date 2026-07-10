"""rotinas de simulação para recompensa acumulada poisson."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm.auto import tqdm

from bandit_inference.poisson_environment import PoissonBanditEnvironment
from bandit_inference.policies import (
    EpsilonGreedyPolicy,
    GreedyPolicy,
    PoissonThompsonSamplingPolicy,
    PoissonUCBPolicy,
    UniformRandomPolicy,
)
from bandit_inference.policies.base import BanditPolicy

PoissonPolicyFactory = Callable[[np.random.Generator], BanditPolicy]


@dataclass
class PoissonExperimentConfig:
    """guardar parâmetros da simulação poisson."""

    n_arms: int = 5
    horizon: int = 365
    n_replications: int = 1000
    lambdas: NDArray[np.float64] = field(
        default_factory=lambda: np.array([18, 20, 21, 23, 27], dtype=float),
    )
    seed: int = 6
    prior_alpha: float = 4.0
    prior_beta: float = 0.2
    ucb_exploration: float = 2.0
    epsilon: float = 0.1
    trajectory_stride: int = 1

    def __post_init__(self) -> None:
        # converter taxas verdadeiras para vetor numérico
        self.lambdas = np.asarray(self.lambdas, dtype=float)

        # validar consistência entre braços e taxas
        if self.lambdas.size != self.n_arms:
            msg = "o número de taxas deve ser igual ao número de braços"
            raise ValueError(msg)

        # validar horizonte e hiperparâmetros
        if self.horizon < self.n_arms:
            msg = "o horizonte deve ser pelo menos igual ao número de braços"
            raise ValueError(msg)
        if self.prior_alpha <= 0 or self.prior_beta <= 0:
            msg = "os hiperparâmetros da gamma devem ser positivos"
            raise ValueError(msg)
        if self.trajectory_stride <= 0:
            msg = "trajectory_stride deve ser positivo"
            raise ValueError(msg)

    @property
    def optimal_arm(self) -> int:
        """retornar índice zero-based do braço ótimo."""
        return int(np.argmax(self.lambdas))


def build_poisson_policy_factories(
    config: PoissonExperimentConfig,
) -> dict[str, PoissonPolicyFactory]:
    """criar fábricas das políticas comparadas na parte 2."""
    return {
        "aleatória uniforme": lambda rng: UniformRandomPolicy(n_arms=config.n_arms, rng=rng),
        "greedy": lambda rng: GreedyPolicy(n_arms=config.n_arms, rng=rng),
        "thompson sampling": lambda rng: PoissonThompsonSamplingPolicy(
            n_arms=config.n_arms,
            prior_alpha=config.prior_alpha,
            prior_beta=config.prior_beta,
            rng=rng,
        ),
        "ucb": lambda rng: PoissonUCBPolicy(
            n_arms=config.n_arms,
            prior_alpha=config.prior_alpha,
            prior_beta=config.prior_beta,
            exploration=config.ucb_exploration,
            rng=rng,
        ),
        "epsilon-greedy": lambda rng: EpsilonGreedyPolicy(
            n_arms=config.n_arms,
            epsilon=config.epsilon,
            rng=rng,
        ),
    }


def _spawn_rngs(seed_sequence: np.random.SeedSequence, n_children: int) -> list[np.random.Generator]:
    """gerar fluxos independentes de aleatoriedade."""
    child_sequences = seed_sequence.spawn(n_children)
    return [np.random.default_rng(child_sequence) for child_sequence in child_sequences]


def run_single_poisson_experiment(
    config: PoissonExperimentConfig,
    policy: BanditPolicy,
    replication: int,
    policy_name: str,
    env_rng: np.random.Generator,
    save_trajectory: bool = True,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    """executar uma replicação poisson para uma política."""
    # inicializar ambiente e estatísticas suficientes
    environment = PoissonBanditEnvironment(lambdas=config.lambdas, rng=env_rng)
    policy.reset()
    counts = np.zeros(config.n_arms, dtype=int)
    reward_sums = np.zeros(config.n_arms, dtype=float)
    cumulative_reward = 0.0
    trajectory_rows: list[dict[str, float | int | str]] = []
    optimal_arm = config.optimal_arm

    # executar decisões sequenciais
    for t in range(1, config.horizon + 1):
        arm = policy.select_arm(t=t, counts=counts, reward_sums=reward_sums)
        reward = environment.pull(arm)

        # atualizar estatísticas observadas
        counts[arm] += 1
        reward_sums[arm] += reward
        cumulative_reward += reward
        policy.update(arm=arm, reward=reward)

        # guardar trajetória para gráficos de recompensa acumulada
        should_record = t == 1 or t == config.horizon or t % config.trajectory_stride == 0
        if save_trajectory and should_record:
            trajectory_rows.append(
                {
                    "replication": replication,
                    "policy": policy_name,
                    "time": t,
                    "chosen_arm": arm + 1,
                    "reward": reward,
                    "cumulative_reward": float(cumulative_reward),
                    "average_reward": float(cumulative_reward / t),
                    "optimal_arm_pulls": int(counts[optimal_arm]),
                    "optimal_arm_proportion": float(counts[optimal_arm] / t),
                },
            )

    # calcular estimativas finais de lambda por braço
    lambda_hats = reward_sums / counts
    estimation_errors = lambda_hats - config.lambdas
    abs_errors = np.abs(estimation_errors)

    # organizar resultados agregados da replicação
    final_row: dict[str, float | int | str] = {
        "replication": replication,
        "policy": policy_name,
        "total_reward": float(cumulative_reward),
        "average_reward": float(cumulative_reward / config.horizon),
        "optimal_arm": optimal_arm + 1,
        "optimal_lambda": float(config.lambdas[optimal_arm]),
        "optimal_arm_pulls": int(counts[optimal_arm]),
        "optimal_arm_proportion": float(counts[optimal_arm] / config.horizon),
        "lambda_rmse": float(np.sqrt(np.mean(estimation_errors**2))),
        "lambda_mae": float(np.mean(abs_errors)),
    }

    # adicionar alocações e estimativas por campanha
    for arm_idx in range(config.n_arms):
        arm_label = arm_idx + 1
        final_row[f"n_arm_{arm_label}"] = int(counts[arm_idx])
        final_row[f"prop_arm_{arm_label}"] = float(counts[arm_idx] / config.horizon)
        final_row[f"lambda_hat_arm_{arm_label}"] = float(lambda_hats[arm_idx])
        final_row[f"lambda_error_arm_{arm_label}"] = float(estimation_errors[arm_idx])
        final_row[f"lambda_abs_error_arm_{arm_label}"] = float(abs_errors[arm_idx])

    return final_row, trajectory_rows


def run_many_poisson_experiments(
    config: PoissonExperimentConfig,
    policy_factories: dict[str, PoissonPolicyFactory] | None = None,
    progress: bool = True,
    save_trajectories: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """executar muitas replicações para todas as políticas poisson."""
    if policy_factories is None:
        policy_factories = build_poisson_policy_factories(config)

    # criar sequência mestre para reprodutibilidade
    seed_sequence = np.random.SeedSequence(config.seed)
    total_runs = config.n_replications * len(policy_factories)
    run_sequences = seed_sequence.spawn(total_runs)

    final_rows: list[dict[str, float | int | str]] = []
    trajectory_rows: list[dict[str, float | int | str]] = []
    iterator = range(config.n_replications)
    if progress:
        iterator = tqdm(iterator, desc="simular replicações poisson")

    run_index = 0
    for replication in iterator:
        for policy_name, policy_factory in policy_factories.items():
            # criar geradores independentes para ambiente e política
            env_rng, policy_rng = _spawn_rngs(run_sequences[run_index], n_children=2)
            policy = policy_factory(policy_rng)

            final_row, current_trajectory_rows = run_single_poisson_experiment(
                config=config,
                policy=policy,
                replication=replication,
                policy_name=policy_name,
                env_rng=env_rng,
                save_trajectory=save_trajectories,
            )
            final_rows.append(final_row)
            trajectory_rows.extend(current_trajectory_rows)
            run_index += 1

    return pd.DataFrame(final_rows), pd.DataFrame(trajectory_rows)
