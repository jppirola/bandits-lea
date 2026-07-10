"""política thompson sampling para recompensas gaussianas com variância conhecida."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bandit_inference.policies.base import BanditPolicy


class GaussianThompsonSamplingPolicy(BanditPolicy):
    """amostrar médias da posterior normal e escolher a maior amostra."""

    def __init__(
        self,
        n_arms: int,
        reward_sigma: float = 1.0,
        prior_mean: float = 0.0,
        prior_var: float = 10.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(n_arms=n_arms, rng=rng)
        self.reward_sigma = reward_sigma
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.posterior_mean: NDArray[np.float64]
        self.posterior_var: NDArray[np.float64]
        self.reset()

    @property
    def name(self) -> str:
        return "thompson sampling"

    def reset(self) -> None:
        # reiniciar a posterior de todos os braços com a prior
        self.posterior_mean = np.full(self.n_arms, self.prior_mean, dtype=float)
        self.posterior_var = np.full(self.n_arms, self.prior_var, dtype=float)

    def select_arm(
        self,
        t: int,
        counts: NDArray[np.int64],
        reward_sums: NDArray[np.float64],
    ) -> int:
        # amostrar uma média plausível para cada braço
        sampled_means = self.rng.normal(
            loc=self.posterior_mean,
            scale=np.sqrt(self.posterior_var),
        )

        # escolher o braço com maior média amostrada
        return self._random_argmax(sampled_means)

    def update(self, arm: int, reward: float) -> None:
        # atualizar posterior normal com variância observacional conhecida
        observation_var = self.reward_sigma**2
        previous_var = self.posterior_var[arm]
        previous_mean = self.posterior_mean[arm]

        new_precision = (1.0 / previous_var) + (1.0 / observation_var)
        new_var = 1.0 / new_precision
        new_mean = new_var * ((previous_mean / previous_var) + (reward / observation_var))

        self.posterior_var[arm] = new_var
        self.posterior_mean[arm] = new_mean
