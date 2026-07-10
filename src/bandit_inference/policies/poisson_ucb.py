"""política ucb bayesiana para recompensas poisson."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bandit_inference.policies.base import BanditPolicy


class PoissonUCBPolicy(BanditPolicy):
    """usar média posterior gamma mais bônus de incerteza."""

    def __init__(
        self,
        n_arms: int,
        prior_alpha: float = 4.0,
        prior_beta: float = 0.2,
        exploration: float = 2.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(n_arms=n_arms, rng=rng)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.exploration = exploration
        self.posterior_alpha: NDArray[np.float64]
        self.posterior_beta: NDArray[np.float64]

        # validar hiperparâmetros e bônus de exploração
        if self.prior_alpha <= 0 or self.prior_beta <= 0:
            msg = "os hiperparâmetros da gamma devem ser positivos"
            raise ValueError(msg)
        if self.exploration < 0:
            msg = "o parâmetro de exploração deve ser não negativo"
            raise ValueError(msg)

        self.reset()

    @property
    def name(self) -> str:
        return "ucb"

    def reset(self) -> None:
        # reiniciar posterior de todos os braços com a prior gamma
        self.posterior_alpha = np.full(self.n_arms, self.prior_alpha, dtype=float)
        self.posterior_beta = np.full(self.n_arms, self.prior_beta, dtype=float)

    def select_arm(
        self,
        t: int,
        counts: NDArray[np.int64],
        reward_sums: NDArray[np.float64],
    ) -> int:
        # calcular média e desvio padrão posteriores de lambda
        posterior_mean = self.posterior_alpha / self.posterior_beta
        posterior_sd = np.sqrt(self.posterior_alpha) / self.posterior_beta

        # aumentar lentamente o bônus para manter exploração ao longo do tempo
        time_factor = np.sqrt(np.log(max(t, 2)))
        scores = posterior_mean + self.exploration * time_factor * posterior_sd

        # escolher braço com maior limite superior posterior
        return self._random_argmax(scores)

    def update(self, arm: int, reward: float) -> None:
        # atualizar posterior gamma-poisson para o braço observado
        self.posterior_alpha[arm] += reward
        self.posterior_beta[arm] += 1.0
