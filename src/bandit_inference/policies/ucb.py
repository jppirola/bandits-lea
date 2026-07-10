"""política upper confidence bound."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bandit_inference.policies.base import BanditPolicy


class UCBPolicy(BanditPolicy):
    """escolher o braço com maior média amostral mais bônus de exploração."""

    def __init__(
        self,
        n_arms: int,
        exploration: float = 2.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(n_arms=n_arms, rng=rng)
        self.exploration = exploration

    @property
    def name(self) -> str:
        return "ucb"

    def select_arm(
        self,
        t: int,
        counts: NDArray[np.int64],
        reward_sums: NDArray[np.float64],
    ) -> int:
        # observar cada braço ao menos uma vez para iniciar o bônus
        unobserved_arm = self._choose_unobserved_arm(counts)
        if unobserved_arm is not None:
            return unobserved_arm

        # calcular médias amostrais por braço
        sample_means = reward_sums / counts

        # calcular bônus de exploração que diminui com o número de alocações
        log_term = np.log(max(t, 2))
        exploration_bonus = self.exploration * np.sqrt(log_term / counts)

        # selecionar braço com maior limite superior
        scores = sample_means + exploration_bonus
        return self._random_argmax(scores)
