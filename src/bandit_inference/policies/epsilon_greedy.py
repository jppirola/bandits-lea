"""política epsilon-greedy baseada nas médias amostrais."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bandit_inference.policies.base import BanditPolicy


class EpsilonGreedyPolicy(BanditPolicy):
    """explorar com probabilidade epsilon e explorar a melhor média no restante."""

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(n_arms=n_arms, rng=rng)
        self.epsilon = epsilon

        # validar probabilidade de exploração
        if not 0 <= self.epsilon <= 1:
            msg = "epsilon deve estar entre 0 e 1"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        return "epsilon-greedy"

    def select_arm(
        self,
        t: int,
        counts: NDArray[np.int64],
        reward_sums: NDArray[np.float64],
    ) -> int:
        # observar cada braço ao menos uma vez para iniciar as médias
        unobserved_arm = self._choose_unobserved_arm(counts)
        if unobserved_arm is not None:
            return unobserved_arm

        # sortear exploração uniforme com probabilidade epsilon
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(low=0, high=self.n_arms))

        # calcular médias amostrais por braço
        sample_means = reward_sums / counts

        # selecionar braço com maior média observada
        return self._random_argmax(sample_means)
