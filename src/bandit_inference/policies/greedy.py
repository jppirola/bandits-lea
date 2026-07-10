"""política greedy baseada nas médias amostrais."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bandit_inference.policies.base import BanditPolicy


class GreedyPolicy(BanditPolicy):
    """escolher sempre o braço com maior média amostral observada."""

    @property
    def name(self) -> str:
        return "greedy"

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

        # calcular médias amostrais por braço
        sample_means = reward_sums / counts

        # selecionar braço com maior média amostral
        return self._random_argmax(sample_means)
