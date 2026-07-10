"""política de alocação uniforme aleatória."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bandit_inference.policies.base import BanditPolicy


class UniformRandomPolicy(BanditPolicy):
    """alocar braços de forma uniforme, com inicialização de todos os braços."""

    @property
    def name(self) -> str:
        return "aleatória uniforme"

    def select_arm(
        self,
        t: int,
        counts: NDArray[np.int64],
        reward_sums: NDArray[np.float64],
    ) -> int:
        # observar cada braço ao menos uma vez para evitar médias indefinidas
        unobserved_arm = self._choose_unobserved_arm(counts)
        if unobserved_arm is not None:
            return unobserved_arm

        # sortear braço com probabilidades iguais
        return int(self.rng.integers(low=0, high=self.n_arms))
