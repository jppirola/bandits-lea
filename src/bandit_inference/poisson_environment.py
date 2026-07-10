"""ambiente poisson para recompensa acumulada."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class PoissonBanditEnvironment:
    """gerar recompensas poisson para cada braço."""

    def __init__(
        self,
        lambdas: NDArray[np.float64],
        rng: np.random.Generator | None = None,
    ) -> None:
        # guardar taxas verdadeiras de cadastro por campanha
        self.lambdas = np.asarray(lambdas, dtype=float)
        self.n_arms = int(self.lambdas.size)
        self.rng = rng if rng is not None else np.random.default_rng()

        # validar taxas positivas
        if np.any(self.lambdas <= 0):
            msg = "todas as taxas poisson devem ser positivas"
            raise ValueError(msg)

    def pull(self, arm: int) -> int:
        """observar recompensa do braço escolhido."""
        # validar índice do braço
        if arm < 0 or arm >= self.n_arms:
            msg = "braço fora do espaço de ações"
            raise ValueError(msg)

        # amostrar número de novos cadastros do dia
        return int(self.rng.poisson(lam=self.lambdas[arm]))
