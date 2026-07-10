"""ambiente gaussiano para o problema de bandits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class GaussianBanditEnvironment:
    """representar um bandit gaussiano com médias fixas e variância conhecida."""

    means: NDArray[np.float64]
    sigma: float = 1.0
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        # converter as médias para vetor numérico
        self.means = np.asarray(self.means, dtype=float)

        # inicializar gerador caso ele não seja informado
        if self.rng is None:
            self.rng = np.random.default_rng()

    @property
    def n_arms(self) -> int:
        """retornar o número de braços."""
        return int(self.means.size)

    def pull(self, arm: int) -> float:
        """sortear a recompensa do braço escolhido."""
        if arm < 0 or arm >= self.n_arms:
            msg = f"braço inválido: {arm}"
            raise ValueError(msg)

        # sortear recompensa normal com média do braço escolhido
        return float(self.rng.normal(loc=self.means[arm], scale=self.sigma))
