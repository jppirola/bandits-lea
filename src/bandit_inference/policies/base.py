"""interfaces comuns para políticas de alocação."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class BanditPolicy(ABC):
    """definir a interface mínima de uma política de alocação."""

    def __init__(self, n_arms: int, rng: np.random.Generator | None = None) -> None:
        self.n_arms = n_arms
        self.rng = rng if rng is not None else np.random.default_rng()

    @property
    @abstractmethod
    def name(self) -> str:
        """retornar o nome da política."""

    def reset(self) -> None:
        """reiniciar estados internos antes de uma nova simulação."""

    @abstractmethod
    def select_arm(
        self,
        t: int,
        counts: NDArray[np.int64],
        reward_sums: NDArray[np.float64],
    ) -> int:
        """selecionar um braço antes de observar a recompensa."""

    def update(self, arm: int, reward: float) -> None:
        """atualizar estados internos após observar a recompensa."""

    def _random_argmax(self, values: NDArray[np.float64]) -> int:
        """sortear uniformemente entre os máximos para quebrar empates."""
        max_value = np.nanmax(values)
        candidates = np.flatnonzero(np.isclose(values, max_value, equal_nan=False))
        return int(self.rng.choice(candidates))

    def _choose_unobserved_arm(self, counts: NDArray[np.int64]) -> int | None:
        """escolher um braço ainda não observado, se existir."""
        unobserved = np.flatnonzero(counts == 0)
        if unobserved.size == 0:
            return None
        return int(self.rng.choice(unobserved))
