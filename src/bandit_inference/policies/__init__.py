"""políticas de alocação implementadas no projeto."""

from bandit_inference.policies.epsilon_greedy import EpsilonGreedyPolicy
from bandit_inference.policies.greedy import GreedyPolicy
from bandit_inference.policies.poisson_thompson import PoissonThompsonSamplingPolicy
from bandit_inference.policies.poisson_ucb import PoissonUCBPolicy
from bandit_inference.policies.random_policy import UniformRandomPolicy
from bandit_inference.policies.thompson import GaussianThompsonSamplingPolicy
from bandit_inference.policies.ucb import UCBPolicy

__all__ = [
    "EpsilonGreedyPolicy",
    "GaussianThompsonSamplingPolicy",
    "GreedyPolicy",
    "PoissonThompsonSamplingPolicy",
    "PoissonUCBPolicy",
    "UCBPolicy",
    "UniformRandomPolicy",
]
