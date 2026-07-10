"""testes mínimos para validar a simulação."""

from __future__ import annotations

from bandit_inference.analysis import summarize_by_policy
from bandit_inference.simulation import ExperimentConfig, run_many_experiments


def test_run_many_experiments_returns_expected_number_of_rows() -> None:
    """verificar se cada replicação gera uma linha por política."""
    config = ExperimentConfig(n_replications=3, horizon=20, seed=123)
    results = run_many_experiments(config=config, progress=False)

    assert results.shape[0] == 12
    assert set(results["policy"]) == {"aleatória uniforme", "greedy", "ucb", "thompson sampling"}


def test_summary_contains_empirical_coverage() -> None:
    """verificar se resumo calcula cobertura empírica."""
    config = ExperimentConfig(n_replications=3, horizon=20, seed=123)
    results = run_many_experiments(config=config, progress=False)
    summary = summarize_by_policy(results)

    assert "empirical_coverage" in summary.columns
    assert summary["empirical_coverage"].between(0, 1).all()
