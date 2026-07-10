"""testes básicos da simulação poisson."""

from __future__ import annotations

from bandit_inference.simulation_poisson import PoissonExperimentConfig, run_many_poisson_experiments


def test_poisson_simulation_has_expected_shape() -> None:
    """verificar se a simulação poisson retorna uma linha por política e replicação."""
    config = PoissonExperimentConfig(horizon=20, n_replications=3, seed=123, trajectory_stride=5)
    results, trajectories = run_many_poisson_experiments(config=config, progress=False)

    assert results.shape[0] == 15
    assert {"policy", "total_reward", "optimal_arm_proportion", "lambda_rmse"}.issubset(
        results.columns,
    )
    assert not trajectories.empty
    assert {"policy", "time", "cumulative_reward"}.issubset(trajectories.columns)


def test_poisson_config_rejects_short_horizon() -> None:
    """verificar validação de horizonte mínimo."""
    try:
        PoissonExperimentConfig(horizon=3)
    except ValueError as error:
        assert "horizonte" in str(error)
    else:
        raise AssertionError("era esperado ValueError para horizonte menor que número de braços")
