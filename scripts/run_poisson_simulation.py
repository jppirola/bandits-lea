"""executar simulações poisson por linha de comando."""

from __future__ import annotations

import argparse
from pathlib import Path

from bandit_inference.analysis_poisson import summarize_poisson_by_policy
from bandit_inference.simulation_poisson import PoissonExperimentConfig, run_many_poisson_experiments
from bandit_inference.visualization_poisson import save_poisson_main_figures


def parse_args() -> argparse.Namespace:
    """ler argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="simular bandit poisson com recompensa acumulada")
    parser.add_argument("--n-replications", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=365)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--prior-alpha", type=float, default=4.0)
    parser.add_argument("--prior-beta", type=float, default=0.2)
    parser.add_argument("--ucb-exploration", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--trajectory-stride", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--table-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--skip-trajectories", action="store_true")
    return parser.parse_args()


def main() -> None:
    """executar simulação, salvar tabelas e salvar figuras."""
    args = parse_args()

    # criar pastas de saída
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    # configurar experimento poisson
    config = PoissonExperimentConfig(
        horizon=args.horizon,
        n_replications=args.n_replications,
        seed=args.seed,
        prior_alpha=args.prior_alpha,
        prior_beta=args.prior_beta,
        ucb_exploration=args.ucb_exploration,
        epsilon=args.epsilon,
        trajectory_stride=args.trajectory_stride,
    )

    # executar simulações
    results, trajectories = run_many_poisson_experiments(
        config=config,
        progress=True,
        save_trajectories=not args.skip_trajectories,
    )
    summary = summarize_poisson_by_policy(results)

    # salvar resultados tabulares
    try:
        results.to_parquet(args.output_dir / "poisson_results.parquet", index=False)
        if not trajectories.empty:
            trajectories.to_parquet(args.output_dir / "poisson_trajectories.parquet", index=False)
    except ImportError:
        print("pyarrow não encontrado; salvar apenas em csv")

    results.to_csv(args.output_dir / "poisson_results.csv", index=False)
    if not trajectories.empty:
        trajectories.to_csv(args.output_dir / "poisson_trajectories.csv", index=False)
    summary.to_csv(args.table_dir / "poisson_summary_by_policy.csv", index=False)

    # salvar figuras principais quando houver trajetórias
    if not trajectories.empty:
        save_poisson_main_figures(results, trajectories, args.figure_dir)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
