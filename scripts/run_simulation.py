"""executar simulações por linha de comando."""

from __future__ import annotations

import argparse
from pathlib import Path

from bandit_inference.analysis import summarize_by_policy
from bandit_inference.simulation import ExperimentConfig, run_many_experiments
from bandit_inference.visualization import (
    plot_allocation_distribution,
    plot_coverage,
    plot_selected_arm_frequency,
    plot_selected_mean_distribution,
    plot_selected_n_vs_mean,
    save_figure,
    set_plot_theme,
)


def parse_args() -> argparse.Namespace:
    """ler argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="simular inferência após seleção em bandits")
    parser.add_argument("--n-replications", type=int, default=2000)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--table-dir", type=Path, default=Path("reports/tables"))
    return parser.parse_args()


def main() -> None:
    """executar simulação, salvar tabelas e salvar figuras."""
    args = parse_args()

    # criar pastas de saída
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    # configurar experimento principal
    config = ExperimentConfig(
        n_arms=5,
        horizon=args.horizon,
        n_replications=args.n_replications,
        sigma=1.0,
        seed=args.seed,
    )

    # executar simulações
    results = run_many_experiments(config=config, progress=True)
    summary = summarize_by_policy(results)

    # salvar resultados tabulares
    try:
        results.to_parquet(args.output_dir / "simulation_results.parquet", index=False)
    except ImportError:
        print("pyarrow não encontrado; salvar apenas em csv")

    results.to_csv(args.output_dir / "simulation_results.csv", index=False)
    summary.to_csv(args.table_dir / "summary_by_policy.csv", index=False)

    # salvar figuras principais
    set_plot_theme()
    plot_coverage(results)
    save_figure(args.figure_dir / "coverage_by_policy.png")

    plot_selected_mean_distribution(results)
    save_figure(args.figure_dir / "selected_mean_distribution.png")

    plot_allocation_distribution(results)
    save_figure(args.figure_dir / "allocation_distribution.png")

    plot_selected_arm_frequency(results)
    save_figure(args.figure_dir / "selected_arm_frequency.png")

    plot_selected_n_vs_mean(results)
    save_figure(args.figure_dir / "selected_n_vs_mean.png")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
