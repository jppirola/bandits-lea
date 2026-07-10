"""funções de visualização para o experimento poisson."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bandit_inference.analysis_poisson import (
    POISSON_POLICY_ORDER,
    build_poisson_allocation_long,
    build_poisson_estimation_long,
    summarize_poisson_by_policy,
    summarize_poisson_trajectory,
)
from bandit_inference.visualization import save_figure, set_plot_theme


CAMPAIGN_LABELS = {
    1: "1: Genérico",
    2: "2: Facilidade",
    3: "3: Economia",
    4: "4: Depoimentos",
    5: "5: Personalizada",
}


def plot_poisson_cumulative_reward(trajectories: pd.DataFrame) -> plt.Axes:
    """criar curvas médias de recompensa acumulada com intervalo de monte carlo."""
    trajectory_summary = summarize_poisson_trajectory(trajectories)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for policy in POISSON_POLICY_ORDER:
        data = trajectory_summary.loc[trajectory_summary["policy"] == policy]
        if data.empty:
            continue

        x = data["time"].to_numpy(dtype=float)
        y = data["cumulative_reward_mean"].to_numpy(dtype=float)
        lower = data["cumulative_reward_lower_90"].to_numpy(dtype=float)
        upper = data["cumulative_reward_upper_90"].to_numpy(dtype=float)
        ax.plot(x, y, linewidth=2, label=policy)
        ax.fill_between(x, lower, upper, alpha=0.15)

    ax.set_xlabel("Dia do experimento")
    ax.set_ylabel("Recompensa acumulada média")
    ax.set_title("Recompensa acumulada ao longo do experimento")
    ax.legend(title="Política", bbox_to_anchor=(1.02, 1), loc="upper left")
    return ax


def plot_poisson_total_reward_boxplot(results: pd.DataFrame) -> plt.Axes:
    """criar boxplot da recompensa acumulada final por política."""
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [
        results.loc[results["policy"] == policy, "total_reward"].to_numpy()
        for policy in POISSON_POLICY_ORDER
    ]

    ax.boxplot(data, tick_labels=POISSON_POLICY_ORDER, showfliers=False)
    ax.set_xlabel("Política de alocação")
    ax.set_ylabel("Recompensa acumulada final")
    ax.set_title("Distribuição da recompensa acumulada final")
    ax.tick_params(axis="x", rotation=15)
    return ax


def plot_poisson_optimal_arm_proportion(results: pd.DataFrame) -> plt.Axes:
    """criar proporção média de alocação na campanha ótima."""
    summary = summarize_poisson_by_policy(results)
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(summary["policy"].astype(str), summary["optimal_arm_proportion_mean"])
    ax.set_xlabel("Política de alocação")
    ax.set_ylabel("Proporção média na campanha ótima")
    ax.set_title("Intensidade de exploração versus alocação na melhor campanha")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)
    return ax


def plot_poisson_allocation_distribution(results: pd.DataFrame) -> plt.Axes:
    """criar distribuição das proporções finais de alocação por campanha."""
    allocation = build_poisson_allocation_long(results)
    fig, ax = plt.subplots(figsize=(12, 5))

    data = []
    positions = []
    n_arms = allocation["arm"].nunique()
    gap = 1

    for policy_idx, policy in enumerate(POISSON_POLICY_ORDER):
        policy_data = allocation.loc[allocation["policy"] == policy]
        start_position = policy_idx * (n_arms + gap)
        for arm in range(1, n_arms + 1):
            values = policy_data.loc[policy_data["arm"] == arm, "allocation_proportion"]
            data.append(values.to_numpy())
            positions.append(start_position + arm)

    ax.boxplot(data, positions=positions, widths=0.75, showfliers=False)

    tick_positions = []
    for policy_idx, _policy in enumerate(POISSON_POLICY_ORDER):
        start_position = policy_idx * (n_arms + gap)
        tick_positions.append(start_position + (n_arms + 1) / 2)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(POISSON_POLICY_ORDER, rotation=15)
    ax.set_xlabel("Política de alocação")
    ax.set_ylabel("Proporção de alocações")
    ax.set_title("Distribuição das alocações finais por campanha")
    ax.text(
        0.01,
        -0.25,
        "Em cada política, as cinco caixas representam as campanhas 1 a 5",
        transform=ax.transAxes,
        fontsize=10,
    )
    return ax


def plot_poisson_lambda_estimates(
    results: pd.DataFrame,
    lambdas: list[float] | None = None,
) -> plt.Axes:
    """criar distribuição das estimativas finais de lambda por campanha."""
    estimates = build_poisson_estimation_long(results)
    if lambdas is None:
        lambdas = [18, 20, 21, 23, 27]

    fig, ax = plt.subplots(figsize=(12, 5))
    data = []
    positions = []
    n_arms = len(lambdas)
    gap = 1

    for policy_idx, policy in enumerate(POISSON_POLICY_ORDER):
        policy_data = estimates.loc[estimates["policy"] == policy]
        start_position = policy_idx * (n_arms + gap)
        for arm in range(1, n_arms + 1):
            values = policy_data.loc[policy_data["arm"] == arm, "lambda_hat"]
            data.append(values.to_numpy())
            positions.append(start_position + arm)

    ax.boxplot(data, positions=positions, widths=0.75, showfliers=False)

    tick_positions = []
    for policy_idx, _policy in enumerate(POISSON_POLICY_ORDER):
        start_position = policy_idx * (n_arms + gap)
        tick_positions.append(start_position + (n_arms + 1) / 2)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(POISSON_POLICY_ORDER, rotation=15)
    ax.set_xlabel("Política de alocação")
    ax.set_ylabel("Estimativa final de $\\lambda_a$")
    ax.set_title("Qualidade das estimativas finais das taxas")

    for lambda_value in lambdas:
        ax.axhline(lambda_value, linestyle="--", linewidth=1, alpha=0.35)

    ax.text(
        0.01,
        -0.25,
        "Linhas tracejadas indicam as taxas verdadeiras das cinco campanhas",
        transform=ax.transAxes,
        fontsize=10,
    )
    return ax


def save_poisson_main_figures(
    results: pd.DataFrame,
    trajectories: pd.DataFrame,
    figure_dir: str | Path,
) -> None:
    """salvar as figuras principais da parte 2."""
    set_plot_theme()
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_poisson_cumulative_reward(trajectories)
    save_figure(figure_dir / "poisson_cumulative_reward.png")
    plt.close()

    plot_poisson_total_reward_boxplot(results)
    save_figure(figure_dir / "poisson_total_reward_boxplot.png")
    plt.close()

    plot_poisson_optimal_arm_proportion(results)
    save_figure(figure_dir / "poisson_optimal_arm_proportion.png")
    plt.close()

    plot_poisson_allocation_distribution(results)
    save_figure(figure_dir / "poisson_allocation_distribution.png")
    plt.close()

    plot_poisson_lambda_estimates(results)
    save_figure(figure_dir / "poisson_lambda_estimates.png")
    plt.close()
