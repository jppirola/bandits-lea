"""funções de visualização para o relatório."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bandit_inference.analysis import (
    POLICY_ORDER,
    build_allocation_long,
    build_selected_arm_frequency,
    summarize_by_policy,
)


def set_plot_theme() -> None:
    """configurar tema visual dos gráficos."""
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25


def save_figure(path: str | Path) -> None:
    """salvar figura com margens ajustadas."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")


def plot_coverage(results: pd.DataFrame, target: float = 0.95) -> plt.Axes:
    """criar gráfico da cobertura empírica por política."""
    summary = summarize_by_policy(results)
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(summary["policy"].astype(str), summary["empirical_coverage"])
    ax.axhline(target, linestyle="--", linewidth=2, label="Cobertura nominal de 95%")
    ax.set_xlabel("Política de alocação")
    ax.set_ylabel("Cobertura empírica")
    ax.set_title("Cobertura do intervalo ingênuo após seleção")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", rotation=15)

    return ax


def plot_selected_mean_distribution(results: pd.DataFrame, bins: int = 45) -> plt.Axes:
    """criar distribuição da média amostral do braço selecionado."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for policy in POLICY_ORDER:
        values = results.loc[results["policy"] == policy, "selected_mean"]
        ax.hist(values, bins=bins, density=True, alpha=0.30, label=policy)

    ax.axvline(0, linestyle="--", linewidth=2, label="Média verdadeira")
    ax.set_xlabel("Média amostral do braço selecionado")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição de $\\hat{\\mu}_{\\hat{a}}$")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    return ax


def plot_allocation_distribution(results: pd.DataFrame) -> plt.Axes:
    """criar boxplot das proporções de alocação por braço."""
    allocation = build_allocation_long(results)
    fig, ax = plt.subplots(figsize=(11, 5))

    data = []
    positions = []
    labels = []
    n_arms = allocation["arm"].nunique()
    gap = 1

    for policy_idx, policy in enumerate(POLICY_ORDER):
        policy_data = allocation.loc[allocation["policy"] == policy]
        start_position = policy_idx * (n_arms + gap)
        for arm in range(1, n_arms + 1):
            values = policy_data.loc[policy_data["arm"] == arm, "allocation_proportion"]
            data.append(values.to_numpy())
            positions.append(start_position + arm)
            labels.append(str(arm))

    ax.boxplot(data, positions=positions, widths=0.75, showfliers=False)

    tick_positions = []
    for policy_idx, _policy in enumerate(POLICY_ORDER):
        start_position = policy_idx * (n_arms + gap)
        tick_positions.append(start_position + (n_arms + 1) / 2)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(POLICY_ORDER, rotation=15)
    ax.set_xlabel("Política de alocação")
    ax.set_ylabel("Proporção de alocações")
    ax.set_title("Distribuição das alocações finais por braço")

    return ax


def plot_selected_arm_frequency(results: pd.DataFrame) -> plt.Axes:
    """criar frequência do braço selecionado ao final do experimento."""
    frequency = build_selected_arm_frequency(results)
    pivot = frequency.pivot(index="selected_arm", columns="policy", values="frequency").fillna(0)
    pivot = pivot.reindex(columns=POLICY_ORDER)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Braço selecionado")
    ax.set_ylabel("Frequência")
    ax.set_title("Frequência de seleção final por braço")
    ax.legend(title="Política", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)

    return ax


def plot_selected_n_vs_mean(results: pd.DataFrame) -> plt.Axes:
    """criar relação entre número de amostras e média selecionada."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for policy in POLICY_ORDER:
        policy_data = results.loc[results["policy"] == policy]
        ax.scatter(
            policy_data["selected_n"],
            policy_data["selected_mean"],
            alpha=0.45,
            s=35,
            label=policy,
        )

    ax.axhline(0, linestyle="--", linewidth=2)
    ax.set_xlabel("Número de observações do braço selecionado")
    ax.set_ylabel("Média amostral do braço selecionado")
    ax.set_title("Tamanho amostral adaptativo e média selecionada")
    ax.legend(title="Política", bbox_to_anchor=(1.02, 1), loc="upper left")

    return ax
