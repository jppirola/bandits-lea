"""funções para resumir simulações poisson."""

from __future__ import annotations

import numpy as np
import pandas as pd


POISSON_POLICY_ORDER = [
    "aleatória uniforme",
    "greedy",
    "thompson sampling",
    "ucb",
    "epsilon-greedy",
]


def summarize_poisson_by_policy(results: pd.DataFrame) -> pd.DataFrame:
    """resumir recompensa, exploração e erro de estimação por política."""
    summary = (
        results.groupby("policy", observed=True)
        .agg(
            n_replications=("total_reward", "size"),
            total_reward_mean=("total_reward", "mean"),
            total_reward_sd=("total_reward", "std"),
            average_reward_mean=("average_reward", "mean"),
            optimal_arm_proportion_mean=("optimal_arm_proportion", "mean"),
            optimal_arm_proportion_sd=("optimal_arm_proportion", "std"),
            lambda_rmse_mean=("lambda_rmse", "mean"),
            lambda_mae_mean=("lambda_mae", "mean"),
        )
        .reset_index()
    )

    # calcular erro padrão e intervalo de monte carlo para recompensa acumulada média
    summary["total_reward_se"] = summary["total_reward_sd"] / np.sqrt(summary["n_replications"])
    summary["total_reward_mc_lower"] = summary["total_reward_mean"] - 1.96 * summary["total_reward_se"]
    summary["total_reward_mc_upper"] = summary["total_reward_mean"] + 1.96 * summary["total_reward_se"]

    # ordenar políticas na ordem do relatório
    summary["policy"] = pd.Categorical(
        summary["policy"],
        categories=POISSON_POLICY_ORDER,
        ordered=True,
    )
    summary = summary.sort_values("policy").reset_index(drop=True)

    return summary


def summarize_poisson_trajectory(trajectories: pd.DataFrame) -> pd.DataFrame:
    """resumir trajetórias médias de recompensa acumulada por política e tempo."""
    summary = (
        trajectories.groupby(["policy", "time"], observed=True)
        .agg(
            cumulative_reward_mean=("cumulative_reward", "mean"),
            cumulative_reward_sd=("cumulative_reward", "std"),
            average_reward_mean=("average_reward", "mean"),
            optimal_arm_proportion_mean=("optimal_arm_proportion", "mean"),
        )
        .reset_index()
    )

    # calcular intervalo de monte carlo de 90% para curvas médias
    n_by_group = trajectories.groupby(["policy", "time"], observed=True).size().rename("n")
    summary = summary.merge(n_by_group.reset_index(), on=["policy", "time"], how="left")
    summary["cumulative_reward_se"] = summary["cumulative_reward_sd"] / np.sqrt(summary["n"])
    summary["cumulative_reward_lower_90"] = (
        summary["cumulative_reward_mean"] - 1.645 * summary["cumulative_reward_se"]
    )
    summary["cumulative_reward_upper_90"] = (
        summary["cumulative_reward_mean"] + 1.645 * summary["cumulative_reward_se"]
    )

    summary["policy"] = pd.Categorical(
        summary["policy"],
        categories=POISSON_POLICY_ORDER,
        ordered=True,
    )
    return summary.sort_values(["policy", "time"]).reset_index(drop=True)


def build_poisson_allocation_long(results: pd.DataFrame) -> pd.DataFrame:
    """transformar proporções finais de alocação para formato longo."""
    prop_cols = [column for column in results.columns if column.startswith("prop_arm_")]
    allocation = results.melt(
        id_vars=["replication", "policy"],
        value_vars=prop_cols,
        var_name="arm",
        value_name="allocation_proportion",
    )

    # extrair número da campanha a partir do nome da coluna
    allocation["arm"] = allocation["arm"].str.replace("prop_arm_", "", regex=False).astype(int)
    allocation["policy"] = pd.Categorical(
        allocation["policy"],
        categories=POISSON_POLICY_ORDER,
        ordered=True,
    )
    return allocation.sort_values(["policy", "replication", "arm"]).reset_index(drop=True)


def build_poisson_estimation_long(results: pd.DataFrame) -> pd.DataFrame:
    """transformar estimativas finais de lambda para formato longo."""
    estimate_cols = [column for column in results.columns if column.startswith("lambda_hat_arm_")]
    estimates = results.melt(
        id_vars=["replication", "policy"],
        value_vars=estimate_cols,
        var_name="arm",
        value_name="lambda_hat",
    )

    # extrair número da campanha a partir do nome da coluna
    estimates["arm"] = estimates["arm"].str.replace("lambda_hat_arm_", "", regex=False).astype(int)
    estimates["policy"] = pd.Categorical(
        estimates["policy"],
        categories=POISSON_POLICY_ORDER,
        ordered=True,
    )
    return estimates.sort_values(["policy", "replication", "arm"]).reset_index(drop=True)
