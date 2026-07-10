"""funções para resumir resultados das simulações."""

from __future__ import annotations

import numpy as np
import pandas as pd


POLICY_ORDER = ["aleatória uniforme", "greedy", "ucb", "thompson sampling"]


def summarize_by_policy(results: pd.DataFrame) -> pd.DataFrame:
    """resumir cobertura, viés de seleção e alocação por política."""
    summary = (
        results.groupby("policy", observed=True)
        .agg(
            n_replications=("covered", "size"),
            empirical_coverage=("covered", "mean"),
            selected_mean_average=("selected_mean", "mean"),
            selected_mean_sd=("selected_mean", "std"),
            selected_n_average=("selected_n", "mean"),
            ci_half_width_average=("ci_half_width", "mean"),
            average_reward=("average_reward", "mean"),
        )
        .reset_index()
    )

    # calcular erro padrão da cobertura empírica
    p_hat = summary["empirical_coverage"]
    n = summary["n_replications"]
    summary["coverage_se"] = np.sqrt(p_hat * (1.0 - p_hat) / n)

    # ordenar políticas na ordem do relatório
    summary["policy"] = pd.Categorical(summary["policy"], categories=POLICY_ORDER, ordered=True)
    summary = summary.sort_values("policy").reset_index(drop=True)

    return summary


def build_allocation_long(results: pd.DataFrame) -> pd.DataFrame:
    """transformar proporções de alocação para formato longo."""
    prop_cols = [column for column in results.columns if column.startswith("prop_arm_")]
    allocation = results.melt(
        id_vars=["replication", "policy"],
        value_vars=prop_cols,
        var_name="arm",
        value_name="allocation_proportion",
    )

    # extrair número do braço a partir do nome da coluna
    allocation["arm"] = allocation["arm"].str.replace("prop_arm_", "", regex=False).astype(int)
    allocation["policy"] = pd.Categorical(allocation["policy"], categories=POLICY_ORDER, ordered=True)
    allocation = allocation.sort_values(["policy", "replication", "arm"]).reset_index(drop=True)

    return allocation


def build_selected_arm_frequency(results: pd.DataFrame) -> pd.DataFrame:
    """calcular frequência de seleção final de cada braço por política."""
    frequency = (
        results.groupby(["policy", "selected_arm"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    frequency["frequency"] = frequency["count"] / frequency.groupby("policy")["count"].transform("sum")
    frequency["policy"] = pd.Categorical(frequency["policy"], categories=POLICY_ORDER, ordered=True)
    return frequency.sort_values(["policy", "selected_arm"]).reset_index(drop=True)
