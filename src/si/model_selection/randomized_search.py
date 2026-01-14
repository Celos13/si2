from __future__ import annotations

import itertools
import copy
from typing import Callable, Any

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation


def randomized_search_cv(
    model: Model,
    dataset: Dataset,
    hyperparameter_grid: dict[str, np.ndarray | list[Any] | tuple[Any, ...]],
    scoring: Callable | None = None,
    cv: int = 3,
    n_iter: int = 10,
    random_state: int | None = 42,
) -> dict[str, Any]:
    """
    Randomized search with cross-validation.

    Returns
    -------
    dict with keys:
        - 'hyperparameters': list[dict]
        - 'scores': list[float]
        - 'best_hyperparameters': dict
        - 'best_score': float
    """
    # 1) validar se os nomes de hyperparâmetros existem no modelo (hasattr)
    for hp_name in hyperparameter_grid.keys():
        if not hasattr(model, hp_name):
            raise ValueError(f"Hyperparameter '{hp_name}' does not exist in the model.")

    # transformar valores em listas (para poder fazer produto cartesiano)
    hp_names = list(hyperparameter_grid.keys())
    hp_values = []
    for name in hp_names:
        vals = hyperparameter_grid[name]
        vals = list(vals)  # funciona para np.ndarray, list, tuple
        if len(vals) == 0:
            raise ValueError(f"Hyperparameter '{name}' has an empty search space.")
        hp_values.append(vals)

    # 2) gerar todas as combinações possíveis e escolher n_iter aleatórias
    all_combinations = list(itertools.product(*hp_values))
    if len(all_combinations) == 0:
        raise ValueError("No hyperparameter combinations were generated.")

    rng = np.random.default_rng(random_state)

    # se n_iter > total de combinações, usamos todas
    n_iter = min(n_iter, len(all_combinations))

    chosen_idx = rng.choice(len(all_combinations), size=n_iter, replace=False)
    chosen_combinations = [all_combinations[i] for i in chosen_idx]

    results_hparams: list[dict[str, Any]] = []
    results_scores: list[float] = []

    best_score = -np.inf
    best_hparams: dict[str, Any] | None = None

    # 3..6) para cada combinação: setattr -> CV -> guardar mean score
    for combo in chosen_combinations:
        # criar dicionário da combinação atual
        current_hparams = dict(zip(hp_names, combo))

        # criar uma cópia do modelo para não “contaminar” combinações
        m = copy.deepcopy(model)


        # 3) setar hyperparâmetros
        for k, v in current_hparams.items():
            # converter floats que eram "inteiros" (ex.: max_iter vindo de linspace)
            if k == "max_iter":
                v = int(round(float(v)))
            setattr(m, k, v)

        # 4) cross validate
        fold_scores = k_fold_cross_validation(m, dataset, cv=cv, scoring=scoring, seed=random_state)

        # 5) mean score
        mean_score = float(np.mean(fold_scores))

        results_hparams.append(current_hparams)
        results_scores.append(mean_score)

        # 7) melhor score
        if mean_score > best_score:
            best_score = mean_score
            best_hparams = current_hparams

    # 8) return dictionary
    return {
        "hyperparameters": results_hparams,
        "scores": results_scores,
        "best_hyperparameters": best_hparams,
        "best_score": best_score,
    }
