from typing import List

import numpy as np

from si.base.model import Model     # Interface base dos modelos (fit, predict, score)
from si.data.dataset import Dataset # Classe Dataset


def k_fold_cross_validation(model: Model, dataset: Dataset, 
                            scoring: callable = None, cv: int = 3,
                            seed: int = None) -> List[float]:
    """
    Perform k-fold cross-validation on the given model and dataset.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    scoring: Callable
        The scoring function to use. If None, the model's score method will be used.
    cv: int
        The number of cross-validation folds.
    seed: int
        The seed to use for the random number generator.

    Returns
    -------
    scores: List[float]
        The scores of the model on each fold.
    """
    # Número total de amostras
    num_samples = dataset.X.shape[0]
    # Tamanho aproximado de cada fold (divisão inteira)
    fold_size = num_samples // cv
    # Lista onde serão acumulados os scores de cada fold
    scores = []

    # Criar um array de índices para embaralhar os dados
    if seed is not None:
        np.random.seed(seed)  # Fixar seed se dado
    indices = np.arange(num_samples)  # [0, 1, ..., num_samples-1]
    np.random.shuffle(indices)        # Embaralhar ordem das amostras

    # Loop sobre o número de folds
    for fold in range(cv):
        # Determinar intervalo de índices do fold atual
        start = fold * fold_size
        end = (fold + 1) * fold_size

        # Índices que farão parte do conjunto de teste neste fold
        test_indices = indices[start:end]
        # Índices de treino: tudo o resto
        train_indices = np.concatenate((indices[:start], indices[end:]))

        # Construir Dataset de treino e teste usando slicing de X e y
        dataset_train = Dataset(dataset.X[train_indices], dataset.y[train_indices])
        dataset_test = Dataset(dataset.X[test_indices], dataset.y[test_indices])

        # Ajustar o modelo aos dados de treino
        model.fit(dataset_train)
        # Se scoring foi fornecido, aplicá-lo; caso contrário, usar model.score
        if scoring is not None:
            fold_score = scoring(dataset_test.y, model.predict(dataset_test))
        else:
            fold_score = model.score(dataset_test)
        # Guardar score deste fold
        scores.append(fold_score)

    # Devolver lista de scores (um por fold)
    return scores
