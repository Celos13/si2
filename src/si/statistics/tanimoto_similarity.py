from typing import Sequence  # (não é usado diretamente, mas foi importado)
import numpy as np


def tanimoto_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the Tanimoto (Jaccard) similarity between a binary vector x
    and a set of binary vectors y.

    Parameters
    ----------
    x : ndarray of shape (n_features,)
        Query binary vector (0/1).
    y : ndarray of shape (n_samples, n_features)
        Binary vectors to compare x with.

    Returns
    -------
    ndarray of shape (n_samples,)
        Tanimoto similarity values between x and each row of y.

    Notes
    -----
    - Both x and y are assumed to be binary.
    - If union is zero for a pair (both vectors all zeros), the similarity
      is defined as 1.0.
    """
    # Converter x e y para arrays booleanos (True = 1, False = 0)
    x_arr = np.asarray(x, dtype=bool)
    y_arr = np.asarray(y, dtype=bool)

    # Verificar que x é 1D
    if x_arr.ndim != 1:
        raise ValueError("x must be a 1D array")
    # Verificar que y é 2D
    if y_arr.ndim != 2:
        raise ValueError("y must be a 2D array")
    # Verificar que nº de features em x e y coincide
    if x_arr.shape[0] != y_arr.shape[1]:
        raise ValueError(
            f"Incompatible shapes: x has {x_arr.shape[0]} features but "
            f"y has {y_arr.shape[1]}."
        )

    # intersection[i] = nº de posições onde x e y[i] têm 1
    intersection = np.logical_and(x_arr, y_arr).sum(axis=1)

    # x_ones = nº de 1s em x
    x_ones = x_arr.sum()
    # y_ones[i] = nº de 1s em y[i]
    y_ones = y_arr.sum(axis=1)

    # união = |x| + |y| - |interseção|
    union = x_ones + y_ones - intersection

    # vetor de similaridade, inicializado a zeros
    sim = np.zeros_like(union, dtype=float)

    # Máscara para pares onde a união é diferente de 0
    non_zero_union = union != 0
    # Nessas posições, aplicar fórmula padrão
    sim[non_zero_union] = intersection[non_zero_union] / union[non_zero_union]

    # Máscara para casos onde união = 0 (ambos vetores são totalmente zero)
    zero_union = union == 0
    # Definir similaridade = 1.0 nesses casos (vetores idênticos)
    sim[zero_union] = 1.0

    return sim
