from typing import Sequence
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
    x_arr = np.asarray(x, dtype=bool)
    y_arr = np.asarray(y, dtype=bool)

    if x_arr.ndim != 1:
        raise ValueError("x must be a 1D array")
    if y_arr.ndim != 2:
        raise ValueError("y must be a 2D array")
    if x_arr.shape[0] != y_arr.shape[1]:
        raise ValueError(
            f"Incompatible shapes: x has {x_arr.shape[0]} features but "
            f"y has {y_arr.shape[1]}."
        )

    intersection = np.logical_and(x_arr, y_arr).sum(axis=1)

    x_ones = x_arr.sum()
    y_ones = y_arr.sum(axis=1)

    union = x_ones + y_ones - intersection

    sim = np.zeros_like(union, dtype=float)

    non_zero_union = union != 0
    sim[non_zero_union] = intersection[non_zero_union] / union[non_zero_union]

    zero_union = union == 0
    sim[zero_union] = 1.0

    return sim

