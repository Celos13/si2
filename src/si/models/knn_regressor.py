from typing import Callable
import numpy as np
from ..base.model import Model
from ..data.dataset import Dataset
from ..statistics.euclidean_distance import euclidean_distance
from ..metrics.rmse import rmse


class KNNRegressor(Model):
    """
    k-Nearest Neighbors regressor.

    Predicts a continuous target as the mean of the targets of the
    k nearest neighbors in the training set.
    """

    def __init__(self, k: int = 5, distance: Callable[[np.ndarray, np.ndarray], np.ndarray] = euclidean_distance):
        """
        Parameters
        ----------
        k : int, default=5
            Number of neighbors to use.
        distance : callable, default=euclidean_distance
            Distance function receiving (x, X_train) and returning
            a 1D array of distances.
        """
        super().__init__()
        self.k = k
        self.distance = distance
        self.dataset: Dataset | None = None

    def _fit(self, dataset: Dataset) -> "KNNRegressor":
        """
        Store the training dataset.

        Parameters
        ----------
        dataset : Dataset
            Training dataset with X and y.

        Returns
        -------
        KNNRegressor
            The fitted model.
        """
        if dataset.y is None:
            raise ValueError("KNNRegressor requer dataset com y (valores alvo).")
        self.dataset = dataset
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict target values for the given dataset.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted target values.
        """
        if self.dataset is None:
            raise RuntimeError("Modelo não está treinado. Chama fit() primeiro.")

        X_train = self.dataset.X
        y_train = self.dataset.y
        X_test = dataset.X

        preds = []

        for x in X_test:
            dists = self.distance(x, X_train)
            nn_idx = np.argsort(dists)[: self.k]
            y_pred = np.mean(y_train[nn_idx])
            preds.append(y_pred)

        return np.array(preds)

    def _score(self, dataset: Dataset, predictions=None) -> float:
        """
        Compute the RMSE score on the given dataset.

        Parameters
        ----------
        dataset : Dataset
        predictions : ndarray or None, default=None
            Optional precomputed predictions.

        Returns
        -------
        float
            RMSE value.
        """
        y_true = dataset.y
        y_pred = self.predict(dataset)
        return rmse(y_true, y_pred)
