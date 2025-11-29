import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """
    Ridge Regression model using the closed-form least-squares solution.

    Optionally standardizes features before fitting.
    """
    def __init__(self, l2_penalty: float = 1.0, scale: bool = True):
        """
        Parameters
        ----------
        l2_penalty : float, default=1.0
            L2 regularization strength (lambda).
        scale : bool, default=True
            If True, standardize features before fitting.
        """
        super().__init__()
        self.l2_penalty = l2_penalty
        self.scale = scale
        self.theta_zero: float | None = None
        self.theta: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, dataset: Dataset) -> "RidgeRegressionLeastSquares":
        """
        Fit the Ridge Regression model using the analytical solution.

        Parameters
        ----------
        dataset : Dataset
            Training dataset with X and y.

        Returns
        -------
        RidgeRegressionLeastSquares
            The fitted model.
        """
        X = dataset.X.copy()
        y = dataset.y.copy()

        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0, ddof=0)
            self.std[self.std == 0] = 1
            X = (X - self.mean) / self.std
        else:
            self.mean = np.zeros(X.shape[1])
            self.std = np.ones(X.shape[1])

        X_bias = np.c_[np.ones(X.shape[0]), X]

        n_features = X_bias.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0 

        A = X_bias.T.dot(X_bias) + self.l2_penalty * I
        b = X_bias.T.dot(y)
        theta_full = np.linalg.inv(A).dot(b)

        self.theta_zero = theta_full[0]
        self.theta = theta_full[1:]

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
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
        X = dataset.X.copy()
        X = (X - self.mean) / self.std

        X_bias = np.c_[np.ones(X.shape[0]), X]

        theta_full = np.r_[self.theta_zero, self.theta]
        return X_bias.dot(theta_full)

    def _score(self, dataset: Dataset, predictions=None) -> float:
        """
        Compute the MSE on the given dataset.

        Parameters
        ----------
        dataset : Dataset
        predictions : ndarray or None, default=None
            Optional precomputed predictions.

        Returns
        -------
        float
            MSE value.
        """
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)

