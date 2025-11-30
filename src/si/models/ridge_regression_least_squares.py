import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """
    Ridge Regression using the analytical (closed-form) least-squares solution.

    This model computes:
        θ = (Xᵀ X + λI)⁻¹ Xᵀ y

    Features can optionally be standardized before fitting. The bias term is NOT
    regularized, matching standard Ridge Regression conventions.

    Parameters
    ----------
    l2_penalty : float, default=1.0
        Strength of the L2 regularization term λ. Larger values shrink the
        coefficients more.
    scale : bool, default=True
        If True, standardize the features to zero mean and unit variance before
        fitting. Standardization usually improves numerical stability.
    """

    def __init__(self, l2_penalty: float = 1.0, scale: bool = True):
        super().__init__()
        self.l2_penalty = l2_penalty
        self.scale = scale

        # learned parameters
        self.theta: np.ndarray | None = None
        self.theta_zero: float | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def _fit(self, dataset: Dataset) -> "RidgeRegressionLeastSquares":
        """
        Fit the Ridge Regression model using the analytical solution.

        Steps:
        1. Standardize X if scale=True.
        2. Add a bias column (column of ones).
        3. Compute the regularized normal equation.

        Parameters
        ----------
        dataset : Dataset
            Training dataset containing X (features) and y (targets).

        Returns
        -------
        RidgeRegressionLeastSquares
            The fitted model.
        """
        X = dataset.X.copy()
        y = dataset.y

        # 1. Standardize features (optional)
        if self.scale:
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            self.std_[self.std_ == 0] = 1  # prevent division by zero
            X = (X - self.mean_) / self.std_
        else:
            # identity transformation
            self.mean_ = np.zeros(X.shape[1])
            self.std_ = np.ones(X.shape[1])

        # 2. Add bias term
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])

        # 3. Regularization matrix (do NOT penalize the bias parameter)
        n_features = X_b.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0  # bias not penalized

        # closed-form ridge regression solution
        A = X_b.T @ X_b + self.l2_penalty * I
        b = X_b.T @ y
        params = np.linalg.solve(A, b)

        # store parameters
        self.theta_zero = float(params[0])
        self.theta = params[1:]

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict target values for the given dataset.

        Prediction:
            y = θ₀ + X θ

        Parameters
        ----------
        dataset : Dataset
            Dataset containing the feature matrix X.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted continuous target values.
        """
        X = dataset.X.copy()

        # apply the same scaling used in fit
        if self.scale:
            X = (X - self.mean_) / self.std_

        return self.theta_zero + X @ self.theta

    def _score(self, dataset: Dataset, predictions: np.ndarray | None = None) -> float:
        """
        Compute the Mean Squared Error (MSE) on the given dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing ground-truth values.
        predictions : ndarray or None, default=None
            Precomputed predictions. If None, predictions are computed.

        Returns
        -------
        float
            MSE value.
        """
        if predictions is None:
            predictions = self.predict(dataset)
        return mse(dataset.y, predictions)


