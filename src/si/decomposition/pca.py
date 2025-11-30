import numpy as np
from ..base.transformer import Transformer
from ..data.dataset import Dataset


class PCA(Transformer):
    """
    Principal Component Analysis (PCA) transformer.

    Learns a linear projection that maps the original features into
    a lower-dimensional space maximizing variance.
    """
    def __init__(self, n_components: int):
        """
        Parameters
        ----------
        n_components : int
            Number of principal components to keep.
        """
        super().__init__()
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None

    def _fit(self, dataset: Dataset) -> "PCA":
        """
        Fit the PCA model to the dataset.

        Parameters
        ----------
        dataset : Dataset
            Input dataset.

        Returns
        -------
        PCA
            The fitted PCA transformer.
        """
        X = dataset.X

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        cov = np.cov(X_centered, rowvar=False)

        eigvals, eigvecs = np.linalg.eigh(cov)

        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        self.components_ = eigvecs[:, :self.n_components]
        self.explained_variance_ = eigvals[:self.n_components]

        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Project the dataset onto the learned principal components.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        Dataset
            Dataset with transformed features (PC1, PC2, ...).
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA must be fitted before calling transform().")

        X = dataset.X
        X_centered = X - self.mean_

        X_new = np.dot(X_centered, self.components_)

        features = [f"PC{i+1}" for i in range(self.n_components)]

        return Dataset(
            X=X_new,
            y=dataset.y,
            features=features,
            label=dataset.label
        )
