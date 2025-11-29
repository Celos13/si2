from typing import Callable
import numpy as np
from ..base.transformer import Transformer
from ..data.dataset import Dataset
from ..statistics.f_classification import f_classification


class SelectPercentile(Transformer):
    """
    Feature selection transformer that keeps the top percentile of features
    according to a scoring function (e.g., f_classification).
    """
    def __init__(self, score_func: Callable[[Dataset], tuple[np.ndarray, np.ndarray]], percentile: float = 10.0):
        """
        Parameters
        ----------
        score_func : callable
            Function that receives a Dataset and returns (F, p) arrays.
        percentile : float, default=10.0
            Percentage of features to keep, in the range (0, 100].
        """
        super().__init__()
        self.score_func = score_func
        self.percentile = percentile
        self.F: np.ndarray | None = None
        self.p: np.ndarray | None = None
        self.selected_features: np.ndarray | None = None

    def _fit(self, dataset: Dataset) -> "SelectPercentile":
        """
        Estimate the F and p values for each feature and select the
        top percentile according to F.

        Parameters
        ----------
        dataset : Dataset
            Input dataset.

        Returns
        -------
        SelectPercentile
            The fitted transformer.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset by selecting only the features chosen
        during fit.

        Parameters
        ----------
        dataset : Dataset
            Input dataset.

        Returns
        -------
        Dataset
            Dataset with reduced feature set.
        """
        n_features = dataset.X.shape[1]
        k = max(1, int(np.ceil(n_features * self.percentile / 100.0)))
        order = np.argsort(self.F)[::-1]
        selected_idx = order[:k]
        selected_idx = np.sort(selected_idx)
        X_new = dataset.X[:, selected_idx]
        
        if dataset.features is not None:
            features_new = [dataset.features[i] for i in selected_idx]
        else:
            features_new = None

        return Dataset(X=X_new, y=dataset.y, features=features_new, label=dataset.label)
