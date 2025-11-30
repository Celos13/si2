from typing import List, Tuple
import numpy as np
from ..base.model import Model
from ..data.dataset import Dataset
from ..metrics.accuracy import accuracy
from .decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier(Model):
    """
    Random Forest classifier ensemble.

    Trains multiple DecisionTreeClassifier models on bootstrap samples
    and random subsets of features, combining their predictions via
    majority voting.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: int | None = None,
        min_sample_split: int = 2,
        max_depth: int | None = None,
        mode: str = "gini",
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        n_estimators : int, default=100
            Number of trees in the forest.
        max_features : int or None, default=None
            Number of features to consider when building each tree.
            If None, use all features.
        min_sample_split : int, default=2
            Minimum number of samples required to split an internal node.
        max_depth : int or None, default=None
            Maximum depth of the individual trees. If None, no limit.
        mode : {"gini", "entropy"}, default="gini"
            Impurity measure used by each DecisionTreeClassifier.
        seed : int or None, default=None
            Random seed for reproducibility.
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees: List[Tuple[np.ndarray, DecisionTreeClassifier]] = []

    def _fit(self, dataset: Dataset) -> "RandomForestClassifier":
        """
        Fit the random forest on the given dataset.

        Each tree is trained on a bootstrap sample of the data and on
        a random subset of features.

        Parameters
        ----------
        dataset : Dataset
            Training dataset with X and y.

        Returns
        -------
        RandomForestClassifier
            The fitted model.
        """
        rng = np.random.default_rng(self.seed)
        X, y = dataset.X, dataset.y
        n_samples, n_features = X.shape
        max_features = self.max_features or n_features

        self.trees = []

        for _ in range(self.n_estimators):
            boot_idx = rng.integers(0, n_samples, size=n_samples)
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]

            feat_idx = rng.choice(n_features, size=max_features, replace=False)
            X_boot_sub = X_boot[:, feat_idx]

            boot_dataset = Dataset(
                X=X_boot_sub,
                y=y_boot,
                features=[dataset.features[i] for i in feat_idx],
                label=dataset.label
            )

            tree_max_depth = self.max_depth if self.max_depth is not None else 1000

            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=tree_max_depth,
                mode=self.mode
            )
            tree.fit(boot_dataset)

            self.trees.append((feat_idx, tree))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels for the given dataset using majority voting
        over all trees in the forest.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X = dataset.X
        n_samples = X.shape[0]

        all_preds = np.zeros((self.n_estimators, n_samples), dtype=object)

        for i, (feat_idx, tree) in enumerate(self.trees):
            X_sub = X[:, feat_idx]
            ds_sub = Dataset(
                X=X_sub,
                y=None,
                features=[dataset.features[j] for j in feat_idx],
                label=dataset.label
            )
            all_preds[i] = tree.predict(ds_sub)

        y_pred = []
        for j in range(n_samples):
            values, counts = np.unique(all_preds[:, j], return_counts=True)
            y_pred.append(values[np.argmax(counts)])

        return np.array(y_pred, dtype=dataset.y.dtype)

    
    def _score(self, dataset: Dataset, predictions=None) -> float:
        """
        Compute the accuracy on the given dataset.

        Parameters
        ----------
        dataset : Dataset
        predictions : ndarray or None, default=None
            Optional precomputed predictions.

        Returns
        -------
        float
            Accuracy score.
        """
        if predictions is None:
            predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)

