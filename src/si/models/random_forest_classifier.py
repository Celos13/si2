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
        if dataset.y is None:
            raise ValueError("RandomForestClassifier requer um dataset com y (labels).")

        X = dataset.X
        y = dataset.y
        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.seed)
        max_features = self.max_features or n_features

        self.trees = []

        for _ in range(self.n_estimators):
            sample_indices = rng.integers(low=0, high=n_samples, size=n_samples)

            feature_indices = rng.choice(
                n_features,
                size=max_features,
                replace=False
            )

            X_boot = X[sample_indices][:, feature_indices]
            y_boot = y[sample_indices]

            features_names = (
                None
                if dataset.features is None
                else [dataset.features[i] for i in feature_indices]
            )

            boot_dataset = Dataset(
                X=X_boot,
                y=y_boot,
                features=features_names,
                label=dataset.label,
            )

            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=self.max_depth,
                mode=self.mode,
                seed=self.seed,
            )
            tree.fit(boot_dataset)

            self.trees.append((feature_indices, tree))

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
        if not self.trees:
            raise RuntimeError("RandomForestClassifier não está treinado. Chama fit() primeiro.")

        X = dataset.X
        n_samples = X.shape[0]
        n_trees = len(self.trees)

        all_preds = np.empty((n_samples, n_trees), dtype=object)

        for j, (feature_indices, tree) in enumerate(self.trees):
            X_sub = X[:, feature_indices]
            tmp_dataset = Dataset(
                X=X_sub,
                y=None,
                features=None,
                label=dataset.label,
            )
            preds = tree.predict(tmp_dataset)
            all_preds[:, j] = preds

        y_pred = []
        for i in range(n_samples):
            values, counts = np.unique(all_preds[i, :], return_counts=True)
            majority_class = values[np.argmax(counts)]
            y_pred.append(majority_class)

        return np.array(y_pred)

    
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


