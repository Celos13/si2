from typing import Tuple
import numpy as np
from si.data.dataset import Dataset

def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

def stratified_train_test_split(
    dataset: Dataset,
    test_size: float = 0.2,
    random_state: int | None = None
) -> Tuple[Dataset, Dataset]:
    """
    Split a labeled dataset into train and test sets while preserving
    the class distribution (stratified split).

    Parameters
    ----------
    dataset : Dataset
        Labeled dataset (y must not be None).
    test_size : float, default=0.2
        Proportion of samples to include in the test split (0 < test_size < 1).
    random_state : int or None, default=None
        Seed for the random number generator.

    Returns
    -------
    (Dataset, Dataset)
        The train and test Dataset objects.
    """
    if dataset.y is None:
        raise ValueError("stratified_train_test_split requires labeled dataset (y is None).")

    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1.")

    X = dataset.X
    y = dataset.y
    n_samples = X.shape[0]

    rng = np.random.default_rng(random_state)

    unique_classes, y_indices = np.unique(y, return_inverse=True)

    train_indices = []
    test_indices = []

    for class_idx, class_label in enumerate(unique_classes):
        class_mask = (y_indices == class_idx)
        class_indices = np.where(class_mask)[0]

        n_class = class_indices.shape[0]
        if n_class == 0:
            continue

        n_test_class = int(np.floor(test_size * n_class))
        if n_test_class == 0 and n_class > 1:
            n_test_class = 1

        perm = rng.permutation(class_indices)

        test_class_idx = perm[:n_test_class]
        train_class_idx = perm[n_test_class:]

        test_indices.append(test_class_idx)
        train_indices.append(train_class_idx)

    if len(test_indices) == 0:
        raise RuntimeError("No samples were assigned to test set. Adjust test_size or data.")

    test_indices = np.concatenate(test_indices)
    train_indices = np.concatenate(train_indices)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    features = dataset.features
    label = dataset.label

    train_dataset = Dataset(X=X_train, y=y_train, features=features, label=label)
    test_dataset = Dataset(X=X_test, y=y_test, features=features, label=label)

    return train_dataset, test_dataset