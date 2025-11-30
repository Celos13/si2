import numpy as np
from si.data.dataset import Dataset
from si.models.random_forest_classifier import RandomForestClassifier


def test_random_forest_classifier_simple_separable_data():
    rng = np.random.default_rng(42)
    X_class0 = rng.normal(loc=-2.0, scale=0.5, size=(50, 2))
    X_class1 = rng.normal(loc=2.0, scale=0.5, size=(50, 2))

    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * 50 + [1] * 50)

    ds = Dataset(X=X, y=y, features=["f1", "f2"], label="class")

    model = RandomForestClassifier(
        n_estimators=10,
        max_features=2,
        min_sample_split=2,
        max_depth=None,
        mode="gini",
        seed=42
    )

    model.fit(ds)
    score = model.score(ds)

    assert score > 0.9
