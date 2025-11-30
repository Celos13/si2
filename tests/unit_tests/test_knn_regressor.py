import numpy as np
from si.data.dataset import Dataset
from si.models.knn_regressor import KNNRegressor  


def test_knn_regressor_learns_simple_linear_relation():
    rng = np.random.default_rng(42)
    X = rng.uniform(-5, 5, size=(100, 1))
    y = 2 * X[:, 0] + 1

    ds_train = Dataset(X=X, y=y, features=["x"], label="y")

    model = KNNRegressor(k=1)
    model.fit(ds_train)

    X_test = np.array([[-1.0], [0.0], [2.0]])
    y_true = 2 * X_test[:, 0] + 1
    ds_test = Dataset(X=X_test, y=y_true, features=["x"], label="y")

    y_pred = model.predict(ds_test)

    assert np.allclose(y_pred, y_true, atol=0.1)



