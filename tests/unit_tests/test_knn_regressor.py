import numpy as np
from si.data.dataset import Dataset
from si.models.knn_regressor import KNNRegressor  # mantém o caminho que já tens


def test_knn_regressor_learns_simple_linear_relation():
    # dados 1D: y = 2x + 1
    rng = np.random.default_rng(42)
    X = rng.uniform(-5, 5, size=(100, 1))
    y = 2 * X[:, 0] + 1  # sem ruído para teste simples

    ds_train = Dataset(X=X, y=y, features=["x"], label="y")

    # k=1 para este teste ficar "perfeito"
    model = KNNRegressor(k=1)
    model.fit(ds_train)

    # usar alguns pontos de teste
    X_test = np.array([[-1.0], [0.0], [2.0]])
    y_true = 2 * X_test[:, 0] + 1
    ds_test = Dataset(X=X_test, y=y_true, features=["x"], label="y")

    y_pred = model.predict(ds_test)

    # como não há ruído, o erro deve ser muito pequeno
    assert np.allclose(y_pred, y_true, atol=0.1)



