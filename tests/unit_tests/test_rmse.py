import numpy as np
from si.metrics.rmse import rmse


def test_rmse_zero_for_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_rmse_known_value():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 4.0])
    # erros: [1, 0, 1] -> MSE = 2/3 -> RMSE = sqrt(2/3)
    expected = np.sqrt(2/3)
    assert np.isclose(rmse(y_true, y_pred), expected)
