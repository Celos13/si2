import numpy as np
from si.metrics.rmse import rmse  # Função de erro RMSE a testar


def test_rmse_zero_for_perfect_prediction():
    # Se a previsão for perfeita (y_pred == y_true), o RMSE deve ser 0

    y = np.array([1.0, 2.0, 3.0])  # valores verdadeiros
    # Aqui estamos a passar o mesmo vetor como previsão
    assert rmse(y, y) == 0.0


def test_rmse_known_value():
    # Testa o RMSE num caso em que o valor esperado é conhecido

    y_true = np.array([1.0, 2.0, 3.0])  # valores verdadeiros
    y_pred = np.array([2.0, 2.0, 4.0])  # previsões

    # Cálculo manual do RMSE:
    # diffs = [1,0,1] → squared = [1,0,1] → mean = 2/3 → sqrt(2/3)
    expected = np.sqrt(2/3)
    # Verifica se rmse(y_true, y_pred) ≈ expected
    assert np.isclose(rmse(y_true, y_pred), expected)

