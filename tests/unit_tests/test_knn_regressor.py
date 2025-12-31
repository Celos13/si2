import numpy as np
from si.data.dataset import Dataset              # Dataset para construir conjuntos de treino/teste
from si.models.knn_regressor import KNNRegressor # Modelo a testar  


def test_knn_regressor_learns_simple_linear_relation():
    # Testa se o KNNRegressor consegue aproximar bem uma relação linear simples y = 2x + 1

    rng = np.random.default_rng(42)  # RNG reprodutível
    X = rng.uniform(-5, 5, size=(100, 1))  # 100 pontos 1D em [-5, 5]
    y = 2 * X[:, 0] + 1                    # relação exata y = 2x + 1 (sem ruído)

    # Criar Dataset de treino
    ds_train = Dataset(X=X, y=y, features=["x"], label="y")

    # Modelo KNN com k=1 vizinho (deve conseguir "memorizar" a relação)
    model = KNNRegressor(k=1)
    model.fit(ds_train)  # Ajusta o modelo (guarda o dataset de treino)

    # Conjunto de teste com alguns valores específicos
    X_test = np.array([[-1.0], [0.0], [2.0]])
    y_true = 2 * X_test[:, 0] + 1  # y = 2x+1 para esses pontos
    ds_test = Dataset(X=X_test, y=y_true, features=["x"], label="y")

    # Previsões do modelo
    y_pred = model.predict(ds_test)

    # Verifica que as previsões são muito próximas dos valores verdadeiros
    assert np.allclose(y_pred, y_true, atol=0.1)



