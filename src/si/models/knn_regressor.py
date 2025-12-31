import numpy as np
from typing import Callable

from si.base.model import Model                 # Classe base Model (fit, predict, score)
from si.data.dataset import Dataset             # Classe Dataset
from si.statistics.euclidean_distance import euclidean_distance  # Distância padrão
from si.metrics.rmse import rmse                # Métrica de avaliação


class KNNRegressor(Model):
    """
    k-Nearest Neighbors regressor.

    This model predicts a continuous target value as the mean of the targets
    of the k nearest neighbors in the training set, according to a given
    distance function.
    """

    def __init__(
        self,
        k: int = 5,
        distance: Callable[[np.ndarray, np.ndarray], np.ndarray] = euclidean_distance,
    ):
        """
        Parameters
        ----------
        k : int, default=5
            Number of neighbors to use.
        distance : callable, default=euclidean_distance
            Function that receives (x, X_train) and returns a 1D array of
            distances from x to each row in X_train.
        """
        super().__init__()         # Inicializa Model
        self.k = k                 # Nº de vizinhos a considerar
        self.distance = distance   # Função de distância a usar
        self._train_dataset: Dataset | None = None  # Dataset de treino será guardado aqui

    def _fit(self, dataset: Dataset) -> "KNNRegressor":
        """
        Store the training dataset.

        Parameters
        ----------
        dataset : Dataset
            Training dataset with X and y.

        Returns
        -------
        KNNRegressor
            The fitted model.
        """
        # KNN é um método "lazy": apenas guarda o dataset de treino
        self._train_dataset = dataset
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict target values for the given dataset using
        the mean of the k nearest neighbors.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted target values.
        """
        # Verificar se já foi feito fit
        if self._train_dataset is None:
            raise ValueError("Model is not fitted. Call 'fit' first.")

        # Extrair dados de treino
        X_train = self._train_dataset.X
        y_train = self._train_dataset.y
        # Dados de teste
        X_test = dataset.X

        # Nº de amostras de teste
        n_test = X_test.shape[0]
        # Vetor para guardar previsões
        y_pred = np.zeros(n_test, dtype=float)

        # Para cada amostra de teste
        for i in range(n_test):
            x = X_test[i]  # Ponto de teste i-ésimo
            # Calcular distâncias entre x e todas as amostras de treino
            dists = self.distance(x, X_train)
            # Obter índices dos k vizinhos mais próximos (distâncias mais pequenas)
            nn_idx = np.argsort(dists)[: self.k]
            # Previsão = média dos y dos vizinhos
            y_pred[i] = float(np.mean(y_train[nn_idx]))
        # Aqui estás a arredondar as previsões a uma casa decimal (não é obrigatório)
        return np.round(y_pred, 1)

    def _score(self, dataset: Dataset, predictions: np.ndarray | None = None) -> float:
        """
        Compute the RMSE on the given dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset with ground-truth target values.
        predictions : ndarray or None, default=None
            Optional precomputed predictions.

        Returns
        -------
        float
            RMSE value.
        """
        # Se previsões não forem dadas, obtê-las chamando predict
        if predictions is None:
            predictions = self.predict(dataset)
        # Calcular RMSE entre y real e y previsto
        return rmse(dataset.y, predictions)
