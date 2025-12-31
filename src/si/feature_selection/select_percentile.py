from typing import Callable  # Para anotar o tipo da função de scoring
import numpy as np

from ..base.transformer import Transformer  # Classe base de transformadores
from ..data.dataset import Dataset          # Classe Dataset
from ..statistics.f_classification import f_classification  # Exemplo de score_func


class SelectPercentile(Transformer):
    """
    Feature selection transformer that keeps the top percentile of features
    according to a scoring function (e.g., f_classification).
    """
    def __init__(self, score_func: Callable[[Dataset], tuple[np.ndarray, np.ndarray]], percentile: float = 10.0):
        """
        Parameters
        ----------
        score_func : callable
            Function that receives a Dataset and returns (F, p) arrays.
        percentile : float, default=10.0
            Percentage of features to keep, in the range (0, 100].
        """
        super().__init__()  # Inicializa Transformer (inclui lógica de fit/transform genérica)
        self.score_func = score_func        # Função de scoring (ex: f_classification)
        self.percentile = percentile        # Percentil de features a manter
        self.F: np.ndarray | None = None    # Scores F calculados no fit
        self.p: np.ndarray | None = None    # p-values (não usados diretamente na seleção)
        self.selected_features: np.ndarray | None = None  # Pode guardar índices

    def _fit(self, dataset: Dataset) -> "SelectPercentile":
        """
        Estimate the F and p values for each feature and select the
        top percentile according to F.

        Parameters
        ----------
        dataset : Dataset
            Input dataset.

        Returns
        -------
        SelectPercentile
            The fitted transformer.
        """
        # Chama a função de scoring que devolve (F, p) para cada feature
        self.F, self.p = self.score_func(dataset)
        # Aqui apenas guardas F e p; a seleção em si é feita em _transform
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset by selecting only the features chosen
        during fit.

        Parameters
        ----------
        dataset : Dataset
            Input dataset.

        Returns
        -------
        Dataset
            Dataset with reduced feature set.
        """
        # Nº total de features disponíveis
        n_features = dataset.X.shape[1]
        # Quantidade de features a manter, calculada a partir do percentil
        k = max(1, int(np.ceil(n_features * self.percentile / 100.0)))
        # Ordenar índices das features por F-descendente (scores maiores primeiro)
        order = np.argsort(self.F)[::-1]
        # Escolher top-k índices
        selected_idx = order[:k]
        # Ordenar os índices selecionados em ordem crescente (para manter ordem lógica de colunas)
        selected_idx = np.sort(selected_idx)
        # Selecionar colunas de X correspondentes às features escolhidas
        X_new = dataset.X[:, selected_idx]
        
        # Ajustar nomes das features, se existirem
        if dataset.features is not None:
            features_new = [dataset.features[i] for i in selected_idx]
        else:
            features_new = None

        # Criar novo Dataset com X reduzido mas mesma y e label
        return Dataset(X=X_new, y=dataset.y, features=features_new, label=dataset.label)
