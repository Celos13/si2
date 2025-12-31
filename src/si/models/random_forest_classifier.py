from typing import List, Tuple
import numpy as np
from ..base.model import Model                # Classe base Model
from ..data.dataset import Dataset            # Classe Dataset
from ..metrics.accuracy import accuracy       # Métrica accuracy
from .decision_tree_classifier import DecisionTreeClassifier  # Árvores base


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
        super().__init__()                  # Inicializa Model
        self.n_estimators = n_estimators    # Nº de árvores na floresta
        self.max_features = max_features    # Nº de features aleatórias por árvore
        self.min_sample_split = min_sample_split  # min_samples_split nas árvores
        self.max_depth = max_depth          # profundidade máxima das árvores
        self.mode = mode                    # critério de impureza ("gini"/"entropy")
        self.seed = seed                    # seed para RNG
        # Lista de tuplos (indices_de_features, árvore_treinada)
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
        # Gerador de números aleatórios (reprodutível se seed não for None)
        rng = np.random.default_rng(self.seed)
        # Extrair X e y
        X, y = dataset.X, dataset.y
        # nº de amostras e nº de features
        n_samples, n_features = X.shape
        # Se max_features for None, usar todas as features
        max_features = self.max_features or n_features

        # Resetar lista de árvores se o modelo for treinado de novo
        self.trees = []

        # Para cada árvore a criar
        for _ in range(self.n_estimators):
            # 1) Amostra bootstrap de índices de linhas (com reposição)
            boot_idx = rng.integers(0, n_samples, size=n_samples)
            # Subconjunto X_boot e y_boot
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]

            # 2) Escolha aleatória de subset de features (sem reposição)
            feat_idx = rng.choice(n_features, size=max_features, replace=False)
            # Subconjunto de X_boot apenas com essas features
            X_boot_sub = X_boot[:, feat_idx]

            # 3) Construir Dataset para esta árvore
            boot_dataset = Dataset(
                X=X_boot_sub,
                y=y_boot,
                features=[dataset.features[i] for i in feat_idx],
                label=dataset.label
            )

            # 4) Determinar profundidade máxima para a árvore
            tree_max_depth = self.max_depth if self.max_depth is not None else 1000

            # 5) Criar e treinar DecisionTreeClassifier
            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=tree_max_depth,
                mode=self.mode
            )
            tree.fit(boot_dataset)

            # 6) Guardar par (features usadas, árvore treinada)
            self.trees.append((feat_idx, tree))

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
        # Matriz de features do dataset de teste
        X = dataset.X
        # Nº de amostras de teste
        n_samples = X.shape[0]

        # Matriz para guardar previsões de todas as árvores
        # shape = (n_estimators, n_samples)
        all_preds = np.zeros((self.n_estimators, n_samples), dtype=object)

        # Para cada árvore, fazer previsões
        for i, (feat_idx, tree) in enumerate(self.trees):
            # Selecionar as mesmas features usadas no treino da árvore
            X_sub = X[:, feat_idx]
            # Construir Dataset só com essas colunas
            ds_sub = Dataset(
                X=X_sub,
                y=None,
                features=[dataset.features[j] for j in feat_idx],
                label=dataset.label
            )
            # Prever com a árvore i-ésima
            all_preds[i] = tree.predict(ds_sub)

        # Agora aplicar votação maioritária por amostra
        y_pred = []
        for j in range(n_samples):
            # all_preds[:, j] = previsões de todas as árvores para amostra j
            values, counts = np.unique(all_preds[:, j], return_counts=True)
            # Escolher label com maior contagem
            y_pred.append(values[np.argmax(counts)])

        # Converter lista para array, usando o mesmo tipo de y original
        return np.array(y_pred, dtype=dataset.y.dtype)

    
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
        # Se não forem dadas previsões, calcular com predict
        if predictions is None:
            predictions = self.predict(dataset)
        # Accuracy entre y verdadeiro e previsto
        return accuracy(dataset.y, predictions)


