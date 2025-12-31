import numpy as np
from ..base.transformer import Transformer  # Classe base de transformadores
from ..data.dataset import Dataset          # Classe Dataset


class PCA(Transformer):
    """
    Principal Component Analysis (PCA) transformer.

    Learns a linear projection that maps the original features into
    a lower-dimensional space maximizing variance.
    """
    def __init__(self, n_components: int):
        """
        Parameters
        ----------
        n_components : int
            Number of principal components to keep.
        """
        super().__init__()              # Inicializa Transformer
        self.n_components = n_components
        self.mean_: np.ndarray | None = None               # Média das features
        self.components_: np.ndarray | None = None         # Vetores próprios (componentes principais)
        self.explained_variance_: np.ndarray | None = None # Autovalores das componentes escolhidas

    def _fit(self, dataset: Dataset) -> "PCA":
        """
        Fit the PCA model to the dataset.

        Parameters
        ----------
        dataset : Dataset
            Input dataset.

        Returns
        -------
        PCA
            The fitted PCA transformer.
        """
        # Extrair matriz de features
        X = dataset.X

        # 1) Calcular média por coluna (feature)
        self.mean_ = X.mean(axis=0)
        # 2) Centrar dados: X_centered tem média zero em cada coluna
        X_centered = X - self.mean_

        # 3) Matriz de covariância, rowvar=False → cada coluna é uma variável
        cov = np.cov(X_centered, rowvar=False)

        # 4) Autovalores e autovetores da matriz de covariância
        eigvals, eigvecs = np.linalg.eigh(cov)

        # 5) Ordenar autovalores do maior para o menor
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        # Reordenar também os vetores próprios de acordo com a mesma ordem
        eigvecs = eigvecs[:, idx]

        # 6) Guardar as primeiras n_components colunas (componentes principais)
        self.components_ = eigvecs[:, :self.n_components]
        # 7) Guardar autovalores correspondentes (variância explicada)
        self.explained_variance_ = eigvals[:self.n_components]

        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Project the dataset onto the learned principal components.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        Dataset
            Dataset with transformed features (PC1, PC2, ...).
        """
        # Verificar se o modelo já foi ajustado
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA must be fitted before calling transform().")

        # X original
        X = dataset.X
        # Centrar usando a média calculada no fit
        X_centered = X - self.mean_

        # Projeção: X_new = X_centered * componentes
        X_new = np.dot(X_centered, self.components_)

        # Gerar nomes de features PC1, PC2, ...
        features = [f"PC{i+1}" for i in range(self.n_components)]

        # Devolver novo Dataset com features transformadas, mas igual y e label
        return Dataset(
            X=X_new,
            y=dataset.y,
            features=features,
            label=dataset.label
        )
