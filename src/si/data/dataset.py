from typing import Tuple, Sequence, Iterable  # Tipos auxiliares para anotações

import numpy as np  # Biblioteca numérica principal
import pandas as pd  # Usado para integração com DataFrames (summary, to_dataframe, etc.)


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None) -> None:
        #Define a classe principal usada em TODO o package.
        #X: matriz de features (obrigatório).
        #y: labels (opcional).
        #features: nomes das colunas, ex.: ["sepal_length", ...].
        #label: nome da coluna alvo.
        """
        Dataset represents a tabular dataset for single output classification.

        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        y: numpy.ndarray (n_samples, 1)
            The label vector
        features: list of str (n_features)
            The feature names
        label: str (1)
            The label name
        """
        # Validar que X não é None
        if X is None:
            raise ValueError("X cannot be None")
        # Se y foi passado, verificar que tem o mesmo nº de linhas que X
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        # Se features foi passada, garantir que o nº de colunas de X coincide com o nº de nomes
        if features is not None and len(X[0]) != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        # Se não vierem nomes de features, criar nomes genéricos feat_0, feat_1, ...
        if features is None:
            features = [f"feat_{str(i)}" for i in range(X.shape[1])]
        # Se y foi passado mas label não, definir um nome padrão "y"
        if y is not None and label is None:
            label = "y"
        # Guardar atributos principais no objeto
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset
        Returns
        -------
        tuple (n_samples, n_features)
        """
        #Devolve (n_samples, n_features)
        return self.X.shape

    def has_label(self) -> bool:
        """
        Returns True if the dataset has a label
        Returns
        -------
        bool
        """
        # Verifica se y não é None
        return self.y is not None

    def get_classes(self) -> np.ndarray:
        """
        Returns the unique classes in the dataset
        Returns
        -------
        numpy.ndarray (n_classes)
        """
        # Só faz sentido se existir y
        if self.has_label():
            # np.unique devolve as classes únicas de y
            return np.unique(self.y)
        else:
            raise ValueError("Dataset does not have a label")

    def get_mean(self) -> np.ndarray:
        """
        Returns the mean of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        # np.nanmean ignora NaNs ao calcular média por coluna
        return np.nanmean(self.X, axis=0)

    def get_variance(self) -> np.ndarray:
        """
        Returns the variance of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        # Variância por coluna, ignorando NaNs
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        """
        Returns the median of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        # Mediana por coluna, ignorando NaNs
        return np.nanmedian(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        """
        Returns the minimum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        # Mínimo por coluna, ignorando NaNs
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """
        Returns the maximum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        # Máximo por coluna, ignorando NaNs
        return np.nanmax(self.X, axis=0)

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the dataset
        Returns
        -------
        pandas.DataFrame (n_features, 5)
        """
        # Construir dicionário com estatísticas básicas por feature
        data = {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": self.get_min(),
            "max": self.get_max(),
            "var": self.get_variance()
        }
        # Cada linha será uma métrica, colunas são as features
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        # Se for passada label, separar X e y usando o nome da coluna
        if label:
            X = df.drop(label, axis=1).to_numpy()  # todas as colunas exceto label
            y = df[label].to_numpy()               # coluna alvo
        else:
            # Caso não haja label, tudo é X e y fica None
            X = df.to_numpy()
            y = None

        # Nomes das colunas do DataFrame tornam-se features
        features = df.columns.tolist()
        # Criar instância de Dataset com esses dados
        return cls(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        # Se não houver y, devolve DataFrame só com features
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            # Caso contrário, cria dataframe com X e adiciona coluna label com y
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df

    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data

        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        # X aleatório contínuo uniforme em [0,1]
        X = np.random.rand(n_samples, n_features)
        # y aleatório inteiro em [0, n_classes)
        y = np.random.randint(0, n_classes, n_samples)
        # Criar Dataset a partir destes dados
        return cls(X, y, features=features, label=label)
    
    def dropna(self) -> "Dataset":
        """
        Remove all samples (rows) that contain at least one NaN value
        in any feature.

        This method updates X (and y if present) in-place and returns
        the modified Dataset instance.

        Returns
        -------
        Dataset
            The Dataset instance without any rows containing NaN values.
        """
        # Cria máscara booleana True para linhas que NÃO têm nenhum NaN
        mask = ~np.any(np.isnan(self.X), axis=1)
        # Mantém apenas essas linhas em X
        self.X = self.X[mask]
        # Se existir y, também remover entradas correspondentes
        if self.y is not None:
            self.y = self.y[mask]
        return self  # permite chaining

    def fillna(self, value: float | str = "mean") -> "Dataset":
        """
        Replace NaN values in X with a given value or with the column mean.

        Parameters
        ----------
        value : float or {"mean"}, default="mean"
            If a float, all NaN entries are replaced by this value.
            If "mean", NaN entries in each feature are replaced by that
            feature's mean.

        Returns
        -------
        Dataset
            The Dataset instance with NaN values filled.
        """
        # Garantir que X é float (para permitir NaN e operações numéricas)
        X = self.X.astype(float).copy()

        # Escolher vetor de valores de preenchimento por coluna
        if isinstance(value, (float, int)):
            # Valor fixo igual para todas as colunas
            fill_vals = np.full(X.shape[1], float(value))
        elif value == "mean":
            # Média de cada coluna, ignorando NaNs
            fill_vals = np.nanmean(X, axis=0)
        elif value == "median":
            # Mediana de cada coluna, ignorando NaNs
            fill_vals = np.nanmedian(X, axis=0)
        else:
            # Caso não seja nenhum dos formatos suportados
            raise ValueError("value must be a float, 'mean' or 'median'")

        # Indices (linhas, colunas) onde X é NaN
        inds = np.where(np.isnan(X))
        # Substituir pelos valores de fill_vals para a coluna correspondente
        X[inds] = np.take(fill_vals, inds[1])
        # Atualizar X no Dataset
        self.X = X
        return self

    def remove_by_index(self, indices: list[int]) -> "Dataset":
        """
        Remove rows at the given indices from the dataset.

        Parameters
        ----------
        indices : list[int]
            Indices of the rows to remove.

        Returns
        -------
        Dataset
            The dataset with the selected rows removed.
        """
        # Converter indices para array NumPy para usar operações vectorizadas
        indices = np.array(indices)

        # Verificar se algum índice está fora dos limites [0, n_samples)
        if np.any(indices < 0) or np.any(indices >= self.X.shape[0]):
            raise IndexError("One or more indices are out of range.")

        # Criar máscara booleana True para todas as linhas
        mask = np.ones(self.X.shape[0], dtype=bool)
        # Colocar False nas posições dos índices a remover
        mask[indices] = False

        # Manter apenas as linhas onde mask é True
        self.X = self.X[mask]
        if self.y is not None:
            self.y = self.y[mask]

        return self


if __name__ == '__main__':
    # Pequeno teste manual quando o ficheiro é executado diretamente
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    features = np.array(['a', 'b', 'c'])
    label = 'y'
    dataset = Dataset(X, y, features, label)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.get_min())
    print(dataset.get_max())
    print(dataset.summary())
