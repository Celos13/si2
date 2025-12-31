from typing import Tuple  # Usado para anotar o tipo de retorno (train, test)
import numpy as np        # Biblioteca numérica usada para aleatorizar e indexar
from si.data.dataset import Dataset  # Classe Dataset que representa o conjunto de dados


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # Fixar a seed do gerador de números aleatórios para ter resultados reprodutíveis
    np.random.seed(random_state)

    # Número total de amostras no dataset (n_samples, n_features) → queremos n_samples
    n_samples = dataset.shape()[0]

    # Número de amostras que vão para o conjunto de teste
    n_test = int(n_samples * test_size)

    # Gera uma permutação aleatória dos índices [0, 1, ..., n_samples - 1]
    permutations = np.random.permutation(n_samples)

    # Primeiros n_test índices vão para o conjunto de teste
    test_idxs = permutations[:n_test]

    # O resto dos índices vai para o conjunto de treino
    train_idxs = permutations[n_test:]

    # Construir o Dataset de treino com as linhas de X e y correspondentes a train_idxs
    train = Dataset(
        dataset.X[train_idxs],        # submatriz X de treino
        dataset.y[train_idxs],        # vetor y de treino
        features=dataset.features,    # mantemos os mesmos nomes de features
        label=dataset.label           # e o mesmo nome de label
    )

    # Construir o Dataset de teste com as linhas de X e y correspondentes a test_idxs
    test = Dataset(
        dataset.X[test_idxs],         # submatriz X de teste
        dataset.y[test_idxs],         # vetor y de teste
        features=dataset.features,
        label=dataset.label
    )

    # Devolver os dois conjuntos: treino e teste
    return train, test


def stratified_train_test_split(
    dataset: Dataset,
    test_size: float = 0.2,
    random_state: int | None = None
) -> Tuple[Dataset, Dataset]:
    """
    Split a labeled dataset into train and test sets while preserving
    the class distribution (stratified split).

    Parameters
    ----------
    dataset : Dataset
        Labeled dataset (y must not be None).
    test_size : float, default=0.2
        Proportion of samples to include in the test split (0 < test_size < 1).
    random_state : int or None, default=None
        Seed for the random number generator.

    Returns
    -------
    (Dataset, Dataset)
        The train and test Dataset objects.
    """
    # Verificar que o dataset tem labels; sem y não é possível estratificar
    if dataset.y is None:
        raise ValueError("stratified_train_test_split requires labeled dataset (y is None).")

    # Validar que test_size está no intervalo (0,1)
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1.")

    # Extrair matriz de features e vetor de labels
    X = dataset.X
    y = dataset.y
    # Número total de amostras
    n_samples = X.shape[0]

    # Criar gerador de números aleatórios; se random_state for None, não é fixo
    rng = np.random.default_rng(random_state)

    # unique_classes: valores únicos de y
    # y_indices: índice da classe (0,1,2,...) para cada amostra
    unique_classes, y_indices = np.unique(y, return_inverse=True)

    # Listas onde vamos acumular os índices para treino e teste
    train_indices = []
    test_indices = []

    # Percorrer cada classe distinta
    for class_idx, class_label in enumerate(unique_classes):
        # Máscara booleana: True nas amostras pertencentes a esta classe
        class_mask = (y_indices == class_idx)
        # Índices das amostras desta classe
        class_indices = np.where(class_mask)[0]

        # Número de amostras desta classe
        n_class = class_indices.shape[0]
        if n_class == 0:
            # Se não houver amostras desta classe (caso extremo), saltar
            continue

        # Número de amostras desta classe que vão para o teste (floor para inteiro)
        n_test_class = int(np.floor(test_size * n_class))
        # Se test_size for pequeno mas a classe tiver mais que 1 amostra, garantir pelo menos 1 no teste
        if n_test_class == 0 and n_class > 1:
            n_test_class = 1

        # Permutar aleatoriamente os índices desta classe
        perm = rng.permutation(class_indices)

        # Primeiros n_test_class índices permutados → conjunto de teste
        test_class_idx = perm[:n_test_class]
        # Restantes índices → conjunto de treino
        train_class_idx = perm[n_test_class:]

        # Acumular esses índices nas listas globais
        test_indices.append(test_class_idx)
        train_indices.append(train_class_idx)

    # Se nenhuma amostra foi colocada em test_indices, algo está errado (test_size muito pequeno ou dados estranhos)
    if len(test_indices) == 0:
        raise RuntimeError("No samples were assigned to test set. Adjust test_size or data.")

    # Concatenar listas de arrays num único array para treino e teste
    test_indices = np.concatenate(test_indices)
    train_indices = np.concatenate(train_indices)

    # Embaralhar final dos índices para não manter ordem por classe
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    # Criar subconjuntos X_train, y_train, X_test, y_test usando os índices calculados
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # Preservar nomes de features e label do dataset original
    features = dataset.features
    label = dataset.label

    # Criar Dataset de treino com os dados estratificados
    train_dataset = Dataset(X=X_train, y=y_train, features=features, label=label)
    # Criar Dataset de teste com os dados estratificados
    test_dataset = Dataset(X=X_test, y=y_test, features=features, label=label)

    # Devolver (treino, teste)
    return train_dataset, test_dataset
