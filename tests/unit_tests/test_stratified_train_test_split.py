import numpy as np
from si.data.dataset import Dataset
from si.model_selection.split import stratified_train_test_split


def test_stratified_split_preserves_class_proportions():
    # criar dataset simples com desbalanceamento
    X = np.random.randn(100, 4)
    y = np.array([0] * 80 + [1] * 20)  # 80% da classe 0, 20% da classe 1

    ds = Dataset(X=X, y=y, features=[f"f{i}" for i in range(4)], label="class")

    train, test = stratified_train_test_split(ds, test_size=0.25, random_state=42)

    # proporções originais
    orig_counts = np.bincount(y)
    orig_ratios = orig_counts / orig_counts.sum()

    # proporções em train
    train_counts = np.bincount(train.y.astype(int))
    train_ratios = train_counts / train_counts.sum()

    # proporções em test
    test_counts = np.bincount(test.y.astype(int))
    test_ratios = test_counts / test_counts.sum()

    # verificar que as proporções estão "próximas"
    assert np.allclose(train_ratios, orig_ratios, atol=0.1)
    assert np.allclose(test_ratios, orig_ratios, atol=0.1)

    # não deve haver interseção entre índices de train e test
    # (comparando via conteúdos X pode ser mais chato, isto é opcional)


def test_stratified_split_requires_labels():
    X = np.random.randn(10, 3)
    ds = Dataset(X=X, y=None, features=["f1", "f2", "f3"], label=None)

    try:
        stratified_train_test_split(ds, test_size=0.3)
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError when y is None"
