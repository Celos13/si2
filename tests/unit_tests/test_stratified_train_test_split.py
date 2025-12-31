import numpy as np
from si.data.dataset import Dataset
from si.model_selection.split import stratified_train_test_split


def test_stratified_split_preserves_class_proportions():
    # Testa se o split estratificado preserva aproximadamente as proporções de classe

    X = np.random.randn(100, 4)             # 100 amostras, 4 features
    y = np.array([0] * 80 + [1] * 20)       # 80% classe 0, 20% classe 1

    # Criar Dataset com essas features e labels
    ds = Dataset(X=X, y=y, features=[f"f{i}" for i in range(4)], label="class")

    # Fazer split estratificado com 25% dos dados para teste
    train, test = stratified_train_test_split(ds, test_size=0.25, random_state=42)

    # Contagens originais por classe
    orig_counts = np.bincount(y)
    orig_ratios = orig_counts / orig_counts.sum()  # proporções originais

    # Contagens no train
    train_counts = np.bincount(train.y.astype(int))
    train_ratios = train_counts / train_counts.sum()

    # Contagens no test
    test_counts = np.bincount(test.y.astype(int))
    test_ratios = test_counts / test_counts.sum()

    # Verificar que as proporções em train e test estão próximas das originais (tolerância 0.1)
    assert np.allclose(train_ratios, orig_ratios, atol=0.1)
    assert np.allclose(test_ratios, orig_ratios, atol=0.1)


def test_stratified_split_requires_labels():
    # Testa se a função lança erro quando não há labels (y=None)

    X = np.random.randn(10, 3)
    # Dataset com y=None e label=None
    ds = Dataset(X=X, y=None, features=["f1", "f2", "f3"], label=None)

    try:
        # Deve lançar ValueError porque não há labels
        stratified_train_test_split(ds, test_size=0.3)
    except ValueError:
        assert True  # Se lançar, o teste passa
    else:
        # Se não lançar erro, o teste falha
        assert False, "Expected ValueError when y is None"
