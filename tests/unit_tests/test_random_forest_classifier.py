import numpy as np
from si.data.dataset import Dataset
from si.models.random_forest_classifier import RandomForestClassifier


def test_random_forest_classifier_simple_separable_data():
    # Testa se o RandomForestClassifier consegue aprender bem um problema linearmente separável simples

    rng = np.random.default_rng(42)  # RNG reprodutível

    # Gerar 50 pontos da classe 0 centrados em (-2, -2) com algum ruído
    X_class0 = rng.normal(loc=-2.0, scale=0.5, size=(50, 2))
    # Gerar 50 pontos da classe 1 centrados em (2, 2)
    X_class1 = rng.normal(loc=2.0, scale=0.5, size=(50, 2))

    # Concatenar as duas classes verticalmente
    X = np.vstack([X_class0, X_class1])
    # Criar vetor de labels: 50 zeros seguidos de 50 uns
    y = np.array([0] * 50 + [1] * 50)

    # Construir Dataset com 2 features e classes 0/1
    ds = Dataset(X=X, y=y, features=["f1", "f2"], label="class")

    # Criar RandomForest com 10 árvores, a usar 2 features, impureza Gini e seed fixa
    model = RandomForestClassifier(
        n_estimators=10,
        max_features=2,
        min_sample_split=2,
        max_depth=None,
        mode="gini",
        seed=42
    )

    # Treinar o modelo no próprio dataset (sem split aqui)
    model.fit(ds)
    # Calcular accuracy no mesmo dataset (treino)
    score = model.score(ds)

    # Verificar que a accuracy é > 0.9 (modelo aprende bem o problema)
    assert score > 0.9

