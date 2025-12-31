import numpy as np
from si.data.dataset import Dataset          # Dataset a usar no teste
from si.decomposition.pca import PCA         # PCA que implementaste


def test_pca_reduces_dimension_and_orders_variance():
    # Testa se o PCA reduz a dimensionalidade corretamente e ordena pela variância

    rng = np.random.default_rng(42)  # RNG reprodutível
    X = rng.normal(size=(100, 5))    # Dados aleatórios 100x5
    # Criar Dataset sem labels, com 5 features nomeadas f0...f4
    ds = Dataset(X=X, y=None, features=[f"f{i}" for i in range(5)], label=None)

    # PCA para reduzir de 5 para 2 componentes principais
    pca = PCA(n_components=2)
    pca.fit(ds)              # Ajusta o PCA (calcula médias, covariância, autovalores, etc.)
    ds_pca = pca.transform(ds)  # Aplica a transformação ao dataset

    # Verifica se a nova X tem shape (100, 2)
    assert ds_pca.X.shape == (100, 2)
    # Verifica se há 2 nomes de features (PC1, PC2)
    assert len(ds_pca.features) == 2

    # Obtém o vetor de variância explicada pelo PCA
    ev = pca.explained_variance_
    # Deve ter shape (2,) já que temos 2 componentes
    assert ev.shape == (2,)
    # A primeira componente deve ter variância ≥ que a segunda
    assert ev[0] >= ev[1]

    # Cálculo manual da covariância e autovalores para conferir com o PCA
    Xc = X - X.mean(axis=0)           # centrar X manualmente
    cov = np.cov(Xc, rowvar=False)    # matriz de covariância
    eigvals, _ = np.linalg.eigh(cov)  # autovalores (e autovetores não usados)
    eigvals = np.sort(eigvals)[::-1][:2]  # ordenar desc e ficar com os 2 maiores

    # Verifica que a explained_variance_ coincide com estes 2 autovalores
    assert np.allclose(ev, eigvals)
