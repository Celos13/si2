import numpy as np
from si.data.dataset import Dataset
from si.decomposition.pca import PCA


def test_pca_reduces_dimension_and_orders_variance():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 5))
    ds = Dataset(X=X, y=None, features=[f"f{i}" for i in range(5)], label=None)

    pca = PCA(n_components=2)
    pca.fit(ds)
    ds_pca = pca.transform(ds)

    assert ds_pca.X.shape == (100, 2)
    assert len(ds_pca.features) == 2

    ev = pca.explained_variance_
    assert ev.shape == (2,)
    assert ev[0] >= ev[1]

    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc, rowvar=False)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.sort(eigvals)[::-1][:2]
    assert np.allclose(ev, eigvals)
