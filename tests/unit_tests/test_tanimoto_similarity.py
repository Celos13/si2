import numpy as np
from si.statistics.tanimoto_similarity import tanimoto_similarity


def test_tanimoto_similarity_basic():
    x = np.array([1, 0, 1, 0], dtype=int)

    y = np.array([
        [1, 0, 1, 0],  
        [1, 1, 0, 0],  
        [0, 0, 0, 0],  
    ], dtype=int)

    sims = tanimoto_similarity(x, y)

    assert np.isclose(sims[0], 1.0)
    assert np.isclose(sims[1], 1/3)
    assert np.isclose(sims[2], 0.0)



def test_tanimoto_raises_on_shape_mismatch():
    x = np.array([1, 0, 1], dtype=int)
    y = np.array([[1, 0, 1, 0]], dtype=int)  

    try:
        tanimoto_similarity(x, y)
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError for incompatible shapes"

def test_tanimoto_both_zero_vectors():
    x = np.array([0, 0, 0, 0], dtype=int)
    y = np.array([[0, 0, 0, 0]], dtype=int)

    sims = tanimoto_similarity(x, y)

    assert np.isclose(sims[0], 1.0)
