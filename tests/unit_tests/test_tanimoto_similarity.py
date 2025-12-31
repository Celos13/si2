import numpy as np  # Usado para criar arrays
from si.statistics.tanimoto_similarity import tanimoto_similarity  # Função a testar


def test_tanimoto_similarity_basic():
    # Teste básico com vetores binários simples

    x = np.array([1, 0, 1, 0], dtype=int)  # Vetor de referência

    y = np.array([
        [1, 0, 1, 0],  # igual a x  → sim ~ 1.0
        [1, 1, 0, 0],  # interseção=1, |x|=2, |y|=2  → 1/(2+2-1)=1/3
        [0, 0, 0, 0],  # todos zeros
    ], dtype=int)

    # Chamar a função tanimoto_similarity para comparar x com cada linha de y
    sims = tanimoto_similarity(x, y)

    # 1ª similaridade deve ser 1.0 (vetores iguais)
    assert np.isclose(sims[0], 1.0)
    # 2ª similaridade deve ser 1/3
    assert np.isclose(sims[1], 1/3)
    # 3ª similaridade esperada aqui (versão antiga do teste) é 0.0
    # (tu depois ajustaste a função para tratar ambos zeros como 1.0)
    assert np.isclose(sims[2], 0.0)


def test_tanimoto_raises_on_shape_mismatch():
    # Testa que a função lança ValueError se os shapes forem incompatíveis

    x = np.array([1, 0, 1], dtype=int)          # 3 features
    y = np.array([[1, 0, 1, 0]], dtype=int)     # 4 features → incompatível

    try:
        tanimoto_similarity(x, y)  # Deve lançar ValueError
    except ValueError:
        assert True  # Se cair aqui, o teste passa
    else:
        # Se não lançar ValueError, o teste falha
        assert False, "Expected ValueError for incompatible shapes"


def test_tanimoto_both_zero_vectors():
    # Testa o caso especial em que ambos os vetores são todos zeros

    x = np.array([0, 0, 0, 0], dtype=int)
    y = np.array([[0, 0, 0, 0]], dtype=int)

    sims = tanimoto_similarity(x, y)

    # Define-se que, se ambos forem zero, a similaridade é 1.0
    assert np.isclose(sims[0], 1.0)
