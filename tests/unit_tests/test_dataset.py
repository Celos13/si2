import unittest  # Módulo de testes unitários da biblioteca standard

import numpy as np  # Usado para criar arrays numéricos

from si.data.dataset import Dataset  # Classe Dataset que queremos testar


class TestDataset(unittest.TestCase):  # Classe de testes que herda de TestCase

    def test_dataset_construction(self):
        # Testa a construção básica de um Dataset e os métodos de estatística

        X = np.array([[1, 2, 3], [4, 5, 6]])  # Matriz de features 2x3
        y = np.array([1, 2])                  # Vetor de labels com 2 amostras

        features = np.array(['a', 'b', 'c'])  # Nomes das features
        label = 'y'                            # Nome da label
        dataset = Dataset(X, y, features, label)  # Criação do Dataset

        # Verifica se a média da primeira coluna é 2.5
        self.assertEqual(2.5, dataset.get_mean()[0])
        # Verifica se o shape do Dataset é (2,3)
        self.assertEqual((2, 3), dataset.shape())
        # Verifica se o dataset tem label (y não é None)
        self.assertTrue(dataset.has_label())
        # Verifica se a primeira classe em get_classes() é 1
        self.assertEqual(1, dataset.get_classes()[0])
        # Verifica variância da primeira coluna (valores 1 e 4 → var=2.25)
        self.assertEqual(2.25, dataset.get_variance()[0])
        # Verifica mínimo da primeira coluna (1)
        self.assertEqual(1, dataset.get_min()[0])
        # Verifica máximo da primeira coluna (4)
        self.assertEqual(4, dataset.get_max()[0])
        # Verifica que no summary() a primeira célula (mean da 1ª feature) é 2.5
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        # Testa o construtor auxiliar Dataset.from_random

        dataset = Dataset.from_random(
            10,       # n_samples
            5,        # n_features
            3,        # n_classes
            features=['a', 'b', 'c', 'd', 'e'],  # nomes das 5 features
            label='y'                            # nome da label
        )
        # Verifica que o shape é (10, 5)
        self.assertEqual((10, 5), dataset.shape())
        # Verifica que o dataset tem label (y não é None)
        self.assertTrue(dataset.has_label())
