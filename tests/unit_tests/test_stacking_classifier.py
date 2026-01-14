import os
from unittest import TestCase
import unittest

from datasets import DATASETS_PATH

from si.ensemble.stacking_classifier import StackingClassifier
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split

from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression
from si.models.decision_tree_classifier import DecisionTreeClassifier


class TestStackingClassifier(unittest.TestCase):
    def setUp(self):
        # Caminho para o dataset pedido no enunciado
        self.csv_file = os.path.join(DATASETS_PATH, "breast_bin", "breast-bin.csv")

        # Ler dataset (última coluna como label)
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        # Split train/test como nos outros testes do projeto
        self.train_ds, self.test_ds = train_test_split(
            self.dataset, test_size=0.2, random_state=42
        )

    def test_stacking_classifier_breast_bin(self):
        # 1) Modelos base (como pedido no slide)
        base_models = [
            KNNClassifier(k=3),
            LogisticRegression(max_iter=2000),
            DecisionTreeClassifier(max_depth=5),
        ]

        # 2) Modelo final (também como pedido: outro KNN)
        final_model = KNNClassifier(k=3)

        # 3) Stacking
        model = StackingClassifier(models=base_models, final_model=final_model)

        # 4) Fit
        model.fit(self.train_ds)

        # 5) Score
        score = model.score(self.test_ds)

        # 6) Verificações (sanity checks)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Limite “razoável” para este dataset (ajusta se o professor exigir outro valor)
        self.assertGreater(score, 0.80)
