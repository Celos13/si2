import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH
from si.io.data_file import read_data_file

from si.model_selection.randomized_search import randomized_search_cv
from si.models.logistic_regression import LogisticRegression


class TestRandomizedSearchCV(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, "breast_bin", "breast-bin.csv")
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    def test_randomized_search_cv_runs_and_returns_expected_keys(self):
        model = LogisticRegression()

        param_dist = {
            "l2_penalty": np.linspace(1, 10, 10),
            "alpha": np.linspace(0.001, 0.0001, 100),
            "max_iter": np.linspace(1000, 2000, 200),
        }

        results = randomized_search_cv(
            model=model,
            dataset=self.dataset,
            hyperparameter_grid=param_dist,
            scoring=None,
            cv=3,
            n_iter=10,
            random_state=42,
        )

        # Verifica chaves obrigatórias do output (como no slide)
        self.assertIn("hyperparameters", results)
        self.assertIn("scores", results)
        self.assertIn("best_hyperparameters", results)
        self.assertIn("best_score", results)

        # Tamanho: n_iter resultados
        self.assertEqual(len(results["hyperparameters"]), 10)
        self.assertEqual(len(results["scores"]), 10)

        # Score válido
        best_score = results["best_score"]
        self.assertGreaterEqual(best_score, 0.0)
        self.assertLessEqual(best_score, 1.0)

        # Best hyperparameters tem as 3 chaves
        best_hparams = results["best_hyperparameters"]
        self.assertEqual(set(best_hparams.keys()), {"l2_penalty", "alpha", "max_iter"})
