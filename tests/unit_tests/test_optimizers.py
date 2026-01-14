import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH
from si.io.data_file import read_data_file
from si.neural_networks.optimizers import Adam


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))


class TestAdamOptimizerWithBreastBin(TestCase):
    def setUp(self):
        # Load breast-bin dataset from repo
        csv_file = os.path.join(DATASETS_PATH, "breast_bin", "breast-bin.csv")
        ds = read_data_file(filename=csv_file, label=True, sep=",")

        X = np.asarray(ds.X, dtype=float)
        y = np.asarray(ds.y)

        # garantir y em {0,1} float
        # (se já vier 0/1, isto não muda nada)
        if y.dtype.kind in ("U", "S", "O"):
            # fallback: map labels to 0/1 if needed
            classes = np.unique(y)
            mapping = {classes[0]: 0.0, classes[-1]: 1.0}
            y = np.array([mapping[v] for v in y], dtype=float)
        else:
            y = y.astype(float)

        # Pequeno subset determinístico para teste rápido e estável
        # (evita testes lentos e instabilidade numérica)
        n = min(80, X.shape[0])
        self.X = X[:n]
        self.y = y[:n]

        # inicialização determinística de pesos
        rng = np.random.default_rng(42)
        self.w0 = rng.normal(scale=0.01, size=self.X.shape[1]).astype(float)
        self.b0 = 0.0

    def test_adam_decreases_logistic_loss_on_breast_bin(self):
        X, y = self.X, self.y
        n_samples = X.shape[0]

        # Um Adam por parâmetro (consistente com o teu framework "stateful per instance")
        opt_w = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        opt_b = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

        w = self.w0.copy()
        b = float(self.b0)

        # Loss inicial
        p = sigmoid(X @ w + b)
        loss0 = binary_cross_entropy(y, p)

        # Fazer algumas iterações (1 pode funcionar, mas 5 torna o teste mais robusto)
        for _ in range(5):
            p = sigmoid(X @ w + b)

            # gradiente da BCE + sigmoid (logistic regression):
            # dL/dz = (p - y)
            dz = (p - y)  # (n,)

            grad_w = (X.T @ dz) / n_samples  # (d,)
            grad_b = float(np.mean(dz))      # escalar

            # update com Adam
            w = opt_w.update(w, grad_w)
            b = float(opt_b.update(np.array(b), np.array(grad_b)))

        # Loss final
        p_new = sigmoid(X @ w + b)
        loss1 = binary_cross_entropy(y, p_new)

        # Deve diminuir
        self.assertLess(loss1, loss0)

    def test_adam_state_shapes_and_timestep(self):
        opt = Adam(learning_rate=0.01)

        w = np.zeros(5, dtype=float)
        g = np.ones(5, dtype=float)

        w1 = opt.update(w, g)
        self.assertEqual(opt.t, 1)
        self.assertIsNotNone(opt.m)
        self.assertIsNotNone(opt.v)
        self.assertEqual(opt.m.shape, w.shape)
        self.assertEqual(opt.v.shape, w.shape)
        self.assertEqual(w1.shape, w.shape)

        w2 = opt.update(w1, g)
        self.assertEqual(opt.t, 2)
        self.assertEqual(opt.m.shape, w.shape)
        self.assertEqual(opt.v.shape, w.shape)
