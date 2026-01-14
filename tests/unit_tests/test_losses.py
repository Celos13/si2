import os
from unittest import TestCase
import numpy as np
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier
from datasets import DATASETS_PATH
from si.neural_networks.losses import BinaryCrossEntropy, MeanSquaredError
from si.neural_networks.losses import CategoricalCrossEntropy

class TestLosses(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_mean_squared_error_loss(self):

        error = MeanSquaredError().loss(self.dataset.y, self.dataset.y)

        self.assertEqual(error, 0)

    def test_mean_squared_error_derivative(self):

        derivative_error = MeanSquaredError().derivative(self.dataset.y, self.dataset.y)

        self.assertEqual(derivative_error.shape[0], self.dataset.shape()[0])

    def test_binary_cross_entropy_loss(self):

        error = BinaryCrossEntropy().loss(self.dataset.y, self.dataset.y)

        self.assertAlmostEqual(error, 0)

    def test_mean_squared_error_derivative(self):

        derivative_error = BinaryCrossEntropy().derivative(self.dataset.y, self.dataset.y)

        self.assertEqual(derivative_error.shape[0], self.dataset.shape()[0])

def softmax_stable(logits: np.ndarray) -> np.ndarray:
    """Softmax estável por linha."""
    x = logits - np.max(logits, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class TestCategoricalCrossEntropy(TestCase):
    def setUp(self):
        # dataset real do repositório
        iris_path = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        ds = read_data_file(filename=iris_path, label=True, sep=",")

        # labels podem ser strings (ex.: Iris-setosa). Fazemos one-hot.
        y = np.asarray(ds.y)
        classes = np.unique(y)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        y_idx = np.array([self.class_to_idx[v] for v in y], dtype=int)

        n_samples = y_idx.shape[0]
        n_classes = len(classes)

        y_true = np.zeros((n_samples, n_classes), dtype=float)
        y_true[np.arange(n_samples), y_idx] = 1.0

        self.y_true = y_true
        self.n_classes = n_classes

        # probabilidades simuladas (como se viessem de uma rede)
        rng = np.random.default_rng(42)
        logits = rng.normal(size=(n_samples, n_classes))
        self.y_pred = softmax_stable(logits)

        self.loss_fn = CategoricalCrossEntropy()

    def test_loss_matches_manual_computation(self):
        # Loss do nosso método
        L = self.loss_fn.loss(self.y_true, self.y_pred)

        # Cálculo manual
        eps = 1e-15
        y_pred_clipped = np.clip(self.y_pred, eps, 1.0 - eps)
        per_sample = -np.sum(self.y_true * np.log(y_pred_clipped), axis=1)
        L_manual = float(np.mean(per_sample))

        self.assertTrue(np.isclose(L, L_manual))

    def test_derivative_shape_and_finite(self):
        grad = self.loss_fn.derivative(self.y_true, self.y_pred)

        self.assertEqual(grad.shape, self.y_pred.shape)
        self.assertTrue(np.isfinite(grad).all())

    def test_derivative_zero_for_non_true_classes_when_one_hot(self):
        # Como y_true é one-hot, grad só deve ser !=0 na classe verdadeira
        grad = self.loss_fn.derivative(self.y_true, self.y_pred)

        non_true_mask = (self.y_true == 0.0)
        self.assertTrue(np.allclose(grad[non_true_mask], 0.0))