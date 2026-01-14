from unittest import TestCase

from datasets import DATASETS_PATH

import os
import numpy as np

from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.neural_networks.activation import ReLUActivation, SigmoidActivation
from si.neural_networks.activation import TanhActivation, SoftmaxActivation


class TestSigmoidLayer(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):

        sigmoid_layer = SigmoidActivation()
        result = sigmoid_layer.activation_function(self.dataset.X)
        self.assertTrue(all([i >= 0 and i <= 1 for j in range(result.shape[1]) for i in result[:, j]]))


    def test_derivative(self):
        sigmoid_layer = SigmoidActivation()
        derivative = sigmoid_layer.derivative(self.dataset.X)
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])


class TestRELULayer(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):

        relu_layer = ReLUActivation()
        result = relu_layer.activation_function(self.dataset.X)
        self.assertTrue(all([i >= 0 for j in range(result.shape[1]) for i in result[:, j]]))


    def test_derivative(self):
        sigmoid_layer = ReLUActivation()
        derivative = sigmoid_layer.derivative(self.dataset.X)
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])

class TestActivations(TestCase):
    def setUp(self):
        # Usar dataset real do repositório (breast_bin)
        csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=csv_file, label=True, sep=",")

        # Usamos apenas X (features)
        self.X = self.dataset.X

    def test_tanh_forward_output_range_with_dataset(self):
        act = TanhActivation()

        act.set_input_shape((self.X.shape[1],))
        out = act.forward_propagation(self.X)

        # tanh(x) ∈ [-1, 1]
        self.assertTrue(np.all(out >= -1.0))
        self.assertTrue(np.all(out <= 1.0))

    def test_tanh_backward_shape_with_dataset(self):
        act = TanhActivation()

        act.set_input_shape((self.X.shape[1],))
        out = act.forward_propagation(self.X)

        error = np.ones_like(out)
        grad = act.backward_propagation(error)

        # gradiente deve ter a mesma forma que o input
        self.assertEqual(grad.shape, self.X.shape)

    def test_softmax_forward_probabilities_sum_to_one(self):
        act = SoftmaxActivation()

        act.set_input_shape((self.X.shape[1],))
        out = act.forward_propagation(self.X)

        # Cada linha deve somar 1 (distribuição de probabilidade)
        row_sums = np.sum(out, axis=1)
        self.assertTrue(np.allclose(row_sums, np.ones_like(row_sums)))

        # Valores entre 0 e 1
        self.assertTrue(np.all(out >= 0.0))
        self.assertTrue(np.all(out <= 1.0))

    def test_softmax_backward_shape_with_dataset(self):
        act = SoftmaxActivation()

        act.set_input_shape((self.X.shape[1],))
        out = act.forward_propagation(self.X)

        error = np.ones_like(out)
        grad = act.backward_propagation(error)

        # Gradiente com a mesma shape
        self.assertEqual(grad.shape, self.X.shape)
