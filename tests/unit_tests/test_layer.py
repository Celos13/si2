from unittest import TestCase
import numpy as np

from si.neural_networks.layers import Dropout


class TestDropoutLayer(TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X = np.random.randn(10, 6)

    def test_dropout_inference_returns_same_input(self):
        drop = Dropout(probability=0.5)

        out = drop.forward_propagation(self.X, training=False)

        # Em inferência, dropout não altera o input
        self.assertTrue(np.allclose(out, self.X))
        # Se o teu Dropout define mask=None em inferência, isto também deve ser verdade
        self.assertIsNone(drop.mask)

    def test_dropout_training_drops_values_and_scales(self):
        # Input de 1s -> com p=0.5, output deve ser 0 ou 2 (porque scale=1/(1-0.5)=2)
        X = np.ones((100, 10))
        drop = Dropout(probability=0.5)

        out = drop.forward_propagation(X, training=True)

        unique_vals = set(np.unique(out).tolist())

        # Deve conter zeros (unidades "dropadas")
        self.assertIn(0.0, unique_vals)
        # Deve conter valores escalados (2.0)
        self.assertIn(2.0, unique_vals)

        # A máscara deve ter a mesma shape do input
        self.assertEqual(drop.mask.shape, X.shape)

    def test_dropout_backward_propagation_masks_gradient(self):
        # Faz forward em treino para gerar mask
        drop = Dropout(probability=0.5)
        _ = drop.forward_propagation(self.X, training=True)

        # Se output_error for tudo 1s, o gradiente de entrada deve ser a mask (0 ou 1)
        output_error = np.ones_like(self.X)
        input_error = drop.backward_propagation(output_error)

        self.assertTrue(np.array_equal(input_error, drop.mask))

    def test_dropout_has_no_parameters(self):
        drop = Dropout(probability=0.5)
        self.assertEqual(drop.parameters(), 0)

    def test_dropout_output_shape_same_as_input(self):
        drop = Dropout(probability=0.5)

        # Alguns projetos usam set_input_shape() antes do forward
        drop.set_input_shape((6,))
        self.assertEqual(drop.output_shape(), (6,))
