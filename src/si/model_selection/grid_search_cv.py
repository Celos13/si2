from unittest import TestCase  # Base para classes de teste

from datasets import DATASETS_PATH  # Caminho base para datasets do projeto

import os  # Usado para construir paths de ficheiros

from si.io.data_file import read_data_file           # Função para ler ficheiros CSV em Dataset
from si.metrics.accuracy import accuracy             # Métrica accuracy (nem é usada diretamente aqui)
from si.model_selection.grid_search_cv import grid_search_cv  # Função a testar
from si.models.logistic_regression import LogisticRegression  # Modelo base para o grid search

import numpy as np


class TestGridSearchCV(TestCase):
    # Classe de testes para a função grid_search_cv

    def setUp(self):
        # Método chamado antes de cada teste, prepara os dados

        # Caminho para o ficheiro CSV breast-bin.csv
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        # Lê o CSV para um Dataset (label=True, última coluna é a label)
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    def test_grid_search_k_fold_cross_validation(self):

        # Modelo base: regressão logística
        model = LogisticRegression()

        # Grelha de hiperparâmetros para explorar
        parameter_grid_ = {
            'l2_penalty': (1, 10),        # duas opções para λ
            'alpha': (0.001, 0.0001),     # duas opções para alpha (step do gradiente)
            'max_iter': (1000, 2000)      # duas opções para max_iter
        }
        # Total de combinações = 2 * 2 * 2 = 8

        # Faz grid search com k-fold cross-validation (cv=3)
        results_ = grid_search_cv(
            model,
            self.dataset,
            hyperparameter_grid=parameter_grid_,
            cv=3
        )

        # Verifica que há 8 scores (um por combinação de hiperparâmetros)
        self.assertEqual(len(results_["scores"]), 8)

        # Extrai o dicionário de melhores hiperparâmetros
        best_hyperparameters = results_['best_hyperparameters']
        # Deve ter 3 chaves: l2_penalty, alpha, max_iter
        self.assertEqual(len(best_hyperparameters), 3)

        # Extrai o melhor score médio
        best_score = results_['best_score']
        # Verifica que o melhor score (arredondado a 2 casas) é 0.97
        self.assertEqual(np.round(best_score, 2), 0.97)
