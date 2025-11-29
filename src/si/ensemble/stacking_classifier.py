import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier(Model):
    """
    Stacking Classifier Ensemble:
    - models: lista de modelos base
    - final_model: modelo final treinado sobre as previsões dos modelos base
    """

    def __init__(self, models, final_model):
        super().__init__()
        self.models = models
        self.final_model = final_model

        # parâmetros estimados
        self.trained_models = None

    def _fit(self, dataset: Dataset):
        """
        1. Treinar os modelos base
        2. Obter previsões dos modelos base → novo dataset
        3. Treinar o modelo final com essas previsões
        """
        X = dataset.X
        y = dataset.y
        n_samples = X.shape[0]

        # 1) treinar modelos base
        self.trained_models = []
        base_predictions = []

        for model in self.models:
            model.fit(dataset)
            self.trained_models.append(model)

            # previsões deste modelo base
            preds = model.predict(dataset)
            base_predictions.append(preds)

        # 2) empilhar previsões (colunas)
        base_pred_matrix = np.column_stack(base_predictions)

        # criar dataset para treinar o final_model
        meta_dataset = Dataset(
            X=base_pred_matrix,
            y=y,
            features=[f"model_{i}_pred" for i in range(len(self.models))]
        )

        # 3) treinar modelo final
        self.final_model.fit(meta_dataset)

        return self

    def _predict(self, dataset: Dataset):
        """
        1. Obter previsões dos modelos base
        2. Criar dataset meta e prever com final_model
        """
        base_predictions = []

        for model in self.trained_models:
            preds = model.predict(dataset)
            base_predictions.append(preds)

        meta_X = np.column_stack(base_predictions)
        meta_dataset = Dataset(X=meta_X, y=None)

        return self.final_model.predict(meta_dataset)

    def _score(self, dataset: Dataset, predictions=None):
        """
        Accuracy dos valores reais vs preditos.
        """
        if predictions is None:
            predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)
