import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier(Model):
    """
    Stacking classifier (ensemble).

    Trains a set of base models, uses their predictions as meta-features,
    and then trains a final model on those meta-features.
    """

    def __init__(self, models: list[Model], final_model: Model):
        """
        Parameters
        ----------
        models : list[Model]
            Base models used to generate meta-features.
        final_model : Model
            Model trained on meta-features to produce final predictions.
        """
        super().__init__()
        self.models = models
        self.final_model = final_model

    def _fit(self, dataset: Dataset):
        """
        Algorithm (from slides):
        1) Train the initial set of models
        2) Get predictions from the initial set of models
        3) Train the final model with the predictions of the initial set of models
        4) Return self
        """
        # 1) train base models
        for m in self.models:
            m.fit(dataset)

        # 2) build meta-features from base predictions
        meta_X = self._base_predictions_as_features(dataset)

        # 3) train final model on meta-features
        meta_ds = Dataset(
            X=meta_X,
            y=dataset.y,
            features=[f"m{i}" for i in range(meta_X.shape[1])],
            label=dataset.label
        )
        self.final_model.fit(meta_ds)

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Algorithm (from slides):
        1) Get predictions from the initial set of models
        2) Get final predictions using the final model and the predictions as input
        """
        meta_X = self._base_predictions_as_features(dataset)

        meta_ds = Dataset(
            X=meta_X,
            y=None,
            features=[f"m{i}" for i in range(meta_X.shape[1])],
            label=None
        )
        return self.final_model.predict(meta_ds)

    def _score(self, dataset: Dataset, predictions=None) -> float:
        """
        From slides:
        1) Get predictions using predict
        2) Compute accuracy between predicted and real labels
        """
        if predictions is None:
            predictions = self.predict(dataset)
        return float(accuracy(dataset.y, predictions))

    def _base_predictions_as_features(self, dataset: Dataset) -> np.ndarray:
        """
        Helper: collect predictions from each base model and stack them
        as columns -> meta-feature matrix shape (n_samples, n_models).
        """
        preds = []
        for m in self.models:
            p = m.predict(dataset)

            # garantir formato (n_samples,)
            p = np.asarray(p).reshape(-1)

            # transformar classes (possivelmente strings) em números se necessário
            # (muitos datasets aqui usam 0/1, por isso normalmente já está ok)
            preds.append(p)

        # empilhar colunas: (n_models, n_samples) -> (n_samples, n_models)
        meta_X = np.vstack(preds).T.astype(float)
        return meta_X

