import logging

import mlflow
import numpy as np
import pandas as pd

from sklearn.base import ClassifierMixin
from zenml import step
from src.evaluation import AUC, ConfusionMatrix
from src.evaluation import FScore, Accuracy
from typing import Tuple
from typing_extensions import Annotated

@step()
def evaluate_model(model: ClassifierMixin,
                   X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_train: np.ndarray,
                   y_test: np.ndarray,
                   ) -> Tuple[Annotated[float, "accuracy"],
                              Annotated[float, "fscore"],
                              Annotated[float, "auc"]]:
    """
    Evaluates the model on the ingested data
    Args:
        df: the ingested data
    """
    try:
        # predicted value of y probabilities
        y_pred_train = model.predict_proba(X_train)
        y_pred_test = model.predict_proba(X_test)

        # predicted values of Y labels
        pred_label_train = model.predict(X_train)
        pred_label_test = model.predict(X_test)

        accuracy_class = Accuracy()
        accuracy = accuracy_class.calculate_scores(y_train, y_test, pred_label_train, pred_label_test)
        mlflow.log_metric("Accuracy", accuracy)

        fscore_class = FScore()
        fscore = fscore_class.calculate_scores(y_train, y_test, pred_label_train, pred_label_test)
        mlflow.log_metric("F1 Score", fscore)

        confusion_class = ConfusionMatrix()
        confusion = confusion_class.calculate_scores(y_test, pred_label_test)
        mlflow.log_param("Confusion Matrix", str(confusion))

        auc_class = AUC()
        auc = auc_class.calculate_scores(y_train, y_test, y_pred_train, y_pred_test)
        mlflow.log_metric("ROC", auc)
        return accuracy, fscore, auc

    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e
