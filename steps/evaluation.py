import logging

import mlflow
import numpy as np
import pandas as pd

from sklearn.base import ClassifierMixin
from zenml import step
#from src.evaluation import AUC_ROC, ConfusionMatrix
from src.evaluation import FScore, Accuracy
from typing_extensions import Annotated


@step()
def evaluate_model(model: ClassifierMixin,
                   X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_train: np.ndarray,
                   y_test: np.ndarray,
                   ) -> Annotated[float, "fscore"]:
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

        '''confusion_class = ConfusionMatrix()
        confusion = confusion_class.calculate_scores(y_test, pred_label_test)
        mlflow.log_metric("Confusion Matrix", confusion)

        auc_roc_class = AUC_ROC()
        auc_roc = auc_roc_class.calculate_scores(y_train, y_test, y_pred_train, y_pred_test)
        mlflow.log_metric("AUC ROC", auc_roc)'''
        return fscore

    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e
