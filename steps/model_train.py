import logging

import mlflow
import numpy as np
import pandas as pd
from zenml import step

from src.model_dev import *
from sklearn.base import ClassifierMixin
from .config import ModelNameConfig
from typing import Union

@step()
def train_model(X_train: pd.DataFrame,
                y_train: np.ndarray,
                config: ModelNameConfig,
                ) -> Union[ClassifierMixin, None]:
    """
    Trains the model on the ingested data

    Args:
        X_train: pd.DataFrame,
        y_train: pd.DataFrame
    """
    try:
        model = None
        if config.model_name == "LogisticRegression":
            mlflow.sklearn.autolog()
            model = LogisticRegressionModel()
            trained_model = model.train(X_train, y_train)
        elif config.model_name == "NaiveBayes":
            mlflow.sklearn.autolog()
            model = NaiveBayesModel()
            trained_model = model.train(X_train, y_train)
        elif config.model_name == "NaiveBayes":
            mlflow.sklearn.autolog()
            model = NaiveBayesModel()
            trained_model = model.train(X_train, y_train)
        elif config.model_name == "DecisionTree":
            mlflow.sklearn.autolog()
            model = DecisionTreeModel()
            trained_model = model.train(X_train, y_train)
        elif config.model_name == "RandomForest":
            mlflow.sklearn.autolog()
            model = RandomForestModel()
            trained_model = model.train(X_train, y_train)
        elif config.model_name == "GradientBoosting":
            mlflow.sklearn.autolog()
            model = GradienBoostModel()
            trained_model = model.train(X_train, y_train)
            print("Training Done")
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
        return trained_model

    except Exception as e:
        logging.error("Error in training model {}".format(e))
        raise e
