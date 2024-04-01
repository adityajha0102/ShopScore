import numpy as np
import pandas as pd
import json
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])


class DeploymentTriggerConfig(BaseParameters):
    """Deployment trigger config"""
    min_accuracy: float = 70.0


'''@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data'''


@step
def deployment_trigger(
        accuracy: float,
        config: DeploymentTriggerConfig,
):
    """Implements a simple model deployment trigger that looks at the input model accuracy amd decide if it is good enough to deploy or not"""
    return accuracy >= config.min_accuracy



@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continous_deployment_pipeline(
        data_path: str,
        min_accuracy: float = 70.0,
        workers: int = 1,
        timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    df = ingest_df(data_path=r"D:\Projects\ShopScore\data\merged_data\merged_data.csv")
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, y_train)
    accuracy, fscore, auc = evaluate_model(model, X_train, X_test, y_train, y_test)
    deployment_decision = deployment_trigger(accuracy)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    pass