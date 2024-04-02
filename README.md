# Predicting Customer Satisfaction and Review Scores for E-commerce Orders Using ZenML 🛒💡📊

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zenml)](https://pypi.org/project/zenml/)

## Problem Statement

In the realm of e-commerce, customer feedback serves as a crucial metric for shaping product recommendations and business strategies. However, the challenge arises when customers abstain from providing review ratings or feedback post-purchase. This lack of feedback impedes e-commerce platforms from accurately gauging customer satisfaction, hindering their ability to make informed decisions regarding product recommendations, inventory management, and vendor relationships.

The objective of this project is to develop a predictive model using ZenML that can anticipate the review score a customer is likely to give for their next order. Leveraging historical data from the Brazilian E-Commerce Public Dataset by Olist, which includes information on 100,000 orders made between 2016 and 2018, the model aims to predict customer satisfaction scores for future purchases.

## Key Objectives

1. **Predict Review Scores**: Develop a machine learning pipeline using ZenML to predict whether a customer is likely to rate a product positively or negatively based on their historical purchasing behavior and other relevant features.

2. **Enhance Product Recommendations**: Utilize predicted review scores to tailor product recommendations, thereby improving the overall shopping experience for customers.

3. **Optimize Business Strategies**: Leverage predictive insights to make informed decisions regarding product inventory, marketing campaigns, and vendor relationships, enabling e-commerce platforms to stay competitive in the market.

4. **Proactive Customer Engagement**: Anticipate customer satisfaction to proactively engage with customers, address concerns, and foster stronger relationships through personalized offers and support services.

## Constraints

1. **High Accuracy**: Ensure the predictive model achieves a high level of accuracy in discerning between positive and negative review scores to facilitate reliable decision-making and recommendation generation.

2. **Low Latency**: Develop a pipeline that provides timely insights into customer satisfaction, allowing for swift actions and interventions within the completion of the order.

3. **Robustness to Outliers**: Design the model to be robust enough to handle outliers and variations in customer behavior, ensuring its effectiveness across diverse scenarios and customer segments.

## :snake: Python Requirements

- ZenML
- scikit-learn
- pandas
- numpy

Let's jump into the Python packages you need. Within the Python environment of your choice, run:

```bash
git clone https://github.com/zenml-io/zenml-projects.git
cd ShopScore
pip install -r requirements.txt
```

Starting with ZenML 0.20.0, ZenML comes bundled with a React-based dashboard. This dashboard allows you
to observe your stacks, stack components and pipeline DAGs in a dashboard interface. To access this, you need to [launch the ZenML Server and Dashboard locally](https://docs.zenml.io/user-guide/starter-guide#explore-the-dashboard), but first you must install the optional dependencies for the ZenML server:

```bash
pip install zenml["server"]
zenml up
```

If you are running the `run_deployment.py` script, you will also need to install some integrations using ZenML:

```bash
zenml integration install mlflow -y
```

The project can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component. Configuring a new stack with the two components are as follows:

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

## :thumbsup: The Solution

In order to build a real-world workflow for predicting the customer satisfaction score for the next order or purchase (which will help make better decisions), it is not enough to just train the model once.

Instead, we are building an end-to-end pipeline for continuously predicting and deploying the machine learning model, alongside a data application that utilizes the latest deployed model for the business to consume.

This pipeline can be deployed to the cloud, scale up according to our needs, and ensure that we track the parameters and data that flow through every pipeline that runs. It includes raw data input, features, results, the machine learning model and model parameters, and prediction outputs. ZenML helps us to build such a pipeline in a simple, yet powerful, way.

In this Project, we give special consideration to the [MLflow integration](https://github.com/zenml-io/zenml/tree/main/examples) of ZenML. In particular, we utilize MLflow tracking to track our metrics and parameters, and MLflow deployment to deploy our model. We also use [Streamlit](https://streamlit.io/) to showcase how this model will be used in a real-world setting.

### Training Pipeline

Our standard training pipeline consists of several steps:

- `ingest_data`: This step will ingest the data and create a `DataFrame`.
- `clean_data`: This step will clean the data and remove the unwanted columns.
- `train_model`: This step will train the model and save the model using [MLflow autologging](https://www.mlflow.org/docs/latest/tracking.html).
- `evaluation`: This step will evaluate the model and save the metrics -- using MLflow autologging -- into the artifact store.

### Deployment Pipeline

We have another pipeline, the `deployment_pipeline.py`, that extends the training pipeline, and implements a continuous deployment workflow. It ingests and processes input data, trains a model and then (re)deploys the prediction server that serves the model if it meets our evaluation criteria. The criteria that we have chosen is a configurable threshold on the [Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) of the training. The first four steps of the pipeline are the same as above, but we have added the following additional ones:

- `deployment_trigger`: The step checks whether the newly trained model meets the criteria set for deployment.
- `model_deployer`: This step deploys the model as a service using MLflow (if deployment criteria is met).

In the deployment pipeline, ZenML's MLflow tracking integration is used for logging the hyperparameter values and the trained model itself and the model evaluation metrics -- as MLflow experiment tracking artifacts -- into the local MLflow backend. This pipeline also launches a local MLflow deployment server to serve the latest MLflow model if its accuracy is above a configured threshold.

The MLflow deployment server runs locally as a daemon process that will continue to run in the background after the example execution is complete. When a new pipeline is run which produces a model that passes the accuracy threshold validation, the pipeline automatically updates the currently running MLflow deployment server to serve the new model instead of the old one.

To round it off, we deploy a Streamlit application that consumes the latest model service asynchronously from the pipeline logic. This can be done easily with ZenML within the Streamlit code:

```python
service = prediction_service_loader(
   pipeline_name="continuous_deployment_pipeline",
   pipeline_step_name="mlflow_model_deployer_step",
   running=False,
)
...
service.predict(...)  # Predict on incoming data from the application
```

While this ZenML Project trains and deploys a model locally, other ZenML integrations such as the [Seldon](https://github.com/zenml-io/zenml/tree/main/examples/seldon_deployment) deployer can also be used in a similar manner to deploy the model in a more production setting (such as on a Kubernetes cluster). We use MLflow here for the convenience of its local deployment.

![training_and_deployment_pipeline](_assets/training_and_deployment_pipeline_updated.png)

## :notebook: Diving into the code

You can run two pipelines as follows:

- Training pipeline:

```bash
python run_pipeline.py
```

- The continuous deployment pipeline:

```bash
python run_deployment.py
```

## 🕹 Demo Streamlit App

If you want to run the Streamlit app in your local system, you can run the following command:-

```bash
streamlit run streamlit_app.py
```

## :question: FAQ

1. When running the continuous deployment pipeline, I get an error stating: `No Step found for the name mlflow_deployer`.

   Solution: It happens because your artifact store is overridden after running the continuous deployment pipeline. So, you need to delete the artifact store and rerun the pipeline. You can get the location of the artifact store by running the following command:

   ```bash
   zenml artifact-store describe
   ```

   and then you can delete the artifact store with the following command:

   **Note**: This is a dangerous / destructive command! Please enter your path carefully, otherwise it may delete other folders from your computer.

   ```bash
   rm -rf PATH
   ```

2. When running the continuous deployment pipeline, I get the following error: `No Environment component with name mlflow is currently registered.`

   Solution: You forgot to install the MLflow integration in your ZenML environment. So, you need to install the MLflow integration by running the following command:

   ```bash
   zenml integration install mlflow -y
   ```
