import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get url from DVC
import dvc.api
path='data/Cleaned_ai4i.csv'
repo='D:\iNeuron\Projects\DVC_MLflow'
version ='v2'


data_url = dvc.api.get_url(
    path=path,
    repo=repo,
    rev=version
)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the Ai4I csv file from the URL

    try:
        data = pd.read_csv(data_url)
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # log data param
    mlflow.set_experiment("AI4I_Experiments_3")
    mlflow.log_param("data_url",data_url)
    mlflow.log_param("data_version", version)
    mlflow.log_param("input_rows", data.shape[0])
    mlflow.log_param("input_col", data.shape[1])

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["AirtemperatureK"], axis=1)
    test_x = test.drop(["AirtemperatureK"], axis=1)
    train_y = train[["AirtemperatureK"]]
    test_y = test[["AirtemperatureK"]]

    # log artifacts
    cols_x = pd.DataFrame(list(train_x.columns))
    cols_x.to_csv('features.csv',header=False,index=False)
    mlflow.log_artifact('features.csv')

    cols_y = pd.DataFrame(list(train_y.columns))
    cols_y.to_csv('target.csv', header=False, index=False)
    mlflow.log_artifact('target.csv')


    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run(nested=True):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetAi4iModel")
        else:
            mlflow.sklearn.log_model(lr, "model")
