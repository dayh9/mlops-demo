import os
import mlflow
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import sys
sys.path.append("src")
print(sys.path)

from data.ingest_data import get_data
from features.preprocess_heart import split_data, get_transformed_data
from common.logger import get_logger

logger = get_logger(__name__)


def get_model():
    """define and return the multi-classication model"""
    # DEFINE YOUR IMPROVED MODEL HERE:
    model = LinearSVC()
    return model


def train_model(data_file, experiment_name="experiment", **kwargs):
    
    start = time.time()
        
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()
    
    df = get_data(data_file)

    train, test = split_data(df, "HeartDisease")


    # x_train, y_train, scaler = get_transformed_data(train)
    # x_test, y_test, _ = get_transformed_data(test, scaler)

    x_train = get_data("data/x_train_heart.csv")
    y_train = get_data("data/y_train_heart.csv")
    x_test = get_data("data/x_test_heart.csv")
    y_test = get_data("data/y_test_heart.csv")
    
    
    with mlflow.start_run() as active_run:
        run_id = active_run.info.run_id
        # add the git commit hash as tag to the experiment run
        # git_hash = os.popen("git rev-parse --verify HEAD").read()[:-2]
        # mlflow.set_tag("git_hash", git_hash)
        
        clf = get_model()
        clf.fit(x_train, y_train)
    
        # return the model uri
        model_uri = mlflow.get_artifact_uri("model")
        mlflow.log_artifact("models/scaler.pkl")
       
    logger.info(f"completed script in {round(time.time() - start, 3)} seconds)")
    """calculate performance metrics"""

    y_pred = clf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    # mlflow.log_artifact("models/scaler.pkl")
    logger.info(f"***** performance {type(clf).__name__} *****")
    logger.info(f'accuracy: {round(accuracy, 3)}')
    logger.info(f'precision: {round(precision, 3)}')
    logger.info(f'recall: {round(recall, 3)}')
    logger.info(f'f1-score: {round(f1, 3)}\n')
    
    return run_id, model_uri


train_model("data/heart.csv")