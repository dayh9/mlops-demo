import argparse
import logging
import os
import sys
import time
from typing import Tuple

import mlflow
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC

sys.path.append("src")
print(sys.path)

from common.logger import get_logger
from data.ingest_data import get_data
from features.preprocess_heart import get_transformed_data, split_data

logger = get_logger(__name__)


def get_model():
    """define and return the multi-classication model"""
    model = LinearSVC()
    return model


def separate_features_labels(data: pd.DataFrame, label: str) -> Tuple[pd.DataFrame]:
    labels = data[label]
    features = data.drop(columns=[label])
    return features, labels


def train_model(
    train_file: str,
    test_file: str,
    label: str,
    scaler_file: str = None,
    experiment_name: str = "experiment",
):
    start = time.time()

    mlflow.set_experiment(experiment_name)
    mlflow.autolog()

    train = get_data(train_file)
    x_train, y_train = separate_features_labels(train, label)
    test = get_data(test_file)
    x_test, y_test = separate_features_labels(test, label)

    with mlflow.start_run() as active_run:
        run_id = active_run.info.run_id
        # add the git commit hash as tag to the experiment run
        # git_hash = os.popen("git rev-parse --verify HEAD").read()[:-2]
        # mlflow.set_tag("git_hash", git_hash)

        clf = get_model()
        clf.fit(x_train, y_train)

        # return the model uri
        model_uri = mlflow.get_artifact_uri("model")
        if scaler_file:
            mlflow.log_artifact(scaler_file)

    logger.info(f"completed script in {round(time.time() - start, 3)} seconds)")
    """calculate performance metrics"""

    y_pred = clf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    logger.info(f"***** performance {type(clf).__name__} *****")
    logger.info(f"accuracy: {round(accuracy, 3)}")
    logger.info(f"precision: {round(precision, 3)}")
    logger.info(f"recall: {round(recall, 3)}")
    logger.info(f"f1-score: {round(f1, 3)}\n")

    return run_id, model_uri


def main(train_file, test_file):
    print(train_file)
    print(test_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate data")
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--scaler-file", type=str)
    parser.add_argument("--experiment-name", type=str)
    args = parser.parse_args()

    train_model(
        args.train_file,
        args.test_file,
        args.label,
        args.scaler_file,
        args.experiment_name,
    )
