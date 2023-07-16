import argparse
import logging
import os
import sys
import time
from typing import Any, Optional, Tuple

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

logger = get_logger(__name__)

DEFAULT_EXPERIMENT = "default_experiment"


def get_model():
    """define and return the multi-classication model"""
    model = LinearSVC()
    return model


def separate_features_labels(data: pd.DataFrame, label: str) -> Tuple[pd.DataFrame, pd.Series]:
    try:
        labels = data[label]
        features = data.drop(columns=[label])
    except KeyError:
        raise Exception(f"Missing label column: {label}")
    return features, labels


def train_model(
    train_file: str,
    test_file: str,
    label: str,
    scaler_file: Optional[str] = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> Tuple[Any, str]:

    start = time.time()

    if mlflow.get_experiment_by_name(experiment_name) is None:
        logger.debug(f"Create experiment: {experiment_name}")
        mlflow.create_experiment(experiment_name, "models")

    logger.debug(f"Set experiment: {experiment_name}")
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()

    logger.debug(f"Get train data from: {train_file}")
    train = get_data(train_file)
    logger.debug(train.info())
    x_train, y_train = separate_features_labels(train, label)
    logger.debug(f"Get test data from: {test_file}")
    test = get_data(test_file)
    logger.debug(test.info())
    x_test, y_test = separate_features_labels(test, label)

    logger.info("Start training")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate data")
    parser.add_argument("-t", "--train-file", type=str, required=True)
    parser.add_argument("-e", "--test-file", type=str, required=True)
    parser.add_argument("-l", "--label", type=str, required=True)
    parser.add_argument("-s", "--scaler-file", type=str)
    parser.add_argument("-n", "--experiment-name", type=str, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()

    train_model(
        args.train_file,
        args.test_file,
        args.label,
        args.scaler_file,
        args.experiment_name,
    )
