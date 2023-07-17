import argparse
import os
import sys
from pickle import dump, load
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append("src")
from common.logger import get_logger

logger = get_logger(__name__)


def scale_features(
    data: pd.DataFrame, scaler: StandardScaler, fit: bool
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scales dataframe with StandardScaler then returns scaled data and scaler. Optionally can beforehand fit this scaler with data."""
    logger.info(f"Transforming features to scale")
    if fit:
        scaler.fit(data)

    scaler.set_output(transform="pandas")
    scaled_set = scaler.transform(data)

    return scaled_set, scaler


def make_features_bool(data_set: pd.DataFrame, true_value_map: dict) -> pd.DataFrame:
    logger.info(f"Transforming features to bool")
    bool_set = pd.DataFrame()
    for k, v in true_value_map.items():
        bool_set[f"{k}_{v}"] = data_set[k].replace({v: 1.0, f"[^{v}]": 0.0}, regex=True)

    return bool_set


def make_features_vectors(data_set: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Transforming features to vectors")
    vectors_sets = []
    for column in data_set.columns:
        vectors_sets.append(
            pd.get_dummies(data_set[column], prefix=column, dtype=float)
        )

    return pd.concat(vectors_sets, axis=1)


def get_transformed_data(
    data: pd.DataFrame, scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    # TODO: pass features in config
    features_to_std = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    features_to_bool = {"Sex": "F", "ExerciseAngina": "Y"}
    features_to_vector = ["ChestPainType", "RestingECG", "ST_Slope"]
    label = "HeartDisease"

    logger.debug(f"features to scale: {features_to_std}")
    fit = False
    if not scaler:
        scaler = StandardScaler()
        fit = True
    scaled_set, fitted_scaler = scale_features(data[features_to_std], scaler, fit)

    logger.debug(f"features to bool: {list(features_to_bool)}")
    bool_set = make_features_bool(data[list(features_to_bool)], features_to_bool)

    logger.debug(f"features to vectors: {features_to_vector}")
    vectors_set = make_features_vectors(data[features_to_vector])

    if label in data.columns:
        labels = data[label]
    else:
        labels = pd.Series(None)

    transformed_data = pd.concat([scaled_set, bool_set, vectors_set, labels], axis=1)
    logger.debug(f"Transformed data: {transformed_data.columns=}")

    return transformed_data, fitted_scaler


# TODO: make reading params from config file
def transform_and_save_data(
    input_dir: str,
    output_dir: str,
    data_file: str,
    models_dir: str,
    scaler_file: Optional[StandardScaler] = None,
) -> None:
    logger.info(f"Transform data from file: {data_file=}")
    file_path = os.path.join(input_dir, data_file)
    logger.debug(f"Loading data file from: {file_path=}")
    data = pd.read_csv(file_path)

    scaler = None
    if scaler_file:
        scaler_path = os.path.join(models_dir, scaler_file)
        logger.info(f"Loading scaler file from: {scaler_path=}")
        scaler = load(open(scaler_path, "rb"))

    transformed_data, fitted_scaler = get_transformed_data(data, scaler)

    if scaler_file is None:
        logger.info(
            f"Saving scaler in folder: {models_dir} fitted on features from: {data_file}"
        )
        if not os.path.exists(models_dir):
            logger.info(f"Creating models_dir: {models_dir}")
            os.makedirs(models_dir)
        scalar_path = os.path.join(models_dir, "heart_scaler.pkl")
        dump(fitted_scaler, open(scalar_path, "wb"))

    if not os.path.exists(output_dir):
        logger.info(f"Creating output_dir: {output_dir}")
        os.makedirs(output_dir)
    logger.info(f"Saving transformed {data_file} file in {output_dir} folder")
    transformed_data.to_csv(os.path.join(output_dir, data_file), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate data")
    parser.add_argument(
        "-i", "--input-dir", type=str, help="Directory with input file", required=True
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Directory to output split files",
        required=True,
    )
    parser.add_argument("-d", "--data-file", type=str, help="Input file to load")
    parser.add_argument(
        "-m",
        "--models-dir",
        type=str,
        help="Folder path to store scaler and other artifacts",
        required=True,
    )
    parser.add_argument("-s", "--scaler-file", type=str, help="Scaler file to load")
    args = parser.parse_args()

    transform_and_save_data(
        args.input_dir, args.output_dir, args.data_file, args.models_dir, args.scaler_file
    )
