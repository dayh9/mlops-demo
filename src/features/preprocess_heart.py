import argparse
import os
import sys
from pickle import dump, load

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append("src")
from common.logger import get_logger

logger = get_logger(__name__)


def scale_features(data: pd.DataFrame, scaler: StandardScaler, fit: bool):
    """Scales dataframe with StandardScaler then returns scaled data and scaler. Optionally can beforehand fit this scaler with data."""
    if fit:
        scaler.fit(data)

    scaler.set_output(transform="pandas")
    scaled_set = scaler.transform(data)

    return scaled_set, scaler


def make_features_bool(data_set, true_value_map):
    bool_set = pd.DataFrame()
    for k, v in true_value_map.items():
        bool_set[f"{k}_{v}"] = data_set[k].replace({v: 1, f"[^{v}]": 0}, regex=True)

    return bool_set


def make_features_vectors(data_set):
    vectors_sets = []
    for column in data_set.columns:
        vectors_sets.append(
            pd.get_dummies(data_set[column], prefix=column, dtype=float)
        )

    return pd.concat(vectors_sets, axis=1)


def split_data(data_set, class_column, split=0.2, random_state=None):
    split_1, split_2 = train_test_split(
        data_set,
        test_size=split,
        random_state=random_state,
        stratify=data_set[class_column],
        shuffle=True,
    )
    return split_1, split_2


def get_transformed_data(data: pd.DataFrame, label: str, scaler: StandardScaler = None):
    logger.debug(f"get_transformed_data args: {data.columns=} {scaler=}")

    # TODO: pass features in config
    features_to_std = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    features_to_bool = {"Sex": "F", "ExerciseAngina": "Y"}
    features_to_vector = ["ChestPainType", "RestingECG", "ST_Slope"]

    logger.info(f"Transforming features to scale")
    logger.debug(f"features to scale: {features_to_std}")
    fit = False
    if not scaler:
        scaler = StandardScaler()
        fit = True
    scaled_set, fitted_scaler = scale_features(data[features_to_std], scaler, fit)

    logger.info(f"Transforming features to bool")
    logger.debug(f"features to bool: {list(features_to_bool)}")
    bool_set = make_features_bool(data[list(features_to_bool)], features_to_bool)

    logger.info(f"Transforming features to vectors")
    logger.debug(f"features to vectors: {features_to_vector}")
    vectors_set = make_features_vectors(data[features_to_vector])

    if label in data.columns:
        labels = data[label]
    else:
        labels = pd.DataFrame(None)

    transformed_data = pd.concat([scaled_set, bool_set, vectors_set, labels], axis=1)

    return transformed_data, fitted_scaler


# TODO: make reading params from config file
def transform_and_save_data(
    input_dir, output_dir, data_file, models_dir=None, scaler_file=None
):
    file_path = os.path.join(input_dir, data_file)
    logger.info(f"Loading data from file: {file_path}")
    data = pd.read_csv(file_path)

    scaler = None
    if scaler_file:
        logger.info(f"Loading scaler from file: {scaler_file}")
        scaler = load(open(scaler_file, "rb"))

    transformed_data, fitted_scaler = get_transformed_data(data, scaler)

    if scaler_file is None:
        logger.info(
            f"Saving scaler in {models_dir} fitted on features from {data_file}"
        )
        if models_dir is None:
            raise Exception("Provide models_dir to specify where to save scalar")
        if not os.path.exists(models_dir):
            logger.info(f"Creating models_dir: {models_dir}")
            os.makedirs(models_dir)
        scalar_path = os.path.join(models_dir, "heart_scaler.pkl")
        dump(fitted_scaler, open(scalar_path, "wb"))

    logger.info(f"Saving transformed {data_file} file in {output_dir} folder")
    if not os.path.exists(output_dir):
        logger.info(f"Creating output_dir: {output_dir}")
        os.makedirs(output_dir)
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
    parser.add_argument("-f", "--file", type=str, help="Input file to load")
    parser.add_argument(
        "-m",
        "--models-dir",
        type=str,
        help="Folder path to store scaler and other artifacts",
    )
    parser.add_argument("-s", "--scaler-file", type=str, help="Scaler file to load")
    args = parser.parse_args()
    logger.debug(f"Args: {args}")

    transform_and_save_data(
        args.input_dir, args.output_dir, args.file, args.models_dir, args.scaler_file
    )
