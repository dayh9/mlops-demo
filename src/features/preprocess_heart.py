import argparse
import os
from pickle import dump, load
import sys

import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
        bool_set[f"{k}_{v}"] = data_set[k].replace({v:1, f"[^{v}]":0}, regex=True)

    return bool_set


def make_features_vectors(data_set):
    vectors_sets = []
    for column in data_set.columns:
        vectors_sets.append(pd.get_dummies(data_set[column], prefix=column, dtype=float))
    
    return pd.concat(vectors_sets, axis=1)


def split_data(data_set, class_column, split=0.2, random_state=None):
    split_1, split_2 = train_test_split(data_set, test_size=split, random_state=random_state,
                                stratify=data_set[class_column], shuffle=True)
    return split_1, split_2


def get_transformed_data(df, scaler=None):
    logger.debug(f"get_transformed_data args: {df.columns=} {scaler=}")

    # TODO: pass features in config
    features_to_std = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    features_to_bool = {'Sex': 'F', 'ExerciseAngina': 'Y'}
    features_to_vector = ['ChestPainType', 'RestingECG', 'ST_Slope']
    
    logger.info(f"Transforming features to scale")
    logger.debug(f"features to scale: {features_to_std}")
    scaled_set, fitted_scaler = scale_features(df[features_to_std], scaler, fit=True)

    logger.info(f"Transforming features to bool")
    logger.debug(f"features to bool: {list(features_to_bool)}")
    bool_set = make_features_bool(df[list(features_to_bool)], features_to_bool)

    logger.info(f"Transforming features to vectors")
    logger.debug(f"features to vectors: {features_to_vector}")
    vectors_set = make_features_vectors(df[features_to_vector])

    features = pd.concat([scaled_set, bool_set, vectors_set], axis=1)
# 
    # if "HeartDisease" in df.columns:
    #     labels = df["HeartDisease"]
    # else:
    #     labels = pd.DataFrame(None)
    # return features, labels, fitted_scaler


# TODO: make reading params from config file
def transform_and_save_data(data_folder, input_file, models_folder=None, scaler_file=None):
    file_path = os.path.join(data_folder, input_file)
    # read csv
    logger.info(f"Loading {file_path}")
    df = pd.read_csv(file_path)

    scaler = None
    if scaler_file:
        # read pkl to scaler
        logger.info(f"Loading scaler from file: {scaler_file}")
        scaler = load(open(scaler_file, "rb"))
    
    x, y, fitted_scaler = get_transformed_data(df, scaler)

    if scaler_file is None:
        logger.info(f"Saving scaler in {models_folder} fitted on features from {input_file}")
        if models_folder is None:
            raise Exception("Provide models_folder to specify where to save scalar")
        scalar_path = os.path.join(models_folder, "scaler.pkl")
        dump(fitted_scaler, open(scalar_path, "wb"))

    logger.info(f"Saving transformed features and labels of {input_file} file in {data_folder} folder")
    x.to_csv(f"{data_folder}/x_{input_file}", index=False)
    y.to_csv(f"{data_folder}/y_{input_file}", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate data")
    parser.add_argument("-d", "--data_folder", type=str, help="Folder path to store data")
    parser.add_argument("-f", "--input_file", type=str, help="Input file to load")
    parser.add_argument("-m", "--models_folder", type=str, help="Folder path to store models and other artifacts")
    parser.add_argument("-s", "--scaler_file", type=str, help="Scaler file to load")
 
    args = parser.parse_args()
    logger.debug(f"Args: {args}")
    transform_and_save_data(args.data_folder, args.input_file, args.models_folder, args.scaler_file)
