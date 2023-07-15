import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append("src")
from common.logger import get_logger

logger = get_logger(__name__)


# TODO: introduce config files later
def split_and_save_data(
    input_dir: str,
    output_dir: str,
    file: str,
    label: str,
    test_size: float = 0.2,
    random_state: int = None,
) -> None:
    file_path = os.path.join(input_dir, file)
    logger.info(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)

    logger.info(f"Splitting {file_path}")
    split_1, split_2 = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[label],
        shuffle=True,
    )

    if not os.path.exists(output_dir):
        logger.info(f"Creating output_dir: {output_dir}")
        os.makedirs(output_dir)

    logger.info(f"Saving train and test {file} files in {output_dir} folder")
    split_1.to_csv(f"{output_dir}/train_{file}")
    split_2.to_csv(f"{output_dir}/test_{file}")


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
    parser.add_argument("-f", "--file", type=str, help="Input file name", required=True)
    parser.add_argument(
        "-l", "--label", type=str, help="Label column name", required=True
    )
    parser.add_argument("-t", "--test-size", type=str, help="Split size")
    parser.add_argument("-r", "--random-state", type=str, help="Split random state")
    args = parser.parse_args()

    split_and_save_data(
        args.input_dir,
        args.output_dir,
        args.file,
        args.label,
        args.test_size,
        args.random_state,
    )
