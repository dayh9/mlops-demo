import argparse
import pandas as pd
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_data(input_file: str) -> pd.DataFrame:
    logger.info(f"Loading {input_file}")
    df = pd.read_csv(input_file)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate data')
    parser.add_argument('--input_file', type=str, help='Input file')

    args = parser.parse_args()
    get_data(args.input_folder)