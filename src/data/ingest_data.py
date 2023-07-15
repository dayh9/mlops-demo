import sys

import pandas as pd

sys.path.append("src")
from common.logger import get_logger

logger = get_logger(__name__)


def get_data(data_file: str) -> pd.DataFrame:
    logger.info(f"Loading data_file: {data_file}")
    data = pd.read_csv(data_file)
    return data
