import logging
import os


def get_logger(name) -> logging.Logger:
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    level = os.getenv("LOG_LEVEL", logging.INFO)
    logger.setLevel(int(level))
    logger.addHandler(handler)
    return logger
