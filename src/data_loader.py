"""Data loading utilities for the Telco Churn dataset."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "telco_churn.csv"
PROCESSED_DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "telco_churn_cleaned.csv"


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco Churn CSV from disk.

    Args:
        path: Path to the raw CSV file.

    Returns:
        Raw DataFrame with original columns.
    """
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns", *df.shape)
    return df


def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned/processed dataset.

    Args:
        path: Path to the processed CSV file.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    logger.info("Loading processed data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns", *df.shape)
    return df
