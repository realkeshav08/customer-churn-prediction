"""Preprocessing pipeline for the Telco Churn dataset."""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "telco_churn_cleaned.csv"


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges from object to float, coercing whitespace to NaN.

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame with TotalCharges as float64.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    logger.info("TotalCharges: coerced %d whitespace entries to NaN", n_missing)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the 11 rows where TotalCharges is NaN (new customers, tenure=0).

    Args:
        df: DataFrame with TotalCharges already converted to numeric.

    Returns:
        DataFrame with missing rows removed.
    """
    n_before = len(df)
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    logger.info("Dropped %d rows with missing TotalCharges (%d remaining)", n_dropped, len(df))
    return df


def standardize_senior_citizen(df: pd.DataFrame) -> pd.DataFrame:
    """Recode SeniorCitizen from 0/1 integer to 'No'/'Yes' string.

    This makes the column consistent with all other binary Yes/No columns
    so downstream encoding treats them uniformly.

    Args:
        df: DataFrame containing SeniorCitizen column.

    Returns:
        DataFrame with SeniorCitizen recoded.
    """
    df = df.copy()
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Churn column from Yes/No string to 1/0 integer.

    Args:
        df: DataFrame with Churn as string.

    Returns:
        DataFrame with Churn as int (1 = churned, 0 = retained).
    """
    df = df.copy()
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    churn_rate = df["Churn"].mean() * 100
    logger.info("Target encoded: overall churn rate = %.2f%%", churn_rate)
    return df


def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Remove customerID — it is a row identifier with no predictive value.

    Args:
        df: DataFrame containing customerID.

    Returns:
        DataFrame without customerID.
    """
    return df.drop(columns=["customerID"], errors="ignore")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline.

    Steps:
        1. Fix TotalCharges dtype
        2. Handle missing values
        3. Standardize SeniorCitizen encoding
        4. Encode target variable
        5. Drop customerID

    Args:
        df: Raw Telco Churn DataFrame.

    Returns:
        Cleaned DataFrame ready for EDA and feature engineering.
    """
    logger.info("Starting preprocessing pipeline — input shape: %s", df.shape)
    df = fix_total_charges(df)
    df = handle_missing_values(df)
    df = standardize_senior_citizen(df)
    df = encode_target(df)
    df = drop_id_column(df)
    logger.info("Preprocessing complete — output shape: %s", df.shape)
    return df


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_DATA_PATH) -> None:
    """Persist the cleaned DataFrame to CSV.

    Args:
        df: Cleaned DataFrame.
        path: Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved processed data to %s", path)
