"""Feature engineering pipeline — produces 25+ features for modeling."""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

SERVICE_COLS = [
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bin tenure into 4 cohort buckets.

    Args:
        df: DataFrame with tenure column.

    Returns:
        DataFrame with new tenure_group column.
    """
    df = df.copy()
    bins = [0, 12, 24, 48, df["tenure"].max() + 1]
    labels = ["0-12", "13-24", "25-48", "49+"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=True)
    return df


def add_avg_monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average monthly spend = TotalCharges / (tenure + 1).

    Args:
        df: DataFrame with TotalCharges and tenure.

    Returns:
        DataFrame with avg_monthly_spend column.
    """
    df = df.copy()
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    return df


def add_is_high_value(df: pd.DataFrame, threshold: float = 75.0) -> pd.DataFrame:
    """Flag customers whose MonthlyCharges exceed the threshold.

    Args:
        df: DataFrame with MonthlyCharges.
        threshold: Monthly charge threshold above which customer is 'high value'.

    Returns:
        DataFrame with is_high_value binary column.
    """
    df = df.copy()
    df["is_high_value"] = (df["MonthlyCharges"] > threshold).astype(int)
    return df


def add_num_services(df: pd.DataFrame) -> pd.DataFrame:
    """Count the number of add-on services each customer subscribes to.

    InternetService counts as 1 if not 'No', others count as 1 if 'Yes'.

    Args:
        df: DataFrame with service columns.

    Returns:
        DataFrame with num_services integer column (0–8).
    """
    df = df.copy()
    svc_flags = pd.DataFrame(index=df.index)
    for col in SERVICE_COLS:
        if col in df.columns:
            svc_flags[col] = (df[col] == "Yes").astype(int)
    if "InternetService" in df.columns:
        svc_flags["InternetService"] = (df["InternetService"] != "No").astype(int)
    df["num_services"] = svc_flags.sum(axis=1)
    return df


def add_is_senior_alone(df: pd.DataFrame) -> pd.DataFrame:
    """Flag senior citizens with no partner (high-risk segment).

    Args:
        df: DataFrame with SeniorCitizen and Partner columns.

    Returns:
        DataFrame with is_senior_alone binary column.
    """
    df = df.copy()
    df["is_senior_alone"] = (
        (df["SeniorCitizen"] == "Yes") & (df["Partner"] == "No")
    ).astype(int)
    return df


def add_contract_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal risk score: month-to-month=3, one year=2, two year=1.

    Month-to-month customers churn at much higher rates; this captures
    the ordinal relationship without needing one-hot encoding here.

    Args:
        df: DataFrame with Contract column.

    Returns:
        DataFrame with contract_risk_score column.
    """
    df = df.copy()
    mapping = {"Month-to-month": 3, "One year": 2, "Two year": 1}
    df["contract_risk_score"] = df["Contract"].map(mapping)
    return df


def add_auto_pay_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag customers who use automatic payment methods (lower churn risk).

    Args:
        df: DataFrame with PaymentMethod column.

    Returns:
        DataFrame with auto_pay_flag binary column.
    """
    df = df.copy()
    auto_methods = {"Bank transfer (automatic)", "Credit card (automatic)"}
    df["auto_pay_flag"] = df["PaymentMethod"].isin(auto_methods).astype(int)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all object/category columns (drop_first=True).

    Excludes the target column 'Churn'.

    Args:
        df: DataFrame with categorical columns still as strings.

    Returns:
        DataFrame with all categoricals expanded into dummies.
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != "Churn"]
    logger.info("One-hot encoding %d categorical columns: %s", len(cat_cols), cat_cols)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def scale_numerics(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize tenure, MonthlyCharges, TotalCharges, and derived numeric features.

    Args:
        df: DataFrame after encoding.
        scaler: Pre-fitted StandardScaler (pass during inference).
        fit: If True, fit a new scaler on df; otherwise use provided scaler.

    Returns:
        Tuple of (scaled DataFrame, fitted scaler).
    """
    num_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "avg_monthly_spend",
        "num_services",
        "contract_risk_score",
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    df = df.copy()
    if scaler is None or fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    logger.info("Scaled %d numeric columns", len(num_cols))
    return df, scaler


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Full feature engineering pipeline.

    Applies all engineered features, then encodes and scales.

    Args:
        df: Cleaned DataFrame from preprocessing pipeline.

    Returns:
        Tuple of (feature-engineered DataFrame, fitted StandardScaler).
    """
    logger.info("Starting feature engineering — input shape: %s", df.shape)
    df = add_tenure_group(df)
    df = add_avg_monthly_spend(df)
    df = add_is_high_value(df)
    df = add_num_services(df)
    df = add_is_senior_alone(df)
    df = add_contract_risk_score(df)
    df = add_auto_pay_flag(df)
    df = encode_categoricals(df)
    df, scaler = scale_numerics(df)
    feature_cols = [c for c in df.columns if c != "Churn"]
    logger.info("Feature engineering complete — total features: %d", len(feature_cols))
    return df, scaler
