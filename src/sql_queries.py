"""SQL analysis helpers — loads cleaned data into SQLite and runs analytical queries."""

import logging
import sqlite3
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "telco_churn_cleaned.csv"
DB_PATH = Path(__file__).parent.parent / "data" / "churn.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open (or create) a SQLite connection.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        sqlite3.Connection object.
    """
    conn = sqlite3.connect(db_path)
    return conn


def load_into_sqlite(
    df: pd.DataFrame = None,
    conn: sqlite3.Connection = None,
    table: str = "customers",
) -> Tuple[sqlite3.Connection, pd.DataFrame]:
    """Load the cleaned dataset into a SQLite table.

    Args:
        df: Cleaned DataFrame; if None loads from processed CSV.
        conn: Existing connection; if None opens an in-memory DB.
        table: Table name to write.

    Returns:
        Tuple of (connection, DataFrame).
    """
    if df is None:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    if conn is None:
        conn = sqlite3.connect(":memory:")
    df.to_sql(table, conn, if_exists="replace", index=False)
    logger.info("Loaded %d rows into SQLite table '%s'", len(df), table)
    return conn, df


def q1_overall_churn_rate(conn: sqlite3.Connection) -> pd.DataFrame:
    """Total customers, total churned, and overall churn rate."""
    sql = """
    SELECT
        COUNT(*)                          AS total_customers,
        SUM(Churn)                        AS total_churned,
        ROUND(AVG(CAST(Churn AS REAL)) * 100, 2) AS churn_rate_pct
    FROM customers;
    """
    return pd.read_sql_query(sql, conn)


def q2_churn_by_contract(conn: sqlite3.Connection) -> pd.DataFrame:
    """Churn rate broken down by Contract type with count and percentage."""
    sql = """
    SELECT
        Contract,
        COUNT(*)                                   AS total_customers,
        SUM(Churn)                                 AS churned,
        ROUND(AVG(CAST(Churn AS REAL)) * 100, 2)  AS churn_rate_pct
    FROM customers
    GROUP BY Contract
    ORDER BY churn_rate_pct DESC;
    """
    return pd.read_sql_query(sql, conn)


def q3_avg_charges_churn_vs_retained(conn: sqlite3.Connection) -> pd.DataFrame:
    """Average monthly and total charges for churned vs retained customers."""
    sql = """
    SELECT
        CASE WHEN Churn = 1 THEN 'Churned' ELSE 'Retained' END AS status,
        ROUND(AVG(MonthlyCharges), 2)  AS avg_monthly_charges,
        ROUND(AVG(TotalCharges), 2)    AS avg_total_charges,
        COUNT(*)                       AS customer_count
    FROM customers
    GROUP BY Churn;
    """
    return pd.read_sql_query(sql, conn)


def q4_top_segments_by_churn_rate(conn: sqlite3.Connection) -> pd.DataFrame:
    """Top 5 customer segments (contract × internet service) by churn rate."""
    sql = """
    SELECT
        Contract,
        InternetService,
        COUNT(*)                                   AS total,
        SUM(Churn)                                 AS churned,
        ROUND(AVG(CAST(Churn AS REAL)) * 100, 2)  AS churn_rate_pct
    FROM customers
    GROUP BY Contract, InternetService
    HAVING total > 50
    ORDER BY churn_rate_pct DESC
    LIMIT 5;
    """
    return pd.read_sql_query(sql, conn)


def q5_tenure_cohort_analysis(conn: sqlite3.Connection) -> pd.DataFrame:
    """Churn rate by tenure bucket using CASE WHEN."""
    sql = """
    SELECT
        CASE
            WHEN tenure BETWEEN 0  AND 12 THEN '0-12 months'
            WHEN tenure BETWEEN 13 AND 24 THEN '13-24 months'
            WHEN tenure BETWEEN 25 AND 48 THEN '25-48 months'
            ELSE '49+ months'
        END AS tenure_bucket,
        COUNT(*)                                   AS customers,
        SUM(Churn)                                 AS churned,
        ROUND(AVG(CAST(Churn AS REAL)) * 100, 2)  AS churn_rate_pct
    FROM customers
    GROUP BY tenure_bucket
    ORDER BY MIN(tenure);
    """
    return pd.read_sql_query(sql, conn)


def q6_rank_customers_by_charges(conn: sqlite3.Connection) -> pd.DataFrame:
    """Window function: rank customers by monthly charges within each contract type."""
    sql = """
    SELECT
        rowid        AS customer_row,
        Contract,
        MonthlyCharges,
        RANK() OVER (
            PARTITION BY Contract
            ORDER BY MonthlyCharges DESC
        ) AS charge_rank_in_contract
    FROM customers
    ORDER BY Contract, charge_rank_in_contract
    LIMIT 20;
    """
    return pd.read_sql_query(sql, conn)


def q7_high_risk_cte_revenue(conn: sqlite3.Connection) -> pd.DataFrame:
    """CTE: identify high-risk customers and calculate their total revenue at risk."""
    sql = """
    WITH high_risk AS (
        SELECT
            rowid          AS customer_row,
            MonthlyCharges,
            tenure,
            TechSupport
        FROM customers
        WHERE Contract = 'Month-to-month'
          AND TechSupport = 'No'
          AND tenure < 12
    )
    SELECT
        COUNT(*)                        AS high_risk_customers,
        ROUND(SUM(MonthlyCharges), 2)   AS monthly_revenue_at_risk,
        ROUND(AVG(MonthlyCharges), 2)   AS avg_monthly_charge
    FROM high_risk;
    """
    return pd.read_sql_query(sql, conn)


def q8_compare_to_contract_avg(conn: sqlite3.Connection) -> pd.DataFrame:
    """Subquery: compare each customer's charge to the average for their contract type."""
    sql = """
    SELECT
        c.rowid                                                AS customer_row,
        c.Contract,
        c.MonthlyCharges,
        ROUND(avg_by_contract.avg_charge, 2)                   AS contract_avg_charge,
        ROUND(c.MonthlyCharges - avg_by_contract.avg_charge, 2) AS diff_from_avg
    FROM customers c
    JOIN (
        SELECT Contract, AVG(MonthlyCharges) AS avg_charge
        FROM customers
        GROUP BY Contract
    ) avg_by_contract ON c.Contract = avg_by_contract.Contract
    ORDER BY ABS(diff_from_avg) DESC
    LIMIT 15;
    """
    return pd.read_sql_query(sql, conn)


def run_all_queries(conn: sqlite3.Connection) -> dict:
    """Execute all 8 SQL queries and return results as a dict.

    Args:
        conn: SQLite connection with 'customers' table loaded.

    Returns:
        Dict mapping query name to result DataFrame.
    """
    return {
        "q1_overall_churn_rate": q1_overall_churn_rate(conn),
        "q2_churn_by_contract": q2_churn_by_contract(conn),
        "q3_avg_charges": q3_avg_charges_churn_vs_retained(conn),
        "q4_top_segments": q4_top_segments_by_churn_rate(conn),
        "q5_tenure_cohort": q5_tenure_cohort_analysis(conn),
        "q6_window_rank": q6_rank_customers_by_charges(conn),
        "q7_high_risk_cte": q7_high_risk_cte_revenue(conn),
        "q8_subquery_compare": q8_compare_to_contract_avg(conn),
    }
