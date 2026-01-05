import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

# ---------------------------
# Setup
# ---------------------------
TICKER_SYMBOL = "AAPL"
ticker = yf.Ticker(TICKER_SYMBOL)

RAW_DATA_PATH = Path("data/raw")
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Functions
# ---------------------------
def get_financial_statements(ticker_obj):
    """
    Pull annual financial statements from Yahoo Finance via yfinance.
    Returns: (income_stmt, balance_sheet, cash_flow) as DataFrames.
    """
    income_stmt = ticker_obj.financials
    balance_sheet = ticker_obj.balance_sheet
    cash_flow = ticker_obj.cashflow
    return income_stmt, balance_sheet, cash_flow


def clean_statement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column dates -> years and sort years ascending.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    df.columns = pd.to_datetime(df.columns, errors="coerce").year
    df = df.loc[:, df.columns.notna()]  # drop columns that failed date parsing
    df = df.sort_index(axis=1)
    return df


def save_statement(df: pd.DataFrame, filename: str):
    if df is None or df.empty:
        raise ValueError(f"{filename} is empty. Yahoo may not have returned data.")

    file_path = RAW_DATA_PATH / filename
    df.to_csv(file_path)
    print(f"Saved {filename} -> {file_path}")


def save_metadata():
    meta = {
        "ticker": TICKER_SYMBOL,
        "download_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "source": "yfinance / Yahoo Finance",
    }
    meta_path = RAW_DATA_PATH / "metadata.csv"
    pd.DataFrame([meta]).to_csv(meta_path, index=False)
    print(f"Saved metadata -> {meta_path}")


# -----
