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
    Uses the newer API methods that return more historical data.
    Returns: (income_stmt, balance_sheet, cash_flow) as DataFrames.
    """
    # Use these methods for annual data (they return more years)
    income_stmt = ticker_obj.income_stmt  # Annual income statement
    balance_sheet = ticker_obj.balance_sheet  # Annual balance sheet
    cash_flow = ticker_obj.cashflow  # Annual cash flow
    
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
    print(f"  Years available: {list(df.columns)}")
    print(f"  Total years: {len(df.columns)}")

def save_metadata():
    meta = {
        "ticker": TICKER_SYMBOL,
        "download_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "source": "yfinance / Yahoo Finance",
    }
    meta_path = RAW_DATA_PATH / "metadata.csv"
    pd.DataFrame([meta]).to_csv(meta_path, index=False)
    print(f"Saved metadata -> {meta_path}")

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    print(f"Fetching financial data for {TICKER_SYMBOL}...")
    
    # Fetch statements
    income, balance, cashflow = get_financial_statements(ticker)
    
    # Clean and save
    print("\nCleaning and saving statements...")
    income_clean = clean_statement(income)
    balance_clean = clean_statement(balance)
    cashflow_clean = clean_statement(cashflow)
    
    save_statement(income_clean, "income_statement.csv")
    save_statement(balance_clean, "balance_sheet.csv")
    save_statement(cashflow_clean, "cash_flow.csv")
    
    save_metadata()
    
    print(f"\n✓ Data collection complete for {TICKER_SYMBOL}")