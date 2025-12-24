import yfinance as yf
import pandas as pd
from pathlib import Path

# ---------------------------
# Setup
# ---------------------------
ticker = yf.Ticker("AAPL")

RAW_DATA_PATH = Path("data/raw")
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Functions
# ---------------------------
def get_financial_statements(ticker):
    income_stmt = ticker.financials
    balance_sheet = ticker.balance_sheet
    cash_flow = ticker.cashflow
    return income_stmt, balance_sheet, cash_flow


def clean_statement(df):
    df = df.copy()
    df.columns = pd.to_datetime(df.columns).year
    df = df.sort_index(axis=1)
    return df


def save_statement(df, filename):
    file_path = RAW_DATA_PATH / filename
    df.to_csv(file_path)
    print(f"Saved {filename}")


# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    income, balance, cashflow = get_financial_statements(ticker)

    income = clean_statement(income)
    balance = clean_statement(balance)
    cashflow = clean_statement(cashflow)

    save_statement(income, "income_statement.csv")
    save_statement(balance, "balance_sheet.csv")
    save_statement(cashflow, "cash_flow.csv")

    print("Income Statement Preview:")
    print(income.head())

