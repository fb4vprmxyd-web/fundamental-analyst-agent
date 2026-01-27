import pandas as pd
from pathlib import Path
from datetime import datetime
import os
import time
from dotenv import load_dotenv
from alpha_vantage.fundamentaldata import FundamentalData

# Load environment variables
load_dotenv()

# ---------------------------
# Setup
# ---------------------------
TICKER_SYMBOL = "AAPL"
RAW_DATA_PATH = Path("data/raw")
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Initialize Alpha Vantage
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not API_KEY:
    raise ValueError("ALPHA_VANTAGE_API_KEY not found in .env file")

fd = FundamentalData(key=API_KEY, output_format='pandas')

# ---------------------------
# Functions
# ---------------------------
def get_financial_statements(ticker_symbol):
    """
    Pull annual financial statements from Alpha Vantage.
    Returns: (income_stmt, balance_sheet, cash_flow) as DataFrames.
    """
    print(f"Fetching income statement for {ticker_symbol}...")
    income_stmt, _ = fd.get_income_statement_annual(ticker_symbol)
    
    print("Waiting 12 seconds (API rate limit)...")
    time.sleep(12)  # Alpha Vantage free tier: 5 calls per minute
    
    print(f"Fetching balance sheet for {ticker_symbol}...")
    balance_sheet, _ = fd.get_balance_sheet_annual(ticker_symbol)
    
    print("Waiting 12 seconds (API rate limit)...")
    time.sleep(12)
    
    print(f"Fetching cash flow for {ticker_symbol}...")
    cash_flow, _ = fd.get_cash_flow_annual(ticker_symbol)
    
    return income_stmt, balance_sheet, cash_flow

def transform_alpha_vantage_to_yfinance_format(df, statement_type):
    """
    Transform Alpha Vantage format to match yfinance format.
    Alpha Vantage returns data with 'fiscalDateEnding' column.
    Limits to most recent 5 years.
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    # Extract year from fiscalDateEnding
    if 'fiscalDateEnding' in df.columns:
        df['year'] = pd.to_datetime(df['fiscalDateEnding']).dt.year
    else:
        raise ValueError("fiscalDateEnding column not found")
    
    # Drop the date column and set year as column
    df = df.drop('fiscalDateEnding', axis=1)
    df = df.set_index('year')
    
    # Transpose so years are columns (like yfinance)
    df = df.T
    
    # Sort columns by year ascending
    df = df[sorted(df.columns)]
    
    # LIMIT TO MOST RECENT 5 YEARS
    if len(df.columns) > 5:
        df = df[df.columns[-5:]]  # Keep only last 5 years
    
    # Convert all values to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Map Alpha Vantage field names to yfinance names
    field_mapping = get_field_mapping(statement_type)
    df = df.rename(index=field_mapping)
    
    return df

def get_field_mapping(statement_type):
    """Map Alpha Vantage field names to yfinance-compatible names."""
    
    if statement_type == 'income':
        return {
            'totalRevenue': 'Total Revenue',
            'costOfRevenue': 'Cost Of Revenue',
            'grossProfit': 'Gross Profit',
            'operatingIncome': 'EBIT',  # Map to EBIT (Operating Income = EBIT)
            'ebit': 'Operating Income',  # Keep both names available
            'netIncome': 'Net Income',
            'researchAndDevelopment': 'Research And Development',
            'sellingGeneralAndAdministrative': 'Selling General And Administration',
            'incomeTaxExpense': 'Tax Provision',
            'incomeBeforeTax': 'Pretax Income',
            'operatingExpenses': 'Operating Expense',
            'interestExpense': 'Interest Expense',
            'interestIncome': 'Interest Income',
            'netInterestIncome': 'Net Interest Income',
        }
    
    elif statement_type == 'balance':
        return {
            'totalAssets': 'Total Assets',
            'totalLiabilities': 'Total Liabilities Net Minority Interest',
            'totalCurrentAssets': 'Total Current Assets',
            'totalCurrentLiabilities': 'Total Current Liabilities',
            'cashAndCashEquivalentsAtCarryingValue': 'Cash And Cash Equivalents',
            'cashAndShortTermInvestments': 'Cash And Short Term Investments',
            'longTermDebt': 'Long Term Debt',
            'longTermDebtNoncurrent': 'Long Term Debt And Capital Lease Obligation',
            'shortTermDebt': 'Short Term Debt',
            'currentDebt': 'Current Debt',
            'currentLongTermDebt': 'Current Debt And Capital Lease Obligation',
            'shortLongTermDebtTotal': 'Total Debt',
            'totalShareholderEquity': 'Total Stockholders Equity',
        }
    
    elif statement_type == 'cashflow':
        return {
            'operatingCashflow': 'Operating Cash Flow',
            'capitalExpenditures': 'Capital Expenditure',
            'depreciationDepletionAndAmortization': 'Depreciation And Amortization',
            'depreciation': 'Depreciation',
            'dividendPayout': 'Cash Dividends Paid',
            'netIncome': 'Net Income From Continuing Operations',
            'changeInWorkingCapital': 'Change In Working Capital',
        }
    
    return {}

def save_statement(df: pd.DataFrame, filename: str):
    if df is None or df.empty:
        raise ValueError(f"{filename} is empty. Alpha Vantage may not have returned data.")
    
    file_path = RAW_DATA_PATH / filename
    df.to_csv(file_path)
    print(f"Saved {filename} -> {file_path}")
    print(f"  Years available: {list(df.columns)}")
    print(f"  Total years: {len(df.columns)}")

def save_metadata():
    meta = {
        "ticker": TICKER_SYMBOL,
        "download_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "source": "Alpha Vantage",
    }
    meta_path = RAW_DATA_PATH / "metadata.csv"
    pd.DataFrame([meta]).to_csv(meta_path, index=False)
    print(f"Saved metadata -> {meta_path}")

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    # Check if CSV files already exist
    csv_files = [
        RAW_DATA_PATH / "income_statement.csv",
        RAW_DATA_PATH / "balance_sheet.csv",
        RAW_DATA_PATH / "cash_flow.csv",
    ]
    
    if all(f.exists() for f in csv_files):
        print(f"✓ CSV files already exist for {TICKER_SYMBOL}")
        print("Skipping API calls. Using cached data.\n")
        
        try:
            income = pd.read_csv(RAW_DATA_PATH / "income_statement.csv", index_col=0)
            balance = pd.read_csv(RAW_DATA_PATH / "balance_sheet.csv", index_col=0)
            cashflow = pd.read_csv(RAW_DATA_PATH / "cash_flow.csv", index_col=0)
            
            print("✓ Data collection complete (using cached files)")
        except Exception as e:
            print(f"✗ Error reading cached files: {e}")
    else:
        print(f"Fetching financial data for {TICKER_SYMBOL} from Alpha Vantage...")
        print("Note: This will take ~30 seconds due to API rate limits...\n")
        
        try:
            # Fetch statements
            income, balance, cashflow = get_financial_statements(TICKER_SYMBOL)
            
            # Transform to yfinance-compatible format
            print("\nTransforming data to yfinance format...")
            income_clean = transform_alpha_vantage_to_yfinance_format(income, 'income')
            balance_clean = transform_alpha_vantage_to_yfinance_format(balance, 'balance')
            cashflow_clean = transform_alpha_vantage_to_yfinance_format(cashflow, 'cashflow')
            
            # Save
            print("\nSaving statements...")
            save_statement(income_clean, "income_statement.csv")
            save_statement(balance_clean, "balance_sheet.csv")
            save_statement(cashflow_clean, "cash_flow.csv")
            
            save_metadata()
            
            print(f"\n✓ Data collection complete for {TICKER_SYMBOL}")
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            print("\nTroubleshooting:")
            print("1. Check your ALPHA_VANTAGE_API_KEY in .env file")
            print("2. Verify you haven't exceeded API rate limits (25 calls/day for free tier)")
            print("3. Check internet connection")