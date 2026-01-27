#!/usr/bin/env python3
"""
Manually update market data cache for a ticker.
Run this once to populate cache, then run fundamental_analyzer.py to use cached data.
"""
import sys
sys.path.insert(0, 'src')

from fundamental_analyzer import FundamentalAnalyzer

if __name__ == "__main__":
    ticker = "AAPL"
    print(f"Updating market data cache for {ticker}...")
    
    try:
        analyzer = FundamentalAnalyzer(ticker)
        analyzer.update_market_data_cache()
        print(f"✓ Cache updated successfully!")
        print("\nNow you can run: python src/fundamental_analyzer.py")
        print("It will use cached data to avoid rate limits.")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
