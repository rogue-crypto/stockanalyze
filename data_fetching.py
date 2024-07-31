import yfinance as yf
import pandas as pd
import numpy as np

def fetch_stock_data(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    data['Return'] = data['Close'].pct_change()
    data['Profit'] = np.where(data['Return'] > 0, 1, 0)
    return data

def get_all_stock_symbols():
    # Fetch a comprehensive list of stock symbols.
    # This can be customized to use an actual stock list or API.
    stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NFLX', 'FB', 'NVDA', 'BABA', 'V']
    return stock_symbols

# Example usage:
if __name__ == "__main__":
    data = fetch_stock_data('AAPL')
    print(data.head())
