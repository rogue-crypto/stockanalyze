def process_data(data):
    data = data.dropna()
    X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = data['Profit']
    return X, y

# Example usage:
if __name__ == "__main__":
    from data_fetching import fetch_stock_data
    data = fetch_stock_data('AAPL')
    X, y = process_data(data)
    print(X.head(), y.head())
