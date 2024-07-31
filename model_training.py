from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test

# Example usage:
if __name__ == "__main__":
    from data_fetching import fetch_stock_data
    from data_processing import process_data
    
    data = fetch_stock_data('AAPL')
    X, y = process_data(data)
    model, X_train, y_train, X_test, y_test = train_model(X, y)
    print(model.coef_)
