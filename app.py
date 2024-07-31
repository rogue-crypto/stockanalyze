from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import io
import base64
from data_fetching import fetch_stock_data, get_all_stock_symbols
from data_processing import process_data
from model_training import train_model
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

def generate_stock_data(symbol):
    # Fetch stock data
    data = fetch_stock_data(symbol)
    
    # Process data
    X, y = process_data(data)
    
    # Train model
    model, X_train, y_train, X_test, y_test = train_model(X, y)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Predict future stock prices
    future_days = 30  # Predict for the next 30 days
    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, future_days + 1)]
    future_X = np.array([X.iloc[-1]] * future_days)
    future_pred = model.predict(future_X)
    
    # Plot results
    plt.figure(figsize=(16, 8))
    plt.plot(data.index, data['Close'], label='Actual', color='blue')
    plt.plot(data.index[-len(y_pred):], y_pred, label='Predicted', color='green')
    plt.plot(future_dates, future_pred, label='Future Prediction', linestyle='dashed', color='orange')
    plt.legend()
    plt.title(f'Stock Price Prediction for {symbol}')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    # Determine time to buy and sell
    time_to_buy = future_dates[np.argmax(future_pred)].strftime('%Y-%m-%d')
    time_to_sell = future_dates[np.argmin(future_pred)].strftime('%Y-%m-%d')

    return {
        'symbol': symbol,
        'plot': plot_url,
        'predicted_profit': f"${future_pred.sum() * 1000:.2f}",
        'time_to_buy': time_to_buy,
        'time_to_sell': time_to_sell,
        'future_pred': future_pred.tolist()
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    symbol = request.args.get('symbol')
    stock_data = generate_stock_data(symbol)
    return jsonify(stock_data)

@app.route('/most_profitable_stocks')
def most_profitable_stocks():
    all_symbols = get_all_stock_symbols()
    stocks_data = [generate_stock_data(symbol) for symbol in all_symbols]
    stocks_data.sort(key=lambda x: np.max(x['future_pred']), reverse=True)
    return jsonify(stocks_data[:5])  # Return top 5 most profitable stocks

@app.route('/stock_info/<symbol>', methods=['GET'])
def stock_info(symbol):
    data = fetch_stock_data(symbol)
    current_price = data['Close'].iloc[-1]
    previous_price = data['Close'].iloc[-2]
    in_profit = current_price > previous_price

    return render_template('stock_info.html', symbol=symbol, in_profit=in_profit, current_price=current_price)

@app.route('/predict_profit/<symbol>', methods=['POST'])
def predict_profit(symbol):
    data = request.get_json()
    buy_date = datetime.strptime(data['buyDate'], '%Y-%m-%d')
    sell_date = datetime.strptime(data['sellDate'], '%Y-%m-%d')
    amount = float(data['amount'])

    stock_data = fetch_stock_data(symbol)
    buy_price = stock_data.loc[buy_date]['Close']
    sell_price = stock_data.loc[sell_date]['Close']
    predicted_profit = (sell_price - buy_price) * amount

    return jsonify({'predicted_profit': f"${predicted_profit:.2f}"})

if __name__ == '__main__':
    app.run(debug=True)
