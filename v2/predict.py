import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
from io import StringIO
import os

print(tf.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def download_data(url):
    response = requests.get(url)
    data = pd.read_csv(StringIO(response.text))
    return data

def preprocess_data(data):
    data = data.drop('TIME', axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler

def create_sequences(data, seq_length, steps_ahead=60):
    X, y = [], []
    for i in range(len(data) - seq_length - steps_ahead + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length+steps_ahead-1])
    return np.array(X), np.array(y)

def make_predictions(model, X_test, scaler):
    predicted_values = model.predict(X_test)
    predicted_values_original_scale = scaler.inverse_transform(predicted_values)
    return predicted_values_original_scale

def calculate_mae(predicted_values_original_scale, y_test, scaler):
    y_test_original_scale = scaler.inverse_transform(y_test)
    predicted_last_price = predicted_values_original_scale[:, 2]
    actual_last_price = y_test_original_scale[:, 2]
    errors_in_dollars = predicted_last_price - actual_last_price
    mae_in_dollars = np.mean(np.abs(errors_in_dollars))
    return mae_in_dollars

def main():
    # Load the pre-trained model
    model = load_model('bitcoin_lstm_model.h5')
    
    # Download and preprocess the data
    url = "https://bitcoin-data-collective-rzeraat.vercel.app/api/download_btc"
    data = download_data(url)
    data_normalized, scaler = preprocess_data(data)
    
    # Create sequences for the entire dataset (for demonstration purposes)
    seq_length = 60
    X, y = create_sequences(data_normalized, seq_length)
    
    # Make predictions
    predicted_values_original_scale = make_predictions(model, X, scaler)
    
    # Calculate MAE
    mae_in_dollars = calculate_mae(predicted_values_original_scale, y, scaler)
    
    # Print the predicted dollar values
    print("Predicted Dollar Values:")
    for value in predicted_values_original_scale[:, 2]:  # Assuming LAST_PRICE is the third column
        print(f"${value:.2f}")
    
    print(f"\nMean Absolute Error in dollars: ${mae_in_dollars:.2f}")
    model.summary()

if __name__ == "__main__":
    main()
