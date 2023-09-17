import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
from io import StringIO
import os

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

def split_data(data_normalized):
    train_data, temp = train_test_split(data_normalized, test_size=0.3, shuffle=False)
    val_data, test_data = train_test_split(temp, test_size=0.67, shuffle=False)
    return train_data, val_data, test_data

def create_sequences(data, seq_length, steps_ahead=60):
    X, y = [], []
    for i in range(len(data) - seq_length - steps_ahead + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length+steps_ahead-1])
    return np.array(X), np.array(y)

def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    
    # Extracting the price column (assuming it's the third column)
    y_test_price = y_test[:, 2]
    predictions_price = predictions[:, 2]
    
    # Inverting the scaling to get the original price values
    y_test_original = scaler.inverse_transform(y_test)[:, 2]
    predictions_original = scaler.inverse_transform(predictions)[:, 2]
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_original, predictions_original)
    mse = mean_squared_error(y_test_original, predictions_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, predictions_original)
    
    return mae, mse, rmse, r2

def main():
    url = "https://bitcoin-data-collective-rzeraat.vercel.app/api/download_btc"
    data = download_data(url)
    data_normalized, scaler = preprocess_data(data)
    _, _, test_data = split_data(data_normalized)
    seq_length = 60
    X_test, y_test = create_sequences(test_data, seq_length)
    
    # Load the trained model
    model = load_model('bitcoin_lstm_model.h5')
    
    # Evaluate the model
    mae, mse, rmse, r2 = evaluate_model(model, X_test, y_test, scaler)
    
    # Print evaluation metrics
    print(f"Mean Absolute Error (MAE) for Price: ${mae:.2f}")
    print(f"Mean Squared Error (MSE) for Price: {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE) for Price: {rmse:.4f}")
    print(f"R-squared (R2) for Price: {r2:.4f}")

if __name__ == "__main__":
    main()
