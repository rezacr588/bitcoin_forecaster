import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
from io import StringIO
import os
from datetime import datetime, timedelta
from .helper import download_data

print(tf.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess_data(data):
    timestamps = data['TIME'].values  # Save the timestamps
    data = data.drop('TIME', axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler, timestamps

def main():
    # Load the pre-trained model
    model = load_model('bitcoin_lstm_model.h5')
    
    # Download and preprocess the data
    url = "https://bitcoin-data-collective-rzeraat.vercel.app/api/download_btc"
    data = download_data(url)
    data_normalized, scaler, timestamps = preprocess_data(data)
    
    # Convert Unix timestamps to readable datetime format
    readable_timestamps = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps]
    
    # Extract the last 60 data points to predict 60 minutes into the future
    last_60_data_points = data_normalized[-60:]
    last_data_point_time = readable_timestamps[-1]
    
    # Get the last actual value in its original scale
    last_actual_value = scaler.inverse_transform([data_normalized[-1]])[0][2]
    
    # Print the last data point's price and timestamp
    print(f"Last data point timestamp: {last_data_point_time}")
    print(f"Last data point price: ${last_actual_value:.2f}\n")
    
    # Predict the value 60 minutes into the future
    predicted_value_normalized = model.predict(last_60_data_points.reshape(1, 60, 4))
    predicted_value_original_scale = scaler.inverse_transform(predicted_value_normalized)
    
    # Calculate the percentage difference from the last data point
    predicted_value = predicted_value_original_scale[0][2]
    percentage_diff_from_last = ((predicted_value - last_actual_value) / last_actual_value) * 100
    
    # Print the results
    future_time = datetime.strptime(last_data_point_time, '%Y-%m-%d %H:%M:%S') + timedelta(minutes=60)
    print(f"Predicted value of Bitcoin at {future_time.strftime('%Y-%m-%d %H:%M:%S')}: ${predicted_value:.2f}")
    print(f"Percentage Difference from Last Data Point: {percentage_diff_from_last:.2f}%")

if __name__ == "__main__":
    main()
