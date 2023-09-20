import numpy as np
import pandas as pd
from keras.models import load_model
from joblib import load
import requests
from io import StringIO
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def download_data(url):
    response = requests.get(url)
    data = pd.read_csv(StringIO(response.text))
    return data

def preprocess_data(data, feature_scaler, target_scaler):
    # Separate the target column
    target = data['LAST_PRICE']
    data = data.drop(['LAST_PRICE', 'TIME'], axis=1)
    
    # Normalize the features
    data_normalized = feature_scaler.transform(data)
    
    # Normalize the target
    target_normalized = target_scaler.transform(target.values.reshape(-1, 1))
    
    return data_normalized, target_normalized

def predict_next_60_minutes(model, last_60_minutes_data, target_scaler):
    current_sequence = last_60_minutes_data.reshape(1, last_60_minutes_data.shape[0], last_60_minutes_data.shape[1])
    
    # Predict the next 60 minutes in one go
    predictions = model.predict(current_sequence)
    
    # Convert predictions to their original scale
    predictions_original = target_scaler.inverse_transform(predictions[0])
    
    return predictions_original

def convert_to_local_time(timestamp):
    utc_time = datetime.utcfromtimestamp(timestamp)
    local_time = utc_time + timedelta(hours=3)  # Convert from UTC+0 to UTC+3
    return local_time.strftime('%H:%M:%S')

def visualize_predictions(timestamps, last_prediction):
    # Convert the last timestamp to local time and add 60 minutes
    last_time = convert_to_local_time(timestamps[-10])
    last_datetime = datetime.strptime(last_time, '%H:%M:%S') + timedelta(minutes=60)
    
    # Generate local times for the next 10 minutes
    local_times = [(last_datetime + timedelta(minutes=i)).strftime('%H:%M:%S') for i in range(1, 11)]
    
    # Plot the last prediction
    plt.figure(figsize=(10, 5))
    plt.plot(local_times, last_prediction[-10:], label='Predicted Prices', color='blue')
    plt.xlabel('Time (in H:M:S)')
    plt.ylabel('Bitcoin Price')
    plt.title('Bitcoin Price Predictions for the Next 10 Minutes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Load the trained model and scalers
    model = load_model('bitcoin_lstm_model.h5')
    feature_scaler = load('feature_scaler.pkl')
    target_scaler = load('target_scaler.pkl')
    
    # Download the latest data
    url = "https://bitcoin-data-collective-rzeraat.vercel.app/api/download_btc"
    data = download_data(url)
    
    # Preprocess the data
    data_normalized, _ = preprocess_data(data, feature_scaler, target_scaler)
    
    # Predict the next 60 minutes
    last_60_minutes_data = data_normalized[-60:]
    predictions_60 = predict_next_60_minutes(model, last_60_minutes_data, target_scaler)
    
    # Extract the last 10 timestamps from the original data
    last_10_timestamps = data['TIME'].values[-60:]
    print(predictions_60[-1])
    
    # Visualize the last 10 minutes of predictions for the last sequence
    visualize_predictions(last_10_timestamps, predictions_60[-1])  # Focus on the last row and last 10 values

if __name__ == "__main__":
    main()
