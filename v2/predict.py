import numpy as np
import pandas as pd
from joblib import load
from keras.models import load_model
import requests
from io import StringIO
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def download_data(url):
    response = requests.get(url)
    data = pd.read_csv(StringIO(response.text))
    return data

def convert_to_local_time(timestamp):
    utc_time = datetime.utcfromtimestamp(timestamp)
    local_time = utc_time + timedelta(hours=3)  # Convert from UTC+0 to UTC+3
    return local_time.strftime('%H:%M:%S')

def predict_next_60_minutes(model, last_60_minutes_data, target_scaler):
    current_sequence = last_60_minutes_data.reshape(1, last_60_minutes_data.shape[0], last_60_minutes_data.shape[1])
    predictions = model.predict(current_sequence)
    predictions_original = target_scaler.inverse_transform(predictions[0])
    return predictions_original

def visualize_predictions(timestamps, last_prediction):
    last_time = convert_to_local_time(timestamps[-10])
    last_datetime = datetime.strptime(last_time, '%H:%M:%S') + timedelta(minutes=60)
    local_times = [(last_datetime + timedelta(minutes=i)).strftime('%H:%M:%S') for i in range(1, 11)]
    
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
    url = "https://bitcoin-data-collective-rzeraat.vercel.app/api/download_btc"
    data = download_data(url)
    
    # Load the saved scalers
    scaler = load('feature_scaler.pkl')
    target_scaler = load('target_scaler.pkl')
    
    # Preprocess the unseen data using the loaded scalers
    data_normalized = scaler.transform(data.drop(['LAST_PRICE', 'TIME'], axis=1))
    
    # Load the saved model
    model = load_model('bitcoin_lstm_model.h5')
    
    # Predict the next 60 minutes
    last_60_minutes_data = data_normalized[-60:]
    predictions_60 = predict_next_60_minutes(model, last_60_minutes_data, target_scaler)
    
    # Extract the last 10 timestamps from the original data
    last_10_timestamps = data['TIME'].values[-10:]
    
    # Visualize the last 10 minutes of predictions for the last sequence
    visualize_predictions(last_10_timestamps, predictions_60[-1][-10:])

if __name__ == "__main__":
    main()
