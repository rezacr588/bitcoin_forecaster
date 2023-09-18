import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import requests
from io import StringIO
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import datetime
from datetime import datetime, timedelta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def download_data(url):
    response = requests.get(url)
    data = pd.read_csv(StringIO(response.text))
    return data

def preprocess_data(data):
    # Separate the target column
    target = data['LAST_PRICE']
    data = data.drop(['LAST_PRICE', 'TIME'], axis=1)  # Assuming 'TIME' is the timestamp and not used as a feature for training
    
    # Normalize the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)
    
    # Normalize the target
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_normalized = target_scaler.fit_transform(target.values.reshape(-1, 1))
    
    return data_normalized, target_normalized, scaler, target_scaler

def split_data(data_normalized, target_normalized):
    X_train, X_temp, y_train, y_temp = train_test_split(data_normalized, target_normalized, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.67, shuffle=False)
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_sequences(data, target, seq_length, steps_ahead=60):
    X, y = [], []
    for i in range(len(data) - seq_length - steps_ahead + 1):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length:i+seq_length+steps_ahead])
    return np.array(X), np.array(y)

def get_model(X_train):
    if os.path.exists('bitcoin_lstm_model.h5'):
        model = load_model('bitcoin_lstm_model.h5')
    else:
        model = Sequential()
        
        # First LSTM layer with dropout and batch normalization
        model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        
        # Second LSTM layer
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        
        # Third LSTM layer
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        
        # Fourth LSTM layer
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        
        # Dense layer
        model.add(Dense(60))  # Output sequence of 60 prices
        
        # Compile the model
        model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=60, validation_data=(X_val, y_val), shuffle=False, callbacks=[early_stop])
    model.save('bitcoin_lstm_model.h5')
    return history

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
    url = "https://bitcoin-data-collective-rzeraat.vercel.app/api/download_btc"
    data = download_data(url)
    data_normalized, target_normalized, scaler, target_scaler = preprocess_data(data)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(data_normalized, target_normalized)
    seq_length = 60
    X_train, y_train = create_sequences(X_train, y_train, seq_length)
    X_val, y_val = create_sequences(X_val, y_val, seq_length)
    X_test, y_test = create_sequences(X_test, y_test, seq_length)

    model = get_model(X_train)
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Print training progress details
    for epoch, loss, val_loss in zip(range(1, len(history.history['loss']) + 1), history.history['loss'], history.history['val_loss']):
        print(f"Epoch {epoch}/{len(history.history['loss'])} - loss: {loss:.4f} - val_loss: {val_loss:.4f}")
    
    # Print the model summary
    model.summary()
    
    # Predict the next 60 minutes
    last_60_minutes_data = data_normalized[-60:]
    predictions_60 = predict_next_60_minutes(model, last_60_minutes_data, target_scaler)
    
    # Extract the last 10 timestamps from the original data
    last_10_timestamps = data['TIME'].values[-10:]
    
    # Visualize the last 10 minutes of predictions for the last sequence
    visualize_predictions(last_10_timestamps, predictions_60[-1][-10:])  # Focus on the last row and last 10 values

if __name__ == "__main__":
    main()
