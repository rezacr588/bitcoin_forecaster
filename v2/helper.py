import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import requests
from io import StringIO
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def download_data(url):
    response = requests.get(url)
    data = pd.read_csv(StringIO(response.text))
    return data

def preprocess_data(data):
    # Assuming 'TIME' is the timestamp and not used as a feature for training
    data = data.drop('TIME', axis=1)
    
    # Separate the target column
    target = data['last_price']
    data = data.drop('last_price', axis=1)
    
    # Normalize the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)
    
    # Normalize the target
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_normalized = target_scaler.fit_transform(target.values.reshape(-1, 1))
    
    return data_normalized, target_normalized, scaler, target_scaler

def evaluate_model(model, X_test, y_test, target_scaler):
    predictions = model.predict(X_test)
    
    # Inverting the scaling to get the original price values
    y_test_original = target_scaler.inverse_transform(y_test)
    predictions_original = target_scaler.inverse_transform(predictions)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_original, predictions_original)
    mse = mean_squared_error(y_test_original, predictions_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, predictions_original)
    
    return mae, mse, rmse, r2

def split_data(data_normalized, target_normalized):
    X_train, X_temp, y_train, y_temp = train_test_split(data_normalized, target_normalized, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.67, shuffle=False)
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)
