import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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
