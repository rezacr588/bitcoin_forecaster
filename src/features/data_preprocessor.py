import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import pickle

def preprocess_data(data, sequence_length=10, train_size=0.7, val_size=0.15, save_scaler=True):
    """
    Preprocess the data for LSTM training.
    ...
    """
    
    # Extract the PRICE column
    prices = data['PRICE'].values.reshape(-1, 1)
    
    train_len = int(len(prices) * train_size)
    val_len = int(len(prices) * val_size)
    
    train_prices = prices[:train_len]
    val_prices = prices[train_len:train_len + val_len]
    test_prices = prices[train_len + val_len - sequence_length:]
    
    scaler = MinMaxScaler()
    train_prices_scaled = scaler.fit_transform(train_prices)
    val_prices_scaled = scaler.transform(val_prices)
    test_prices_scaled = scaler.transform(test_prices)
    
    if save_scaler:
        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
    
    train_generator = TimeseriesGenerator(train_prices_scaled, train_prices_scaled, length=sequence_length, batch_size=1)
    val_generator = TimeseriesGenerator(val_prices_scaled, val_prices_scaled, length=sequence_length, batch_size=1)
    test_generator = TimeseriesGenerator(test_prices_scaled, test_prices_scaled, length=sequence_length, batch_size=1)
    
    return train_generator, val_generator, test_generator, scaler, test_prices_scaled
