# Assuming this is in /bitcoin_forecaster/utils/data_preprocessor.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import os

def preprocess_data(closing_prices, sequence_length=10, train_size=0.8):
    """
    Preprocess the data for LSTM training.
    
    Parameters:
    - closing_prices: Array of closing prices.
    - sequence_length: Number of time steps for LSTM sequences.
    - train_size: Proportion of data to be used for training.
    
    Returns:
    - train_generator: TimeseriesGenerator object for training data.
    - test_generator: TimeseriesGenerator object for test data.
    - scaler: MinMaxScaler object used for data normalization.
    """
    
    # Split data into training and test sets
    train_len = int(len(closing_prices) * train_size)
    train_data = closing_prices[:train_len]
    test_data = closing_prices[train_len - sequence_length:]

    # Reshape the closing prices
    train_data = train_data.reshape(-1, 1)
    test_data = test_data.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # Create sequences
    train_generator = TimeseriesGenerator(train_scaled, train_scaled, length=sequence_length, batch_size=1)
    test_generator = TimeseriesGenerator(test_scaled, test_scaled, length=sequence_length, batch_size=1)
    
    return train_generator, test_generator, scaler
