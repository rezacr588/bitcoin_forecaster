# Assuming this is in /bitcoin_forecaster/utils/data_preprocessor.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import os

import numpy as np

def preprocess_data(data, sequence_length=10, train_size=0.8):
    """
    Preprocess the data for LSTM training.
    
    Parameters:
    - data: DataFrame containing the data.
    - sequence_length: Number of time steps for LSTM sequences.
    - train_size: Proportion of data to be used for training.
    
    Returns:
    - train_generator: TimeseriesGenerator object for training data.
    - test_generator: TimeseriesGenerator object for test data.
    - scaler: MinMaxScaler object used for data normalization.
    """
    
    # Extract features and target
    features = data[['open', 'high', 'low', 'close', 'volume', 'quote_volume']].values
    target = data['close'].values

    # Split data into training and test sets
    train_len = int(len(target) * train_size)
    train_features = features[:train_len]
    test_features = features[train_len - sequence_length:]
    train_target = target[:train_len]
    test_target = target[train_len - sequence_length:]

    # Normalize the data
    scaler = MinMaxScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    train_target_scaled = scaler.fit_transform(train_target.reshape(-1, 1))
    test_target_scaled = scaler.transform(test_target.reshape(-1, 1))

    # Create sequences
    train_generator = TimeseriesGenerator(train_features_scaled, train_target_scaled, length=sequence_length, batch_size=1)
    test_generator = TimeseriesGenerator(test_features_scaled, test_target_scaled, length=sequence_length, batch_size=1)
    
    # Save the preprocessed data
    train_df = pd.DataFrame(train_features_scaled, columns=['open', 'high', 'low', 'close', 'volume', 'quote_volume'])
    train_df['target'] = train_target_scaled
    train_df.to_csv("data/processed/train_data.csv", index=False)

    test_df = pd.DataFrame(test_features_scaled, columns=['open', 'high', 'low', 'close', 'volume', 'quote_volume'])
    test_df['target'] = test_target_scaled
    test_df.to_csv("data/processed/test_data.csv", index=False)
    
    return train_generator, test_generator, scaler