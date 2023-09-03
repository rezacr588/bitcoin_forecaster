import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import pickle

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
    
    # Extract target
    target = data['close'].values

    # Split data into training and test sets
    train_len = int(len(target) * train_size)
    train_target = target[:train_len]
    test_target = target[train_len - sequence_length:]

    # Normalize the target
    scaler = MinMaxScaler()
    train_target_scaled = scaler.fit_transform(train_target.reshape(-1, 1))
    test_target_scaled = scaler.transform(test_target.reshape(-1, 1))

    # Save the scaler
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Create sequences
    train_generator = TimeseriesGenerator(train_target_scaled, train_target_scaled, length=sequence_length, batch_size=1)
    test_generator = TimeseriesGenerator(test_target_scaled, test_target_scaled, length=sequence_length, batch_size=1)
    
    return train_generator, test_generator, scaler
