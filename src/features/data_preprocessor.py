import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import pickle

def preprocess_data(data, sequence_length=10, train_size=0.7, val_size=0.15, save_scaler=True, save_preprocessed=True):
    """
    Preprocess the data for LSTM training.
    
    Parameters:
    - data: DataFrame containing the data.
    - sequence_length: Number of time steps for LSTM sequences.
    - train_size: Proportion of data to be used for training.
    - val_size: Proportion of data to be used for validation.
    - save_scaler: Boolean to control if the scaler should be saved.
    - save_preprocessed: Boolean to control if the preprocessed data should be saved.
    
    Returns:
    - train_generator: TimeseriesGenerator object for training data.
    - val_generator: TimeseriesGenerator object for validation data.
    - test_generator: TimeseriesGenerator object for test data.
    - scaler: MinMaxScaler object used for data normalization.
    """
    
    features = data[['open', 'high', 'low', 'close', 'volume']].values
    
    train_len = int(len(features) * train_size)
    val_len = int(len(features) * val_size)
    
    train_features = features[:train_len]
    val_features = features[train_len:train_len + val_len]
    test_features = features[train_len + val_len - sequence_length:]
    
    scaler = MinMaxScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)
    
    if save_scaler:
        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    if save_preprocessed:
        train_df = pd.DataFrame(train_features_scaled, columns=['open', 'high', 'low', 'close', 'volume'])
        val_df = pd.DataFrame(val_features_scaled, columns=['open', 'high', 'low', 'close', 'volume'])
        test_df = pd.DataFrame(test_features_scaled, columns=['open', 'high', 'low', 'close', 'volume'])
        train_df.to_csv("data/processed/train_preprocessed.csv", index=False)
        val_df.to_csv("data/processed/val_preprocessed.csv", index=False)
        test_df.to_csv("data/processed/test_preprocessed.csv", index=False)
    
    train_generator = TimeseriesGenerator(train_features_scaled, train_features_scaled[:, 3], length=sequence_length, batch_size=1)
    val_generator = TimeseriesGenerator(val_features_scaled, val_features_scaled[:, 3], length=sequence_length, batch_size=1)
    test_generator = TimeseriesGenerator(test_features_scaled, test_features_scaled[:, 3], length=sequence_length, batch_size=1)
    
    return train_generator, val_generator, test_generator, scaler
