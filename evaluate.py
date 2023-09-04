import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import pickle
from src.data.data_fetcher import fetch_bitcoin_prices
import config

def evaluate():
    model_path = "models/bitcoin_lstm_model.h5"
    scaler_path = "models/scaler.pkl"
    sequence_length = config.SEQUENCE_LENGTH

    # Load the saved model
    model = load_model(model_path)
    
    # Load the unseen data
    data = fetch_bitcoin_prices()
    
    # Extract relevant features
    features = data[['open', 'high', 'low', 'close', 'volume']].values
    
    # Load the scaler used for the training data
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # Scale the unseen data using the loaded scaler
    features_scaled = scaler.transform(features)
    
    # Split the unseen data into validation and test sets
    val_size = int(0.2 * len(features_scaled))  # 20% of the data for validation
    val_features_scaled = features_scaled[-(val_size + sequence_length):-sequence_length]
    test_features_scaled = features_scaled[-sequence_length:]
    
    # Create TimeseriesGenerators for validation and test data
    val_generator = TimeseriesGenerator(val_features_scaled, val_features_scaled[:, 3], length=sequence_length, batch_size=1)
    test_generator = TimeseriesGenerator(test_features_scaled, test_features_scaled[:, 3], length=sequence_length, batch_size=1)
    
    # Evaluate the model on the test set to get MSE
    loss = model.evaluate(test_generator)
    
    # Calculate RMSE from the loss
    rmse = np.sqrt(loss)
    
    print(f"Test RMSE: {rmse}")
