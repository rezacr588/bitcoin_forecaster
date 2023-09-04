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
    sequence_length = config.SEQUENCE_LENGTH  # or any other sequence length you've used

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
    
    # Create a TimeseriesGenerator for the unseen data
    test_generator = TimeseriesGenerator(features_scaled, features_scaled[:, 3], length=sequence_length, batch_size=1)
    
    # Evaluate the model on the unseen data to get MSE
    loss = model.evaluate(test_generator)
    
    # Calculate RMSE from the loss
    rmse = np.sqrt(loss)
    
    print(f"Test RMSE: {rmse}")
