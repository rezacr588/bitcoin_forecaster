import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import pickle
from src.data.data_fetcher import fetch_and_save_csv
import config

def evaluate():
    model_path = "models/bitcoin_lstm_model.h5"
    scaler_path = "models/scaler.pkl"
    sequence_length = config.SEQUENCE_LENGTH

    # Load the saved model
    model = load_model(model_path)
    
    # Load the unseen data
    data = fetch_and_save_csv()

    # Ensure the data has the 'price' column
    if 'price' not in data.columns:
        print("Fetched data does not have the 'price' column.")
        return
    
    # Extract the 'price' feature
    features = data[['price']].values
    
    # Load the scaler used for the training data
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # Scale the unseen data using the loaded scaler
    features_scaled = scaler.transform(features)
    
    # Split the unseen data into validation and test sets
    val_size = int(0.2 * len(features_scaled))  # 20% of the data for validation
    val_features_scaled = features_scaled[-(val_size + sequence_length):-sequence_length]
    test_features_scaled = features_scaled[-(val_size + 2*sequence_length):]
    
    # Create TimeseriesGenerators for validation and test data
    val_generator = TimeseriesGenerator(val_features_scaled, val_features_scaled, length=sequence_length, batch_size=1)
    test_generator = TimeseriesGenerator(test_features_scaled, test_features_scaled, length=sequence_length, batch_size=1)
    
    # Evaluate the model on the test set to get MSE
    loss = model.evaluate(test_generator)
    
    # Calculate RMSE from the loss
    rmse = np.sqrt(loss)
    
    print(f"Test RMSE: {rmse}")

    return rmse  # Optionally return the RMSE for further use
