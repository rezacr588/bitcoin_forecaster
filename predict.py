import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import pickle
from src.data.data_fetcher import fetch_and_save_csv
from src.models.model_utils import predict_next_hour
import config

def predict():
    model_path = "models/bitcoin_lstm_model.h5"
    scaler_path = "models/scaler.pkl"
    sequence_length = config.SEQUENCE_LENGTH

    # Load the saved model
    model = load_model(model_path)
    
    # Load the unseen data
    data = fetch_and_save_csv()
    
    # Extract relevant features
    features = data[['open', 'high', 'low', 'close', 'volume']].values
    
    # Load the scaler used for the training data
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # Scale the unseen data using the loaded scaler
    features_scaled = scaler.transform(features)
    
    # Create a TimeseriesGenerator for the unseen data
    test_generator = TimeseriesGenerator(features_scaled, features_scaled[:, 3], length=sequence_length, batch_size=1)
    
    # Use the model to make predictions on the unseen data
    predictions_scaled = model.predict(test_generator)
    
    # Inverse transform the predictions to get them in the original scale
    predictions = scaler.inverse_transform(np.hstack((features_scaled[-len(predictions_scaled):, :-1], predictions_scaled)))
    predicted_close_prices = predictions[:, 3]
    print("Predicted close prices:")
    print(predicted_close_prices)
