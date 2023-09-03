from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import config

def build_lstm_model(sequence_length, n_features=5):
    """
    Build and return an LSTM model for time series prediction.
    
    Parameters:
    - sequence_length: Number of time steps the LSTM should consider for prediction.
    - n_features: Number of features in the dataset.
    
    Returns:
    - model: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, input_shape=(sequence_length, n_features), return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, generator, epochs=50):
    """
    Train the LSTM model.
    
    Parameters:
    - model: LSTM model to be trained.
    - generator: TimeseriesGenerator object for training data.
    - epochs: Number of training epochs.
    
    Returns:
    - history: Training history.
    """
    history = model.fit(generator, epochs=epochs)
    return history

def predict_next_hour(model, last_sequence_scaled, scaler):
    """
    Predict the next hour's Bitcoin price.
    
    Parameters:
    - model: Trained LSTM model.
    - last_sequence_scaled: Last scaled sequence of data to base the prediction on.
    - scaler: MinMaxScaler object used to scale the data.
    
    Returns:
    - predicted_price: Predicted price for the next hour.
    """
    
    # Reshape the last_sequence_scaled to match the input shape for LSTM
    last_sequence_reshaped = last_sequence_scaled.reshape((1, last_sequence_scaled.shape[0], last_sequence_scaled.shape[1]))
    
    predicted_scaled = model.predict(last_sequence_reshaped)
    predicted_price = scaler.inverse_transform(predicted_scaled)
    return predicted_price[0][0]