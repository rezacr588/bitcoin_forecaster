from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import config

def build_lstm_model(sequence_length):
    """
    Build and return an LSTM model for time series prediction.
    
    Parameters:
    - sequence_length: Number of time steps the LSTM should consider for prediction.
    
    Returns:
    - model: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, input_shape=(sequence_length, 1), return_sequences=True))  # 50 LSTM units, with return_sequences=True for potential stacking
    model.add(LSTM(50))  # Another LSTM layer with 50 units
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

def predict_next_hour(model, last_sequence, scaler):
    """
    Predict the next hour's Bitcoin price.
    
    Parameters:
    - model: Trained LSTM model.
    - last_sequence: Last sequence of data to base the prediction on.
    - scaler: MinMaxScaler object used to scale the data.
    
    Returns:
    - predicted_price: Predicted price for the next hour.
    """
    # Scale the last_sequence
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
    
    # Reshape the last_sequence_scaled to match the input shape for LSTM
    last_sequence_scaled = last_sequence_scaled.reshape((1, last_sequence_scaled.shape[0], 1))
    
    predicted_scaled = model.predict(last_sequence_scaled)
    predicted_price = scaler.inverse_transform(predicted_scaled)
    return predicted_price[0][0]


def evaluate_model(model, test_generator, scaler):
    """
    Evaluate the LSTM model using various metrics.
    
    Parameters:
    - model: Trained LSTM model.
    - test_generator: TimeseriesGenerator object for test data.
    - scaler: MinMaxScaler object used for data normalization.
    
    Returns:
    - metrics: Dictionary containing MAE, MSE, RMSE, and R2 values.
    """
    # Get true values and predictions
    true_values = []
    predictions = []
    for i in range(len(test_generator)):
        x, y = test_generator[i]
        true_values.append(y[0][0])
        predictions.append(model.predict(x)[0][0])
    
    true_values = scaler.inverse_transform(np.array(true_values).reshape(-1, 1))
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Calculate metrics
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predictions)
    
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }
    
    return metrics
