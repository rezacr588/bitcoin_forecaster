from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import config

def build_lstm_model(sequence_length, n_features=6):
    """
    Build and return an LSTM model for time series prediction.
    
    Parameters:
    - sequence_length: Number of time steps the LSTM should consider for prediction.
    - n_features: Number of features in the dataset.
    
    Returns:
    - model: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=(sequence_length, n_features)))
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

def predict_next_hour(model, data, scaler):
    # Extract the last sequence from the data
    last_sequence_df = data.tail(config.SEQUENCE_LENGTH)
    
    # Ensure the sequence contains all the required features
    last_sequence = last_sequence_df[['open', 'high', 'low', 'close', 'volume', 'quote_volume']].values
    
    # Scale the last_sequence
    last_sequence_scaled = scaler.transform(last_sequence)
    
    # Reshape the last_sequence_scaled to match the input shape for LSTM
    last_sequence_scaled = last_sequence_scaled.reshape((1, last_sequence_scaled.shape[0], last_sequence_scaled.shape[1]))
    
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
