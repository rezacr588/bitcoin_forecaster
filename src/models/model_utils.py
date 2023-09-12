from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import config
from sklearn.preprocessing import MinMaxScaler

def build_lstm_model(sequence_length, n_features=1):  # Default to 1 feature
    """
    Build and return an LSTM model for time series prediction.
    ...
    """
    model = Sequential()
    model.add(LSTM(config.UNITS, input_shape=(sequence_length, n_features), return_sequences=True))
    model.add(LSTM(config.UNITS))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, generator, validation_data=None, epochs=50):
    """
    Train the LSTM model.
    
    Parameters:
    - model: LSTM model to be trained.
    - generator: TimeseriesGenerator object for training data.
    - validation_data: Validation data generator (optional).
    - epochs: Number of training epochs.
    
    Returns:
    - history: Training history.
    """
    
    # Using ReduceLROnPlateau callback
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    callbacks = [reduce_lr]
    
    history = model.fit(generator, validation_data=validation_data, epochs=epochs, callbacks=callbacks)
    return history

def predict_next_hour(model, initial_sequence, scaler):
    """
    Predict the Bitcoin price for the next hour (60 minutes).
    
    Parameters:
    - model: Trained LSTM model.
    - initial_sequence: Last 60 minutes of scaled data.
    - scaler: MinMaxScaler object used for data normalization.
    
    Returns:
    - predictions: List of predicted prices for the next hour.
    """
    
    predictions = []
    current_sequence = initial_sequence.copy()
    
    for _ in range(60):  # Predict next 60 minutes
        # Reshape the sequence to match the input shape for LSTM
        current_sequence_reshaped = current_sequence.reshape((1, current_sequence.shape[0], current_sequence.shape[1]))
        
        # Predict the next minute
        predicted_scaled = model.predict(current_sequence_reshaped)
        
        # Inverse transform the prediction to original scale
        predicted_price = scaler.inverse_transform(predicted_scaled)
        
        # Append the predicted price to the predictions list
        predictions.append(predicted_price[0][0])
        
        # Append the predicted scaled value to the end of the current sequence and remove the first value
        current_sequence = np.append(current_sequence[1:], predicted_scaled, axis=0)
    
    return predictions

def evaluate_model(model, test_generator):

    # Evaluate the model on the test set to get MSE
    loss = model.evaluate(test_generator)

    # Calculate RMSE from the loss
    rmse = np.sqrt(loss)

    # Convert RMSE to dollar value
    print(f"Test MSE (scaled): {loss}")
    print(f"Test RMSE (scaled): {rmse}")
