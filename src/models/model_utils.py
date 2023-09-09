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

def evaluate_model(model, test_generator, scaler):

    # Evaluate the model on the test set to get MSE
    loss = model.evaluate(test_generator)

    # Calculate RMSE from the loss
    rmse = np.sqrt(loss)

    # Convert RMSE to dollar value
    rmse_in_dollar = rmse * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]

    print(f"Test RMSE (scaled): {rmse}")
    print(f"Test RMSE in dollar value: ${rmse_in_dollar:.2f}")

    return rmse  # Optionally return the RMSE for further use