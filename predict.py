from keras.models import load_model
import pickle
from src.data.data_fetcher import fetch_bitcoin_prices
import config

def preprocess_unseen_data(data, sequence_length, scaler_path="models/scaler.pkl"):
    """
    Preprocess the unseen data for LSTM prediction.
    
    Parameters:
    - data: DataFrame containing the unseen data.
    - sequence_length: Number of time steps for LSTM sequences.
    - scaler_path: Path to the saved scaler.
    
    Returns:
    - last_sequence_scaled: The last sequence from the unseen data, scaled and ready for prediction.
    - scaler: The scaler used for normalization.
    """
    
    # Load the saved scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Extract features from the unseen data
    features = data[['open', 'high', 'low', 'close', 'volume']].values

    # Normalize the features using the saved scaler
    features_scaled = scaler.transform(features)

    # Extract the last sequence for prediction
    last_sequence_scaled = features_scaled[-sequence_length:]
    
    return last_sequence_scaled, scaler

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
    return predicted_price[0]

def predict():
    """
    Predict on unseen data using the trained LSTM model.
    
    Parameters:
    - data: DataFrame containing the unseen data.
    - model_path: Path to the trained LSTM model.
    - sequence_length: Number of time steps for LSTM sequences.
    
    Returns:
    - predicted_price: Predicted price for the next hour.
    """
    unseen_data = fetch_bitcoin_prices()
    model_path="models/bitcoin_lstm_model.h5"
    sequence_length=config.SEQUENCE_LENGTH
    # Preprocess the unseen data
    last_sequence_scaled, scaler = preprocess_unseen_data(unseen_data, sequence_length)

    # Load the trained LSTM model
    model = load_model(model_path)

    # Predict using the LSTM model
    predicted_price = predict_next_hour(model, last_sequence_scaled, scaler)
    print(f"Predicted Bitcoin Price for the Next Hour: ${predicted_price:.2f}")
