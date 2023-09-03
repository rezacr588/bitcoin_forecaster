from keras.models import load_model
import joblib
from src.data.data_fetcher import fetch_bitcoin_prices
import config

def load_resources(model_path, scaler_path):
    """Load the trained model and the scaler."""
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def extract_last_sequence(data, sequence_length):
    """Extract the last sequence from the data."""
    return data['close'].tail(sequence_length).values.astype('float32').reshape(-1, 1)

def predict_next_hour(model, last_sequence, scaler):
    """Predict the next hour's Bitcoin price."""
    # Ensure the input shape matches the model's input shape
    last_sequence_reshaped = last_sequence.reshape((1, last_sequence.shape[0], 1))
    
    # Predict the next hour's price using the model
    predicted_scaled = model.predict(last_sequence_reshaped)
    
    # Inverse transform the prediction to get the actual price
    predicted_price = scaler.inverse_transform(predicted_scaled)
    
    return predicted_price[0][0]

def predict():
    # Constants
    MODEL_PATH = "models/bitcoin_lstm_model.h5"
    SCALER_PATH = "models/scaler.pkl"

    # Load necessary resources
    model, scaler = load_resources(MODEL_PATH, SCALER_PATH)

    # Fetch the latest data
    data = fetch_bitcoin_prices()

    # Extract the last sequence for prediction
    last_sequence = extract_last_sequence(data, config.SEQUENCE_LENGTH)

    # Predict the next hour's price using the predict_next_hour function
    predicted_price = predict_next_hour(model, last_sequence, scaler)

    print(f"Predicted Bitcoin Price for the Next Hour: ${predicted_price:.2f}")

if __name__ == "__main__":
    predict()
