import os
from keras.models import load_model
import config

from src.data.data_fetcher import fetch_bitcoin_prices
from src.features.data_preprocessor import preprocess_data
from src.models.model_utils import predict_next_hour

def predict():
    # Fetch data
    data = fetch_bitcoin_prices()

    # Load the trained model
    model_path = "models/bitcoin_lstm_model.h5"
    model = load_model(model_path)

    # Preprocess data (to get the scaler)
    _, _, scaler = preprocess_data(data, config.SEQUENCE_LENGTH)

    # Predict next hour
    last_sequence = data[['open', 'high', 'low', 'close', 'volume', 'quote_volume']].values[-config.SEQUENCE_LENGTH:]
    predicted_price = predict_next_hour(model, last_sequence, scaler)
    print(f"Predicted Price for the Next Hour: {predicted_price}")

if __name__ == "__main__":
    predict()
