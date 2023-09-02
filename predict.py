from keras.models import load_model
import joblib
from src.data.data_fetcher import fetch_bitcoin_prices
from src.features.data_preprocessor import preprocess_data
from src.models.model_utils import predict_next_hour

def predict():
    # Fetch data
    data = fetch_bitcoin_prices()

    # Load the model
    model_path = "models/bitcoin_lstm_model.h5"
    model = load_model(model_path)

    # Load the scaler
    scaler = joblib.load('scaler.pkl')

    # Predict the next hour's price
    predicted_price = predict_next_hour(model, data, scaler)
    print(f"Predicted Bitcoin Price for the Next Hour: ${predicted_price:.2f}")

if __name__ == "__main__":
    predict()
