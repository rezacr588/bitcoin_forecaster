import os
from keras.models import load_model
import config

from src.data.data_fetcher import fetch_bitcoin_prices
from src.features.data_preprocessor import preprocess_data
from src.models.model_utils import evaluate_model

def evaluate():
    # Fetch data
    data = fetch_bitcoin_prices()

    # Preprocess data
    _, test_generator, scaler = preprocess_data(data, config.SEQUENCE_LENGTH)

    # Load the trained model
    model_path = "models/bitcoin_lstm_model.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        print("Model not found. Please train the model first.")
        return

    # Evaluate the model
    metrics = evaluate_model(model, test_generator, scaler)
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    evaluate()
