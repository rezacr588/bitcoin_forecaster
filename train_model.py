import os
from keras.models import load_model
from keras.callbacks import EarlyStopping
import config

from src.data.data_fetcher import fetch_bitcoin_prices
from src.features.data_preprocessor import preprocess_data
from src.models.model_utils import build_lstm_model, evaluate_model

def train():
    # Fetch data
    data = fetch_bitcoin_prices()

    # Preprocess data
    train_generator, test_generator, scaler = preprocess_data(data, config.SEQUENCE_LENGTH)

    # Check if model exists, if so, load it. Otherwise, build a new one.
    model_path = "models/bitcoin_lstm_model.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = build_lstm_model(config.SEQUENCE_LENGTH)

    # Add early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train model
    model.fit(train_generator, epochs=config.EPOCHS, validation_data=test_generator, callbacks=[early_stopping])

    # Save the trained model
    model.save(model_path)

    # Evaluate the model
    metrics = evaluate_model(model, test_generator, scaler)
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    train()
