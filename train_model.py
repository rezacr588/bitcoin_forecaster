import os
from keras.models import load_model
from keras.callbacks import EarlyStopping
import config
from src.data.data_fetcher import fetch_bitcoin_prices
from src.features.data_preprocessor import preprocess_data
from src.models.model_utils import build_lstm_model, predict_next_hour
import numpy as np

def train():
    # Fetch data
    data = fetch_bitcoin_prices()

    # Preprocess data
    train_generator, val_generator, test_generator, scaler = preprocess_data(data, config.SEQUENCE_LENGTH)

    # Determine the number of features from the preprocessed data
    x, _ = train_generator[0]
    n_features = x.shape[2]

    # Check if model exists, if so, load it. Otherwise, build a new one.
    model_path = "models/bitcoin_lstm_model.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = build_lstm_model(config.SEQUENCE_LENGTH, n_features)

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train model with EarlyStopping
    model.fit(train_generator, epochs=config.EPOCHS, validation_data=val_generator, callbacks=[early_stopping])

    # Save the trained model
    model.save(model_path)
    
    # Evaluate the model on the test set to get the loss (MSE)
    loss = model.evaluate(test_generator)

    # Calculate RMSE from the loss
    rmse = np.sqrt(loss)
    
    print(f"Test LOSS: {loss}, Test RMSE: {rmse}")


if __name__ == "__main__":
    train()
