import os
from keras.models import load_model
from keras.callbacks import EarlyStopping
import config
from src.data.data_fetcher import fetch_and_save_csv
from src.features.data_preprocessor import preprocess_data
from src.models.model_utils import build_lstm_model, predict_next_hour, evaluate_model
import numpy as np

def train():
    # Fetch data
    data = fetch_and_save_csv()

    # Preprocess data
    train_generator, val_generator, test_generator, scaler, test_features_scaled = preprocess_data(data, config.SEQUENCE_LENGTH)

    # Determine the number of features from the preprocessed data
    x, _ = train_generator[0]
    # Check if model exists, if so, load it. Otherwise, build a new one.
    model_path = "models/bitcoin_lstm_model.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = build_lstm_model(config.SEQUENCE_LENGTH)

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train model with EarlyStopping
    model.fit(train_generator, epochs=config.EPOCHS, validation_data=val_generator, callbacks=[early_stopping])

    # Save the trained model
    model.save(model_path)
    
    # Evaluate the model
    evaluate_model(model, test_generator, scaler)

    # Predict the next hour's closing price
    last_sequence = test_features_scaled[-config.SEQUENCE_LENGTH:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    predicted_scaled = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform(predicted_scaled)
    
    print(f"Predicted closing price for the next hour: {predicted_price[0][0]} USD")

if __name__ == "__main__":
    train()
