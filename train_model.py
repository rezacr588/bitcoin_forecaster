import os
from keras.models import load_model
from keras.callbacks import EarlyStopping
import config
from src.data.data_fetcher import fetch_and_save_csv
from src.features.data_preprocessor import preprocess_data
from src.models.model_utils import build_lstm_model, evaluate_model
import numpy as np
import tensorflow as tf
import datetime

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
    # Define the TensorBoard callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # Train model with EarlyStopping
    model.fit(train_generator, epochs=config.EPOCHS, validation_data=val_generator, callbacks=[early_stopping, tensorboard_callback])

    # Save the trained model
    model.save(model_path)
    
    # Evaluate the model
    evaluate_model(model, test_generator)

if __name__ == "__main__":
    train()
