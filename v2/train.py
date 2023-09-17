import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import requests
from io import StringIO
import os
from helper import download_data, preprocess_data, split_data, create_sequences 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_model(X_train):
    if os.path.exists('bitcoin_lstm_model.h5'):
        model = load_model('bitcoin_lstm_model.h5')
    else:
        model = Sequential()
        model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(4))
        model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_val, y_val), shuffle=False, callbacks=[early_stop])
    model.save('bitcoin_lstm_model.h5')
    return history

def main():
    url = "https://bitcoin-data-collective-rzeraat.vercel.app/api/download_btc"
    data = download_data(url)
    data_normalized, scaler = preprocess_data(data)
    train_data, val_data, test_data = split_data(data_normalized)
    seq_length = 60
    X_train, y_train = create_sequences(train_data, seq_length)
    X_val, y_val = create_sequences(val_data, seq_length)
    model = get_model(X_train)
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Print training progress details
    for epoch, loss, val_loss in zip(range(1, len(history.history['loss']) + 1), history.history['loss'], history.history['val_loss']):
        print(f"Epoch {epoch}/{len(history.history['loss'])} - loss: {loss:.4f} - val_loss: {val_loss:.4f}")
    
    # Print the model summary
    model.summary()

if __name__ == "__main__":
    main()
