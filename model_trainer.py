import numpy as np
from keras.models import Model, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.layers import MultiHeadAttention
import tensorflow as tf
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from data_handler import DataHandler
import os
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model_path = 'bitcoin_lstm_model.h5'

    def create_advanced_model(self, input_shape, units=50, l1_value=0.01, l2_value=0.01, dropout_rate=0.2, learning_rate=0.001):
        inputs = Input(shape=input_shape)
        x = Bidirectional(LSTM(units, return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=l1_value, l2=l2_value)))(inputs)
        x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)
        attn_layer = MultiHeadAttention(num_heads=2, key_dim=units)
        attn_out = attn_layer(query=x, key=x, value=x)
        x = tf.keras.layers.Add()([x, attn_out])
        x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)
        x = Bidirectional(LSTM(units, return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=l1_value, l2=l2_value)))(x)
        x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)
        x = Bidirectional(LSTM(units, return_sequences=False, kernel_regularizer=regularizers.l1_l2(l1=l1_value, l2=l2_value)))(x)
        x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)
        outputs = Dense(60, kernel_regularizer=regularizers.l1_l2(l1=l1_value, l2=l2_value))(x)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def visualize_training_progress(self, history):
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss', color='blue')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def objective(self, params, X, y):
        units = int(params['units'])
        dropout_rate = params['dropout_rate']
        l1_value = params['l1_value']
        l2_value = params['l2_value']
        learning_rate = params['learning_rate']
        model = self.create_advanced_model((X.shape[1], X.shape[2]), units, l1_value, l2_value, dropout_rate, learning_rate)
        history = model.fit(X, y, epochs=10, batch_size=60, validation_split=0.2, shuffle=False, verbose=0)
        val_loss = history.history['val_loss'][-1]
        return {'loss': val_loss, 'status': STATUS_OK}

    def train_model(self):
        if os.path.exists(self.model_path):
            print("Loading existing model...")
            model = load_model(self.model_path)
        else:
            print("Creating a new model...")
            space = {
                'units': hp.quniform('units', 30, 500, 10),
                'dropout_rate': hp.uniform('dropout_rate', 0.05, 0.6),
                'l1_value': hp.loguniform('l1_value', np.log(0.00001), np.log(0.1)),
                'l2_value': hp.loguniform('l2_value', np.log(0.00001), np.log(0.1)),
                'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.1))
            }
            trials = Trials()
            best = fmin(lambda params: self.objective(params, self.X, self.y), space, algo=tpe.suggest, max_evals=50, trials=trials)
            best_units = int(best['units'])
            best_dropout_rate = best['dropout_rate']
            best_l1_value = best['l1_value']
            best_l2_value = best['l2_value']
            best_learning_rate = best['learning_rate']
            model = self.create_advanced_model((self.X.shape[1], self.X.shape[2]), best_units, best_l1_value, best_l2_value, best_dropout_rate, best_learning_rate)

        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        history = model.fit(self.X, self.y, epochs=50, batch_size=60, shuffle=False, callbacks=[early_stop])
        model.save(self.model_path)
        model.summary()
        self.visualize_training_progress(history)
        return model
