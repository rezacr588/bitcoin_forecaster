from keras.models import load_model
from src.data.data_fetcher import fetch_bitcoin_prices
from src.features.data_preprocessor import preprocess_data
from src.models.model_utils import build_lstm_model, predict_next_hour
from keras.callbacks import EarlyStopping
import config
import os

# Fetch data
data = fetch_bitcoin_prices()

# Preprocess data
train_generator, test_generator, scaler = preprocess_data(data['close'].values, config.SEQUENCE_LENGTH)

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

# Predict next hour
last_sequence = data['close'].values[-config.SEQUENCE_LENGTH:]
predicted_price = predict_next_hour(model, last_sequence, scaler)
print(f"Predicted Price for the Next Hour: {predicted_price}")
