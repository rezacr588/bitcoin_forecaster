import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import requests
from io import StringIO

# Download the data
url = "https://bitcoin-data-collective-rzeraat.vercel.app/api/download_btc"
response = requests.get(url)
data = pd.read_csv(StringIO(response.text))

# Drop the TIME column (assuming it exists in the new dataset)
data = data.drop('TIME', axis=1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Split the data
train_data, test_data = train_test_split(data_normalized, test_size=0.2, shuffle=False)

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 5
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(50))
model.add(Dense(4))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test), shuffle=False)

# Make predictions
predicted_values = model.predict(X_test)

# Inverse transform the predicted values to get them back to the original scale
predicted_values_original_scale = scaler.inverse_transform(predicted_values)
y_test_original_scale = scaler.inverse_transform(y_test)

print(predicted_values_original_scale)
print(y_test_original_scale)

# Extract the LAST_PRICE values from predictions and actual values
# Assuming LAST_PRICE is the third column, change the index accordingly if different
predicted_last_price = predicted_values_original_scale[:, 2]
actual_last_price = y_test_original_scale[:, 2]

# Calculate the prediction errors in dollar terms
errors_in_dollars = predicted_last_price - actual_last_price

# Calculate the Mean Absolute Error (MAE) in dollars
mae_in_dollars = np.mean(np.abs(errors_in_dollars))

print(f"Mean Absolute Error in dollars: ${mae_in_dollars:.2f}")
