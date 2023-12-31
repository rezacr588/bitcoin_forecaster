import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from joblib import dump, load
import numpy as np

class Utils:
    @staticmethod
    def convert_to_local_time(timestamp):
        utc_time = datetime.utcfromtimestamp(timestamp)
        local_time = utc_time + timedelta(hours=3)  # Convert from UTC+0 to UTC+3
        return local_time.strftime('%H:%M:%S')

    @staticmethod
    def visualize_predictions(timestamps, last_prediction, title="Bitcoin Price Predictions"):
        # Convert the last timestamp to local time and add 60 minutes
        last_time = Utils.convert_to_local_time(timestamps[-10])
        last_datetime = datetime.strptime(last_time, '%H:%M:%S') + timedelta(minutes=60)
    
        # Generate local times for the next 10 minutes
        local_times = [(last_datetime + timedelta(minutes=i)).strftime('%H:%M:%S') for i in range(1, 11)]
    
        # Plot the last prediction
        plt.figure(figsize=(10, 5))
        plt.plot(local_times, last_prediction[-10:], label='Predicted Prices', color='blue')
        plt.xlabel('Time (in H:M:S)')
        plt.ylabel('Bitcoin Price')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def save_scalers(scalers, filenames):
        for scaler, filename in zip(scalers, filenames):
            dump(scaler, filename)

    @staticmethod
    def load_scalers(filenames):
        scalers = []
        for filename in filenames:
            scalers.append(load(filename))
        return scalers

    @staticmethod
    def create_sequences(data, target, seq_length, steps_ahead=60):
        X, y = [], []
        for i in range(len(data) - seq_length - steps_ahead + 1):
            X.append(data[i:i+seq_length])
            y.append(target[i+seq_length:i+seq_length+steps_ahead])
        return np.array(X), np.array(y)

    @staticmethod
    def predict_next_60_minutes(model, last_60_minutes_data, target_scaler):
        current_sequence = last_60_minutes_data.reshape(1, last_60_minutes_data.shape[0], last_60_minutes_data.shape[1])
        predictions = model.predict(current_sequence)
        predictions_original = target_scaler.inverse_transform(predictions[0])
        return predictions_original
