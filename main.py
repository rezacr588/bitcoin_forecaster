from data_handler import DataHandler
from model_trainer import ModelTrainer
from utils import Utils

def main():
    # 1. Download and preprocess the data
    url = "https://bitcoin-data-collective-rzeraat.vercel.app/api/download_btc"
    data_handler = DataHandler(url)
    data_normalized, target_normalized, scaler, target_scaler = data_handler.preprocess_data()

    # Save the scalers
    Utils.save_scalers([scaler, target_scaler], ['feature_scaler.pkl', 'target_scaler.pkl'])

    # 2. Create sequences for training
    seq_length = 60
    X, y = Utils.create_sequences(data_normalized, target_normalized, seq_length)

    # 3. Initialize the model trainer
    model_trainer = ModelTrainer(X, y)

    # 4. Cross-validate the model
    model_trainer.cross_validate(n_splits=5)

    # 5. Train the model on the entire dataset
    model = model_trainer.train_model()

    # 6. Make predictions for the next 60 minutes
    last_60_minutes_data = data_normalized[-60:]
    predictions_60 = Utils.predict_next_60_minutes(model, last_60_minutes_data, target_scaler)

    # 7. Visualize the predictions
    last_10_timestamps = data_handler.data['TIME'].values[-10:]
    first_10_timestamps = data_handler.data['TIME'].values[:10]

    Utils.visualize_predictions(last_10_timestamps, predictions_60[-1][-10:], title="Bitcoin Price Predictions for the Next 10 Minutes")
    Utils.visualize_predictions(first_10_timestamps, predictions_60[-1][:10], title="Bitcoin Price Predictions for the First 10 Minutes")

if __name__ == "__main__":
    main()
