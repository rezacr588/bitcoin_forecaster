from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    explained_variance_score, max_error, mean_absolute_percentage_error, 
    mean_squared_log_error, median_absolute_error
)
import numpy as np
import pandas as pd
from src.data.data_fetcher import fetch_bitcoin_prices
from src.features.data_preprocessor import preprocess_data
import config

def evaluate_comprehensive(model, test_generator, scaler):
    """
    Evaluate the LSTM model using a comprehensive set of metrics.
    """
    true_values = []
    predictions = []
    
    for i in range(len(test_generator)):
        x, y = test_generator[i]
        pred = model.predict(x)
        
        true_values.append(y[0])  # y is of shape (1, num_features)
        predictions.append(pred[0])  # pred is of shape (1, num_features)

    true_values = np.array(true_values)
    predictions = np.array(predictions)

    # Inverse transform the predictions
    true_close_prices = scaler.inverse_transform(true_values)[:, 3]
    predicted_close_prices = scaler.inverse_transform(predictions)[:, 3]


    # Calculate metrics
    metrics = {
        "MAE": mean_absolute_error(true_close_prices, predicted_close_prices),
        "MSE": mean_squared_error(true_close_prices, predicted_close_prices),
        "RMSE": np.sqrt(mean_squared_error(true_close_prices, predicted_close_prices)),
        "R2": r2_score(true_close_prices, predicted_close_prices),
        "Explained Variance": explained_variance_score(true_close_prices, predicted_close_prices),
        "Max Error": max_error(true_close_prices, predicted_close_prices),
        "Mean Absolute Percentage Error": mean_absolute_percentage_error(true_close_prices, predicted_close_prices),
        "Mean Squared Log Error": mean_squared_log_error(true_close_prices, predicted_close_prices),
        "Median Absolute Error": median_absolute_error(true_close_prices, predicted_close_prices)
    }
    
    return metrics

def interpret_metrics(metrics):
    """
    Interpret the metrics and provide explanations.
    
    Parameters:
    - metrics: Dictionary containing various evaluation metrics.
    
    Returns:
    - interpretations: Dictionary containing explanations and evaluations for each metric.
    """
    interpretations = {}
    
    # MAE
    interpretations["MAE"] = {
        "explanation": "Mean Absolute Error: Represents the average absolute difference between the actual and predicted values.",
        "evaluation": "Good" if metrics["MAE"] < 0.1 else "Poor"
    }
    
    # MSE
    interpretations["MSE"] = {
        "explanation": "Mean Squared Error: Represents the average squared difference between the actual and predicted values.",
        "evaluation": "Good" if metrics["MSE"] < 0.1 else "Poor"
    }
    
    # RMSE
    interpretations["RMSE"] = {
        "explanation": "Root Mean Squared Error: Square root of MSE. Represents the standard deviation of the residuals.",
        "evaluation": "Good" if metrics["RMSE"] < 0.1 else "Poor"
    }
    
    # R2
    interpretations["R2"] = {
        "explanation": "R-squared: Represents the proportion of variance in the dependent variable that's explained by the independent variables.",
        "evaluation": "Good" if metrics["R2"] > 0.9 else "Poor"
    }
    
    # Explained Variance
    interpretations["Explained Variance"] = {
        "explanation": "Explained Variance: Measures the proportion of total variance captured by the model.",
        "evaluation": "Good" if metrics["Explained Variance"] > 0.9 else "Poor"
    }
    
    # Max Error
    interpretations["Max Error"] = {
        "explanation": "Max Error: Represents the maximum residual error between the predicted and actual values.",
        "evaluation": "Good" if metrics["Max Error"] < 0.1 else "Poor"
    }
    
    # Mean Absolute Percentage Error
    interpretations["Mean Absolute Percentage Error"] = {
        "explanation": "Mean Absolute Percentage Error: Represents the average percentage error between the predicted and actual values.",
        "evaluation": "Good" if metrics["Mean Absolute Percentage Error"] < 10 else "Poor"
    }
    
    # Mean Squared Log Error
    interpretations["Mean Squared Log Error"] = {
        "explanation": "Mean Squared Log Error: Measures the squared log error between the predicted and actual values.",
        "evaluation": "Good" if metrics["Mean Squared Log Error"] < 0.1 else "Poor"
    }
    
    # Median Absolute Error
    interpretations["Median Absolute Error"] = {
        "explanation": "Median Absolute Error: Represents the median absolute difference between the predicted and actual values.",
        "evaluation": "Good" if metrics["Median Absolute Error"] < 0.1 else "Poor"
    }
    
    return interpretations

def evaluate():
    # Load the data and preprocess
    data = fetch_bitcoin_prices()
    _, test_generator, scaler = preprocess_data(data, config.SEQUENCE_LENGTH, save_scaler=False)

    # Load the trained model
    model_path = "models/bitcoin_lstm_model.h5"
    model = load_model(model_path)

    # Evaluate the model using comprehensive metrics
    metrics = evaluate_comprehensive(model, test_generator, scaler)
    
    # Interpret the metrics
    interpretations = interpret_metrics(metrics)
    
    # Print the metrics with explanations
    for metric, values in interpretations.items():
        print(f"{metric} ({values['explanation']}): {metrics[metric]:.4f}")
        print(f"Evaluation: {values['evaluation']}\n")
