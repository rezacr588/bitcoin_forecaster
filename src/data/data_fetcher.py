# /bitcoin_forecaster/utils/data_fetcher.py

import requests
import pandas as pd
import os
import config

def fetch_bitcoin_prices():
    # Define the endpoint URL (Cryptowatch API for hourly data)
    url = config.API_ENDPOINT
    
    # Define the parameters: 3600 seconds for hourly data and a span of 6 months
    # Note: The 'after' parameter might need adjustments based on the exact date range you want.
    params = {
        "periods": "3600",
        "after": int((pd.Timestamp.now() - pd.DateOffset(months=6)).timestamp())
    }
    
    # Make the API request
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise an error for failed requests
    
    # Extract the data
    data = response.json()["result"]["3600"]
    
    # Convert the data to a DataFrame
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"])
    
    # Convert the timestamp to a readable date format
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
    
    # Define the path to save the CSV file
    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "bitcoin_hourly_prices.csv")
    
    # Save the data to a CSV file
    df.to_csv(save_path, index=False)
    
    print(f"Data fetched and saved to '{save_path}'")
    
    # Return the DataFrame
    return df

if __name__ == "__main__":
    fetch_bitcoin_prices()
