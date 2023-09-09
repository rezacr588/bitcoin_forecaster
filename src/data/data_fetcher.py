# /bitcoin_forecaster/utils/data_fetcher.py

import requests
import pandas as pd
import os
import config
import io  # <-- Import the io module

def fetch_and_save_csv():
    # Define the endpoint URL (Your API endpoint to download the CSV)
    url = config.API_ENDPOINT + "/download"
    
    # Make the API request to get the CSV data
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for failed requests
    
    # Convert the CSV data to a DataFrame using io.StringIO
    csv_data = response.text
    df = pd.read_csv(io.StringIO(csv_data)) # pylint: disable=abstract-class-instantiated
    
    # Define the path to save the CSV file
    save_path = os.path.join("data", "raw", "bitcoin_prices.csv")
    
    # Save the data to a CSV file
    df.to_csv(save_path, index=False)
    
    print(f"Data fetched and saved to '{save_path}'")
    
    # Return the DataFrame
    return df