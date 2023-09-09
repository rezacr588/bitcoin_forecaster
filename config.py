import pandas as pd
import os

API_ENDPOINT = "https://bitcoin-data-collective.vercel.app/api"
SEQUENCE_LENGTH = 60 
EPOCHS = 50
UNITS = 150
SAVE_PATH = os.path.join("data", "raw", "bitcoin_hourly_prices.csv")
