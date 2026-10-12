
import requests
import pandas as pd

# Read one real feature row from your synthetic test dataset
# Predict daily potable water saved if the system exist, then estimate annual savings from a 365-day forecast.
X_test = pd.read_csv("06_water_savings_test_X.csv")

sample = X_test.iloc[0].to_dict()

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={"features": sample}
)

print("HTTP status:", response.status_code)
print(response.json())
