import requests
import pandas as pd

# Your API key from Financial Modeling Prep
API_KEY =  "xxxxxxxxxxG6x2nowAqKxxxxxxT" # Replace with your actual API key
symbol = 'AAPL'
limit = 40  # Number of quarters to fetch

# API endpoint
url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period=quarter&limit={limit}&apikey={API_KEY}"

# Send request
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data)
    print(df[['date', 'revenue', 'netIncome', 'eps']])
else:
    print("Failed to fetch data:", response.status_code, response.text)
