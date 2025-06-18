import requests
import json
import datetime

# API key and base URL
api_key = "m8TZJWQFGH7G6x2nowAqKdzDfAyakr0T"
base_url = "https://financialmodelingprep.com/api/v3/stock_news"

# Set the tickers
tickers = "MSFT"

# Function to fetch stock news for a given month and year
def fetch_stock_news(year, month):
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year + 1}-01-01" if month == 12 else f"{year}-{month + 1:02d}-01"
    url = f"{base_url}?tickers={tickers}&page=1&from={start_date}&to={end_date}&apikey={api_key}"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None

# List to store all news data
news_collection = []

# Loop through each year from 2015 to current year
for year in range(2015, datetime.datetime.now().year + 1):
    for month in range(1, 13):
        print(f"Fetching news for {year}-{month:02d}")
        news_data = fetch_stock_news(year, month)
        if news_data:
            news_collection.extend(news_data)

# Save data to a single JSON file
with open("msft_stock.json", "w") as file:
    json.dump(news_collection, file, indent=4)

print("News collection complete. Data saved to 'msft_stock_news.json'.")
