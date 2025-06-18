import json
import pandas as pd

# Path to the JSON file
json_file = "/content/msft_stock_news_2015_to_present.json"

# Open and load JSON file
with open(json_file, "r", encoding="utf-8") as file:
    try:
        data = json.load(file)  # Load JSON data
    except json.JSONDecodeError as e:
        print(f"Error loading JSON: {e}")
        data = []

# Convert JSON data to a DataFrame
df = pd.DataFrame(data)


df
