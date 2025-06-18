import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import re
import json 
import pandas as pd
# Download necessary NLTK data (only needed once)
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to clean text and remove stop words
def clean_text(text):
    # Check if text is a string and not None
    if isinstance(text, str):
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)

        # Convert to lowercase
        text = text.lower()
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [w for w in words if not w in stop_words]
        text = " ".join(words)

        return text
    else:
        # Handle None or non-string values by returning an empty string or a placeholder
        return "" # or any suitable placeholder


json_file = "msft_stock_news.json"

# Open and load JSON file
with open(json_file, "r", encoding="utf-8") as file:
    try:
        data = json.load(file)  # Load JSON data
    except json.JSONDecodeError as e:
        print(f"Error loading JSON: {e}")
        data = []

# Convert JSON data to a DataFrame
df = pd.DataFrame(data)


# Apply the cleaning function to the 'text' column
df['cleaned_text'] = df['text'].apply(clean_text)

# Create a function to assign sentiment labels based on sentiment score
def assign_sentiment_label(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


# Function to calculate sentiment score
def get_sentiment_score(text):
    scores = analyzer.polarity_scores(text)
    return scores['compound']

# Calculate sentiment scores for cleaned text and store it in the 'sentiment' column
df['sentiment_score'] = df['cleaned_text'].apply(get_sentiment_score)
# Apply the function to create the 'sentiment_label' column
df['sentiment_label'] = df['sentiment_score'].apply(assign_sentiment_label)

# Ensure 'publishedDate' is datetime
df['publishedDate'] = pd.to_datetime(df['publishedDate'])

# Sort by symbol and time
df = df.sort_values(['symbol', 'publishedDate'])

# Set the index to publishedDate for rolling window
df.set_index('publishedDate', inplace=True)

df['news_count_last_24h'] = (
    df.groupby('symbol')['title']
    .rolling('24h').count()
    .reset_index(level=0, drop=True)
)

# Average sentiment score over last 24 hours
df['average_sentiment_last_24h'] = (
    df.groupby('symbol')['sentiment_score']
    .rolling('24h').mean()
    .reset_index(level=0, drop=True)
)

# Sentiment volatility over last 24 hours
df['news_volatility_impact'] = (
    df.groupby('symbol')['sentiment_score']
    .rolling('24h').std()
    .reset_index(level=0, drop=True)
)


df = df.reset_index()

df.to_csv('msft_stock_news_sentiment.csv', index=False) 