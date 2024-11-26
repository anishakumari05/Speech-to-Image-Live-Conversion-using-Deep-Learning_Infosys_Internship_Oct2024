
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")

# Initialize the sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def analyze_sentiment(text):
    scores = sentiment_analyzer.polarity_scores(text)
    sentiment = "Positive" if scores["compound"] > 0.05 else "Negative" if scores["compound"] < -0.05 else "Neutral"
    return sentiment, scores

# Input text
text = input("Enter a text to analyze its sentiment: ")

# Analyze sentiment
sentiment, scores = analyze_sentiment(text)

# Display results
print(f"Sentiment: {sentiment}")
print(f"Sentiment Scores: {scores}")
