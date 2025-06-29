import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self):
        self.data = None
        self.vectorizer = None
        self.encoder = None

    def load_data(self, filepath="data/Tweets.csv"):
        """Loads data from a CSV file and filters out null tweets."""
        data = pd.read_csv(filepath)
        
        # Filter out rows where tweet is null
        initial_count = len(data)
        data = data.dropna(subset=['Tweet'])
        filtered_count = len(data)
        
        print(f"Loaded {initial_count} tweets from {filepath}")
        print(f"After filtering null tweets: {filtered_count} tweets")
        print(f"Removed {initial_count - filtered_count} rows with null tweets")
        
        self.data = data
        return data

    @staticmethod
    def remove_characters(text) -> str:
        """Remove URLs, numbers, and non-letters from a given string"""
        # Handle non-string input
        if not isinstance(text, str):
            return ""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove all non-alphabetic characters (including numbers and punctuation)
        text = re.sub(r'[^a-zA-Z]', '', text)
        return text.strip()

    def clean_text(self, text) -> str:
        text = self.remove_characters(text)
        return text.strip()

    def vectorize_text(self, tweets: list[str]):
        self.vectorizer = TfidfVectorizer(max_features=2500, min_df=1, max_df=0.8)
        return self.vectorizer.fit_transform(tweets).toarray()

    def label_encoder(self, parties):
        self.encoder = LabelEncoder()
        return self.encoder.fit_transform(parties)

    def preprocess_tweets(self):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.data.Tweet = self.data.Tweet.apply(self.clean_text)
        return self.vectorize_text(self.data.Tweet.values)

    def preprocess_parties(self):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.data.Party = self.data.Party.apply(self.clean_text)
        return self.label_encoder(self.data.Party.values)

