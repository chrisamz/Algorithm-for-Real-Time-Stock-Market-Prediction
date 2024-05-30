# sentiment_analysis.py

"""
Sentiment Analysis Module for Real-Time Stock Market Prediction

This module contains functions for analyzing the sentiment of news articles
to gauge market sentiment and inform stock market predictions.

Techniques Used:
- Text Classification
- Sentiment Scoring

Libraries/Tools:
- NLTK
- spaCy
- Transformers (BERT)
"""

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from transformers import pipeline

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

class SentimentAnalysis:
    def __init__(self, model='vader'):
        """
        Initialize the SentimentAnalysis class.
        
        :param model: str, sentiment analysis model to use ('vader', 'bert')
        """
        self.model = model
        if model == 'vader':
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        elif model == 'bert':
            self.sentiment_analyzer = pipeline('sentiment-analysis')
        else:
            raise ValueError(f"Model {model} not supported.")

    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of the input text.
        
        :param text: str, input text
        :return: dict, sentiment scores
        """
        if self.model == 'vader':
            return self.sentiment_analyzer.polarity_scores(text)
        elif self.model == 'bert':
            result = self.sentiment_analyzer(text)[0]
            return {'label': result['label'], 'score': result['score']}
        else:
            raise ValueError(f"Model {self.model} not supported.")

    def analyze_sentiment_bulk(self, texts):
        """
        Analyze the sentiment of multiple texts.
        
        :param texts: list, list of input texts
        :return: list, list of sentiment scores
        """
        return [self.analyze_sentiment(text) for text in texts]

    def preprocess_news_data(self, filepath):
        """
        Preprocess news data by analyzing sentiment scores for each article.
        
        :param filepath: str, path to the input news data file
        :return: DataFrame, news data with sentiment scores
        """
        data = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
        data['sentiment'] = data['headline'].apply(lambda x: self.analyze_sentiment(x)['compound'] if self.model == 'vader' else self.analyze_sentiment(x)['score'])
        return data

if __name__ == "__main__":
    news_data_filepath = 'data/raw/news_data.csv'
    sentiment_model = 'vader'  # Change to 'bert' for BERT-based sentiment analysis

    sentiment_analysis = SentimentAnalysis(model=sentiment_model)
    
    # Preprocess news data
    preprocessed_news_data = sentiment_analysis.preprocess_news_data(news_data_filepath)
    preprocessed_news_data.to_csv('data/processed/preprocessed_news_data_with_sentiment.csv')
    print("News data preprocessing with sentiment analysis completed and saved to 'data/processed/preprocessed_news_data_with_sentiment.csv'.")
