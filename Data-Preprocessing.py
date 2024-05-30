# data_preprocessing.py

"""
Data Preprocessing Module for Real-Time Stock Market Prediction

This module contains functions for collecting, cleaning, normalizing, and preparing
historical stock data and news articles for further analysis and modeling.

Techniques Used:
- Data cleaning
- Normalization
- Feature extraction
- Handling missing data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

class DataPreprocessing:
    def __init__(self):
        """
        Initialize the DataPreprocessing class.
        """
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def load_stock_data(self, filepath):
        """
        Load stock data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        return pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')

    def load_news_data(self, filepath):
        """
        Load news data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        return pd.read_csv(filepath, parse_dates=['date'], index_col='date')

    def clean_stock_data(self, data):
        """
        Clean the stock data by removing duplicates and handling missing values.
        
        :param data: DataFrame, input data
        :return: DataFrame, cleaned data
        """
        data = data.drop_duplicates()
        data = pd.DataFrame(self.imputer.fit_transform(data), columns=data.columns, index=data.index)
        return data

    def normalize_stock_data(self, data, columns):
        """
        Normalize the specified columns in the stock data.
        
        :param data: DataFrame, input data
        :param columns: list, columns to be normalized
        :return: DataFrame, normalized data
        """
        data[columns] = self.scaler.fit_transform(data[columns])
        return data

    def extract_features_from_news(self, data):
        """
        Extract sentiment scores from news articles.
        
        :param data: DataFrame, input data
        :return: DataFrame, data with sentiment scores
        """
        data['sentiment'] = data['headline'].apply(lambda x: self.sentiment_analyzer.polarity_scores(x)['compound'])
        return data

    def preprocess_stock_data(self, filepath, columns_to_normalize):
        """
        Execute the full preprocessing pipeline for stock data.
        
        :param filepath: str, path to the input data file
        :param columns_to_normalize: list, columns to be normalized
        :return: DataFrame, preprocessed data
        """
        data = self.load_stock_data(filepath)
        data = self.clean_stock_data(data)
        data = self.normalize_stock_data(data, columns_to_normalize)
        return data

    def preprocess_news_data(self, filepath):
        """
        Execute the full preprocessing pipeline for news data.
        
        :param filepath: str, path to the input data file
        :return: DataFrame, preprocessed data
        """
        data = self.load_news_data(filepath)
        data = self.extract_features_from_news(data)
        return data

if __name__ == "__main__":
    stock_data_filepath = 'data/raw/stock_data.csv'
    news_data_filepath = 'data/raw/news_data.csv'
    columns_to_normalize = ['Open', 'High', 'Low', 'Close', 'Volume']

    preprocessing = DataPreprocessing()

    # Preprocess stock data
    preprocessed_stock_data = preprocessing.preprocess_stock_data(stock_data_filepath, columns_to_normalize)
    preprocessed_stock_data.to_csv('data/processed/preprocessed_stock_data.csv')
    print("Stock data preprocessing completed and saved to 'data/processed/preprocessed_stock_data.csv'.")

    # Preprocess news data
    preprocessed_news_data = preprocessing.preprocess_news_data(news_data_filepath)
    preprocessed_news_data.to_csv('data/processed/preprocessed_news_data.csv')
    print("News data preprocessing completed and saved to 'data/processed/preprocessed_news_data.csv'.")
