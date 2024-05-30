# predictive_modeling.py

"""
Predictive Modeling Module for Real-Time Stock Market Prediction

This module contains functions for building, training, and evaluating predictive models
to forecast future stock prices and market trends.

Techniques Used:
- Regression
- Classification
- Ensemble Methods

Algorithms Used:
- Linear Regression
- Random Forest
- Gradient Boosting
- LSTM

Metrics Used:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import joblib

class PredictiveModeling:
    def __init__(self):
        """
        Initialize the PredictiveModeling class.
        """
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(),
            'gradient_boosting': GradientBoostingRegressor()
        }

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        return pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')

    def split_data(self, data, target_column, test_size=0.2):
        """
        Split the data into training and testing sets.
        
        :param data: DataFrame, input data
        :param target_column: str, name of the target column
        :param test_size: float, proportion of the data to include in the test split
        :return: tuples, training and testing sets (X_train, X_test, y_train, y_test)
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=42)

    def train_model(self, model_name, X_train, y_train):
        """
        Train the specified model.
        
        :param model_name: str, name of the model to train
        :param X_train: DataFrame, training features
        :param y_train: Series, training target
        """
        model = self.models.get(model_name)
        if model:
            model.fit(X_train, y_train)
            return model
        else:
            raise ValueError(f"Model {model_name} not found.")

    def train_lstm(self, data, target_column, n_lag=1, n_ahead=1, n_epochs=50, n_batch=1, n_neurons=50):
        """
        Train an LSTM model.
        
        :param data: DataFrame, time series data
        :param target_column: str, name of the target column
        :param n_lag: int, number of lag observations as input
        :param n_ahead: int, number of observations as output
        :param n_epochs: int, number of epochs for training
        :param n_batch: int, number of batches for training
        :param n_neurons: int, number of neurons in the LSTM layer
        :return: LSTM model
        """
        X, y = self.create_lagged_features(data[target_column], n_lag, n_ahead)
        X = X.reshape(X.shape[0], 1, X.shape[1])
        
        model = Sequential()
        model.add(LSTM(n_neurons, input_shape=(1, n_lag)))
        model.add(Dense(n_ahead))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=2, shuffle=False)
        
        self.models['lstm'] = model
        return model

    def create_lagged_features(self, data, n_lag=1, n_ahead=1):
        """
        Create lagged features for time series data.
        
        :param data: Series, time series data
        :param n_lag: int, number of lag observations as input
        :param n_ahead: int, number of observations as output
        :return: tuple, input and output arrays for the model
        """
        X, y = [], []
        for i in range(len(data) - n_lag - n_ahead + 1):
            X.append(data[i:(i + n_lag)].values)
            y.append(data[(i + n_lag):(i + n_lag + n_ahead)].values)
        return np.array(X), np.array(y)

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the specified model using various metrics.
        
        :param model: trained model
        :param X_test: DataFrame, testing features
        :param y_test: Series, testing target
        :return: dict, evaluation metrics
        """
        y_pred = model.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        return metrics

    def evaluate_lstm(self, model, data, target_column, n_lag=1, n_ahead=1):
        """
        Evaluate the LSTM model using various metrics.
        
        :param model: trained LSTM model
        :param data: DataFrame, input data
        :param target_column: str, name of the target column
        :param n_lag: int, number of lag observations as input
        :param n_ahead: int, number of observations as output
        :return: dict, evaluation metrics
        """
        X, y_true = self.create_lagged_features(data[target_column], n_lag, n_ahead)
        X = X.reshape(X.shape[0], 1, X.shape[1])
        y_pred = model.predict(X).flatten()
        
        metrics = {
            'mae': mean_absolute_error(y_true.flatten(), y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true.flatten(), y_pred))
        }
        return metrics

    def save_model(self, model_name, filepath):
        """
        Save the trained model to a file.
        
        :param model_name: str, name of the model to save
        :param filepath: str, path to save the model
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found.")
        joblib.dump(model, filepath)

    def load_model(self, model_name, filepath):
        """
        Load a trained model from a file.
        
        :param model_name: str, name to assign to the loaded model
        :param filepath: str, path to the saved model
        """
        self.models[model_name] = joblib.load(filepath)

if __name__ == "__main__":
    filepath = 'data/processed/preprocessed_stock_data.csv'
    target_column = 'Close'

    predictive_modeling = PredictiveModeling()
    data = predictive_modeling.load_data(filepath)
    X_train, X_test, y_train, y_test = predictive_modeling.split_data(data, target_column)

    # Train and evaluate linear regression model
    lr_model = predictive_modeling.train_model('linear_regression', X_train, y_train)
    lr_metrics = predictive_modeling.evaluate_model(lr_model, X_test, y_test)
    print("Linear Regression Evaluation:", lr_metrics)

    # Train and evaluate random forest model
    rf_model = predictive_modeling.train_model('random_forest', X_train, y_train)
    rf_metrics = predictive_modeling.evaluate_model(rf_model, X_test, y_test)
    print("Random Forest Evaluation:", rf_metrics)

    # Train and evaluate gradient boosting model
    gb_model = predictive_modeling.train_model('gradient_boosting', X_train, y_train)
    gb_metrics = predictive_modeling.evaluate_model(gb_model, X_test, y_test)
    print("Gradient Boosting Evaluation:", gb_metrics)

    # Train and evaluate LSTM model
    lstm_model = predictive_modeling.train_lstm(data, target_column)
    lstm_metrics = predictive_modeling.evaluate_lstm(lstm_model, data, target_column)
    print("LSTM Model Evaluation:", lstm_metrics)

    # Save models
    predictive_modeling.save_model('linear_regression', 'models/linear_regression_model.pkl')
    predictive_modeling.save_model('random_forest', 'models/random_forest_model.pkl')
    predictive_modeling.save_model('gradient_boosting', 'models/gradient_boosting_model.pkl')
    predictive_modeling.save_model('lstm', 'models/lstm_model.pkl')
