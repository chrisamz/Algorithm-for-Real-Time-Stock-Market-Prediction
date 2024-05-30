# evaluation.py

"""
Evaluation Module for Real-Time Stock Market Prediction

This module contains functions for evaluating the performance of predictive models
and real-time data processing systems using appropriate metrics.

Techniques Used:
- Model Evaluation
- Real-Time System Evaluation

Metrics Used:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

class ModelEvaluation:
    def __init__(self):
        """
        Initialize the ModelEvaluation class.
        """
        self.models = {}

    def load_data(self, filepath):
        """
        Load test data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        return pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')

    def load_model(self, model_name, filepath):
        """
        Load a trained model from a file.
        
        :param model_name: str, name to assign to the loaded model
        :param filepath: str, path to the saved model
        """
        self.models[model_name] = joblib.load(filepath)

    def evaluate_regression_model(self, model_name, X_test, y_test):
        """
        Evaluate the specified regression model using various metrics.
        
        :param model_name: str, name of the model to evaluate
        :param X_test: DataFrame, testing features
        :param y_test: Series, testing target
        :return: dict, evaluation metrics
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found.")
        
        y_pred = model.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        return metrics

    def evaluate_classification_model(self, model_name, X_test, y_test):
        """
        Evaluate the specified classification model using various metrics.
        
        :param model_name: str, name of the model to evaluate
        :param X_test: DataFrame, testing features
        :param y_test: Series, testing target
        :return: dict, evaluation metrics
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found.")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        return metrics

    def evaluate_real_time_system(self, true_values, predicted_values):
        """
        Evaluate the real-time processing system using various metrics.
        
        :param true_values: list, actual values
        :param predicted_values: list, predicted values
        :return: dict, evaluation metrics
        """
        metrics = {
            'mae': mean_absolute_error(true_values, predicted_values),
            'rmse': np.sqrt(mean_squared_error(true_values, predicted_values))
        }
        return metrics

if __name__ == "__main__":
    test_data_filepath = 'data/processed/preprocessed_stock_data_test.csv'
    target_column = 'Close'

    evaluator = ModelEvaluation()
    data = evaluator.load_data(test_data_filepath)
    X_test = data.drop(columns=[target_column])
    y_test = data[target_column]

    # Load models
    evaluator.load_model('linear_regression', 'models/linear_regression_model.pkl')
    evaluator.load_model('random_forest', 'models/random_forest_model.pkl')
    evaluator.load_model('gradient_boosting', 'models/gradient_boosting_model.pkl')
    evaluator.load_model('lstm', 'models/lstm_model.pkl')

    # Evaluate regression models
    lr_metrics = evaluator.evaluate_regression_model('linear_regression', X_test, y_test)
    rf_metrics = evaluator.evaluate_regression_model('random_forest', X_test, y_test)
    gb_metrics = evaluator.evaluate_regression_model('gradient_boosting', X_test, y_test)
    lstm_metrics = evaluator.evaluate_regression_model('lstm', X_test, y_test)  # Assuming LSTM is used for regression

    print("Linear Regression Evaluation:", lr_metrics)
    print("Random Forest Evaluation:", rf_metrics)
    print("Gradient Boosting Evaluation:", gb_metrics)
    print("LSTM Model Evaluation:", lstm_metrics)

    # Example real-time system evaluation
    true_values = y_test.values[:30]
    predicted_values = y_test.values[:30] * 0.95  # Example predicted values (95% of true values)
    rt_metrics = evaluator.evaluate_real_time_system(true_values, predicted_values)
    print("Real-Time System Evaluation:", rt_metrics)
