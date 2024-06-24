import logging
from typing import Optional, Tuple, List, Dict, Any, Callable, Union

import numpy as np
import pandas as pd

import streamlit as st

from sktime.performance_metrics.forecasting import (
    mean_absolute_percentage_error,
    mean_squared_percentage_error,
    mean_absolute_scaled_error,
    mean_squared_scaled_error
)
from skforecast.model_selection import backtesting_forecaster

from utils.modeling.search import *

# Logger setup
logger = logging.getLogger(__name__)


def evaluate_train_data(train: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates the trained model on the training set. 
    """
    
    # Log the start of model evaluation
    logger.info("Evaluating model")
    
    # Change the frequency of the data based on the weekend drop session state
    try:
        if st.session_state.forecast_freq.lower() == "daily" and st.session_state.weekend_drop.lower() == "yes":
            forecast_freq = "B"
        elif st.session_state.forecast_freq.lower() == "daily" and st.session_state.weekend_drop.lower() == "no":
            forecast_freq = "D"
        elif st.session_state.forecast_freq.lower() == "weekly":
            forecast_freq = "W"
        elif st.session_state.forecast_freq.lower() == "monthly":
            forecast_freq = "M"
        else:
            raise ValueError("Unknown Frequency!")
    except Exception as e:
        model_search_bar.error(e, icon="🚨")
        logger.error(e)
    
    # Check if the model type is either Prophet or Naive
    if st.session_state.model_type in ["Prophet", "Naive - Mean", "Naive - Drift"]:

        # Select the best model based on the model type
        if st.session_state.model_type == "Prophet":
            best_model = st.session_state.prophet_best_model
        elif st.session_state.model_type in ["Naive - Mean", "Naive - Drift"]:
            best_model = st.session_state.naive_best_model

        # Generate forecast horizon (negative for in-sample prediction)
        fh = np.arange(-len(train.copy(deep=True).asfreq('D').resample(forecast_freq).mean()) + 1, 1)
        # Generate predictions and intervals
        pred_train = best_model.predict(fh=fh).reset_index(drop=True)
        pred_train.columns = ["y_pred"]
        
        pred_train_int = best_model.predict_interval(fh=fh).reset_index(drop=True)
        pred_train_int.columns = ["min_pred", "max_pred"]
        
        if st.session_state.model_type in ["Naive - Mean", "Naive - Drift"]:
            pred_train_int.fillna(0, inplace=True)
        
        # Combine predictions and intervals
        df_predictions = pd.concat([pred_train, pred_train_int], axis=1)
        
        # Align the predictions with the actual data
        df_predictions = pd.concat([train.copy(deep=True).asfreq('D').resample(forecast_freq).mean().reset_index(), df_predictions], axis=1)
        df_predictions.rename(columns={"y": "y_true"}, inplace=True)
        df_predictions = df_predictions[["ds", "y_true", "y_pred", "min_pred", "max_pred"]]
        
        # Check if the lengths of the forecast horizon and the predictions match
        assert len(fh) == len(df_predictions)
        
    # Check if the model type is either Random Forest, XGBoost, or Ensemble Tree
    elif st.session_state.model_type in ["Random Forest", "XGBoost", "Ensemble Tree"]:
        
        # Select the best model and the best estimator based on the model type
        if st.session_state.model_type == "Random Forest":
            best_model = st.session_state.rf_best_model
            best_estimator = st.session_state.rf_best_estimator
        elif st.session_state.model_type == "XGBoost":
            best_model = st.session_state.xgb_best_model
            best_estimator = st.session_state.xgb_best_estimator
        elif st.session_state.model_type == "Ensemble Tree":
            best_model = st.session_state.ensemble_best_model
            best_estimator = st.session_state.ensemble_best_estimator
        
        exog_cols = st.session_state.exog_cols
        
        # Generate date features for the train data
        train_data = train.copy(deep=True).asfreq('D').resample(forecast_freq).mean()
        df_train = generate_date_features(train_data.copy(deep=True))

        # Training Set Predictions
        metric, df_predictions = backtesting_forecaster(forecaster=best_estimator,
                                                        y=df_train['y'],
                                                        exog=df_train[exog_cols],
                                                        initial_train_size=None,
                                                        steps=1,
                                                        interval=[5, 95],
                                                        n_boot=50,
                                                        metric='mean_squared_error',
                                                        refit=False,
                                                        verbose=False)

        # Add the true values to the predictions DataFrame
        df_predictions['y_true'] = df_train['y'][len(df_train) - len(df_predictions):]
        df_predictions = df_predictions.reset_index()
        df_predictions.columns = ["ds", "y_pred", "min_pred", "max_pred", "y_true"]
        
        # Get the history DataFrame
        df_history = train_data[:len(df_train) - len(df_predictions)]
        df_history = df_history[['y']].reset_index()
        df_history.columns = ["ds", "y_true"]
        
        # Combine the history and predictions DataFrames
        df_predictions = pd.concat([df_history, df_predictions], sort=True)
        df_predictions = df_predictions.reset_index(drop=True)

        # If there's a weekend drop, filter out weekends from the predictions DataFrame
        if st.session_state.forecast_freq.lower() == "daily" and st.session_state.weekend_drop.lower() == "yes":
            mask = df_predictions["ds"].apply(lambda x: x.dayofweek in [5, 6])
            df_predictions = df_predictions[~mask]
    else:
        # If the model type is neither of the specified types, raise an error
        raise ValueError("Error: Unknown Model!")

    # Log the completion of model evaluation
    logger.info("Model evaluation completed")
    return df_predictions


def create_date_range(start_date: pd.Timestamp, periods: int, drop_weekend: str) -> pd.DatetimeIndex:
    """
    Create date range with or without weekends.
    """
    
    logger.info("Creating date range")

    freq = "D" #if drop_weekend.lower() == "yes" else "D"
    return pd.date_range(start=start_date + pd.Timedelta(days=1), periods=periods, freq=freq)


def generate_forecast(forecast_period: int, weekend_drop: str, train: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Generate forecast for the given period.
    """
    
    # Log the start of forecast generation
    logger.info("Generating forecast")

    # Save model type in a variable to avoid multiple state accesses
    model_type = st.session_state.model_type
    
    # Change the frequency of the data based on the weekend drop session state
    try:
        if st.session_state.forecast_freq.lower() == "daily" and st.session_state.weekend_drop.lower() == "yes":
            forecast_freq = "B"
            forecast_days = 1
        elif st.session_state.forecast_freq.lower() == "daily" and st.session_state.weekend_drop.lower() == "no":
            forecast_freq = "D"
            forecast_days = 1
        elif st.session_state.forecast_freq.lower() == "weekly":
            forecast_freq = "W"
            forecast_days = 5
        elif st.session_state.forecast_freq.lower() == "monthly":
            forecast_freq = "M"
            forecast_days = 28
        else:
            raise ValueError("Unknown Frequency!")
    except Exception as e:
        model_search_bar.error(e, icon="🚨")
        logger.error(e)
        
    # Change frequency of change the train data
    train_data = train.copy(deep=True).asfreq('D').resample(forecast_freq).mean()
    
    # Generate forecast horizon
    fh = np.arange(0, forecast_period)

    # Generate date range
    fh_date_range = pd.date_range(train_data.index[-1] + pd.Timedelta(days=1), periods=len(fh), freq=forecast_freq)

    # Check if the model type is either Prophet or Naive
    if model_type in ["Prophet", "Naive - Mean", "Naive - Drift"]:
        
        # Select the best model based on the model type
        best_model = st.session_state.prophet_best_model if model_type == "Prophet" else st.session_state.naive_best_model

        # Generate predictions and intervals
        forecast_pred = best_model.predict(fh=fh).reset_index(drop=True)
        forecast_pred.columns = ["y_pred"]

        forecast_pred_int = best_model.predict_interval(fh=fh).reset_index(drop=True)
        forecast_pred_int.columns = ["min_pred", "max_pred"]

        # Assert the lengths of date range, predictions and intervals are equal
        assert len(fh_date_range) == len(forecast_pred) == len(forecast_pred_int), "Error: Length mismatch!"

        # Combine predictions and intervals
        df_forecast = pd.concat([forecast_pred, forecast_pred_int], axis=1)
        df_forecast["ds"] = fh_date_range
        df_forecast["y_true"] = np.nan
        
    # Check if the model type is either Random Forest, XGBoost, or Ensemble Tree
    elif model_type in ["Random Forest", "XGBoost", "Ensemble Tree"]:
        
        # Select the best model based on the model type
        if model_type == "Random Forest":
            best_model = st.session_state.rf_best_model
        elif model_type == "XGBoost":
            best_model = st.session_state.xgb_best_model
        elif model_type == "Ensemble Tree":
            best_model = st.session_state.ensemble_best_model
        
        exog_cols = st.session_state.exog_cols
        
        # Initialize forecast DataFrame with the date range
        df_forecast = pd.DataFrame(index=fh_date_range)

        # Generate date features for the forecast
        df_forecast = generate_date_features(df_forecast)

        # Add exogenous features
        if st.session_state.external_features.lower() == "yes":
            df_forecast = df_forecast.join(forecast.set_index('ds'), how='left')
        
        # Predict intervals
        df_forecast = best_model.predict_interval(last_window=train_data['y'], steps=len(df_forecast),
                                                  exog=df_forecast[exog_cols],
                                                  interval=[5, 95], n_boot=50)
        
        df_forecast = df_forecast.reset_index()
        df_forecast.columns = ["ds", "y_pred", "min_pred", "max_pred"]
        df_forecast["y_true"] = np.nan

        # Assert the lengths of date range and forecast DataFrame are equal
        assert len(fh_date_range) == len(df_forecast), "Error: Length mismatch!"
        
        # If there's a weekend drop, filter out weekends from the predictions DataFrame
        if st.session_state.forecast_freq.lower() == "daily" and st.session_state.weekend_drop.lower() == "yes":
            mask = df_forecast["ds"].apply(lambda x: x.dayofweek in [5, 6])
            df_predictions = df_forecast[~mask]

    # If the model type is neither of the specified types, raise an error
    else:
        raise ValueError("Error: Unknown Model!")

    # Log the completion of forecast generation
    logger.info("Forecast generation completed")
    
    df_forecast = df_forecast[["ds", "y_true", "y_pred", "min_pred", "max_pred"]]
    
    return df_forecast


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from the Date column.
    """
    
    logger.info("Extracting date features")

    # Extract various features from the Date column
    df["Year"] = df["Date"].dt.year
    df["Quarter"] = df["Date"].dt.quarter
    df["Month"] = df["Date"].dt.strftime("%B")
    df["Weekday Number"] = df["Date"].dt.weekday
    df["Day"] = df["Date"].dt.strftime("%A")
    df["Month Week"] = df["Date"].dt.isocalendar().week

    logger.info("Date features extracted")
    
    return df


def concatenate_predictions_forecasts(df_predictions: pd.DataFrame, df_forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate predictions and forecasts and perform additional transformations.
    """
    
    logger.info("Concatenating predictions and forecasts")

    # Concatenate predictions and forecasts
    df_combined = pd.concat([df_predictions, df_forecast], sort=True)

    # Rename columns
    columns_mapping = {
        "ds": "Date",
        "y_true": "Actual",
        "y_pred": "Forecast",
        "min_pred": "Lower Forecast",
        "max_pred": "Upper Forecast"
    }
    df_combined = df_combined.rename(columns=columns_mapping)

    # Extract date features
    df_combined = extract_date_features(df_combined)
    
    # Calculate Absolute Percentage Error (APE)
    df_combined["Absolute Percentage Error"] = np.round(
        np.abs((df_combined["Actual"] - df_combined["Forecast"]) / df_combined["Actual"]) * 100, 3
    )

    # Define order of columns
    columns_order = ["Date", "Year", "Quarter", "Month", "Weekday Number", "Day", "Month Week",
                     "Actual", "Forecast", "Absolute Percentage Error", "Lower Forecast", "Upper Forecast"]

    logger.info("Predictions and forecasts concatenated")
    
    # Return DataFrame with ordered columns
    return df_combined[columns_order].round(1)


def compute_prediction_coverage(predicted_data: pd.DataFrame) -> float:
    """
    Compute prediction coverage.
    """
    
    # Calculate coverage
    coverage = np.where(
                    (predicted_data.loc[predicted_data.index, 'y_true'] >= predicted_data['min_pred']) &
                    (predicted_data.loc[predicted_data.index, 'y_true'] <= predicted_data['max_pred']),
                    True,
                    False
                ).mean()

    return coverage


def compute_metrics(df_predictions: pd.DataFrame) -> dict:
    """
    Compute different metrics for the predictions.
    """
    
    logger.info("Calculating and displaying metrics")
    
    # Drop NA values from DataFrame
    df_predictions = df_predictions.dropna()
    
    # Compute metrics
    mape = mean_absolute_percentage_error(df_predictions["y_true"], df_predictions["y_pred"])
    rmspe = mean_squared_percentage_error(df_predictions["y_true"], df_predictions["y_pred"], square_root=True)
    mase = mean_absolute_scaled_error(df_predictions["y_true"], df_predictions["y_pred"], y_train=df_predictions["y_true"])
    rmsse = mean_squared_scaled_error(df_predictions["y_true"], df_predictions["y_pred"], y_train=df_predictions["y_true"], square_root=True)
    coverage = compute_prediction_coverage(df_predictions)
    
    metrics = {
        "MAPE": mape,
        "RMSPE": rmspe,
        "MASE": mase,
        "RMSSE": rmsse,
        "Coverage": coverage
    }
    
    return metrics

    
__all__ = ['evaluate_train_data', 'generate_forecast', 'concatenate_predictions_forecasts', 'compute_metrics', 'extract_date_features']
