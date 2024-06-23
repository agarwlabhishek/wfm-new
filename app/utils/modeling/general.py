import os
import yaml

import pandas as pd
import warnings
import holidays

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.naive import NaiveForecaster


def resample_dataframe(df, forecast_freq='D'):
    """
    Resample and compute the mean for the dataframes based on a specified frequency.
    """
    
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)
    df = df.resample(forecast_freq).mean()
    
    return df.reset_index()


def determine_params(forecast_freq):
    """
    Determines lag window and test step size based on the frequency and weekend inclusion.
    """
    # Define settings for different scenarios using dictionaries
    # initial_window_size, lag_window_range, rolling_window_range, test_size, test_steps
    freq_settings = {
        "D": (90, [7, 15, 30, 60, 90], [3, 7, 15, 30, 60, 90], 30, 3),
        "B": (60, [5, 10, 20, 40, 60], [3, 5, 10, 20, 40, 60], 20, 2),
        "W": (12, [4, 8, 12], [4, 8, 12, 16, 20, 24], 6, 1),
        "M": (3, [3], [3, 6, 9, 12], 3, 1)
    }
    
    try:
        # Select the appropriate settings based on forecast frequency and weekend drop
        if forecast_freq in freq_settings:
            if isinstance(freq_settings[forecast_freq], dict):
                # Handle daily frequency differently based on weekend inclusion
                return freq_settings[forecast_freq][weekend_drop]
            else:
                return freq_settings[forecast_freq]
        else:
            raise ValueError(f"Unknown Frequency: {forecast_freq}")
    except Exception as e:
        error_message = f"Failed to determine lag window and test set size: {e}"
        logger.error(error_message)
        raise Exception(error_message)
        

def generate_date_features(df: pd.DataFrame, freq='D', country_name=None) -> pd.DataFrame:
    """
    Add time-based features to a DataFrame based on its DateTime index, considering the frequency of data.
    """
    
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        error_message = "DataFrame must have a DateTimeIndex"
        logger.error("DataFrame must have a DateTimeIndex")
        raise ValueError(error_message)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress warnings during feature generation

        # Generate features based on the specified frequency
        if freq in ['D', 'B']:
            # Features specific to daily data
            df['day_of_week'] = df.index.dayofweek + 1  # Monday=1, Sunday=7
            df['day_of_year'] = df.index.dayofyear
            df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

        if freq in ['D', 'B', 'W']:  # Weekly features include week_of_year
            df['week_of_year'] = df.index.isocalendar().week.astype(int)

        # Features applicable to all frequencies
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['year'] = df.index.year

        # Calculate holidays if country_name is provided
        if country_name:
            country_holidays = holidays.CountryHoliday(country_name)
            if freq in ['D', 'B']:
                # Mark holidays for daily data
                df['is_holiday'] = df.index.map(lambda date: int(date in country_holidays))
            elif freq == 'W':
                # Count holidays in a week for weekly data
                df['is_holiday'] = df.index.map(lambda week_start: sum(
                    1 for day in pd.date_range(start=week_start - pd.Timedelta(days=6), end=week_start)
                    if day in country_holidays))
            elif freq == 'M':
                # Count holidays in a month for monthly data
                df['is_holiday'] = df.index.map(lambda month_start: sum(
                    1 for day in pd.date_range(start=month_start.replace(day=1), end=month_start)
                    if day in country_holidays))

    return df.reset_index()


def load_model_params_and_create_instance(model_type, current_dir):
    """
    Load model parameters from a YAML file and create a model instance based on model type.
    """
    # Dictionary to map model types to their respective classes and YAML files
    model_config = {
        'random_forest': (RandomForestRegressor(), 'random_forest.yaml'),
        'xgboost': (XGBRegressor(), 'xgboost.yaml'),
        'prophet': (Prophet(), 'prophet.yaml'),
        'naive': (NaiveForecaster(), 'naive.yaml')
    }
    
    # Ensure the model type is supported
    if model_type not in model_config:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model_class, yaml_file = model_config[model_type]
    file_path = os.path.join(current_dir, 'params', yaml_file)
    
    # Load parameters from the YAML file, handling file and parsing errors
    try:
        with open(file_path, 'r') as file:
            param_grid = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The configuration file {yaml_file} was not found in {file_path}")
    except yaml.YAMLError as e:
        raise Exception(f"Error parsing the YAML file: {e}")

    return model_class, param_grid