import logging

import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset, BusinessDay, MonthEnd, Week, Day
from typing import Optional, Dict

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def convert_data_types(data: pd.DataFrame) -> (bool, str):
    """
    Convert the data types of the columns in the data.
    Returns a boolean indicating success and an optional error message on failure.
    """
    try:
        data["ds"] = pd.to_datetime(data["ds"], dayfirst=True)
        data["y"] = pd.to_numeric(data["y"], errors="coerce")
    except Exception as e:
        error_message = f"Incorrect datatype. Please ensure the date and volume columns have the correct data types: {e}"
        logger.error(error_message)
        raise Exception(error_message)

    if data["ds"].isnull().any():
        error_message = "Null values found in the date column. Please check the data."
        logger.error(error_message)
        raise Exception(error_message)

    if (data["y"] < 0).any():
        error_message = "Volume cannot be negative. Please check the data."
        logger.error(error_message)
        raise Exception(error_message)

    return True


def validate_input_file(uploaded_file, external_features: bool = False) -> (Optional[pd.DataFrame], Optional[str]):
    """
    Validate the uploaded CSV file by checking the number of columns, column types,
    and data types. Returns a DataFrame and an optional error message.
    """
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        error_message = f"Error reading the CSV file: {e}"
        logger.error(error_message)
        raise Exception(error_message)

    data.rename(columns={data.columns[0]: "ds", data.columns[1]: "y"}, inplace=True)
    
    if not external_features:
        data = data[["ds", "y"]]

    success = convert_data_types(data)

    data["y"] = data["y"].fillna(0)

    return data.sort_values(by="ds")


def calculate_end_date(run_params):
    """
    Calculate the end date of a forecast range based on the start date, period, and frequency.
    """
    # Parse the start date
    start_date = pd.to_datetime(run_params['forecast_start_date'])
    
    # Create the appropriate offset based on the frequency
    freq = run_params['forecast_freq']
    period = run_params['forecast_period']
    
    if freq == 'B':
        offset = BusinessDay(n=period)
    elif freq == 'M':
        offset = MonthEnd(n=period)
    elif freq == 'W':
        offset = Week(n=period)
    elif freq == 'D':
        offset = Day(n=period)
    else:
        raise ValueError("Invalid forecast frequency specified.")
    
    # Calculate the end date
    end_date = start_date + offset
    
    return end_date


def find_dates(historical_df, run_params):
    """
    Initializes and returns run parameters based on historical data.
    """
    # Set historical start and end dates from the DataFrame
    run_params["historical_start_date"] = historical_df['ds'].min()
    run_params["historical_end_date"] = historical_df['ds'].max()

    # Set forecasting start and end dates
    run_params["forecast_start_date"] = historical_df['ds'].max() + pd.Timedelta(days=1)
    run_params["forecast_end_date"] = calculate_end_date(run_params)

    return run_params


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

def validate_dataframes(optimal_df, forecast_df, run_params):
    """
    Validates the column and row counts of dataframes based on specified parameters.
    """
    try:
        # Validate column counts based on whether external features are used
        if run_params["external_features"]:
            assert optimal_df.shape[1] > 2 and forecast_df.shape[1] > 2, \
                "Uploaded Historical or Forecast Data does not have the required number of columns!"
        else:
            assert optimal_df.shape[1] == 2 and forecast_df.shape[1] == 2, \
                "Uploaded Historical or Forecast Data does not have the required number of columns!"

        # Ensure non-empty data structure
        assert optimal_df.shape[0] > 0, "Uploaded Historical Data does not have enough rows!"
        assert forecast_df.shape[0] >= run_params["forecast_period"], \
            "Forecast Data does not have enough rows based on the forecast period!"

        # Ensure same number of columns
        assert optimal_df.shape[1] == forecast_df.shape[1], \
            "Uploaded Historical and Forecast Data do not have the same number of columns"
    
    except AssertionError as e:
        raise ValueError(f"Invalid input data format: {e}")
        
        
def resample_dataframe(df, forecast_freq='D'):
    """
    Resample and compute the mean for the dataframes based on a specified frequency.
    """
    
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)
    df = df.resample(forecast_freq).sum()
    
    return df.reset_index()
