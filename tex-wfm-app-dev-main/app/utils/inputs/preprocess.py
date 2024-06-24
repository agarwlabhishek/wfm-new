import datetime
import logging
from typing import Optional, Dict

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import f
import streamlit as st

from utils.inputs.validation import *


# Create a logger
logger = logging.getLogger(__name__)


def chow_test(data1: pd.DataFrame, data2: pd.DataFrame) -> bool:
    """
    Performs Chow Test to check if the coefficients in two linear regressions 
    on different data sets are equal.
    """
    # Perform two separate OLS regressions
    model1 = sm.OLS(data1.y, sm.add_constant(range(len(data1)))).fit()
    model2 = sm.OLS(data2.y, sm.add_constant(range(len(data2)))).fit()

    # Perform a pooled OLS regression
    pooled = pd.concat([data1, data2])
    model_pooled = sm.OLS(pooled.y, sm.add_constant(range(len(pooled)))).fit()

    # Calculate F statistic
    N = len(pooled)
    k = 2  # Number of parameters
    SSR1 = model1.ssr
    SSR2 = model2.ssr
    SSRpooled = model_pooled.ssr
    F = ((SSRpooled - (SSR1 + SSR2)) / k) / ((SSR1 + SSR2) / (N - 2 * k))
    
    df1 = k
    df2 = N - 2*k
    p_value = f.sf(F, df1, df2)
    
    # Return True if test is passed, False otherwise
    return p_value > 0.05  # 0.05 is the usual significance level


def find_optimal_period(data: pd.DataFrame) -> int:
    """
    Finds the optimal time period for data based on Chow Test. The function implements
    both an expanding and rolling window approach and selects the longest window 
    size that passed the Chow Test.
    """

    # Ensure the data is sorted in descending order by date
    data = data.sort_values(by='ds', ascending=False)

    results = []  # Initialize a list to store the window sizes that passed the Chow Test

    # Loop over several predefined periods
    for period in [1, 3, 6, 9, 12]:

        window_size = period * 30  # Assuming each month has 30 days, calculate the window size
        start = 0  # Starting index for the current window
        end = window_size  # Ending index for the current window

        # Initialize flags to track if there's a previous window and if the current window passed the Chow Test
        has_previous = True
        passed_test = True

        # Perform Chow Test on expanding windows until the end of data is reached, 
        # or there's no previous window, or the current window fails the Chow Test
        while end < len(data) and has_previous and passed_test:

            # Extract the current window data
            window_data = data.iloc[start:end]

            # Extract the previous window data
            prev_window_data = data.iloc[end:min(end + window_size, len(data))]

            # If there's no previous window data, set the flag to False and continue
            if len(prev_window_data) == 0:
                has_previous = False
                continue

            # Perform Chow Test on the current and previous window
            passed_test = chow_test(window_data, prev_window_data)

            # If the current window passes the Chow Test, expand the window for the next iteration
            if passed_test:
                end += window_size

        # Add the size of the longest window that passed the Chow Test to the results list
        results.append(end)

    # Repeat the above process for rolling windows
    for period in [1, 3, 6, 9, 12]:
        window_size = period * 30  # Assuming each month has 30 days
        start = 0
        end = window_size
        has_previous = True
        passed_test = True
        
        # Perform Chow Test on expanding windows until the end of data is reached, 
        # or there's no previous window, or the current window fails the Chow Test
        while end < len(data) and has_previous and passed_test:
            window_data = data.iloc[start:end]
            prev_window_data = data.iloc[end:min(end + window_size, len(data))]

            # Extract the current window data
            if len(prev_window_data) == 0:
                has_previous = False
                continue

            # Perform Chow Test on the current and previous window
            passed_test = chow_test(window_data, prev_window_data)

            # If the current window passes the Chow Test, expand the window for the next iteration
            if passed_test:
                start += window_size
                end += window_size

        results.append(end)

    # Select the maximum window size that passed the Chow Test as the optimal period
    optimal_period = max(results)
    logger.info(f"Optimal amount of data to use is from the most recent {optimal_period} days.")

    return optimal_period


def find_start_date(df, column_name):
    # Find indices where the value in the specified column is 0 or None
    zero_indices = df[(df[column_name] == 0) | (df[column_name].isnull())].index
    
    # Check if there are no zero or None sequences
    if len(zero_indices) == 0:
        return -1
    
    # Compute differences between consecutive zero or None indices
    diffs = np.diff(zero_indices)
    
    # Find the start index of the last zero or None sequence
    start_index = zero_indices[np.where(diffs > 1)[0][-1] + 1] if np.any(diffs > 1) else zero_indices[0]

    # Check if the sum of values in the column from the start index onwards is still 0 or None
    if df.loc[start_index:][column_name].sum() == 0 or df.loc[start_index:][column_name].isnull().all():
        return start_index
    else:
        return -1


def process_input_file(uploaded_file) -> pd.DataFrame:
    """
    Handles file changes and returns a processed DataFrame.
    """
    # Validate input file
    train_full = validate_input_file(uploaded_file)
    
    # Find the start index of a sequence where all subsequent values are zero till the end
    start_index = find_start_date(train_full, 'y')
    logger.info(f"Start Date Forecast: {train_full['ds'].iloc[start_index]}")

    # Split train and forecast set
    forecast_full = train_full[start_index:]
    train_full = train_full[:start_index]
    
    # Find optimal window 
    st.session_state.optimal_window_size = find_optimal_period(train_full) + 180
    st.session_state.optimal_window_date = train_full[-st.session_state.optimal_window_size:]['ds'].min()
    
    # Ensure that all dates are present in the range (no missing dates)
    date_range = pd.date_range(start=min(train_full['ds']), end=max(train_full['ds']))
    missing_dates = date_range.difference(train_full['ds'])

    if not missing_dates.empty:
        warning_message = "There are missing dates in your data. Imputing missing dates."
        logger.info(warning_message)

        # Creating a new dataframe with complete date range
        complete_data = pd.DataFrame(date_range, columns=['ds'])
        # Merging the new dataframe with the original data
        train_full = pd.merge(complete_data, train_full, how='left', on='ds')
        # Replacing NaN values with zero
        train_full['y'].fillna(0, inplace=True)
        
        # Try to convert data types
        if not convert_data_types(train_full):
            raise ValueError(error_message)
        
        st.session_state.train_impute_dates = True
    
    # Fill missing values with zero
    train_full = train_full.fillna(0)
    train_full.loc[train_full['y'] == 0, 'y'] = 1
    
    # Set the frequency of the datetime index
    train_full.set_index("ds", inplace=True)
    train_full = train_full.asfreq('D')
    
    return train_full, forecast_full


__all__ = ['process_input_file']
