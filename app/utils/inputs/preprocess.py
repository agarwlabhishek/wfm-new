import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional

# Configure logging to output at INFO level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_start_date(df: pd.DataFrame, column_name: str) -> int:
    """
    Identify the start index of the last sequence where all subsequent values in the specified column
    are zero or missing until the end of the DataFrame.
    """
    try:
        zero_indices = df[df[column_name].fillna(0) == 0].index
        if not zero_indices.any():
            logger.info("No zero indices found in column.")
            return -1  # Return -1 if no zeros found

        gaps = np.diff(zero_indices)
        if not np.any(gaps > 1):
            logger.info("All zeros are contiguous from the first found zero.")
            return zero_indices[0]  # All zeros are contiguous

        # Locate last significant gap
        last_gap_idx = np.where(gaps > 1)[0][-1] + 1
        start_index = zero_indices[last_gap_idx]
        if pd.isnull(df.loc[start_index:, column_name]).all() or (df.loc[start_index:, column_name] == 0).all():
            return start_index
        return -1
    except Exception as e:
        logger.error(f"Error in finding start date: {e}")
        return None
    

def process_input_file(train_full: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Processes the input DataFrame to separate training data from forecast data, handle missing dates,
    and ensure data type consistency.
    """
    try:
        start_index = find_start_date(train_full, 'y')
        if start_index == -1:
            logger.info("No valid start date found for zero or missing values sequence.")
            start_index = len(train_full)  # Use full data if no valid start date found
        elif start_index is None:
            error_message = "Failed to determine the start date due to an error."
            logger.error(error_message)
            raise ValueError(error_message)

        logger.info(f"Start Date Forecast: {train_full.loc[start_index, 'ds'] if start_index < len(train_full) else 'No Start Date'}")
        # Splitting the data into forecast and training sets
        forecast_full = train_full.loc[start_index:]
        train_full = train_full.loc[:start_index - 1]

        # Reindex the train_full DataFrame to include all days and fill missing dates with zeros
        train_date_range = pd.date_range(start=train_full['ds'].min(), end=train_full['ds'].max(), freq='D')
        train_full = train_full.set_index('ds').reindex(train_date_range).fillna(0).reset_index().rename(columns={'index': 'ds'})
        train_full.set_index("ds", inplace=True)
        train_full = train_full.asfreq('D').reset_index()

        # Reindex the forecast_full DataFrame to include all days and fill missing dates with zeros
        if not forecast_full.empty:
            forecast_date_range = pd.date_range(start=forecast_full['ds'].min(), end=forecast_full['ds'].max(), freq='D')
            forecast_full = forecast_full.set_index('ds').reindex(forecast_date_range).fillna(0).reset_index().rename(columns={'index': 'ds'})
            forecast_full.set_index("ds", inplace=True)
            forecast_full = forecast_full.asfreq('D').reset_index()

        return train_full, forecast_full, None
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        logger.error(error_message)
        return None, None, error_message
    
    
__all__ = ['process_input_file']