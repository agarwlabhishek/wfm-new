import logging

import pandas as pd
import numpy as np
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
        data["ds"] = pd.to_datetime(data["ds"], errors="coerce")
        data["y"] = pd.to_numeric(data["y"], errors="coerce")
    except Exception as e:
        error_message = f"Incorrect datatype. Please ensure the date and volume columns have the correct data types: {e}"
        logger.error(error_message)
        return False, error_message

    if data["ds"].isnull().any():
        error_message = "Null values found in the date column. Please check the data."
        logger.error(error_message)
        return False, error_message

    if (data["y"] < 0).any():
        error_message = "Volume cannot be negative. Please check the data."
        logger.error(error_message)
        return False, error_message

    return True, None  # Conversion successful, no error message


def validate_input_file(uploaded_file, external_features: bool = False) -> (Optional[pd.DataFrame], Optional[str]):
    """
    Validate the uploaded CSV file by checking the number of columns, column types,
    and data types. Returns a DataFrame and an optional error message.
    """
    try:
        data = pd.read_csv(uploaded_file, parse_dates=[0])
    except Exception as e:
        error_message = f"Error reading the CSV file: {e}"
        logger.error(error_message)
        return None, error_message

    if not np.issubdtype(data.iloc[:, 0].dtype, np.datetime64) or not np.issubdtype(data.iloc[:, 1].dtype, np.number):
        error_message = "The first column should be of date type and the second column should be numeric."
        logger.error(error_message)
        return None, error_message

    data.rename(columns={data.columns[0]: "ds", data.columns[1]: "y"}, inplace=True)

    if not external_features:
        data = data[["ds", "y"]]

    success, error_message = convert_data_types(data)
    if not success:
        return None, error_message

    if any(data['y'] < 0):
        error_message = "All volume values should be non-negative."
        logger.error(error_message)
        return None, error_message

    return data, None


__all__ = ['validate_input_file']