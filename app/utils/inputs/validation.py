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


__all__ = ["validate_input_file"]