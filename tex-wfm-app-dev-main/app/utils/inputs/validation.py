import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import streamlit as st

# Create a logger
logger = logging.getLogger(__name__)


def convert_data_types(data: pd.DataFrame) -> bool:
    """
    Convert the data types of the columns in the data.
    """
    try:
        # Check if the date and volume columns have the correct data types
        data["ds"] = pd.to_datetime(data["ds"], errors="coerce")
        data["y"] = pd.to_numeric(data["y"], errors="coerce")
    except Exception as e:
        error_message = f"Incorrect datatype. Please ensure the date and volume columns have the correct data types: {e}"
        st.error(error_message, icon="🚨")
        logger.error(error_message)
        return False

    # Checking for null values after conversion
    if data["ds"].isnull().values.any():
        error_message = "Null values found in the date column. Please check the data."
        st.error(error_message, icon="🚨")
        logger.error(error_message)
        return False

    # Checking for negative volumes, as it's not a valid input
    if (data["y"] < 0).any():
        error_message = "Volume cannot be negative. Please check the data."
        st.error(error_message, icon="🚨")
        logger.error(error_message)
        return False

    return True


def validate_input_file(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Validate the uploaded CSV file by checking the number of columns, column types,
    and data types.
    """
    try:
        data = pd.read_csv(uploaded_file, parse_dates=[0])
    except Exception as e:
        error_message = f"Error reading the CSV file: {e}"
        st.error(error_message, icon="🚨")
        logger.error(error_message)
        raise ValueError(error_message)

    # Check if first column is datetime-like and second column is numeric
    if not np.issubdtype(data.iloc[:, 0].dtype, np.datetime64) or not np.issubdtype(data.iloc[:, 1].dtype, np.number):
        error_message = "The first column should be of date type and the second column should be numeric."
        st.error(error_message, icon="🚨")
        logger.error(error_message)
        raise ValueError(error_message)

    # Rename the columns to 'ds' and 'y'
    data.rename(columns={data.columns[0]: "ds", data.columns[1]: "y"}, inplace=True)

    # If external_features is no then discard the rest of the columns
    if st.session_state.external_features.lower() == "no":
        data = data[["ds", "y"]]

    # Try to convert data types
    if not convert_data_types(data):
        raise ValueError(error_message)

    # Ensure that all 'y' values are non-negative
    if any(data['y'] < 0):
        error_message = "All volume values should be non-negative."
        st.error(error_message, icon="🚨")
        logger.error(error_message)
        raise ValueError(error_message)

    return data


__all__ = ['validate_input_file', 'convert_data_types']