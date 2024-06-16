import pandas as pd
import statsmodels.api as sm
from scipy.stats import f
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data(data):
    """
    Converts datetime to a numeric value, ensuring compatibility for regression.
    """
    try:
        data = data.copy()
        data.reset_index(drop=True, inplace=True)  # Reset index to ensure it's numeric
        data['time_index'] = (data['ds'] - data['ds'].min()).dt.days  # Create a numeric time index
        return data
    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
        return None  # Optionally, return None to indicate failure
    

def chow_test(data1: pd.DataFrame, data2: pd.DataFrame, alpha: float = 0.05) -> bool:
    """
    Performs Chow Test to check if the coefficients in two linear regressions on different data sets are equal.
    """
    try:
        X1 = sm.add_constant(data1['time_index'])
        X2 = sm.add_constant(data2['time_index'])
        y1 = data1['y']
        y2 = data2['y']

        model1 = sm.OLS(y1, X1).fit()
        model2 = sm.OLS(y2, X2).fit()

        X_pooled = sm.add_constant(pd.concat([data1['time_index'], data2['time_index']]))
        y_pooled = pd.concat([y1, y2])
        model_pooled = sm.OLS(y_pooled, X_pooled).fit()

        N = len(y_pooled)
        k = 2 # X1.shape[1]
        SSR1 = model1.ssr
        SSR2 = model2.ssr
        SSR_pooled = model_pooled.ssr
        F = ((SSR_pooled - (SSR1 + SSR2)) / k) / ((SSR1 + SSR2) / (N - 2 * k))
        p_value = f.sf(F, k, N - 2 * k)

        return p_value > alpha
    except Exception as e:
        error_message = f"Error in Chow Test: {e}"
        logger.error(error_message)
        return None, error_message
    

def find_optimal_window(data: pd.DataFrame, periods: list = [1, 3, 6, 9, 12], alpha: float = 0.05) -> int:
    """
    Finds the optimal time period for data based on Chow Test.
    """
    try:
        data = prepare_data(data)
        if data is None:
            raise ValueError("Data preparation failed.")

        def test_period(months):
            try:
                window_size = months * 30  # Assume 30 days per month
                max_window_size = 0  # To keep track of the maximum valid window size

                for start in range(0, len(data), window_size):  # Increment start by window size each loop
                    end = start + window_size
                    if end > len(data):  # Check if end exceeds the data length
                        end = len(data)  # Adjust end to the length of the data

                    current_window_size = end - start
                    if current_window_size > max_window_size and chow_test(data.iloc[start:end], data.iloc[end:min(end + window_size, len(data))], alpha):
                        max_window_size = current_window_size

                return max_window_size
            except Exception as e:
                error_message = f"Error testing period of {months} months: {e}"
                logger.error(error_message)
                return None, error_message

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(test_period, periods))

        return max(results), None
    except Exception as e:
        error_message = f"Error finding optimal window: {e}"
        logger.error(error_message)
        return None, error_message
    
    
__all__ = ['find_optimal_window']