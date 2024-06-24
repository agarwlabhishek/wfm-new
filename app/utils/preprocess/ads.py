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
        error_message = f"Failed to prepare data: {e}"
        logger.error(error_message)
        raise Exception(error_message)
    

def chow_test(data1: pd.DataFrame, data2: pd.DataFrame, alpha: float = 0.05) -> bool:
    """
    Performs Chow Test to check if the coefficients in two linear regressions on different data sets are equal.
    """
    try:
        if data1.empty or data2.empty:
            raise ValueError("One of the input datasets is empty.")
        
        X1 = sm.add_constant(data1['time_index'])
        y1 = data1['y']
        model1 = sm.OLS(y1, X1).fit()

        X2 = sm.add_constant(data2['time_index'])
        y2 = data2['y']
        model2 = sm.OLS(y2, X2).fit()

        X_pooled = sm.add_constant(pd.concat([data1['time_index'], data2['time_index']]))
        y_pooled = pd.concat([y1, y2])
        model_pooled = sm.OLS(y_pooled, X_pooled).fit()

        N = len(y_pooled)
        k = X_pooled.shape[1]  # Correct number of regressors including constant
        SSR1 = model1.ssr
        SSR2 = model2.ssr
        SSR_pooled = model_pooled.ssr
        F = ((SSR_pooled - (SSR1 + SSR2)) / k) / ((SSR1 + SSR2) / (N - 2 * k))
        p_value = f.sf(F, k, N - 2 * k)

        return p_value > alpha
    except Exception as e:
        error_message = f"Error in Chow Test: {e}"
        logger.error(error_message)
        raise Exception(error_message)
    

def find_optimal_window(data: pd.DataFrame, periods: list = [1, 3, 6, 9, 12], alpha: float = 0.05):
    """
    Finds the optimal time period for data based on Chow Test.
    Propagates errors by returning them with the result.
    """
    try:
        data = prepare_data(data)

        def test_period(months):
            window_size = months * 30  # Assume 30 days per month
            max_window_size = 0  # To track the maximum valid window size

            for start in range(0, len(data) - window_size, window_size):
                end = start + window_size
                next_end = min(end + window_size, len(data))

                if next_end <= end:  # Ensure there is data for the second window
                    continue  # Skip this iteration as the second window would be empty

                test_result = chow_test(data.iloc[start:end], data.iloc[end:next_end], alpha)

                current_window_size = end - start
                if current_window_size > max_window_size and test_result:
                    max_window_size = current_window_size

            return max_window_size


        with ThreadPoolExecutor() as executor:
            results = list(executor.map(test_period, periods))

        return max(results)
    except Exception as e:
        error_message = f"Error finding optimal window: {e}"
        logger.error(error_message)
        raise Exception(error_message)
    
    
__all__ = ['find_optimal_window']