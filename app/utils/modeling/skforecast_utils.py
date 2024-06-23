import os
import yaml
import holidays
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.model_selection import grid_search_forecaster
from sktime.performance_metrics.forecasting import (
    mean_absolute_percentage_error,
    mean_squared_percentage_error,
    mean_absolute_scaled_error,
    mean_squared_scaled_error
)
from typing import Optional, Tuple, List, Dict, Any, Callable, Union


from utils.modeling.general import *


def find_best_model_skforecast(lag_window_range, model, train_df, param_grid, run_params):
    """
    Perform a grid search to optimize model parameters over different lag windows and return the best model configuration.
    """

    def _filter_holidays(y_true: pd.Series, y_pred: np.ndarray) -> Tuple[pd.Series, np.ndarray]:
        """
        Filters out holidays from the true and predicted series based on the specified country.
        """
        country_holidays = holidays.CountryHoliday(run_params["country_name"])
        holiday_mask = y_true.index.isin(country_holidays)
        return y_true[~holiday_mask], y_pred[~holiday_mask]


    def _custom_mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Custom MAPE calculation considering country holidays.
        """
        
        # if run_params["country_name"]:
        #    y_true, y_pred = _filter_holidays(y_true, y_pred)
            
        return mean_absolute_percentage_error(y_true, y_pred)


    def _custom_mspe(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Custom MSPE calculation considering country holidays.
        """
        # if run_params["country_name"]:
        #    y_true, y_pred = _filter_holidays(y_true, y_pred)
        
        return mean_squared_percentage_error(y_true, y_pred)


    def _custom_predictors(y: pd.Series) -> np.ndarray:
        """
        Generate predictors using rolling statistics and lagged values from input array y.
        """
        predictors = []

        # Pre-compute rolling statistics only once for each window size
        for rolling_window in run_params["rolling_window_range"]:
            # Slice the relevant window segment once to avoid repetitive slicing
            window_segment = y[-rolling_window:]
            predictors.extend([np.mean(window_segment), 
                               np.std(window_segment), 
                               np.min(window_segment), 
                               np.max(window_segment)])

        # Extend with lagged values using slicing
        if lag_window > 0:
            predictors.extend(y[-1:-lag_window-1:-1])

        # Combine all predictors into one array
        return np.hstack(predictors)


    def _custom_weights(index: pd.DatetimeIndex) -> np.ndarray:
        """
        Return a list of weights for each index in the DataFrame, considering weekends and holidays.
        """
        # Initialize all weights as 1
        weights = np.ones(len(index))

        # Identify weekend days
        weekend_indices = index.dayofweek >= 5
        weights[weekend_indices] = run_params["weekend_weight"]

        # Check for holidays if a country is specified
        if run_params["country_name"]:
            country_holidays = holidays.CountryHoliday(run_params["country_name"])
            holiday_indices = index.isin(country_holidays)
            weights[holiday_indices] = run_params["holiday_weight"]

        return weights


    def _grid_search_forecaster(model: BaseEstimator, y: pd.Series, exog_data: pd.DataFrame, 
                                param_grid: Dict, lag_window: int) -> pd.DataFrame:
        """
        Create a forecaster and perform a grid search to find the best model.
        """

        forecaster = ForecasterAutoregCustom(regressor=model,
                                             fun_predictors=_custom_predictors,
                                             window_size=run_params["initial_window_size"],
                                             weight_func=_custom_weights)

        return grid_search_forecaster(forecaster=forecaster,
                                      y=y,
                                      exog=exog_data,
                                      param_grid=param_grid,
                                      steps=run_params["test_steps"],
                                      fixed_train_size=False,
                                      refit=True,
                                      metric=[_custom_mape, _custom_mspe],
                                      initial_train_size=len(y)-run_params["test_size"],
                                      return_best=False,
                                      verbose=False)


    def _get_best_parameters(search_results: pd.DataFrame) -> Tuple[Dict, float, int]:
        """
        Get the parameters, the best score, and the lag window for the best-performing model from the search results.
        """
        metric_key = "_custom_mspe" if run_params["metric_key"].lower() == 'mspe' else "_custom_mape"
        best_result = search_results.loc[search_results[metric_key].idxmin()]
        best_params = best_result["params"]
        best_score = best_result[metric_key]
        lag_window = best_result["lag_window"]
        return best_params, best_score, lag_window

    # Loop through lags windows
    results_list = []
    for lag_window in lag_window_range:
        results_grid = _grid_search_forecaster(model, train_df["y"], train_df[run_params["exog_cols_all"]],
                                               param_grid, lag_window)
        results_grid["lag_window"] = lag_window
        results_list.append(results_grid)

    # Compile and process results
    search_results = pd.concat(results_list, ignore_index=True)
    best_params, best_score, lag_window = _get_best_parameters(search_results)
    best_dict = {"best_params": best_params, "best_score": best_score, "lag_window": lag_window}
    
    # Instantiate and train the best model
    forecaster = ForecasterAutoregCustom(regressor=model.set_params(**best_params),
                                         fun_predictors=_custom_predictors,
                                         window_size=run_params["initial_window_size"],
                                         weight_func=_custom_weights)

    return best_dict, search_results, forecaster


def generate_forecast_skforecast(best_model, run_params, y, forecast_df, forecast_start_date, forecast_period):
    """
    Generates a forecast DataFrame with prediction intervals.
    """

    # Predict intervals
    df_forecast = best_model.predict_interval(last_window=y, steps=forecast_period,
                                              exog=forecast_df[run_params["exog_cols_all"]],
                                              interval=[5, 95], n_boot=50)
    df_forecast.reset_index(inplace=True)
    df_forecast.rename(columns={"index": "ds", "pred": "y_pred", "lower_bound": "min_pred", "upper_bound": "max_pred"},
                       inplace=True)
    
    return df_forecast