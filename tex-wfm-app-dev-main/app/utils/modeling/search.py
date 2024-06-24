import sys
import os
import time
import warnings
import logging
from typing import Optional, Tuple, List, Dict, Any, Callable, Union

import numpy as np
import pandas as pd
import yaml
import holidays

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from scipy.optimize import minimize
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.model_selection import grid_search_forecaster
from sktime.performance_metrics.forecasting import (
    mean_absolute_percentage_error,
    mean_squared_percentage_error,
    mean_absolute_scaled_error,
    mean_squared_scaled_error
)
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    ForecastingGridSearchCV
)
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.naive import NaiveForecaster, NaiveVariance

import streamlit as st

from tqdm import tqdm



# Create a logger
logger = logging.getLogger(__name__)


def generate_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional time-based features to a DataFrame.
    """
    
    with warnings.catch_warnings():
        # Ignore warnings
        warnings.simplefilter("ignore")
        
        # Extract various features from the DateTimeIndex
        if st.session_state.forecast_freq.lower() == "daily":
            df['day_of_week'] = df.index.dayofweek + 1
            df['day_of_year'] = df.index.dayofyear
            df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        if st.session_state.forecast_freq.lower() in ["daily", "weekly"]:
            df['week_of_year'] = df.index.isocalendar().week.astype(int)
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['year'] = df.index.year
        
        # Define a function to count holidays within a given period
        def count_holidays(period_start, period_end):
            return sum(1 for date in pd.date_range(start=period_start, end=period_end) if date in country_holidays)
        
        # Check for holidays if a country name is specified
        if st.session_state.country_name.lower() != "none":
            # Get holidays for the specified country
            country_holidays = holidays.CountryHoliday(st.session_state.country_name)
            
            # Determine the frequency and apply the function
            if st.session_state.forecast_freq.lower() == "daily":
                df['is_holiday'] = df.index.map(lambda date: int(date in country_holidays))
            elif st.session_state.forecast_freq.lower() == "weekly":
                df['is_holiday'] = df.index.map(lambda date: count_holidays(date - pd.Timedelta('6D'), date))
            elif st.session_state.forecast_freq.lower() == "monthly":
                df['is_holiday'] = df.index.map(lambda date: count_holidays(date.replace(day=1), date))

    return df


def grid_search_skforecast(lag_window_range: List[int], 
                           model: BaseEstimator, 
                           train_data: pd.DataFrame, 
                           exog_cols: List[str], 
                           param_grid: Dict) -> Tuple[pd.DataFrame, Dict, ForecasterAutoregCustom]:
    """
    Function to perform grid search over different window sizes.
    """
    
    if st.session_state.forecast_freq.lower() == "daily":
        window_size = 90
        rolling_window_range = [7, 15, 30, 60, 90]
        validation_steps = 15
        validation_size = 60
    elif st.session_state.forecast_freq.lower() == "weekly":
        window_size = 12
        rolling_window_range = [4, 8, 12]
        validation_steps = 2
        validation_size = 8
    elif st.session_state.forecast_freq.lower() == "monthly":
        window_size = 3
        rolling_window_range = [3]
        validation_steps = 1
        validation_size = 3
    
    def _load_params() -> Dict:
        """
        Load parameters from a yaml file.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'params', 'grid_search.yaml')
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_country() -> str:
        """
        Determine the country based on session state.
        """
        country = st.session_state.country_name
        return None if country.lower() == "none" else country

    def _create_forecaster(model: BaseEstimator, y: pd.Series, exog_cols: pd.DataFrame, 
                           param_grid: Dict, lag_window: int, validation_steps: int, validation_size: int) -> pd.DataFrame:
        """
        Create a forecaster and perform a grid search to find the best model.
        """
        
        forecaster = ForecasterAutoregCustom(regressor=model,
                                             fun_predictors=_custom_predictors,
                                             window_size=window_size,
                                             weight_func=_custom_weights)

        return grid_search_forecaster(forecaster=forecaster,
                                      y=y,
                                      exog=exog_cols,
                                      param_grid=param_grid,
                                      steps=validation_steps,
                                      fixed_train_size=False,
                                      refit=True,
                                      metric=[_custom_mape, _custom_mspe],
                                      initial_train_size=len(y)-validation_size,
                                      return_best=False,
                                      verbose=False)


    def _get_best_model_parameters_and_metrics(search_results: pd.DataFrame, metric_key: str) -> Tuple[Dict, float, int]:
        """
        Get the parameters and metrics for the best model.
        """
        best_params = search_results.iloc[0]['params']
        best_score = search_results.iloc[0][metric_key]
        lag_window = search_results.iloc[0]['lag_window']
        return best_params, best_score, lag_window
    
    def _filter_holidays(y_true: pd.Series, y_pred: np.ndarray, country: str) -> Tuple[pd.Series, np.ndarray]:
        """
        Filters out holidays from the true and predicted series based on the specified country.
        """
        country_holidays = holidays.CountryHoliday(country)
        holiday_mask = y_true.index.map(lambda date: int(date in country_holidays))
        holiday_mask = (holiday_mask == 1)

        y_true = pd.Series([act for act, mask in zip(y_true, holiday_mask) if not mask])
        y_pred = np.array([pred for pred, mask in zip(y_pred, holiday_mask) if not mask])

        return y_true, y_pred
    
    def _custom_mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Custom MAPE calculation considering country holidays.
        """
        # Filter out holidays
        if country:
            y_true, y_pred = _filter_holidays(y_true, y_pred, country)

        # Calculate errors
        return mean_absolute_percentage_error(y_true, y_pred)

    def _custom_mspe(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Custom MSPE calculation considering country holidays.
        """
        # Filter out holidays
        if country:
            y_true, y_pred = _filter_holidays(y_true, y_pred, country)

        # Calculate errors
        return mean_squared_percentage_error(y_true, y_pred)

    def _custom_predictors(y: pd.Series) -> np.ndarray:
        """
        Function to create custom predictors for a time series.
        """
        predictors = []

        # Calculate rolling statistics for specific rolling window sizes
        for rolling_window in rolling_window_range:
            predictors.extend([np.mean(y[-rolling_window:]), 
                               np.std(y[-rolling_window:]), 
                               np.min(y[-rolling_window:]), 
                               np.max(y[-rolling_window:])])

        # Create lags
        predictors.extend(y[-1:-lag_window-1:-1])

        # Combine all predictors into one array
        return np.hstack(predictors)

    def _custom_weights(index: pd.DatetimeIndex, country: str=None) -> np.ndarray:
        """
        Return a list of weights for each index in the DataFrame.
        """
        # Start with all weights as 1
        weights = np.ones(len(index))

        # Find indices of weekends
        weekend_indices = index.dayofweek.isin([5, 6])

        # Check for holidays if country is specified
        if country:
            country_holidays = holidays.CountryHoliday(country)
            holiday_indices = index.map(lambda date: date in country_holidays)
            weights[holiday_indices] = holiday_weight

        # Set weights to 2 if either a weekend or a holiday
        weights[weekend_indices] = weekend_weight

        return weights
    
    # Load parameters from yaml file
    params = _load_params()
    
    metric_key = '_custom_mspe' if st.session_state.objective_type.lower() == 'forecasting' else '_custom_mape'
    weekend_weight = params['weekend_weight']
    holiday_weight = params['holiday_weight']

    # Determine the country based on session state
    country = _get_country()

    # Grid search over different window sizes
    total_iterations = len(lag_window_range)
    pbar = tqdm(total=total_iterations, desc="Grid Search Progress")

    # Perform grid search for each lag window and compile results
    results_list = []
    for lag_window in lag_window_range:
        results_grid = _create_forecaster(model, train_data['y'], train_data[exog_cols], 
                                          param_grid, lag_window, validation_steps, validation_size)
        results_grid['lag_window'] = lag_window
        results_list.append(results_grid)
        pbar.update()

    pbar.close()

    # Get the best model parameters and metrics
    search_results = pd.concat(results_list, ignore_index=True).sort_values(metric_key)
    best_params, best_score, lag_window = _get_best_model_parameters_and_metrics(search_results, metric_key)
    best_dict = {"best_params": best_params, "best_score": best_score, "lag_window": lag_window}

    # Instantiate and train the best model
    forecaster = ForecasterAutoregCustom(regressor=model.set_params(**best_params),
                                         fun_predictors=_custom_predictors,
                                         window_size=int(window_size))

    return search_results, best_dict, forecaster


def grid_search_sktime(model: Callable, train: pd.DataFrame, param_grid: Dict[str, List]) -> Tuple[Dict, Dict, Callable]:
    """
    Conduct a grid search on the parameter space to find the optimal parameters for the model.
    """
    
    # Get the current directory and construct file path to the parameters file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'params', 'grid_search.yaml')

    # Open the yaml file and load the parameters
    with open(file_path, 'r') as f:
        params = yaml.safe_load(f)

    if st.session_state.forecast_freq.lower() == "daily":
        validation_steps = 15
        validation_size = 60
    elif st.session_state.forecast_freq.lower() == "weekly":
        validation_steps = 2
        validation_size = 8
    elif st.session_state.forecast_freq.lower() == "monthly":
        validation_steps = 1
        validation_size = 3

    # Retrieve country name from the streamlit session
    country = st.session_state.country_name
    if country.lower() == "none":
        country = None
        params['add_country_holidays'] = [None]
    else:
        params['add_country_holidays'] = [{'country_name': country}]

    # Create a cross validation object with expanding windows
    cv = ExpandingWindowSplitter(initial_window=len(train)-validation_size,
                                 fh=validation_steps,
                                 step_length=validation_steps)
    
    # Perform grid search over the model's parameters space
    gscv = ForecastingGridSearchCV(
        forecaster=model,
        param_grid=param_grid,
        cv=cv,
        strategy="refit",
        scoring=mean_absolute_percentage_error,
        n_jobs=1,
        verbose=1
    )

    # Fit the model with grid search
    gscv.fit(train)
    
    # Get the cross-validation results
    search_results = gscv.cv_results_

    # Compile best model's parameters and metrics into a dictionary
    best_dict = {
        "best_params": gscv.best_params_,
        "best_score": gscv.best_score_
    }
    
    print(f"Best model's parameters and metrics: {best_dict}")

    # Instantiate and train the best model
    forecaster = gscv.best_forecaster_

    # Return search results, best model parameters and metrics, and best model
    return search_results, best_dict, forecaster


def compute_error_metrics(
    actual: List[float], 
    predicted: List[float], 
    train_actual: List[float], 
    holiday_mask: Union[List[bool], None] = None
) -> Dict[str, float]:
    """
    Compute error metrics between actual and predicted values, with the option to exclude holidays.
    """
    # Check if holiday mask is provided and matches the length of actual data
    if holiday_mask is not None and len(holiday_mask) == len(actual):
        actual = [act for act, mask in zip(actual, holiday_mask) if not mask]
        predicted = [pred for pred, mask in zip(predicted, holiday_mask) if not mask]
    
    # Calculate Mean Absolute Percentage Error
    mape = mean_absolute_percentage_error(actual, predicted)
    
    # Calculate Root Mean Squared Percentage Error
    rmspe = mean_squared_percentage_error(actual, predicted, square_root=True)
    
    # Calculate Mean Absolute Scaled Error
    mase = mean_absolute_scaled_error(actual, predicted, y_train=train_actual)
    
    # Calculate Root Mean Squared Scaled Error
    rmsse = mean_squared_scaled_error(actual, predicted, y_train=train_actual, square_root=True)
    
    # Calculate Root Mean Squared Error
    rmse = mean_squared_error(actual, predicted, squared=False)
    
    # Calculate Mean Absolute Error
    mae = mean_absolute_error(actual, predicted)
    
    # Compile the error metrics into a dictionary
    results = {
        'mape': mape,
        'rmspe': rmspe,
        'mase': mase,
        'rmsse': rmsse,
        'rmse': rmse,
        'mae': mae
    }
    
    return results


def compute_prediction_coverage(test_data: pd.DataFrame, predicted_data: pd.DataFrame, holiday_mask: Union[List[bool], None] = None) -> float:
    """
    Compute the prediction coverage, with the option to exclude holidays.
    """
    # Check if holiday mask is provided and matches the length of test data
    if holiday_mask is not None:
        if len(holiday_mask) == len(test_data):
            test_data = test_data[~holiday_mask]
            predicted_data = predicted_data[~holiday_mask]

    # Calculate coverage by finding the mean percentage of predictions within the bounds
    coverage = np.where(
                    (test_data.loc[predicted_data.index, 'y'] >= predicted_data['lower_bound']) &
                    (test_data.loc[predicted_data.index, 'y'] <= predicted_data['upper_bound']),
                    True,
                    False
                ).mean() * 100

    return coverage


def calculate_optimal_weights(actual: List[float], predictions: List[List[float]], holiday_mask: Union[List[bool], None] = None) -> np.ndarray:
    """
    Calculate the optimal weights for a set of predictions based on actual data.
    """

    def mape_metrics(actual: List[float], predicted: List[float], holiday_mask: Union[List[bool], None] = None) -> float:
        """
        Calculate the Mean Absolute Percentage Error (MAPE) between actual and predicted data.
        """
        # Check if holiday mask is provided and matches the length of actual data
        if holiday_mask is not None:
            if len(holiday_mask) == len(actual):
                # Exclude holidays
                actual = [act for act, mask in zip(actual, holiday_mask) if not mask]
                predicted = [pred for pred, mask in zip(predicted, holiday_mask) if not mask]

        # Calculate MAPE
        mape = mean_absolute_percentage_error(actual, predicted)

        return mape

    # Convert the predictions and actual data into numpy arrays for manipulation
    predictions = np.array(predictions)
    actual = np.array(actual)

    # Check if holiday mask is provided and matches the length of actual data
    if holiday_mask is not None:
        assert len(holiday_mask) == len(actual)

    def objective(weights: np.ndarray) -> float:
        """
        Define an objective function to minimize.
        """
        # Ensure weights are non-negative and normalized to 1
        weights = np.abs(weights)
        weights /= np.sum(weights)

        # Calculate the weighted sum of predictions
        weighted_preds = np.dot(weights, predictions)

        # Calculate the Mean Absolute Percentage Error (MAPE) as our metric to minimize
        metrics = mape_metrics(actual, weighted_preds, holiday_mask)

        return metrics

    # Initialize the weights evenly across all predictions
    initial_weights = np.array([1.0/len(predictions)]*len(predictions))

    # Set bounds to ensure weights are between 0 and 1
    bounds = [(0.0, 1.0)] * len(predictions)

    # Set constraints to ensure weights sum up to 1
    constraints = {'type': 'eq', 'fun': lambda w: 1 - sum(w)}

    # Use Scipy's SLSQP method to find the optimal weights that minimize the objective function
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Check the result of the optimization
    if result.success:
        optimal_weights = result.x
    else:
        raise ValueError("Optimization for weights did not converge:", result.message)

    return optimal_weights

    
def find_best_model(train_set: pd.DataFrame, st_tab) -> None:
    """
    Find the best model by performing a grid search if rerun_search is True. Fit the best model with the 
    entire data and store it in the session state. Retrieve the best model and parameters from the session state.
    """
    logger.info("Starting search for best model..")

    # Create a progress bar for the model search
    model_search_bar = st_tab.progress(0)
    model_search_bar.progress(0/5, text=f"⌛ Estimated Time Remaining: Calculating")

    # Record the start time of the model search
    init_time = time.time()
    start_time = time.time()

    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change the frequency of the data based on the weekend drop session state
    try:
        if st.session_state.forecast_freq.lower() == "daily" and st.session_state.weekend_drop.lower() == "yes":
            forecast_freq = "B"
            lag_window_range = [7, 15, 30, 60, 90]
            test_steps = 30
        elif st.session_state.forecast_freq.lower() == "daily" and st.session_state.weekend_drop.lower() == "no":
            forecast_freq = "D"
            lag_window_range = [7, 15, 30, 60, 90]
            test_steps = 30
        elif st.session_state.forecast_freq.lower() == "weekly":
            forecast_freq = "W"
            lag_window_range = [4, 8, 12]
            test_steps = 4
        elif st.session_state.forecast_freq.lower() == "monthly":
            forecast_freq = "M"
            lag_window_range = [3]
            test_steps = 3
        else:
            raise ValueError("Unknown Frequency!")
    except Exception as e:
        model_search_bar.error(e, icon="🚨")
        logger.error(e)

        
    #####################################################################################
    # Load parameters from the grid_search.yaml file
    file_path = os.path.join(current_dir, 'params', 'grid_search.yaml')
    with open(file_path, 'r') as f:
        params = yaml.safe_load(f)

    # Split the train set into train and test data based on the number of test steps and the frequency
    train_data = train_set.copy(deep=True).asfreq('D').resample(forecast_freq).mean()[:-test_steps]
    test_data = train_set.copy(deep=True).asfreq('D').resample(forecast_freq).mean()[-test_steps:]
    
    train_set = train_set.copy(deep=True).asfreq('D').resample(forecast_freq).mean()

    logger.info(f"Train Date Range : {train_data.index.min()} --- {train_data.index.max()}  (n={len(train_data)})")
    logger.info(f"Test Date Range  : {test_data.index.min()} --- {test_data.index.max()}  (n={len(test_data)})")
    
    # Get the names of the exogenous variables from the train data
    exog_cols = list((train_data.columns).difference(['y']))
    st.session_state.exog_cols_actual = exog_cols

    # Generate date features for the train and test data
    train_filtered = generate_date_features(train_set.copy(deep=True))
    train_data = generate_date_features(train_data)
    test_data = generate_date_features(test_data)

    # Get the names of the exogenous variables from the train data
    exog_cols = list((train_data.columns).difference(['y']))
    st.session_state.exog_cols = exog_cols

    # Get the actual target values from the test data
    actual = test_data['y']

    #####################################################################################
    # If the session state country is "none", set holiday_mask to None,
    # else create a holiday mask from the test data
    if st.session_state.country_name.lower() == "none":
        holiday_mask = None
    else:
        holiday_mask = (test_data['is_holiday'] == 1).values
        logger.warning("WARNING: Holidays are excluded from model evaluation!")

    # Set prediction interval parameters
    pi_lower = 5
    pi_upper = 95
    pi_n_boots = 50

    #####################################################################################

    # Load parameters from the xgboost.yaml file
    file_path = os.path.join(current_dir, 'params', 'xgboost.yaml')
    with open(file_path, 'r') as file:
        param_grid_xgb = yaml.safe_load(file)

    # Create an instance of the XGBoost regressor
    model = XGBRegressor()

    # Perform grid search on the XGBoost regressor
    search_results_xgb, best_dict_xgb, forecaster_xgb = grid_search_skforecast(
        lag_window_range, model, train_data, exog_cols, param_grid_xgb
    )

    logger.info(f"XGBOOST - Best score: {best_dict_xgb['best_score']}, Best params: {best_dict_xgb['best_params']}")

    # Store the best estimator and model parameters in the session state
    st.session_state.xgb_best_estimator = forecaster_xgb
    st.session_state.xgb_best_model_params = best_dict_xgb

    # Fit the best model on the train data and compute error metrics on the test data
    best_model_xgb = forecaster_xgb
    best_model_xgb.fit(y=train_data['y'], exog=train_data[exog_cols])

    # Generate prediction intervals for the test data
    predictions_xgb = forecaster_xgb.predict_interval(
        steps=test_steps, exog=test_data[exog_cols], interval=[pi_lower, pi_upper], n_boot=pi_n_boots
    )

    # Compute error metrics and prediction coverage for the test data
    st.session_state.xgb_test_metrics = compute_error_metrics(
        actual, predictions_xgb['pred'], train_data['y'], holiday_mask
    )
    st.session_state.xgb_test_coverage = compute_prediction_coverage(test_data, predictions_xgb, holiday_mask)

    # Fit the best model on the filtered train data
    best_model_xgb = forecaster_xgb
    best_model_xgb.fit(y=train_filtered['y'], exog=train_filtered[exog_cols])

    # Store the best model in the session state
    st.session_state.xgb_best_model = best_model_xgb
    
    # Store the feature importance of the best model
    st.session_state.xgb_best_model_fi = best_model_xgb.get_feature_importances()
    
    # Update the progress bar with the time taken
    end_time = time.time()
    remaining_time = round(6*(end_time - start_time)/60, 2)
    start_time = time.time()
    st.session_state.search_time = round((end_time - init_time)/60, 2)

    model_search_bar.progress(1/5, text=f"⌛ 1/5 Completed! - Estimated Time Remaining: {remaining_time} minutes")

    #####################################################################################

    # Load parameters from the YAML file
    file_path = os.path.join(current_dir, 'params', 'random_forest.yaml')
    with open(file_path, 'r') as file:
        param_grid_rf = yaml.safe_load(file)

    # Grid search for RandomForestRegressor
    model = RandomForestRegressor()
    search_results_rf, best_dict_rf, forecaster_rf = grid_search_skforecast(lag_window_range,
                                                                 model, train_data, exog_cols, param_grid_rf)

    logger.info(f"RANDOM FOREST - Best score: {best_dict_rf['best_score']}, Best params: {best_dict_rf['best_params']}")

    # Store the best estimator and model parameters in the session state
    st.session_state.rf_best_estimator = forecaster_rf
    st.session_state.rf_best_model_params = best_dict_rf

    # Fit the best model on the train data and compute error metrics on the test data
    best_model_rf = forecaster_rf
    best_model_rf.fit(y=train_data['y'], exog=train_data[exog_cols])

    # Generate prediction intervals for the test data
    predictions_rf = forecaster_rf.predict_interval(steps=test_steps, exog=test_data[exog_cols],
                                                    interval=[pi_lower, pi_upper], n_boot=pi_n_boots)

    # Compute error metrics and prediction coverage for the test data
    st.session_state.rf_test_metrics = compute_error_metrics(actual, predictions_rf['pred'], train_data['y'], holiday_mask)
    st.session_state.rf_test_coverage = compute_prediction_coverage(test_data, predictions_rf, holiday_mask)

    # Fit the best model on the full data
    best_model_rf = forecaster_rf
    best_model_rf.fit(y=train_filtered['y'], exog=train_filtered[exog_cols])

    # Store the best model in the session state
    st.session_state.rf_best_model = best_model_rf
    
    # Store the feature importance of the best model
    st.session_state.rf_best_model_fi = best_model_rf.get_feature_importances()

    # Record end time, calculate time taken
    end_time = time.time()
    remaining_time = round(5*(end_time - start_time)/60, 2)
    start_time = time.time()
    st.session_state.search_time = round((end_time - init_time)/60, 2)

    model_search_bar.progress(2/5, text=f"⌛ 2/5 Completed! - Estimated Time Remaining: {remaining_time} minutes")

    #####################################################################################

    # Ensemble Method
    # Combine the predictions from XGBoost and Random Forest
    predictions_ensemble = [predictions_xgb['pred'], predictions_rf['pred']]

    # Compute the optimal weights for the ensemble method
    optimal_weights = calculate_optimal_weights(actual, predictions_ensemble)
    logger.info(f"Optimal weights for Ensemble Method: {optimal_weights}")

    # Define the parameter grid for the ensemble method
    param_grid_ensemble = {
        'weights': [[0, 1], [1, 0], [0.5, 0.5], optimal_weights]
    }

    # Define the ensemble model with XGBoost and Random Forest as estimators
    model = VotingRegressor(
        estimators=[
            ('xgb', XGBRegressor(**st.session_state.xgb_best_model_params['best_params'])),
            ('rf', RandomForestRegressor(**st.session_state.rf_best_model_params['best_params']))
        ]
    )

    # Perform grid search on the ensemble model
    search_results_ensemble, best_dict_ensemble, forecaster_ensemble = grid_search_skforecast(
        lag_window_range, model, train_data, exog_cols, param_grid_ensemble
    )

    logger.info(f"ENSEMBLE - Best score: {best_dict_ensemble['best_score']}, Best params: {best_dict_ensemble['best_params']}")

    # Store the best estimator and model parameters for the ensemble method in the session state
    st.session_state.ensemble_best_estimator = forecaster_ensemble
    st.session_state.ensemble_best_model_params = best_dict_ensemble

    # Fit the ensemble model on the train data and compute error metrics on the test data
    best_model_ensemble = forecaster_ensemble
    best_model_ensemble.fit(y=train_data['y'], exog=train_data[exog_cols])

    # Generate prediction intervals for the test data
    predictions_ensemble = forecaster_ensemble.predict_interval(
        steps=test_steps, exog=test_data[exog_cols], interval=[pi_lower, pi_upper], n_boot=pi_n_boots
    )

    # Compute error metrics and prediction coverage for the test data
    st.session_state.ensemble_test_metrics = compute_error_metrics(
        actual, predictions_ensemble['pred'], train_data['y'], holiday_mask
    )
    st.session_state.ensemble_test_coverage = compute_prediction_coverage(test_data, predictions_ensemble, holiday_mask)

    # Fit the ensemble model on the filtered train data
    best_model_ensemble = forecaster_ensemble
    best_model_ensemble.fit(y=train_filtered['y'], exog=train_filtered[exog_cols])

    # Store the best ensemble model in the session state
    st.session_state.ensemble_best_model = best_model_ensemble

    # Update the progress bar with the time taken
    end_time = time.time()
    remaining_time = round(4*(end_time - start_time)/60, 2)
    start_time = time.time()
    st.session_state.search_time = round((end_time - init_time)/60, 2)

    model_search_bar.progress(3/5, text=f"⌛ 3/5 Completed! - Estimated Time Remaining: {remaining_time} minutes")
    
    #####################################################################################

    # Split the train set into train and test data based on the number of test steps and the frequency
    train_data = train_set.copy(deep=True).asfreq('D').resample(forecast_freq).mean()[:-test_steps]
    test_data = train_set.copy(deep=True).asfreq('D').resample(forecast_freq).mean()[-test_steps:]
    
    train_set = train_set.copy(deep=True).asfreq('D').resample(forecast_freq).mean()
    
    # Dropping all columns except the target
    train_data = train_data.drop(columns=train_data.columns.difference(['y']), axis=1)
    test_data = test_data.drop(columns=train_data.columns.difference(['y']), axis=1)
    
    train_set = train_set.drop(columns=train_set.columns.difference(['y']), axis=1)

    logger.info(f"Train Date Range : {train_data.index.min()} --- {train_data.index.max()}  (n={len(train_data)})")
    logger.info(f"Test Date Range  : {test_data.index.min()} --- {test_data.index.max()}  (n={len(test_data)})")

    # Get the actual target values for the test data
    actual = test_data['y']

    # Generate a forecast horizon as a range from 0 to the length of the test data
    fh = np.arange(0, len(test_data))

    # Generate a date range for the test data
    test_date_range = pd.date_range(test_data.index[0], periods=len(fh), freq=forecast_freq)

    #####################################################################################

    # Load parameters for Prophet from the YAML file
    file_path = os.path.join(current_dir, 'params', 'prophet.yaml')
    with open(file_path, 'r') as file:
        param_grid_prophet = yaml.safe_load(file)

    # Create an instance of Prophet
    model = Prophet()

    # Perform grid search on Prophet with the loaded parameters
    search_results_prophet, best_dict_prophet, forecaster_prophet = grid_search_sktime(model, train_data, param_grid_prophet)

    # Log the best score and parameters for Prophet
    logger.info(f"PROPHET - Best score: {best_dict_prophet['best_score']}, Best params: {best_dict_prophet['best_params']}")

    # Store the best estimator and model parameters for Prophet in the session state
    st.session_state.prophet_best_estimator = forecaster_prophet
    st.session_state.prophet_best_model_params = best_dict_prophet

    # Fit the best model from Prophet on the train data
    best_model_prophet = forecaster_prophet
    best_model_prophet.fit(y=train_data)

    # Generate predictions and prediction intervals for the test data
    forecast_pred = best_model_prophet.predict(fh=fh).reset_index(drop=True)
    forecast_pred.columns = ["pred"]

    forecast_pred_int = best_model_prophet.predict_interval(fh=fh).reset_index(drop=True)
    forecast_pred_int.columns = ["lower_bound", "upper_bound"]

    # Combine the predictions and prediction intervals
    predictions_prophet = pd.concat([forecast_pred, forecast_pred_int], axis=1)
    predictions_prophet.index = test_date_range

    assert len(test_data) == len(predictions_prophet)

    # Compute error metrics and prediction coverage for the test data
    st.session_state.prophet_test_metrics = compute_error_metrics(
        actual, predictions_prophet['pred'], train_data['y'], holiday_mask
    )
    
    st.session_state.prophet_test_coverage = compute_prediction_coverage(test_data, predictions_prophet, holiday_mask)

    # Fit the best model from Prophet on the full data
    best_model_prophet = forecaster_prophet
    best_model_prophet.fit(y=train_set.copy(deep=True).asfreq('D').resample(forecast_freq).mean())

    # Store the best Prophet model in the session state
    st.session_state.prophet_best_model = best_model_prophet

    # Update the progress bar with the time taken
    end_time = time.time()
    remaining_time = round((end_time - start_time)/10/60, 2)
    start_time = time.time()
    st.session_state.search_time = round((end_time - init_time)/60, 2)

    model_search_bar.progress(4/5, text=f"⌛ 4/5 Completed! - Estimated Time Remaining: {remaining_time} minutes")

    #####################################################################################

    # Load parameters for NaiveForecaster from the YAML file
    file_path = os.path.join(current_dir, 'params', 'naive.yaml')
    with open(file_path, 'r') as file:
        param_grid_naive = yaml.safe_load(file)

    # Create an instance of NaiveForecaster
    model = NaiveForecaster()

    # Perform grid search on NaiveForecaster with the loaded parameters
    search_results_naive, best_dict_naive, forecaster_naive = grid_search_sktime(model, train_data, param_grid_naive)

    # Log the best score and parameters for NaiveForecaster
    logger.info(f"NAIVE - Best score: {best_dict_naive['best_score']}, Best params: {best_dict_naive['best_params']}")

    # Store the best estimator and model parameters for NaiveForecaster in the session state
    st.session_state.naive_best_estimator = NaiveVariance(forecaster_naive)
    st.session_state.naive_best_model_params = best_dict_naive

    # Fit the best model from NaiveForecaster on the train data
    best_model_naive = NaiveVariance(forecaster_naive)
    best_model_naive.fit(y=train_data)

    # Generate predictions and prediction intervals for the test data
    forecast_pred = best_model_naive.predict(fh=fh).reset_index(drop=True)
    forecast_pred.columns = ["pred"]

    forecast_pred_int = best_model_naive.predict_interval(fh=fh).reset_index(drop=True)
    forecast_pred_int.columns = ["lower_bound", "upper_bound"]

    # Combine the predictions and prediction intervals
    predictions_naive = pd.concat([forecast_pred, forecast_pred_int], axis=1)
    predictions_naive.index = test_date_range

    # Ensure the lengths of test data and predictions are the same
    assert len(test_data) == len(predictions_naive)

    # Compute error metrics and prediction coverage for the test data
    st.session_state.naive_test_metrics = compute_error_metrics(
        actual, predictions_naive['pred'], train_data['y'], holiday_mask
    )
    st.session_state.naive_test_coverage = compute_prediction_coverage(test_data, predictions_naive, holiday_mask)

    # Fit the best model from NaiveForecaster on the full data
    best_model_naive = NaiveVariance(forecaster_naive)
    best_model_naive.fit(y=train_set.copy(deep=True).asfreq('D').resample(forecast_freq).mean())

    # Store the best NaiveForecaster model in the session state
    st.session_state.naive_best_model = best_model_naive

    # Calculate and store the total time taken for the search
    end_time = time.time()
    st.session_state.search_time = round((end_time - init_time)/60, 2)

    # Complete the progress bar
    model_search_bar.progress(5/5, text=f"⌛ 5/5 Completed in {st.session_state.search_time} minutes!")

    #####################################################################################
    # Create a list to store the performance metrics of each model
    metrics_data: List[Dict[str, Union[str, float]]] = [
        {'Model': 'Random Forest',
         'MAPE': st.session_state.rf_test_metrics['mape'],
         'RMSPE': st.session_state.rf_test_metrics['rmspe'],
         'MASE': st.session_state.rf_test_metrics['mase'],
         'RMSSE': st.session_state.rf_test_metrics['rmsse'],
         'Coverage': st.session_state.rf_test_coverage},
        {'Model': 'XGBoost',
         'MAPE': st.session_state.xgb_test_metrics['mape'],
         'RMSPE': st.session_state.xgb_test_metrics['rmspe'],
         'MASE': st.session_state.xgb_test_metrics['mase'],
         'RMSSE': st.session_state.xgb_test_metrics['rmsse'],
         'Coverage': st.session_state.xgb_test_coverage},
        {'Model': 'Ensemble Tree',
         'MAPE': st.session_state.ensemble_test_metrics['mape'],
         'RMSPE': st.session_state.ensemble_test_metrics['rmspe'],
         'MASE': st.session_state.ensemble_test_metrics['mase'],
         'RMSSE': st.session_state.ensemble_test_metrics['rmsse'],
         'Coverage': st.session_state.ensemble_test_coverage},
        {'Model': 'Prophet',
         'MAPE': st.session_state.prophet_test_metrics['mape'],
         'RMSPE': st.session_state.prophet_test_metrics['rmspe'],
         'MASE': st.session_state.prophet_test_metrics['mase'],
         'RMSSE': st.session_state.prophet_test_metrics['rmsse'],
         'Coverage': st.session_state.prophet_test_coverage},
        {'Model': f'Naive - {st.session_state.naive_best_model_params["best_params"]["strategy"].title()}',
         'MAPE': st.session_state.naive_test_metrics['mape'],
         'RMSPE': st.session_state.naive_test_metrics['rmspe'],
         'MASE': st.session_state.naive_test_metrics['mase'],
         'RMSSE': st.session_state.naive_test_metrics['rmsse'],
         'Coverage': st.session_state.naive_test_coverage}
    ]

    # Convert the list of dictionaries into a DataFrame for easy manipulation
    metrics_df = pd.DataFrame(metrics_data)

    # Round off the values in the DataFrame to 3 decimal places for better readability
    metrics_df = metrics_df.round(2)

    # Sort the DataFrame based on the performance metrics in the order of preference
    # MASE > RMSSE > Coverage > MAPE > RMSPE
    metric_order = ['MASE', 'RMSSE', 'Coverage', 'MAPE', 'RMSPE', 'Model']
    ascending_order = [True, True, False, True, True, False]

    metrics_df = metrics_df.sort_values(by=metric_order, ascending=ascending_order).reset_index(drop=True)
    
    metrics_df['Rank'] = metrics_df.index + 1
    
    metric_order.remove('Model')
    metrics_df = metrics_df[['Rank', 'Model'] + metric_order]

    # The best model is the one that is first in the sorted DataFrame
    best_model_type: str = metrics_df[metrics_df['Model'].isin(['Random Forest', 'XGBoost', 'Ensemble Tree'])].iloc[0]['Model']

    # Store the best model type and the comparison metrics in the session state for later use
    st.session_state.best_model_type = best_model_type
    st.session_state.model_comparisons = metrics_df
    

# Expose find_best_model to other modules
__all__ = ['generate_date_features', 'find_best_model', 'compute_error_metrics', 'compute_prediction_coverage']
