import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
from sktime.forecasting.model_selection import ExpandingWindowSplitter, ForecastingGridSearchCV
from sktime.performance_metrics.forecasting import mean_squared_percentage_error, mean_absolute_percentage_error
from sktime.forecasting.naive import NaiveForecaster


def find_best_model_sktime(y, run_params, model, param_grid):
    """
    Perform a forecasting grid search with expanding window cross-validation.
    """
    # Create a cross-validation object with expanding windows
    cv = ExpandingWindowSplitter(initial_window=len(y)-run_params["test_size"],
                                 fh=run_params["test_steps"],
                                 step_length=run_params["test_steps"])

    if model != NaiveForecaster():
    
        # Adjust parameter grid based on country name for holidays
        if run_params["country_name"]:
            param_grid["add_country_holidays"] = [{"country_name": run_params["country_name"]}]
        else:
            param_grid["add_country_holidays"] = [None]
    
    # Select metric based on run_params
    metric_key = (mean_squared_percentage_error if run_params["metric_key"].lower() == "mspe"
                  else mean_absolute_percentage_error)

    # Set up the grid search with the model, param_grid, and scoring
    gscv = ForecastingGridSearchCV(
        forecaster=model,
        param_grid=param_grid,
        cv=cv,
        strategy="refit",
        scoring=metric_key,
        n_jobs=-1,
        verbose=1
    )

    # Fit the model using grid search
    gscv.fit(y)

    # Compile best model"s parameters and metrics into a dictionary
    best_dict = {
        "best_params": gscv.best_params_,
        "best_score": gscv.best_score_
    }

    # Return the best parameters, CV results, and the best forecaster
    return best_dict, gscv.cv_results_, gscv.best_forecaster_


def generate_forecast_sktime(best_model, forecast_period):
    """
    Generates a DataFrame containing forecast predictions, prediction intervals, and a date range.
    """
    # Create forecast horizon and date range simultaneously
    forecast_horizon = np.arange(1, forecast_period+1)
    
    # Generate predictions
    forecast_pred = best_model.predict(fh=forecast_horizon).reset_index(drop=False)
    forecast_pred.columns = ["ds", "y_pred"]

    # Generate prediction intervals
    forecast_pred_int = best_model.predict_interval(fh=forecast_horizon, coverage=0.95).reset_index(drop=False)
    forecast_pred_int.columns = ["ds", "min_pred", "max_pred"]

    # Combine predictions and intervals, add the date range
    forecast_results = forecast_pred.merge(forecast_pred_int, on="ds")
    
    # Ensure the lengths are equal
    assert len(forecast_results) == forecast_period, "Error: Length mismatch!"
    
    return forecast_results