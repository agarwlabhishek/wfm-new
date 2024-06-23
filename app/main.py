import os
import io
import sys
import json
import time
import random
import logging

import numpy as np
import pandas as pd
import datetime

import plotly.io as pio
import streamlit as st
import seaborn as sns
import zipfile

import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

from utils.manager.login import *
from utils.inputs.validation import *
from utils.inputs.ads import *
from utils.modeling.general import *
from utils.modeling.skforecast_utils import *
from utils.modeling.sktime_utils import *
from utils.modeling.plot import *

# Set up the logging configuration for cmdstanpy
logger = logging.getLogger()

# Add NullHandler with CRITICAL log level
null_handler = logging.NullHandler()
null_handler.setLevel(logging.CRITICAL)
logger.addHandler(null_handler)

# Add StreamHandler with INFO log level
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

logger.propagate = False

    
st.set_page_config(
    page_title="Forecasting - TEXAS",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Developed by TeX team at Allianz Partners. Please contact abhishek.agarwal@allianz.com for any queries/ complaints."
    } 
)

col_1, col_2 =  st.sidebar.columns([5, 1])

col_1.title("WFM Forecasting :chart_with_upwards_trend:")
col_2.caption("")
col_2.write("v2.0.0")
st.sidebar.write("""The tool leverages advanced statistical & machine learning models to deliver automated, precise predictions for optimizing operational efficiency.""")

# Set logged in flag as False
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'input_submitted' not in st.session_state:
    st.session_state.input_submitted = False

# Checking the login status
try:
    if not st.session_state.logged_in:
        login_page()

except Exception as e:
    st.error(f"Error in checking login status: {e}")
    st.stop()

if st.session_state.logged_in:
    # Retrieve input from the sidebar
    st.sidebar.title("Inputs :file_folder:")


    with st.sidebar.form('user_inputs'):

        # Add dropdown for Country
        country_name = st.selectbox("Holidays - Country Code :world_map:", [None, "BR", "CA", "CN", "DE", "ES", "FR", "IN", "IT", "UK", "US"])

        # Add dropdown for frequency
        forecast_freq = st.selectbox("Forecast Frequency :repeat:", ["B", "D", "W", "M"])

        # Add dropdown for data selection
        data_selection = st.radio("Automatic Data Selection :scissors:", ["No", "Yes"])
        
        # Add file uploaders to the sidebar
        # uploaded_historical_file = st.file_uploader("Historical Data (.csv) :arrow_up: **(MANDATORY)**", type=["csv"])
        # uploaded_forecast_file = st.file_uploader("Forecast Data (.csv) :arrow_up: **(OPTIONAL)**", type=["csv"])
        
        uploaded_historical_file = "Agency Services_multi.csv"
        uploaded_forecast_file = "Agency Services_multi_forecast.csv"

        submitted = st.form_submit_button("Submit")

        if submitted:
            st.session_state.input_submitted = True

if st.session_state.input_submitted:
    if uploaded_historical_file is None:
        st.error(':rotating_light: Please upload the historical data as .csv file to use the tool!')
        st.stop()
    else:
        
        if 'country_name' not in st.session_state or country_name != st.session_state.country_name:
            st.session_state.country_name = country_name
            st.session_state.run_search = True

        if 'forecast_freq' not in st.session_state or forecast_freq != st.session_state.forecast_freq:
            st.session_state.forecast_freq = forecast_freq
            st.session_state.run_search = True
            
            if forecast_freq == "D":
                st.session_state.forecast_period = 92
            elif forecast_freq == "B":
                st.session_state.forecast_period = 66
            elif forecast_freq == "W":
                st.session_state.forecast_period = 26
            elif forecast_freq == "M":
                forecast_period = 12


        if 'data_selection' not in st.session_state or data_selection != st.session_state.data_selection:
            st.session_state.data_selection = data_selection
            st.session_state.run_search = True
        
        if 'uploaded_historical_file' not in st.session_state or uploaded_historical_file != st.session_state.uploaded_historical_file:
            st.session_state.uploaded_historical_file = uploaded_historical_file
            st.session_state.run_search = True
            
        if 'uploaded_forecast_file' not in st.session_state or uploaded_forecast_file != st.session_state.uploaded_forecast_file:
            st.session_state.uploaded_forecast_file = uploaded_forecast_file
            st.session_state.run_search = True
            if uploaded_forecast_file is None:
                st.session_state.external_features = False
            else:
                st.session_state.external_features = True
            
        run_params = {
                "country_name": st.session_state.country_name,
                "forecast_freq": st.session_state.forecast_freq,
                "forecast_period": st.session_state.forecast_period,
                "data_selection": st.session_state.data_selection,
                "external_features": st.session_state.external_features,
                "weekend_weight": 5,
                "holiday_weight": 10,
                "metric_key": "mspe"
            }
        

        # Set up tabs for modelling, analysis, and download results
        modelling_tab, analysis_tab, download_tab = st.tabs(["Modelling :bar_chart:", 
                                                             "Analysis :microscope:", 
                                                             "Download Results :arrow_down:"])

        #####################################################################################

        try:
            # Validate the histoical data
            historical_df = validate_input_file(uploaded_historical_file, run_params["external_features"])
            logging.info(f"Historical Data Size: {historical_df.shape}")
            # Find the min data for optimal train data
            run_params["historical_start_date"] = historical_df['ds'].min()
            run_params["historical_end_date"] = historical_df['ds'].max()
            run_params["forecast_start_date"] = historical_df['ds'].max() + pd.Timedelta(days=1)
            run_params["forecast_end_date"] = historical_df['ds'].max() + pd.Timedelta(days=run_params["forecast_period"])

        except Exception as e:
            # Log this exception or handle it further up the call stack
            raise Exception(f"An error occurred while validating the uploaded historical data: {str(e)}")
            
            
        try:
            if run_params["external_features"]:
                # Validate the forecast data
                forecast_df = validate_input_file(uploaded_forecast_file, run_params["external_features"])
                logging.info(f"Forecast Data Size: {forecast_df.shape}")

                assert forecast_df['ds'].min() == run_params["forecast_start_date"], 'Forecast Start Data is not aligned with Historical End Date'

                assert forecast_df['ds'].max() >= run_params["forecast_end_date"], 'Forecast Data is not available for entire Forecast Period'

            else:
                forecast_df = pd.DataFrame(columns=historical_df.columns)

        except Exception as e:
            # Log this exception or handle it further up the call stack
            raise Exception(f"An error occurred while validating the uploaded forecast data: {str(e)}")
            
        #####################################################################################
        
        try:
            if run_params["data_selection"]:

                # Find optimal window 
                optimal_window_size = find_optimal_window(historical_df)

                logging.info(f"Optimal Window Size: {optimal_window_size}")

                # Add 180 days for feature engineering to optimal window
                optimal_window_size += 180

            else:
                optimal_window_size = len(historical_df)

        except Exception as e:
            # Log this exception or handle it further up the call stack
            raise Exception(f"An error occurred while finding the optimal window: {str(e)}")


        # Truncate the train set based on optimal window
        optimal_df = historical_df[-optimal_window_size:].copy(deep=True)

        logging.info(f"Optimal Train Data Size: {optimal_df.shape}")

        # Find the min data for optimal train data
        run_params["optimal_window_size"] = optimal_window_size
        run_params["optimal_start_date"] = optimal_df['ds'].min()
        run_params["optimal_end_date"] = optimal_df['ds'].max()
        
        st.success(f"Automatic Data Selection: Historical data has been truncated to **{optimal_window_size}** most recent days!")
        
        #####################################################################################
        
        try:
            # Validate column counts based on whether external features are used
            if run_params["external_features"]:
                assert optimal_df.shape[1] > 2 and forecast_df.shape[1] > 2, "Uploaded Historical or Forecast Data does have required number of columns!"
            else:
                assert optimal_df.shape[1] == 2 and forecast_df.shape[1] == 2, "Uploaded Historical or Forecast Data does have required number of columns!"
            # Ensure non-empty data structure
            assert optimal_df.shape[1] > 0, "Uploaded Historical Data does not have enough rows!"
            # Ensure same number of columns
            assert optimal_df.shape[1] == forecast_df.shape[1], "Uploaded Historical and Forecast Data do not have the same number of columns"
        except Exception as e:
            raise ValueError(f"Invalid input data format: {e}")
        
        st.success(f"Data Validation: Input data has been validated sucessfully!")
        
        
        try:
            # Resample data
            optimal_df = resample_dataframe(optimal_df, run_params["forecast_freq"])
            forecast_df = resample_dataframe(forecast_df, run_params["forecast_freq"])
        except Exception as e:
            raise Exception(f"Failed to set the data frequency to {forecast_freq}: {e}")
            
            
        try:
            # Generate date features
            optimal_df = generate_date_features(optimal_df, forecast_freq, country_name)
            forecast_df = generate_date_features(forecast_df, forecast_freq, country_name)
        except Exception as e:
            raise ValueError(f"Failed to generate features using 'ds': {e}")
        
        
        # Get the names of the exogenous variables from the train data
        run_params["exog_cols_all"]  = list((optimal_df.columns).difference(['y', 'ds']))
        
        
        try:
            initial_window_size, lag_window_range, rolling_window_range, test_size, test_steps = determine_params(forecast_freq)
            logger.info(f"Initial Window Size: {initial_window_size}, Lag Window Range: {lag_window_range}")
            logger.info(f"Test Size: {test_size}, Test Steps: {test_steps}")
        except Exception as e:
            raise Exception(e)
            
        run_params.update({
            "initial_window_size": initial_window_size,
            "lag_window_range": lag_window_range,
            "rolling_window_range": rolling_window_range,
            "test_size": test_size,
            "test_steps": test_steps
        })
        
        
        try:
            test_df = optimal_df[-test_size:].copy(deep=True)
            test_df = test_df.set_index('ds').resample(run_params["forecast_freq"]).sum()
            test_df = test_df.fillna(0)

            train_df = optimal_df[:-test_size].copy(deep=True)
            train_df = train_df.set_index('ds').resample(run_params["forecast_freq"]).sum()
            train_df = train_df.fillna(0)

            assert len(train_df) + len(test_df) == len(optimal_df)

            run_params["train_start_date"] = train_df.index.min()
            run_params["train_end_date"] = train_df.index.max()

            run_params["test_start_date"] = test_df.index.min()
            run_params["test_end_date"] = test_df.index.max()

            optimal_df = optimal_df.set_index('ds').resample(run_params["forecast_freq"]).sum()
            optimal_df = optimal_df.fillna(0)

            forecast_df = forecast_df.set_index('ds').resample(run_params["forecast_freq"]).sum()
            forecast_df = forecast_df.fillna(0)
        except Exception as e:
            raise ValueError(f"Failed to split into train and test: {e}")
            
            
        dates_df = pd.DataFrame({
            'Type': ['Uploaded Period', 'Optimal Period', 'Forecast Period'],
            'From': [run_params["historical_start_date"].strftime('%d-%m-%Y'), run_params["optimal_start_date"].strftime('%d-%m-%Y'), run_params["forecast_start_date"].strftime('%d-%m-%Y')],
            'To': [run_params["historical_end_date"].strftime('%d-%m-%Y'), run_params["optimal_end_date"].strftime('%d-%m-%Y'), run_params["forecast_end_date"].strftime('%d-%m-%Y')]
        })

        modelling_tab.dataframe(dates_df)
         
        #####################################################################################
            
        if st.session_state.run_search:
            
            current_dir = 'utils/modeling'

            model_types = {
                'prophet': 'sktime',
                'naive': 'sktime',
                'random_forest': 'skforecast',
                'xgboost': 'skforecast'
            }

            search_results = {}

            for model_type, package_type in model_types.items():
                # Load parameters for grid search
                model, param_grid = load_model_params_and_create_instance(model_type, current_dir)

                if package_type == 'sktime':
                    # Find best model
                    best_configuration, all_results, best_model = find_best_model_sktime(
                        train_df['y'], run_params, model, param_grid
                    )

                elif package_type == 'skforecast':
                    # Find best model
                    best_configuration, all_results, best_model = find_best_model_skforecast(
                        lag_window_range, model, train_df, param_grid, run_params
                    )

                else:
                    raise Exception('Unknown package type!')

                # Save best model and config
                search_results[model_type] = {
                    'best_model': best_model,
                    'best_configuration': best_configuration,
                    'all_results': all_results,
                    'package_type': package_type
                }
            
            test_eval = {}

            for model_type, model_results in search_results.items():
                if model_results['package_type'] == 'sktime':
                    best_model = search_results[model_type]['best_model']
                    best_model.fit(y=train_df['y'])
                    predictions_df = generate_forecast_sktime(best_model, len(test_df))
                elif model_results['package_type'] == 'skforecast':
                    best_model = search_results[model_type]['best_model']
                    best_model.fit(y=train_df['y'], exog=train_df[run_params["exog_cols_all"]])
                    predictions_df = generate_forecast_skforecast(best_model, run_params, train_df['y'],
                                                                  test_df.drop('y', axis=1),
                                                                  run_params["test_start_date"],
                                                                  len(test_df))
                else:
                    raise Exception('Unknown package type!')

                test_eval[model_type] = compute_metrics(predictions_df.merge(test_df.reset_index()), train_df["y"])

            # Convert the list of dictionaries into a DataFrame for easy manipulation
            metrics_df = pd.DataFrame(test_eval).T

            # Round off the values in the DataFrame to 3 decimal places for better readability
            metrics_df = metrics_df.round(3)

            # Sort the DataFrame based on the performance metrics in the order of preference
            # MASE > RMSSE > Coverage > MAPE > RMSPE
            metric_order = ['MASE', 'RMSSE', 'Coverage', 'MAPE', 'RMSPE']
            ascending_order = [True, True, False, True, True]

            metrics_df = metrics_df.sort_values(by=metric_order, ascending=ascending_order).reset_index()

            metrics_df.rename(columns={'index': 'Model'}, inplace=True)
            
            # Generate Forecasts
            forecasts_dict = {}

            for model_type, model_results in search_results.items():
                if model_results['package_type'] == 'sktime':
                    best_model = search_results[model_type]['best_model']
                    best_model.fit(y=optimal_df['y'])
                    predictions = generate_forecast_sktime(best_model, run_params['forecast_period'])
                elif model_results['package_type'] == 'skforecast':
                    best_model = search_results[model_type]['best_model']
                    best_model.fit(y=optimal_df['y'], exog=optimal_df[run_params["exog_cols_all"]])
                    predictions = generate_forecast_skforecast(best_model, run_params, optimal_df['y'],
                                                                             forecast_df.drop('y', axis=1),
                                                                             run_params["forecast_start_date"],
                                                                             run_params['forecast_period'])
                else:
                    raise Exception('Unknown package type!')

                forecasts_dict[model_type] = predictions.head(run_params['forecast_period'])
            
            st.session_state.metrics_df = metrics_df
            st.session_state.forecasts_dict = forecasts_dict
            st.session_state.run_search = False
            
            st.success(F"Grid Search completed in min(s)")
            
        
        model_types = {
                'Prophet': 'prophet',
                'Naive': 'naive',
                'Random Forest': 'random_forest',
                'XGBoost': 'xgboost'
        }
        
        # Show the best model
        st.success(f":first_place_medal: Best Model: {st.session_state.metrics_df.iloc[0]['Model']}")

        # Creating the custom color map
        cmap_custom = sns.light_palette("seagreen", as_cmap=True)

        # Creating a reversed custom color map for the 'Coverage' column
        cmap_custom_reversed = sns.light_palette("seagreen", as_cmap=True).reversed()
        
        # Load test ste evaluation metrics
        metrics_df = st.session_state.metrics_df

        # Apply the background gradient to the metrics columns except 'Coverage'
        metrics_df = metrics_df.style.background_gradient(
            cmap=cmap_custom_reversed, subset=['RMSSE', 'MASE', 'MAPE', 'RMSPE']
        )

        # Apply the reversed background gradient to the 'Coverage' column
        metrics_df = metrics_df.background_gradient(
            cmap=cmap_custom, subset=['Coverage']
        )
        
        # Show Test Set Metrics
        st.dataframe(metrics_df)
        
        # Choose Model Type
        selected_model = st.selectbox("Choose Model Type :brain:", model_types.keys(), key="model_dropdown")
        selected_model = model_types[selected_model]
        
        # Extract relevant data from the historical and forecast DataFrames
        historical_data = optimal_df.reset_index()[['ds', 'y']]
        forecast_data = st.session_state.forecasts_dict[selected_model][['ds', 'y_pred', 'min_pred', 'max_pred']]
        
        
        # Generate forecast plot
        fig = plot_forecasts(historical_data, forecast_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Combine all data
        historical_data['type'] = 0 # 1 for historical
        forecast_data['type'] = 1  # 1 for forecast
        # Prepare forecast data
        forecast_data['y'] = forecast_data['y_pred']
        forecast_data = forecast_data.drop(columns=['y_pred'])

        # Merge the two DataFrames
        combined_data = pd.concat([historical_data, forecast_data], sort=True)
        combined_data = combined_data.sort_values(by='ds')
        
        freq_types = {
            'Month': 'month',
            'Day of Week': 'day_of_week',
            'Week of Year': 'week_of_year'
        }
        
        agg_types = ['Mean', 'Sum', 'Min', 'Max']
        
        col_1, col_2, _ = st.columns([2, 2, 6])
        selected_freq = col_1.selectbox("Select Frequency Type", freq_types.keys(), key="plot_freq_dropdown")
        selected_freq = freq_types[selected_freq]
        selected_agg = col_2.selectbox("Choose Model Type", agg_types, key="table_freq_dropdown").lower()
        
        col_1, col_2 = st.columns([2, 2])
        
        pivot_table = create_pivot_table(combined_data, selected_freq, selected_agg)
        col_1.dataframe(pivot_table)
        
        fig = plot_time_series(pivot_table)
        col_2.plotly_chart(fig, use_container_width=True)