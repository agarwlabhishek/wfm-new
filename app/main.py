import os
import io
import sys
import json
import time
import random
import logging

import numpy as np
import pandas as pd

import plotly.io as pio
import streamlit as st
import seaborn as sns
import zipfile

import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

from utils.general import *
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


if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

if not st.session_state.logged_in:
    # Checking the login status
    try:
        login_page()
    except Exception as e:
        logger.error(f"Error in checking login status: {e}")
        raise e
    else:
        logging.info("Login successful!")


if 'input_submitted' not in st.session_state:
    st.session_state.input_submitted = False

if st.session_state.logged_in:
    # Retrieve input from the sidebar
    st.sidebar.title("Inputs :file_folder:")


    with st.sidebar.form('user_inputs'):

        # Add dropdown for Country
        country_name = st.selectbox("Holidays - Country Code :world_map:", [None, "BR", "CA", "CN", "DE", "ES", "FR", "IN", "IT", "UK", "US"])

        # Add dropdown for frequency
        forecast_freq = st.selectbox("Forecast Frequency :repeat:", ["B", "D", "W", "M"])

        # Add dropdown for data selection
        data_selection = st.radio("Automatic Data Selection :scissors:", [False, True])
        
        # Add file uploaders to the sidebar
        # uploaded_historical_file = st.file_uploader("Historical Data (.csv) :arrow_up: **(MANDATORY)**", type=["csv"])
        # uploaded_forecast_file = st.file_uploader("Forecast Data (.csv) :arrow_up: **(OPTIONAL)**", type=["csv"])
        
        uploaded_historical_file = "Agency Services.csv"
        uploaded_forecast_file = None

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
            
            # Dictionary to map forecast frequencies to periods
            forecast_periods = {
                "D": 92,  # Daily frequency
                "B": 66,  # Business days frequency
                "W": 26,  # Weekly frequency
                "M": 12   # Monthly frequency
            }
            
            # Set forecast period based on the frequency
            st.session_state.forecast_period = forecast_periods.get(forecast_freq)

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
        
        # Create a Pandas Excel writer for master result file
        result_buffer = io.BytesIO()
        writer = pd.ExcelWriter(result_buffer, engine="xlsxwriter")

        # Create a Zip File for storing html files
        zip_buffer = io.BytesIO()
        zipf = zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED)

        # Retrieve current timestamp
        dt_now = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        

        # Set up tabs for modelling, analysis, and download results
        modelling_tab, analysis_tab, download_tab = st.tabs(["Modelling :bar_chart:", 
                                                             "Analysis :microscope:", 
                                                             "Download Results :arrow_down:"])

        #####################################################################################

        try:
            # Validate the histoical data
            historical_df = validate_input_file(uploaded_historical_file, run_params["external_features"])
            logging.info(f"Historical Data Size: {historical_df.shape}")
            # Find the corresponding dates
            run_params = find_dates(historical_df, run_params)

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
                
                modelling_tab.success(f":white_check_mark: Automatic Data Selection: Historical data has been truncated to **{optimal_window_size}** most recent days!")
                
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
        
        # Populate forecast dataframe if needed
        if not run_params["external_features"]:
            date_range = pd.date_range(start=run_params['forecast_start_date'],
                                                       end=run_params['forecast_end_date'])
            forecast_df['ds'] = date_range
        
        # Validate dataframes
        validate_dataframes(optimal_df, forecast_df, run_params)
        
        modelling_tab.success(f":white_check_mark: Data Validation: Input data has been validated sucessfully!")
        
        #####################################################################################
        
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
            # Determine parameters based on forecast frequency
            initial_window_size, lag_window_range, rolling_window_range, test_size, test_steps = determine_params(forecast_freq)

            # Log all parameters at once
            logger.info(f"Params - Initial Window Size: {initial_window_size}, Lag Window Range: {lag_window_range}, "
                        f"Rolling Window Range: {rolling_window_range}, Test Size: {test_size}, Test Steps: {test_steps}")

            # Update run_params directly with the values
            run_params.update({
                "initial_window_size": initial_window_size,
                "lag_window_range": lag_window_range,
                "rolling_window_range": rolling_window_range,
                "test_size": test_size,
                "test_steps": test_steps
            })
        except Exception as e:
            # Provide a specific error message if there's an exception
            raise Exception(f"Error determining parameters: {e}")

        try:
            # Resample and clean the dataframes for test and train
            test_df = optimal_df[-run_params["test_size"]:].set_index('ds').resample(run_params["forecast_freq"]).sum().fillna(0)
            train_df = optimal_df[:-run_params["test_size"]].set_index('ds').resample(run_params["forecast_freq"]).sum().fillna(0)

            # Assert to ensure no data is lost during split
            assert len(train_df) + len(test_df) == len(optimal_df)

            # Update training and testing periods in run_params
            run_params.update({
                "train_start_date": train_df.index.min(),
                "train_end_date": train_df.index.max(),
                "test_start_date": test_df.index.min(),
                "test_end_date": test_df.index.max()
            })

            # Resample and clean the entire optimal_df and forecast_df
            optimal_df = optimal_df.set_index('ds').resample(run_params["forecast_freq"]).sum().fillna(0)
            forecast_df = forecast_df.set_index('ds').resample(run_params["forecast_freq"]).sum().fillna(0)

        except Exception as e:
            raise ValueError(f"Failed to split into train and test: {e}")
    
    
        modelling_tab.subheader(f"Data Overview :eyes:")
        
        dates_df = pd.DataFrame({
            'Type': ['Uploaded Period', 'Optimal Period', 'Forecast Period'],
            'From': [run_params["historical_start_date"].strftime('%d-%m-%Y'), run_params["optimal_start_date"].strftime('%d-%m-%Y'), run_params["forecast_start_date"].strftime('%d-%m-%Y')],
            'To': [run_params["historical_end_date"].strftime('%d-%m-%Y'), run_params["optimal_end_date"].strftime('%d-%m-%Y'), run_params["forecast_end_date"].strftime('%d-%m-%Y')]
        })

        modelling_tab.dataframe(dates_df)

        modelling_tab.markdown(f'**Exogenous Features**: :blue[{", ".join(run_params["exog_cols_all"])}]')
        
        col_1, _ = modelling_tab.columns([2, 10])
        
        # Add download buttons for plot and data
        csv_download_button(col_1, dates_df, "Data Overview")
        
        # Add to Excel File
        dates_df.to_excel(writer, sheet_name="Data Overview", index=False)
        
        modelling_tab.divider()
         
        #####################################################################################
        
        # Display the parameters for the best model
        modelling_tab.subheader(f"Model Selection :mag_right:")

        current_dir = 'utils/modeling'
        
        model_types = {
                'Prophet': {
                    'fname': 'prophet',
                    'package_type': 'sktime'
                },
                'Naive': {
                    'fname': 'naive',
                    'package_type': 'sktime'
                },
                'Random Forest': {
                    'fname': 'random_forest',
                    'package_type': 'skforecast'
                },
                'XGBoost': {
                    'fname': 'xgboost',
                    'package_type': 'skforecast'
                }
        }

        if st.session_state.run_search:
            waiting_messages = [
                    "Hang tight! Forecasting in progress! 😊",
                    "Patience is key! Training in progress! 💪",
                    "Almost there!  Smarter by the second! 🚀",
                    "Stay tuned! Fine-tuning skills! 📡 ",
                    "Magic in progress! Brewing predictions! 🐣",
                    "Behind the scenes magic! Appreciate your patience! 🙏",
                    "Remember, coffee breaks are essential for survival ☕️ (and sanity).",
                ]
            
            random_idx = random.randint(0, len(waiting_messages)-1)
            
            with modelling_tab:
                with st.spinner(waiting_messages[random_idx]):
            
                    # Create a progress bar for the model search
                    model_search_bar = modelling_tab.progress(0)
                    model_search_bar.progress(0/len(model_types), text=f"⌛ Estimated Time Remaining: Calculating")

                    # Record the start time of the model search
                    init_time = time.time()
                    start_time = time.time()

                    search_results = {}

                    for idx, (model_name, model_values) in enumerate(model_types.items()):
                        model_fname = model_values['fname']
                        package_type = model_values['package_type']
                        
                        # Load parameters for grid search
                        model, param_grid = load_model_params_and_create_instance(model_fname, current_dir)

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
                        search_results[model_name] = {
                            'best_model': best_model,
                            'best_configuration': best_configuration,
                            'all_results': all_results,
                            'package_type': package_type
                        }

                        # Update the progress bar with the time taken
                        end_time = time.time()
                        remaining_time = round((len(model_types) - idx+1)*(end_time - start_time)/60, 2)
                        start_time = time.time()

                        model_search_bar.progress((idx+1)/len(model_types), text=f"⌛ {idx+1}/{len(model_types)} Completed! - Estimated Time Remaining: {remaining_time} minutes")

                    # Test Set Evaluation
                    test_eval = {}

                    for model_name, model_results in search_results.items():
                        if model_results['package_type'] == 'sktime':
                            best_model = search_results[model_name]['best_model']
                            best_model.fit(y=train_df['y'])
                            predictions_df = generate_forecast_sktime(best_model, len(test_df))
                        elif model_results['package_type'] == 'skforecast':
                            best_model = search_results[model_name]['best_model']
                            best_model.fit(y=train_df['y'], exog=train_df[run_params["exog_cols_all"]])
                            predictions_df = generate_forecast_skforecast(best_model, run_params, train_df['y'],
                                                                          test_df.drop('y', axis=1),
                                                                          run_params["test_start_date"],
                                                                          len(test_df))
                        else:
                            raise Exception('Unknown package type!')

                        predictions_df = predictions_df.merge(test_df.reset_index())
                        predictions_df['y'] = predictions_df['y'].apply(lambda x: 1 if x == 0 else x)
                        predictions_df['y_pred'] = predictions_df['y_pred'].apply(lambda x: 1 if x == 0 else x)

                        test_eval[model_name] = compute_metrics(predictions_df, train_df["y"])
                     
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
                    st.session_state.metrics_df = metrics_df

                    #####################################################################################

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

                    
                    st.session_state.forecasts_dict = forecasts_dict
                    st.session_state.run_search = False

                    end_time = time.time()
                    modelling_tab.success(f":stopwatch: Search Completed in {round((end_time - init_time)/60, 2)} minutes") 


        #####################################################################################
        
        # Show the best model
        modelling_tab.success(f":first_place_medal: Best Model: {st.session_state.metrics_df.iloc[0]['Model']}")

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
        modelling_tab.dataframe(metrics_df)
        
        col_1, _ = modelling_tab.columns([2, 10])

        # Add download buttons for plot and data
        csv_download_button(col_1, st.session_state.metrics_df, "Model Selection")
        
        # Add to excel file
        metrics_df.to_excel(writer, sheet_name="Model Selection", index=False)
        
        modelling_tab.divider()
        
        #####################################################################################
        
        modelling_tab.subheader("Forecasts Generated :chart_with_upwards_trend:")
        
        col_1, _ = modelling_tab.columns([2, 10])
        
        # Choose Model Type
        selected_model = col_1.selectbox("Choose Model Type :brain:", st.session_state.metrics_df['Model'].tolist(), key="model_dropdown")
        
        # Extract relevant data from the historical and forecast DataFrames
        historical_data = optimal_df.reset_index()[['ds', 'y']]
        forecast_data = st.session_state.forecasts_dict[selected_model][['ds', 'y_pred', 'min_pred', 'max_pred']]
        
        # Generate forecast plot
        fig = plot_forecasts(historical_data, forecast_data)
        modelling_tab.plotly_chart(fig, use_container_width=True)
        
        img_fname = "Historical_Forecast_Plot"
        
        # Prepare an HTML buffer to store the exported plot
        html_buffer = io.StringIO()
        fig.write_html(html_buffer, include_plotlyjs='cdn')
        
        col_1, col_2, col_3, _ = modelling_tab.columns([2, 2, 2, 6])

        # Add download buttons for plot and data
        csv_download_button(col_1, historical_data, "Historical Data")
        csv_download_button(col_2, forecast_data, "Forecast Data")
        html_download_button(col_3, html_buffer, img_fname)
        
        # Add to excel
        historical_data.to_excel(writer, sheet_name="Historical Data", index=False)
        forecast_data.to_excel(writer, sheet_name="Forecast Data", index=False)
        writer.book.add_worksheet(img_fname).insert_image(f'A1', f'{img_fname}', {'image_data': io.BytesIO(pio.to_image(fig, format='png'))})
        
        modelling_tab.divider()
        
        #####################################################################################
        
        analysis_tab.subheader("Year-Over-Year Comparsions :scales:")
        
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
        
        col_1, col_2, col_3, _ = analysis_tab.columns([2, 2, 2, 8])
        selected_freq = col_1.selectbox("Frequency Type", freq_types.keys(), key="plot_freq_dropdown")
        selected_freq = freq_types[selected_freq]
        selected_agg = col_2.selectbox("Aggregation Function", agg_types, key="table_freq_dropdown").lower()
        
        min_value=run_params["historical_start_date"]
        max_value=run_params["forecast_end_date"]
        date_range = col_3.date_input(
                        "Date Range",
                        (min_value, max_value),
                        min_value=min_value,
                        max_value=max_value,
                        format="MM/DD/YYYY",
                    )
        
        col_1, col_2 = analysis_tab.columns([4, 6])
        
        combined_data = combined_data[(combined_data['ds'] >= pd.to_datetime(date_range[0])) &
                                     (combined_data['ds'] <= pd.to_datetime(date_range[1]))]
        
        pivot_table = create_pivot_table(combined_data, selected_freq, selected_agg)
        col_1.dataframe(pivot_table, use_container_width=True)
        
        fig = plot_time_series(pivot_table)
        col_2.plotly_chart(fig, use_container_width=True)

        # Define file names for data and plot
        data_fname = f'{selected_freq}_{selected_agg}_YoY_Data'
        img_fname = f'{selected_freq}_{selected_agg}_YoY_Plot'
        
        # Prepare an HTML buffer to store the exported plot
        html_buffer = io.StringIO()
        fig.write_html(html_buffer, include_plotlyjs='cdn')
        
        col_1, col_2 = analysis_tab.columns([4, 6])

        # Add download buttons for plot and data
        csv_download_button(col_1, pivot_table, data_fname)
        html_download_button(col_2, html_buffer, img_fname)
        
        # Add to excel sheet
        pivot_table.to_excel(writer, sheet_name=data_fname)
        writer.sheets[data_fname].insert_image(f'A{len(pivot_table)+2}', f'{img_fname}', {'image_data': io.BytesIO(pio.to_image(fig, format='png'))})
        
        analysis_tab.divider()
        
        #####################################################################################
        
        # Close the Pandas Excel writer and output the Excel file to the BytesIO object
        writer.close()
        zipf.close()

        # Seek to the beginning of the stream
        result_buffer.seek(0)
        
        # Notify user about the readiness of the result
        download_tab.success('Your results are ready for download!', icon="✅")

        # Define the layout for displaying the results
        col_1, col_2, col_3 = download_tab.columns([2, 2, 8])

        # Add a button for downloading the results
        col_1.download_button(
            label="Forecasts and Analysis (.xlsx)",
            data=result_buffer,
            file_name=f"{dt_now}_Forecast_Analysis.xlsx",
            mime="application/vnd.ms-excel",
            type="primary"
        )

        # Add a button for downloading the results
        col_2.download_button(
            label="Interactive Plots (.zip)",
            data=zip_buffer.getvalue(),
            file_name=f"{dt_now}_Interactive_Plots.zip",
            mime="application/zip",
            type="primary"
        )

        download_tab.error("🚀 **Please help us save resources and costs: kindly close the tool once you're done.** 🙏✨")