import logging
import io
import json
import random
import time
import sys

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
from utils.inputs.preprocess import *
from utils.modeling.search import *
from utils.modeling.general import *
from utils.modeling.plot import *
from utils.analysis.tables import *
from utils.analysis.plot import *

# Set up the logging configuration for cmdstanpy
logger = logging.getLogger()

# Add NullHandler with CRITICAL log level
null_handler = logging.NullHandler()
null_handler.setLevel(logging.CRITICAL)
logger.addHandler(null_handler)

# Add StreamHandler with INFO log level
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

logger.propagate = False

        
def main():
    
    st.set_page_config(
        page_title="WFM Forecasting - TEXAS",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "Developed by TeX team at Allianz Partners. Please contact abhishek.agarwal@allianz.com for any queries/ complaints."
        } 
    )
    
    title_col_1, title_col_2 = st.columns([0.9, 0.2])
    
    title_col_1.title("Workforce Management Forecasting :chart_with_upwards_trend:")
    title_col_2.subheader("v1.6.5-beta")
    
    # Set logged in flag as False
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'rerun_search' not in st.session_state:    
        st.session_state.rerun_search = True
    if 'evaluate_train' not in st.session_state:
        st.session_state.evaluate_train = True
    if 'evaluate_early_warning' not in st.session_state:
        st.session_state.evaluate_early_warning = False
    if 'generate_forecast' not in st.session_state:
        st.session_state.generate_forecast = True
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
        
        # Add dropdown for Objective
        objective_type = st.sidebar.selectbox("Objective :dart:", ["Forecasting", "Early Warning"])
                
        with st.sidebar.form('user_inputs'):
            
            # Add dropdown for Country
            country_name = st.selectbox("Holidays - Country Code :world_map:", ["None", "BR", "CA", "CN", "DE", "ES", "FR", "IN", "IT", "UK", "US"])
            
            # Add radio for weekend
            weekend_drop = st.selectbox("Exclude Weekends :sunglasses:", ("No", "Yes"))

            # Add dropdown for frequency
            forecast_freq = st.selectbox("Forecast Frequency :repeat:", ["Daily", "Weekly", "Monthly"])

            if forecast_freq.lower() == "daily":
                forecast_period = 92
            elif forecast_freq.lower() == "weekly":
                forecast_period = 26
            elif forecast_freq.lower() == "monthly":
                forecast_period = 12
            
            # Add dropdown for data selection
            data_selection = st.selectbox("Automatic Data Selection :scissors:", ["No", "Yes"])
            
            # Add dropdown for data selection
            external_features = st.selectbox("Additional Features Available :1234:", ["No", "Yes"])
            
            
            
            # Add a file uploader for Planning
            if 'early warning' in objective_type.lower():
                # Add file uploader to the sidebar
                uploaded_file = st.file_uploader("Upload your Actuals (.csv) :arrow_up:", type=["csv"])
                uploaded_file_planning = st.file_uploader("Upload your Planning (.csv) :arrow_up:", type=["csv"])
            else:
                # Add file uploader to the sidebar
                uploaded_file = st.file_uploader("Upload your file (.csv) :arrow_up:", type=["csv"])
            
            show_analysis_tab = st.toggle('Show Analysis', False)
            
            submitted = st.form_submit_button("Submit")
    
            if submitted:
                st.session_state.input_submitted = True
                
    if st.session_state.input_submitted:
        if uploaded_file is None:
            st.error('Please upload the input CSV file to use the tool!')
            st.stop()
        else:
                        
            if 'objective_type' not in st.session_state or objective_type != st.session_state.objective_type:
                st.session_state.objective_type = objective_type
                st.session_state.rerun_search = True
                st.session_state.evaluate_train = True
                st.session_state.generate_forecast = True
                
            if 'country_name' not in st.session_state or country_name != st.session_state.country_name:
                st.session_state.country_name = country_name
                st.session_state.rerun_search = True
                st.session_state.evaluate_train = True
                st.session_state.generate_forecast = True
            
            if 'weekend_drop' not in st.session_state or weekend_drop != st.session_state.weekend_drop:
                st.session_state.weekend_drop = weekend_drop
                st.session_state.rerun_search = True
                st.session_state.evaluate_train = True
                st.session_state.generate_forecast = True
                
            if 'forecast_freq' not in st.session_state or forecast_freq != st.session_state.forecast_freq:
                st.session_state.forecast_freq = forecast_freq
                st.session_state.rerun_search = True
                st.session_state.evaluate_train = True
                st.session_state.generate_forecast = True
                
            if 'forecast_period' not in st.session_state or forecast_period != st.session_state.forecast_period:
                st.session_state.forecast_period = forecast_period
                st.session_state.rerun_search = True
                st.session_state.evaluate_train = True
                st.session_state.generate_forecast = True
                
            if 'data_selection' not in st.session_state or data_selection != st.session_state.data_selection:
                st.session_state.data_selection = data_selection
                st.session_state.rerun_search = True
                st.session_state.evaluate_train = True
                st.session_state.generate_forecast = True
                
            if 'external_features' not in st.session_state or external_features != st.session_state.external_features:
                st.session_state.external_features = external_features
                st.session_state.rerun_search = True
                st.session_state.evaluate_train = True
                st.session_state.generate_forecast = True

            # Set up tabs for modelling, analysis, and download results
            modelling_tab, analysis_tab, download_tab = st.tabs(["Modelling :bar_chart:", 
                                                                 "Analysis :microscope:", 
                                                                 "Download Results :arrow_down:"])

            #####################################################################################

            # Preprocess and validate input file
            train_full, forecast_full = process_input_file(uploaded_file)
            st.session_state.train_init_size = len(train_full)
            
            if 'early warning' in st.session_state.objective_type.lower():
                if uploaded_file_planning is None:
                    st.error('Please upload the Planning CSV file to use the tool with Early Warning as the objective!')
                    st.stop()
                else:
                    # Validate input file for Planning
                    planning_full = validate_input_file(uploaded_file_planning).set_index('ds')

            if st.session_state.forecast_freq.lower() == "daily" and len(train_full) < 300:     
                st.error('The input file does not have enough observations for training a model!')
                st.stop()
            elif st.session_state.forecast_freq.lower() == "weekly" and len(train_full) < 52:
                st.error('The input file does not have enough observations for training a model!')
                st.stop()
            elif st.session_state.forecast_freq.lower() == "monthly" and len(train_full) < 24:
                st.error('The input file does not have enough observations for training a model!')
                st.stop()

            if data_selection.lower() == "yes":
                # Split at optimal window
                train = train_full[train_full.index > st.session_state.optimal_window_date].copy(deep=True)
            else:
                train = train_full.copy(deep=True)
            
            # Compute the default value for holidays
            if st.session_state.country_name != 'None':
                ff_freq_temp = st.session_state.forecast_freq
                st.session_state.forecast_freq = "daily"
                train_data_features = generate_date_features(train.copy(deep=True).asfreq('D').bfill()).reset_index()
                st.session_state.forecast_freq = ff_freq_temp
                holiday_default_value = train_data_features[train_data_features['is_holiday'] > 0].sort_values('ds', ascending=False)['y'].head(min(len(train_data_features[train_data_features['is_holiday'] > 0]), 3)).mean() if not train_data_features[train_data_features['is_holiday'] > 0].empty else 0
                holiday_dates = train_data_features[train_data_features['is_holiday'] > 0]['ds'].tolist()

            st.session_state.train_optimal_size = len(train)

            # Create a Pandas Excel writer for master result file
            result_buffer = io.BytesIO()
            writer = pd.ExcelWriter(result_buffer, engine="xlsxwriter")

            # Create a Zip File for storing html files
            zip_buffer = io.BytesIO()
            zipf = zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED)

            dt_now = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')

            # Define the forecast period
            date_range = pd.date_range(start=train.index[-1] + pd.Timedelta(days=1), 
                                       periods=st.session_state.forecast_period, 
                                       freq="D")
            fc_period = f"{min(date_range).strftime('%d-%m-%Y')}_{max(date_range).strftime('%d-%m-%Y')}"

            # Check if external variables are avilable for entire forecast period
            if st.session_state.external_features.lower() == "yes" and len(date_range) > len(forecast_full):
                error_message = f"External variables not available for entire forecast period. Please select a shorter forecast period."
                st.error(error_message, icon="🚨")
                logger.error(error_message)
                st.stop()

            #####################################################################################

            # Display the parameters for the best model
            modelling_tab.subheader(f"Data Overview :eyes:")

            dates_df = pd.DataFrame({
                'Type': ['Uploaded Period', 'Forecast Period'],
                'From': [train_full.index.min().strftime('%d-%m-%Y'), min(date_range).strftime('%d-%m-%Y')],
                'To': [train_full.index.max().strftime('%d-%m-%Y'), max(date_range).strftime('%d-%m-%Y')]
            })

            modelling_tab.dataframe(dates_df)

            # Get the names of the exogenous variables from the train data
            exog_cols = list((train_full.columns).difference(['y']))
            if len(exog_cols) > 0:
                modelling_tab.markdown(f'**Exogenous Features**: :blue[{", ".join(exog_cols)}]')

            # Display the parameters for the best model
            modelling_tab.subheader(f"Model Selection Based on Test Set :mag_right:")

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

            # Find the best model based on the training data and inputs
            if st.session_state.rerun_search:
                with modelling_tab:
                    with st.spinner(waiting_messages[random_idx]):
                        find_best_model(train, modelling_tab)
                st.session_state.rerun_search = False
                st.session_state.evaluate_train = True
                st.session_state.generate_forecast = True
                if 'early warning' in st.session_state.objective_type.lower():
                    st.session_state.evaluate_early_warning = True
                st.session_state.model_type = st.session_state.model_comparisons['Model'].tolist()[0]
            else:
                # Show the time
                modelling_tab.success(f":stopwatch: Search Completed in {st.session_state.search_time} minutes") 

            # Show the best model
            modelling_tab.success(f":first_place_medal: Best Model: {st.session_state.best_model_type}")

            # Creating the custom color map
            cmap_custom = sns.light_palette("seagreen", as_cmap=True)

            # Creating a reversed custom color map for the 'Coverage' column
            cmap_custom_reversed = sns.light_palette("seagreen", as_cmap=True).reversed()

            # Apply the background gradient to the metrics columns except 'Coverage'
            styled_model_comparisons = st.session_state.model_comparisons.style.background_gradient(
                cmap=cmap_custom_reversed, subset=['RMSSE', 'MASE', 'MAPE', 'RMSPE']
            )

            # Apply the reversed background gradient to the 'Coverage' column
            styled_model_comparisons = styled_model_comparisons.background_gradient(
                cmap=cmap_custom, subset=['Coverage']
            )
            modelling_tab.dataframe(styled_model_comparisons)

            # Save the model parameters to results excel sheet
            data_fname = f"Model_Comparisons_Test_Set"

            st.session_state.model_comparisons.to_excel(writer, sheet_name=data_fname, index=False)

            modelling_tab.download_button(
                label="Export Model Comparsions (.csv)",
                data=st.session_state.model_comparisons.to_csv(index=False),
                file_name=f"{dt_now}_{data_fname}_{fc_period}.csv",
                mime="text/csv"
            )


            with st.sidebar.form('user_model_selection'):
                model_list = st.session_state.model_comparisons['Model'].tolist()
                model_type = st.selectbox("Choose Model Type :brain:", model_list, key="model_dropdown")
                model_submitted =  st.form_submit_button("Update")

                if model_submitted and model_type != st.session_state.model_type:
                    st.session_state.model_type = model_type
                    st.session_state.rerun_search = False
                    st.session_state.evaluate_train = True
                    st.session_state.generate_forecast = True
                    if 'early warning' in st.session_state.objective_type.lower():
                        st.session_state.evaluate_early_warning = True

                best_params = None
                if st.session_state.model_type == "Prophet":
                    best_params = st.session_state.prophet_best_model_params['best_params']
                elif st.session_state.model_type in ["Naive - Mean", "Naive - Drift"]:
                    best_params = st.session_state.naive_best_model_params['best_params']
                elif st.session_state.model_type == "Random Forest":
                    best_params = st.session_state.rf_best_model_params['best_params']
                    best_params['lag_window'] = st.session_state.rf_best_model_params['lag_window']
                elif st.session_state.model_type == "XGBoost":
                    best_params = st.session_state.xgb_best_model_params['best_params']
                    best_params['lag_window'] = st.session_state.xgb_best_model_params['lag_window']
                elif st.session_state.model_type == "Ensemble Tree":
                    best_params_xgb = st.session_state.xgb_best_model_params['best_params']
                    lag_window_xgb = st.session_state.xgb_best_model_params['lag_window']
                    # Add the 'xgb' prefix to the keys in best_params_xgb dictionary
                    best_params = {f'xgb_{key}': value for key, value in best_params_xgb.items()}
                    # Add the lag_window key for xgb
                    best_params['xgb_lag_window'] = lag_window_xgb
                    best_params_rf = st.session_state.rf_best_model_params['best_params']
                    lag_window_rf = st.session_state.rf_best_model_params['lag_window']
                    # Add the 'rf' prefix to the keys in best_params_rf dictionary and merge with the existing best_params
                    best_params.update({f'rf_{key}': value for key, value in best_params_rf.items()})
                    # Add the lag_window key for rf
                    best_params['rf_lag_window'] = lag_window_rf

                # Create a DataFrame with the dictionary values and set 'index' as the column name
                params_df = pd.DataFrame(list(best_params.values()), index=list(best_params.keys()), columns=['Value']).reset_index()

                # Rename the columns if needed
                params_df = params_df.rename(columns={'index': 'Parameter'})

                # Save the model parameters to results excel sheet
                data_fname = f"{st.session_state.model_type}_Parameters"

                params_df.to_excel(writer, sheet_name=data_fname, index=False)

                # Save the model feature importances to results excel sheet
                data_fname = f"{st.session_state.model_type}_Feature_Import."

                if st.session_state.model_type == "Random Forest":
                    st.session_state.rf_best_model_fi.to_excel(writer, sheet_name=data_fname, index=False)
                elif st.session_state.model_type == "XGBoost":
                    st.session_state.xgb_best_model_fi.to_excel(writer, sheet_name=data_fname, index=False)
                elif st.session_state.model_type == "Ensemble Tree":
                    st.session_state.rf_best_model_fi.to_excel(writer, sheet_name=data_fname, index=False)
                    st.session_state.xgb_best_model_fi.to_excel(writer, sheet_name=data_fname, index=False)
                            
                #####################################################################################
            
                # Start of model evaluation
                modelling_tab.subheader(f"Evaluation for {st.session_state.model_type} on Train Set:dart:")

                if st.session_state.evaluate_train:
                    with modelling_tab:
                        # Evaluate the model and make predictions
                        with st.spinner(':hourglass: Evaluating the performance...'):
                            start_time = time.time()  
                            df_predictions = evaluate_train_data(train.copy(deep=True))
                            train_metrics = compute_metrics(df_predictions)
                            st.session_state.evaluation_time = round((time.time() - start_time)/60, 2)
                    st.session_state.df_predictions = df_predictions
                    st.session_state.train_metrics = train_metrics
                    st.session_state.evaluate_train = False
                else:
                    df_predictions = st.session_state.df_predictions
                    train_metrics = st.session_state.train_metrics

                # Show the time
                modelling_tab.success(f":stopwatch: Evaluation Completed in {st.session_state.evaluation_time} minutes for {len(train)} {forecast_freq}") 

                # Display model metrics
                metric_col_1, metric_col_2, metric_col_3, metric_col_4, metric_col_5 = modelling_tab.columns(5)

                # Display metrics
                metric_col_1.metric("Root Mean Squared Scaled Error (RMSSE)", f"{train_metrics['RMSSE']:.2f}")
                metric_col_2.metric("Mean Absolute Scaled Error (MASE)", f"{train_metrics['MASE']:.2f}")
                metric_col_3.metric("Coverage (with Prediction Intervals)", f"{train_metrics['Coverage']*100:.2f}")
                metric_col_4.metric("Mean Absolute Percentage Error (MAPE)", f"{train_metrics['MAPE']:.2f}")
                metric_col_5.metric("Root Mean Squared Percentage Error (RMSPE)", f"{train_metrics['RMSPE']:.2f}")

                if st.session_state.model_type == "Naive":
                    modelling_tab.warning(':warning: The performance metrics above might not be accurate due to limitations of the Naive method!')

                logger.info("Metrics displayed")

                # Generate forecasts based on inputs and the best model
                if st.session_state.generate_forecast:
                    with modelling_tab:
                        with st.spinner(':crystal_ball: Generating forecasts...'):
                            start_time = time.time()
                            df_forecast = generate_forecast(st.session_state.forecast_period, st.session_state.weekend_drop,
                                                            train, forecast_full)
                            st.session_state.forecast_time = round((time.time() - start_time)/60, 2)
                    st.session_state.df_forecast = df_forecast
                    st.session_state.generate_forecast = False
                else:
                    df_forecast = st.session_state.df_forecast

                # Show the time
                modelling_tab.success(f":stopwatch: Forecast Generated in {st.session_state.forecast_time} minutes for {st.session_state.forecast_period} {forecast_freq}")
                
                # Change the frequency of the data based on the weekend drop session state
                try:
                    if st.session_state.forecast_freq.lower() == "daily" and st.session_state.weekend_drop.lower() == "yes":
                        forecast_freq = "B"
                    elif st.session_state.forecast_freq.lower() == "daily" and st.session_state.weekend_drop.lower() == "no":
                        forecast_freq = "D"
                    elif st.session_state.forecast_freq.lower() == "weekly":
                        forecast_freq = "W"
                    elif st.session_state.forecast_freq.lower() == "monthly":
                        forecast_freq = "M"
                    else:
                        raise ValueError("Unknown Frequency!")
                except Exception as e:
                    st.error(e, icon="🚨")
                    logger.error(e)
                
                if st.session_state.country_name != 'None':
                    #### ADJUST HOLIDAY VALUES POSTPROCESSING
                    test_data_features = df_forecast.copy(deep=True).set_index('ds').asfreq('D').bfill()
                    ff_freq_temp = st.session_state.forecast_freq
                    st.session_state.forecast_freq = "daily"
                    test_data_features = generate_date_features(test_data_features).reset_index()
                    st.session_state.forecast_freq = ff_freq_temp
                    df_forecast = df_forecast.copy(deep=True).set_index('ds').asfreq('D').bfill().reset_index()

                    # Step 1: Find all 'ds' in test_data_features where is_holiday > 0
                    holiday_dates.extend(test_data_features[test_data_features['is_holiday'] > 0]['ds'].tolist())

                    # Step 2 & 3: Find corresponding 'ds' in df_forecast and update 'y_true'
                    # Also, calculate the scaling ratio
                    scaling_ratio = holiday_default_value / df_forecast.loc[df_forecast['ds'].isin(holiday_dates), 'y_pred']
                    df_forecast.loc[df_forecast['ds'].isin(holiday_dates), 'y_pred'] = holiday_default_value

                    # Step 4: Apply the scaling ratio to 'min_pred' and 'max_pred'
                    df_forecast.loc[df_forecast['ds'].isin(holiday_dates), 'min_pred'] *= scaling_ratio
                    df_forecast.loc[df_forecast['ds'].isin(holiday_dates), 'max_pred'] *= scaling_ratio

                    # Reset the forecast frequency
                    test_data_features = test_data_features.set_index('ds').resample(forecast_freq).mean()
                    df_forecast = df_forecast.set_index('ds').resample(forecast_freq).mean().reset_index()
                    
                    st.session_state.df_forecast_updated = df_forecast
                else:
                    st.session_state.df_forecast_updated = st.session_state.df_forecast

                # Plot actual and forecast data
                fig = plot_forecast(df_predictions, df_forecast)
                modelling_tab.plotly_chart(fig, use_container_width=True)

                # Save the plot to results excel sheet
                data_fname = 'Training_Data'

                df_predictions = round(df_predictions)
                df_predictions.to_excel(writer, sheet_name=data_fname, index=False)

                # Creating columns for export buttons
                export_row1_col1, export_row1_col2, export_row1_col3, export_row1_col4, export_row1_col5, _ = modelling_tab.columns(6)

                # Button to download data as CSV
                export_row1_col1.download_button(
                    label="Export Train Data (.csv)",
                    data=df_predictions.set_index("ds").to_csv(),
                    file_name=f"{dt_now}_{data_fname}_{fc_period}.csv",
                    mime="text/csv"
                )


                # Define file names for forecast data and plot'
                data_fname = 'Forecast_Data'

                # Round forecast data
                df_forecast = round(df_forecast)

                # Save forecast data to the Excel file
                df_forecast.to_excel(writer, sheet_name=data_fname, index=False)         

                # Button to download forecast data as CSV
                export_row1_col2.download_button(
                    label="Export Forecast Data (.csv)",
                    data=df_forecast.set_index("ds").to_csv(),
                    file_name=f"{dt_now}_{data_fname}_{fc_period}.csv",
                    mime="text/csv"
                )
                
                train_data_features = generate_date_features(train.copy(deep=True).asfreq('D').resample(forecast_freq).bfill())
                
                # Define file names for train features data
                data_fname = 'Train_Features'
                
                # Save data to the Excel file
                train_data_features.to_excel(writer, sheet_name=data_fname, index=False)         

                # Button to download data as CSV
                export_row1_col5.download_button(
                    label="Export Train Features (.csv)",
                    data=train_data_features.to_csv(),
                    file_name=f"{dt_now}_{data_fname}_{fc_period}.csv",
                    mime="text/csv"
                )
                #####################################################################################
                

                df_predictions.set_index('ds', inplace=True)
                df_predictions = df_predictions.resample('D').asfreq().bfill().reset_index()

                df_forecast.set_index('ds', inplace=True)
                df_forecast = df_forecast.resample('D').asfreq().bfill().reset_index()

                df_train = train_full.reset_index()[['ds', 'y']]
                df_train.columns = ['ds', 'y_true']
                df_train = df_train.merge(df_predictions, on=['ds', 'y_true'], how='left')

                # Concatenate predictions and forecasts
                df_combined = pd.concat([df_predictions, df_forecast.fillna(0)], sort=True)
                df_combined = df_combined.set_index('ds').resample('D').asfreq().bfill().reset_index()

                # Rename columns
                columns_mapping = {
                    "ds": "Date",
                    "y_true": "Actual",
                    "y_pred": "Forecast",
                    "min_pred": "Lower Forecast",
                    "max_pred": "Upper Forecast"
                }
                df_combined = df_combined.rename(columns=columns_mapping)

                # Extract date features
                df_combined = extract_date_features(df_combined)

                # Calculate Absolute Percentage Error (APE)
                df_combined["Absolute Percentage Error"] = np.round(
                    np.abs((df_combined["Actual"] - df_combined["Forecast"]) / df_combined["Actual"]) * 100, 3
                )

                # Define order of columns
                columns_order = ["Date", "Year", "Quarter", "Month", "Weekday Number", "Day", "Month Week",
                                 "Actual", "Forecast", "Absolute Percentage Error", "Lower Forecast", "Upper Forecast"]

                logger.info("Predictions and forecasts concatenated")

                # Return DataFrame with ordered columns
                df_combined = df_combined[columns_order].round(1)

                # Define file names for combined data and plot
                img_fname = 'Actual_Forecast_Plot'
                data_fname = 'Actual_Forecast_Data'

                # Save combined data to the Excel file
                df_combined.to_excel(writer, sheet_name=data_fname, index=False)
                writer.book.add_worksheet(img_fname).insert_image(f'A1', f'{img_fname}', {'image_data': io.BytesIO(pio.to_image(fig, format='png'))})

                # Button to download combined data as CSV
                export_row1_col3.download_button(
                    label="Export Combined Data (.csv)",
                    data=df_combined.set_index("Date").to_csv(),
                    file_name=f"{dt_now}_{data_fname}_{fc_period}.csv",
                    mime="text/csv"
                )

                # Export plot to HTML
                html_buffer = io.StringIO()
                fig.write_html(html_buffer, include_plotlyjs='cdn')

                # Button to download combined plot as HTML
                export_row1_col4.download_button(
                    label='Export Interactive Plot (.html)',
                    data=html_buffer.getvalue().encode(),
                    file_name=f"{dt_now}_{img_fname}_{fc_period}.html",
                    mime='text/html'
                )

                # Add HTML file to ZIP
                zipf.writestr(f"{dt_now}_{img_fname}_{fc_period}.html", html_buffer.getvalue().encode())
                
                
                #####################################################################################
                
                if 'early warning' in st.session_state.objective_type.lower():

                    modelling_tab.subheader(f'Early Warning Metrics :traffic_light:')
                    
                    
                    
                    # Set prediction interval parameters
                    pi_lower = 5
                    pi_upper = 95
                    pi_n_boots = 50

                    # Change the frequency of the data based on the weekend drop session state
                    try:
                        if st.session_state.forecast_freq.lower() == "daily" and st.session_state.weekend_drop.lower() == "yes":
                            forecast_freq = "B"
                            st.session_state.test_steps_list = [7, 15, 30, 45, 60]
                        elif st.session_state.forecast_freq.lower() == "daily" and st.session_state.weekend_drop.lower() == "no":
                            forecast_freq = "D"
                            st.session_state.test_steps_list  = [7, 15, 30, 45, 60]
                        elif st.session_state.forecast_freq.lower() == "weekly":
                            forecast_freq = "W"
                            st.session_state.test_steps_list  = [4, 8, 13, 16, 20, 26]
                        elif st.session_state.forecast_freq.lower() == "monthly":
                            forecast_freq = "M"
                            st.session_state.test_steps_list  = [3, 6, 9, 12]
                        else:
                            raise ValueError("Unknown Frequency!")
                    except Exception as e:
                        model_search_bar.error(e, icon="🚨")
                        logger.error(e)
                        
                    modelling_tab.markdown(f'<strong>Backward Looking:</strong> The model is trained on the entire historical dataset, excluding the last {forecast_freq} observations, which form the test set.', unsafe_allow_html=True)
                    modelling_tab.markdown(f'<strong>Forward Looking:</strong> The model is retrained on the complete historical dataset to forecast the next {forecast_freq} observations.', unsafe_allow_html=True)
                    
                if 'early warning' in st.session_state.objective_type.lower() and st.session_state.evaluate_early_warning:
                    
                    # Check if the model type is either Prophet or Naive
                    if st.session_state.model_type in ["Prophet", "Naive - Mean", "Naive - Drift"]:
                        modelling_tab.error('Feature not available for Prophet and Naive. Please change the type of the model.')

                    # Check if the model type is either Random Forest, XGBoost, or Ensemble Tree
                    elif st.session_state.model_type in ["Random Forest", "XGBoost", "Ensemble Tree"]:

                    
                        with modelling_tab:
                            with st.spinner(':warning: Calculating Metrics for Early Warning...'):

                                test_metrics_list = []

                                for test_steps in st.session_state.test_steps_list:
                                    # Split the train set into train and test data based on the number of test steps and the frequency
                                    train_data = train.copy(deep=True).asfreq('D').resample(forecast_freq).mean()[:-test_steps]
                                    test_data = train.copy(deep=True).asfreq('D').resample(forecast_freq).mean()[-test_steps:]
                                    planning_data = planning_full.asfreq('D').resample(forecast_freq).mean()

                                    logger.info(f"Train Date Range : {train_data.index.min()} --- {train_data.index.max()}  (n={len(train_data)})")
                                    logger.info(f"Test Date Range  : {test_data.index.min()} --- {test_data.index.max()}  (n={len(test_data)})")

                                    # Generate date features for the train and test data
                                    train_data = generate_date_features(train_data)
                                    test_data = generate_date_features(test_data)

                                    # Get the actual target values from the test data
                                    actual = test_data['y']

                                    if st.session_state.country_name.lower() == "none":
                                        holiday_mask = None
                                    else:
                                        holiday_mask = (test_data['is_holiday'] == 1).values
                                        logger.warning("WARNING: Holidays are excluded from model evaluation!")


                                    # Select the best model and the best estimator based on the model type
                                    if st.session_state.model_type == "Random Forest":
                                        best_estimator = st.session_state.rf_best_estimator
                                    elif st.session_state.model_type == "XGBoost":
                                        best_estimator = st.session_state.xgb_best_estimator
                                    elif st.session_state.model_type == "Ensemble Tree":
                                        best_estimator = st.session_state.ensemble_best_estimator

                                    # Fit the best model on the train data and compute error metrics on the test data
                                    best_estimator.fit(y=train_data['y'], exog=train_data[st.session_state.exog_cols_actual])

                                    # Generate prediction intervals for the test data
                                    predictions_test = best_estimator.predict_interval(
                                        steps=test_steps, exog=test_data[st.session_state.exog_cols_actual],
                                        interval=[pi_lower, pi_upper], n_boot=pi_n_boots)
                                    
                                    if st.session_state.country_name != 'None':
                                        
                                        #### ADJUST HOLIDAY VALUES POSTPROCESSING
                                        predictions_test = predictions_test.copy(deep=True).asfreq('D').bfill().reset_index()
                                        # Step 2 & 3: Find corresponding 'ds' in df_forecast and update 'y_true'
                                        # Also, calculate the scaling ratio
                                        scaling_ratio = holiday_default_value / predictions_test.loc[predictions_test['index'].isin(holiday_dates), 'pred']
                                        predictions_test.loc[predictions_test['index'].isin(holiday_dates), 'pred'] = holiday_default_value

                                        # Step 4: Apply the scaling ratio to 'min_pred' and 'max_pred'
                                        predictions_test.loc[predictions_test['index'].isin(holiday_dates), 'lower_bound'] *= scaling_ratio
                                        predictions_test.loc[predictions_test['index'].isin(holiday_dates), 'upper_bound'] *= scaling_ratio

                                        # Reset the forecast frequency
                                        predictions_test = predictions_test.set_index('index').resample(forecast_freq).mean()

                                    metrics_test = {}
                                    
                                    # Compute error metrics and prediction coverage for the test data
                                    metrics_test_predicted = compute_error_metrics(
                                        actual, predictions_test['pred'], train_data['y'], holiday_mask)
                                    
                                    for k, v in metrics_test_predicted.items():
                                        metrics_test[f'{k}_pred'] = v

                                    metrics_test["coverage_pred"] = compute_prediction_coverage(test_data, predictions_test, holiday_mask)
                                    
                                    
                                    predictions_test['actual'] = actual

                                    metrics_test['test_steps'] = test_steps

                                    # Check if test dates are subset of planning dates
                                    is_subset = predictions_test.index.isin(planning_data.index).all()

                                    # Perform left join if the condition is met
                                    if is_subset:
                                        predictions_test = predictions_test.join(planning_data, how='left').reset_index()
                                        predictions_test.rename(columns={
                                            'index': 'Date',
                                            'actual': 'Actual',
                                            'pred': 'Predicted',
                                            'upper_bound': 'Predicted - Upper Bound',
                                            'lower_bound': 'Predicted - Lower Bound',
                                            'y': 'Planned'
                                            }, inplace=True)
                                        
                                        # Compute error metrics and prediction coverage for the test data
                                        metrics_test_planned = compute_error_metrics(
                                            predictions_test['Actual'], predictions_test['Planned'], train_data['y'], holiday_mask)

                                        for k, v in metrics_test_planned.items():
                                            metrics_test[f'{k}_plnd'] = v

                                        # Calculate RMSE between Actual and Predicted (RMSE_A-P)
                                        metrics_test['actual_deviation_rmse'] = round(mean_squared_error(predictions_test['Actual'], predictions_test['Predicted'], squared=False), 3)

                                        # Calculate Planning Deviation (PD = RMSE_A-Pld)
                                        metrics_test['planning_deviation_rmse'] = round(mean_squared_error(predictions_test['Actual'], predictions_test['Planned'], squared=False), 3)

                                        # Calculate Expected Deviation (ED = RMSE_Pld-P)
                                        metrics_test['expected_deviation_rmse'] = round(mean_squared_error(predictions_test['Planned'], predictions_test['Predicted'], squared=False), 3)

                                    else:
                                        st.error(f"Planning is incomplete for {forecast_freq}-{test_steps}. Please review the uploaded Planning File.")
                                        st.stop()

                                    # Get forecast data for same period
                                    forecast_data = st.session_state.df_forecast_updated[:test_steps].set_index('ds')

                                    # Check if test dates are subset of planning dates
                                    is_subset = forecast_data.index.isin(planning_data.index).all()
                                    
                                    # Perform left join if the condition is met
                                    if is_subset:
                                        forecast_data = forecast_data.join(planning_data, how='left').reset_index()
                                        forecast_data.rename(columns={
                                            'ds': 'Date',
                                            'y_true': 'Actual',
                                            'y_pred': 'Predicted',
                                            'min_pred': 'Predicted - Lower Bound',
                                            'max_pred': 'Predicted - Upper Bound',
                                            'y': 'Planned'
                                            }, inplace=True)

                                        # Calculate Forecast Deviation (FD = RMSE_Pld-P_future)
                                        metrics_test['forecast_deviation_rmse'] = round(mean_squared_error(forecast_data['Predicted'], forecast_data['Planned'], squared=False), 3)

                                        # Calculate Scaled Deviation (SD = (FD - ED) / PD)
                                        metrics_test['scaled_deviation'] = round(np.abs((metrics_test['forecast_deviation_rmse'] - metrics_test['expected_deviation_rmse']) / np.abs(metrics_test['planning_deviation_rmse'] - metrics_test['actual_deviation_rmse'])), 3)
                                        
                                    else:
                                        st.error(f"Forecast is incomplete for {forecast_freq}-{test_steps}.")
                                        st.stop()

                                    test_metrics_list.append(metrics_test)

                                    merged_df = pd.concat([predictions_test, forecast_data], sort=True)
                                    merged_df = merged_df[['Date', 'Actual', 'Planned', 'Predicted', 'Predicted - Lower Bound', 'Predicted - Upper Bound']]
                                    # Find the first date where 'Actual' is NaN
                                    max_actual_date = max(merged_df[~merged_df['Actual'].isna()]['Date'].tolist())

                                    # Find the minimum date where 'Actual' is not NaN
                                    min_actual_date = min(merged_df[~merged_df['Actual'].isna()]['Date'].tolist())


                                    # Create Plot
                                    fig = go.Figure()

                                    # Add Traces
                                    fig.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Actual'], mode='lines+markers', name='Actual', line=dict(color='rgba(143,188,143,0.75)')))
                                    fig.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Planned'], mode='lines+markers', name='Planned', line=dict(color='orange')))
                                    fig.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Predicted'], mode='lines+markers', name='Predicted', line=dict(color='firebrick', width=2)))
                                    fig.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Predicted - Upper Bound'], mode='lines', name='Predicted Upper Bound', line=dict(color='rgba(158,202,225,0.8)', dash='dash')))
                                    fig.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Predicted - Lower Bound'], mode='lines', name='Predicted Lower Bound', line=dict(color='rgba(158,202,225,0.8)', dash='dash')))

                                    # Add vertical dashed line
                                    fig.add_shape(
                                        go.layout.Shape(type='line', x0=max_actual_date, x1=max_actual_date, y0=0, y1=1, yref='paper', line=dict(color='black', dash='dash'))
                                    )

                                    # Add background color rectangles
                                    fig.add_shape(
                                        go.layout.Shape(type="rect", x0=min(merged_df['Date']), x1=max_actual_date, y0=0, y1=1, yref="paper", fillcolor="lightgrey", opacity=0.2, layer="below", line_width=0)
                                    )
                                    fig.add_shape(
                                        go.layout.Shape(type="rect", x0=max_actual_date, x1=max(merged_df['Date']), y0=0, y1=1, yref="paper", fillcolor="lightblue", opacity=0.2, layer="below", line_width=0)
    )

                                    # Layout settings with annotations
                                    fig.update_layout(
                                        title=f"Actual vs Planned vs Predicted for {forecast_freq}-{test_steps} with Scaled Devation: {metrics_test['scaled_deviation']}",
                                        xaxis_title='Date',
                                        yaxis_title='Value',
                                        xaxis=dict(showline=True, showgrid=False),
                                        yaxis=dict(showgrid=False),
                                         annotations=[
                                            dict(x=min_actual_date, y=1, yref='paper', xref='x', xanchor='left', text='Backward Looking', showarrow=False),
                                            dict(x=max_actual_date, y=1, yref='paper', xref='x', xanchor='left', text='Forward Looking', showarrow=False)
                                        ]
    )
                                    st.session_state[f'{forecast_freq}_{test_steps}_fig'] = fig
                                    st.session_state[f'{forecast_freq}_{test_steps}_data'] = merged_df

                        st.session_state[f'ews_metrics_data'] = test_metrics_list
                        st.session_state.evaluate_early_warning = False    
                    
               
                if 'early warning' in st.session_state.objective_type.lower():
                    # Check if the model type is either Prophet or Naive
                    if st.session_state.model_type in ["Prophet", "Naive - Mean", "Naive - Drift"]:
                        modelling_tab.error('Feature not available for Prophet and Naive. Please change the type of the model.')

                    # Check if the model type is either Random Forest, XGBoost, or Ensemble Tree
                    elif st.session_state.model_type in ["Random Forest", "XGBoost", "Ensemble Tree"]:

                    
                        # Renaming columns
                        rename_dict = {
                            'test_steps': forecast_freq,
                            'mape_pred': 'MAPE - Pred.',
                            'rmspe_pred': 'RMSPE - Pred.',
                            'mase_pred': 'MASE - Pred.',
                            'rmsse_pred': 'RMSSE - Pred.',
                            'rmse_pred': 'RMSE - Pred.',
                            'mae_pred': 'MAE - Pred.',
                            'coverage_pred': 'Coverage - Pred.',
                            'mape_plnd': 'MAPE - Plnd.',
                            'rmspe_plnd': 'RMSPE - Plnd.',
                            'mase_plnd': 'MASE - Plnd.',
                            'rmsse_plnd': 'RMSSE - Plnd.',
                            'rmse_plnd': 'RMSE - Plnd.',
                            'mae_plnd': 'MAE - Plnd.',
                            'actual_deviation_rmse': 'Actual Error (AE)',
                            'planning_deviation_rmse': 'Planning Error (PE)',
                            'expected_deviation_rmse': 'Excepted Error (EE)',
                            'forecast_deviation_rmse': 'Forecast Error (FE)',
                            'scaled_deviation': 'Scaled Devation (SD)'
                        }

                        df_ews_metrics = pd.DataFrame(st.session_state[f'ews_metrics_data']).round(3)
                        df_ews_metrics.rename(columns=rename_dict, inplace=True)
                        df_ews_metrics.set_index(forecast_freq, inplace=True)

                        # Apply the background gradient to the metrics columns except 'Coverage'
                        styled_model_comparisons = df_ews_metrics.style.background_gradient(
                            cmap=cmap_custom_reversed, subset=list(set(rename_dict.values()).difference(set(['Coverage - Pred.', forecast_freq])))
                        )

                        # Apply the reversed background gradient to the 'Coverage' column
                        styled_model_comparisons = styled_model_comparisons.background_gradient(
                            cmap=cmap_custom, subset=['Coverage - Pred.']
                        )

                        modelling_tab.dataframe(styled_model_comparisons)

                        data_fname = f"Early_Warning_Metrics"

                        modelling_tab.download_button(
                                            label="Export Data (.csv)",
                                            data=df_ews_metrics.to_csv(),
                                            file_name=f"{dt_now}_{data_fname}.csv",
                                            mime="text/csv"
                                        )


                        df_ews_metrics.to_excel(writer, sheet_name=data_fname, index=True)

                        for test_steps in st.session_state.test_steps_list:


                            fig = st.session_state[f'{forecast_freq}_{test_steps}_fig']

                            modelling_tab.plotly_chart(fig, use_container_width=True)

                            col_1, col_2 = modelling_tab.columns([2, 5])

                            data_fname = f"Early_Warning_{forecast_freq}-{test_steps}"

                            # Add buttons to download the data and the plot
                            col_1.download_button(
                                label="Export Data (.csv)",
                                data=st.session_state[f'{forecast_freq}_{test_steps}_data'].to_csv(),
                                file_name=f"{dt_now}_{data_fname}.csv",
                                mime="text/csv"
                            )

                            # Export plot to HTML
                            html_buffer = io.StringIO()
                            fig.write_html(html_buffer, include_plotlyjs='cdn')

                            col_2.download_button(
                                label='Export Plot (.html)',
                                data=html_buffer.getvalue().encode(),
                                file_name=f"{dt_now}_{data_fname}.html",
                                mime='text/html'
                            )

                            # Save the predictions to results excel sheet
                            st.session_state[f'{forecast_freq}_{test_steps}_data'].to_excel(writer, sheet_name=data_fname, index=False)
                            writer.book.add_worksheet(f"{data_fname}_Plot").insert_image(f'A1', f"{data_fname}_Plot", {'image_data': io.BytesIO(pio.to_image(fig, format='png'))})

                            # Add HTML file to ZIP
                            zipf.writestr(f"{forecast_freq}-{test_steps}.html", html_buffer.getvalue().encode())
                        
                        forecast_metrics = df_forecast.set_index('ds').resample(forecast_freq).mean()
                        forecast_metrics = forecast_metrics.join(planning_full.asfreq('D').resample(forecast_freq).mean(), how='left')
                        forecast_metrics.rename(columns={
                                'y': 'Planned',
                                'y_true': 'Actual',
                                'y_pred': 'Predicted',
                                'min_pred': 'Predicted - Lower Bound',
                                'max_pred': 'Predicted - Upper Bound'
                            }, inplace=True)
                        
                        # Function to calculate MAE, RMSE, MAPE, RMSPE
                        def calculate_metrics(row):
                            planned = row['Planned']
                            predicted = row['Predicted']

                            mae = np.abs(predicted - planned)
                            rmse = np.sqrt((predicted - planned) ** 2)

                            # Avoid division by zero for MAPE and RMSPE
                            if planned != 0:
                                mape = np.abs((predicted - planned) / planned)
                                rmspe = np.sqrt(((predicted - planned) / planned) ** 2)
                            else:
                                mape = np.nan
                                rmspe = np.nan

                            return pd.Series([mae, rmse, mape, rmspe])

                        # Applying the function to each row
                        forecast_metrics[['MAE', 'RMSE', 'MAPE', 'RMSPE']] = forecast_metrics.apply(calculate_metrics, axis=1)
                        
                        forecast_metrics = forecast_metrics.reset_index()
                        
                        # Create figure
                        fig = go.Figure()

                        # Add Planned (orange) and Predicted (red) line plot
                        fig.add_trace(go.Scatter(x=forecast_metrics['ds'], y=forecast_metrics['Planned'],
                                                 mode='lines+markers', name='Planned',
                                                 line=dict(color='orange')))
                        fig.add_trace(go.Scatter(x=forecast_metrics['ds'], y=forecast_metrics['Predicted'],
                                                 mode='lines+markers', name='Predicted',
                                                 line=dict(color='red')))

                        # Add RMSPE bar plot on secondary y-axis with transparency
                        fig.add_trace(go.Bar(x=forecast_metrics['ds'], y=forecast_metrics['RMSPE'],
                                             name='RMSPE', marker_color='lightblue', yaxis='y2',
                                             opacity=0.5))  # Adjust opacity here

                        # Create a secondary y-axis for RMSPE
                        fig.update_layout(
                            yaxis2=dict(
                                title='RMSPE',
                                overlaying='y',
                                side='right',
                                showgrid=False
                            )
                        )


                        # Set x-axis and primary y-axis titles
                        fig.update_xaxes(title_text='')
                        fig.update_yaxes(title_text='')

                        # Add title
                        fig.update_layout(title_text="Planned vs Predicted Values Over Time for Forecast Horizon", width=1200)

                        # Show the figure
                        modelling_tab.plotly_chart(fig)
                        
                        col_1, col_2, col_3 = modelling_tab.columns([2, 2, 5])

                        data_fname = f"EWS_Forecast_vs_Planned"

                        # Add buttons to download the data and the plot
                        col_1.download_button(
                            label="Export Data (.csv)",
                            data=forecast_metrics.to_csv(),
                            file_name=f"{dt_now}_{data_fname}.csv",
                            mime="text/csv"
                        )

                        # Export plot to HTML
                        html_buffer = io.StringIO()
                        fig.write_html(html_buffer, include_plotlyjs='cdn')

                        col_2.download_button(
                            label='Export Plot (.html)',
                            data=html_buffer.getvalue().encode(),
                            file_name=f"{dt_now}_{data_fname}.html",
                            mime='text/html'
                        )

                        # Save the predictions to results excel sheet
                        forecast_metrics.to_excel(writer, sheet_name=data_fname, index=False)
                        writer.book.add_worksheet(f"{data_fname}_Plot").insert_image(f'A1', f"{data_fname}_Plot", {'image_data': io.BytesIO(pio.to_image(fig, format='png'))})

                        # Add HTML file to ZIP
                        zipf.writestr(f"{data_fname}_Plot.html", html_buffer.getvalue().encode())
                        
                        # Modify the calculate_metrics function to handle a week's data
                        def calculate_weekly_metrics(week_group):
                            if week_group.empty:
                                return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

                            planned = week_group['Planned'].values
                            predicted = week_group['Predicted'].values

                            mae = np.mean(np.abs(predicted - planned))
                            rmse = np.sqrt(np.mean((predicted - planned) ** 2))

                            # Avoid division by zero for MAPE and RMSPE
                            non_zero_planned = planned != 0
                            if np.any(non_zero_planned):
                                mape = np.mean(np.abs((predicted[non_zero_planned] - planned[non_zero_planned]) / planned[non_zero_planned]))
                                rmspe = np.sqrt(np.mean(((predicted[non_zero_planned] - planned[non_zero_planned]) / planned[non_zero_planned]) ** 2))
                            else:
                                mape = np.nan
                                rmspe = np.nan

                            return pd.Series([np.mean(planned), np.mean(predicted), mae, rmse, mape, rmspe])

                        # Set 'ds' as the index and resample the DataFrame by week
                        forecast_metrics.set_index('ds', inplace=True)
                        forecast_metrics = forecast_metrics.asfreq('D').bfill()

                        weekly_metrics = forecast_metrics.resample('W').apply(calculate_weekly_metrics)
                        weekly_metrics.columns = ['Planned - Mean', 'Predicted - Mean', 'MAE', 'RMSE', 'MAPE', 'RMSPE']
                        weekly_metrics.reset_index(inplace=True)
                        
                        data_fname = f"EWS_Forecast_vs_Planned_Weekly"
                        
                        col_3.download_button(
                            label="Export Weekly Data (.csv)",
                            data=weekly_metrics.to_csv(),
                            file_name=f"{dt_now}_{data_fname}.csv",
                            mime="text/csv"
                        )

                        # Save the predictions to results excel sheet
                        weekly_metrics.to_excel(writer, sheet_name=data_fname, index=False)
                        

                #####################################################################################

                # Getting the latest year and the end date of the training data
                current_year = df_train["ds"].dt.year.max()
                train_end_date = df_train["ds"].max()

                # Creating a list of years from the current year to the maximum year in the forecast data
                forecast_years = sorted(list(range(current_year, df_combined["Date"].dt.year.max() + 1)))

                #####################################################################################
                if show_analysis_tab:
                    # Subsection header
                    analysis_tab.header("Month-By-Month")

                    # Select operation
                    mbm_operation = analysis_tab.radio("Choose Operation:", ['Mean', 'Sum'], index=0, key="mbm_operation")

                    # Compute Month-By-Month metrics
                    df_mbm = compute_mbm_table(df_combined, forecast_years, current_year, mbm_operation)
                    df_mbm = df_mbm.loc[:, (df_mbm != 0).any(axis=0)]
                    df_mbm = df_mbm.loc[:, (df_mbm[:-1] != 100).any(axis=0)]

                    # Select years for analysis
                    unique_years = sorted(list(set([col[:4] for col in df_mbm.T.index.unique().tolist() if "Actual" in col or "Forecast" in col])))
                    selected_years = analysis_tab.multiselect("Choose Years:", unique_years, default=unique_years[-5:], key="mbm_years")

                    # Generate the plot for the selected years
                    selected_start_date = train_end_date + datetime.timedelta(days=1)
                    fig = mbm_plot(df_mbm, selected_start_date, selected_years)

                    # Define the layout for displaying the results
                    analysis_row2_col1, analysis_row2_col2 = analysis_tab.columns(2)

                    # Display the data and the plot
                    analysis_row2_col1.subheader("Result")
                    analysis_row2_col1.dataframe(df_mbm, use_container_width=True)
                    analysis_row2_col2.plotly_chart(fig, use_container_width=True)

                    # Define file names for data and plot
                    img_fname = f'{current_year}_Month_By_Month_Plot'
                    data_fname = f'{current_year}_Month_By_Month_Data'

                    # Round off the data and save it to the Excel file
                    df_mbm = round(df_mbm, 1)
                    df_mbm.to_excel(writer, sheet_name=data_fname)
                    writer.sheets[data_fname].insert_image(f'A{len(df_mbm)+2}', f'{img_fname}', {'image_data': io.BytesIO(pio.to_image(fig, format='png'))})

                    # Export plot to HTML
                    html_buffer = io.StringIO()
                    fig.write_html(html_buffer, include_plotlyjs='cdn')

                    # Add buttons to download the data and the plot
                    analysis_row2_col1.download_button(
                        label="Export Data (.csv)",
                        data=df_mbm.to_csv(),
                        file_name=f"{dt_now}_{data_fname}_{fc_period}.csv",
                        mime="text/csv"
                    )

                    analysis_row2_col2.download_button(
                        label='Export Interactive Plot (.html)',
                        data=html_buffer.getvalue().encode(),
                        file_name=f"{dt_now}_{img_fname}_{fc_period}.html",
                        mime='text/html'
                    )

                    # Add HTML file to ZIP
                    zipf.writestr(f"{dt_now}_{img_fname}_{fc_period}.html", html_buffer.getvalue().encode())

                    #####################################################################################

                    # Subsection header
                    analysis_tab.header("Month-By-Month Peak")

                    # Creating a list of years from the current year to the maximum year in the forecast data
                    forecast_years = sorted(list(range(current_year, df_combined["Date"].dt.year.max() + 1)))

                    # Compute Month-By-Month peak metrics
                    df_mbm_peak = compute_peak_mbm_table(df_combined, forecast_years, current_year)
                    df_mbm_peak = df_mbm_peak.loc[:, (df_mbm_peak != 0).any(axis=0)]
                    df_mbm_peak = df_mbm_peak.loc[:, (df_mbm_peak[:-1] != 100).any(axis=0)]

                    # Select years for analysis
                    unique_years = sorted(list(set([col[:4] for col in df_mbm_peak.T.index.unique().tolist() if "Actual" in col or "Forecast" in col])))
                    selected_years = analysis_tab.multiselect("Choose Years:", unique_years, default=unique_years[-5:], key="mbm_peak_years")


                    # Define the start date for the plot
                    selected_start_date = train_end_date + datetime.timedelta(days=1)

                    # Generate the plot for the selected years
                    fig = mbm_plot(df_mbm_peak, selected_start_date, selected_years)

                    # Define the layout for displaying the results
                    analysis_row3_col1, analysis_row3_col2 = analysis_tab.columns(2)

                    # Display the data and the plot
                    analysis_row3_col1.subheader("Result")
                    analysis_row3_col1.dataframe(df_mbm_peak, use_container_width=True)
                    analysis_row3_col2.plotly_chart(fig, use_container_width=True)

                    # Define file names for data and plot
                    img_fname = f'{current_year}_Month_By_Month_Peak_Plot'
                    data_fname = f'{current_year}_Month_By_Month_Peak_Data'

                    # Round off the data and save it to the Excel file
                    df_mbm_peak = round(df_mbm_peak, 1)
                    df_mbm_peak.to_excel(writer, sheet_name=data_fname)
                    writer.sheets[data_fname].insert_image(f'A{len(df_mbm_peak)+2}', f'{img_fname}', {'image_data': io.BytesIO(pio.to_image(fig, format='png'))})

                    # Export plot to HTML
                    html_buffer = io.StringIO()
                    fig.write_html(html_buffer, include_plotlyjs='cdn')

                    # Add buttons to download the data and the plot
                    analysis_row3_col1.download_button(
                        label="Export Data (.csv)",
                        data=df_mbm_peak.to_csv(),
                        file_name=f"{dt_now}_{data_fname}_{fc_period}.csv",
                        mime="text/csv"
                    )

                    analysis_row3_col2.download_button(
                        label='Export Interactive Plot (.html)',
                        data=html_buffer.getvalue().encode(),
                        file_name=f"{dt_now}_{img_fname}_{fc_period}.html",
                        mime='text/html'
                    )

                    # Add HTML file to ZIP
                    zipf.writestr(f"{dt_now}_{img_fname}_{fc_period}.html", html_buffer.getvalue().encode())

                    #####################################################################################

                    # Subsection header
                    analysis_tab.header("Week-By-Week")

                    # Select operation
                    wbw_operation = analysis_tab.radio("Choose Operation:", ['Mean', 'Sum'], index=0, key="wbw_operation")

                    # Create a selectbox for year selection
                    selected_year = analysis_tab.selectbox("Select Year:", forecast_years, key="week_by_week_year")

                    # Compute Week-By-Week metrics
                    df_wbw = compute_wbw_table(df_combined, current_year, selected_year, wbw_operation)
                    df_wbw = df_wbw.loc[:, (df_wbw != 0).any(axis=0)]
                    df_wbw = df_wbw.loc[:, (df_wbw[:-1] != 100).any(axis=0)]

                    # Get the unique years in the data
                    unique_years = sorted(list(set([col[:4] for col in df_wbw.T.index.unique().tolist() if "Actual" in col or "Forecast" in col])))

                    # Create the multiselect box in the sidebar for year selection
                    selected_years = analysis_tab.multiselect("Choose Years:", unique_years, default=unique_years[-3:], key="wbw_years")

                    # Define the start date for the plot
                    selected_start_date = train_end_date + datetime.timedelta(days=1)

                    # Generate the plot for the selected years
                    fig = wbw_plot(df_wbw, selected_start_date, selected_years, selected_year)

                    # Define the layout for displaying the results
                    analysis_row4_col1, analysis_row4_col2 = analysis_tab.columns(2)

                    # Display the data and the plot
                    analysis_row4_col1.subheader("Result")
                    analysis_row4_col1.dataframe(df_wbw, use_container_width=True)
                    analysis_row4_col2.plotly_chart(fig, use_container_width=True)

                    # Define file names for data and plot
                    img_fname = f'{current_year}_Week_By_Week_Plot'
                    data_fname = f'{current_year}_Week_By_Week_Data'

                    # Round off the data and save it to the Excel file
                    df_wbw = round(df_wbw, 1)
                    df_wbw.to_excel(writer, sheet_name=data_fname)
                    writer.sheets[data_fname].insert_image(f'A{len(df_wbw)+2}', f'{img_fname}', {'image_data': io.BytesIO(pio.to_image(fig, format='png'))})

                    # Export plot to HTML
                    html_buffer = io.StringIO()
                    fig.write_html(html_buffer, include_plotlyjs='cdn')

                    # Add buttons to download the data and the plot
                    analysis_row4_col1.download_button(
                        label="Export Data (.csv)",
                        data=df_wbw.to_csv(),
                        file_name=f"{dt_now}_{data_fname}_{fc_period}.csv",
                        mime="text/csv"
                    )

                    analysis_row4_col2.download_button(
                        label='Export Interactive Plot (.html)',
                        data=html_buffer.getvalue().encode(),
                        file_name=f"{dt_now}_{img_fname}_{fc_period}.html",
                        mime='text/html'
                    )

                    # Add HTML file to ZIP
                    zipf.writestr(f"{dt_now}_{img_fname}_{fc_period}.html", html_buffer.getvalue().encode())

                    ####################################################################################

                    # Subsection header
                    analysis_tab.header("Day Peak")

                    # Selection box for the year
                    selected_year = analysis_tab.selectbox("Select Year:", forecast_years, key="day_peak_year")

                    # Generate the peak day table, start date, and any error messages
                    df_peak_table, selected_start_date, error_message = day_peak_table(selected_year, current_year, train_end_date, df_combined)
                    df_peak_table = df_peak_table.loc[:, (df_peak_table != 0).any(axis=0)]
                    df_peak_table = df_peak_table.loc[:, (df_peak_table[:-1] != 100).any(axis=0)]

                    # If an error message is generated, display it
                    if error_message:
                        analysis_tab.error(error_message, icon="🚨")
                    else:
                        # Select days for analysis
                        unique_days = df_peak_table.index.unique().tolist()
                        selected_days = analysis_tab.multiselect("Choose Days:", unique_days, default=unique_days, key="pdm_days")

                        # Generate the plot for the selected days
                        fig = pdm_plot(df_peak_table, selected_start_date, selected_days)

                        # Define the layout for displaying the results
                        analysis_row1_col1, analysis_row1_col2 = analysis_tab.columns(2)

                        # Display the data and the plot
                        analysis_row1_col1.subheader("Result")
                        analysis_row1_col1.dataframe(df_peak_table, use_container_width=True)
                        analysis_row1_col2.plotly_chart(fig, use_container_width=True)

                        # Define file names for data and plot
                        img_fname = f'{current_year}_Day_Peak_Plot'
                        data_fname = f'{current_year}_Day_Peak_Data'

                        # Round off the data and save it to the Excel file
                        df_peak_table = round(df_peak_table, 1)
                        df_peak_table.to_excel(writer, sheet_name=data_fname)
                        writer.sheets[data_fname].insert_image(f'A{len(df_peak_table)+2}', f'{img_fname}', {'image_data': io.BytesIO(pio.to_image(fig, format='png'))})

                        # Export plot to HTML
                        html_buffer = io.StringIO()
                        fig.write_html(html_buffer, include_plotlyjs='cdn')

                        # Add buttons to download the data and the plot
                        analysis_row1_col1.download_button(
                            label="Export Data (.csv)",
                            data=df_peak_table.to_csv(),
                            file_name=f"{dt_now}_{data_fname}_{fc_period}.csv",
                            mime="text/csv"
                        )

                        analysis_row1_col2.download_button(
                            label='Export Interactive Plot (.html)',
                            data=html_buffer.getvalue().encode(),
                            file_name=f"{dt_now}_{img_fname}_{fc_period}.html",
                            mime='text/html'
                        )

                        # Add HTML file to ZIP
                        zipf.writestr(f"{dt_now}_{img_fname}_{fc_period}.html", html_buffer.getvalue().encode())

                ####################################################################################

                # Close the Pandas Excel writer and output the Excel file to the BytesIO object
                writer.close()

                zipf.close()

                # Seek to the beginning of the stream
                result_buffer.seek(0)

                # Notify user about the readiness of the result
                download_tab.success('Your result is ready for download!', icon="✅")

                # Define the layout for displaying the results
                download_row1_col1, download_row1_col2 = download_tab.columns(2)

                # Add a button for downloading the results
                download_row1_col1.download_button(
                    label="Forecasts and Analysis (.xlsx)",
                    data=result_buffer,
                    file_name=f"{dt_now}_{'Forecast_Analysis'}_{fc_period}.xlsx",
                    mime="application/vnd.ms-excel"
                )

                # Add a button for downloading the results
                download_row1_col1.download_button(
                    label="Interactive Plots (.zip)",
                    data=zip_buffer.getvalue(),
                    file_name=f"{dt_now}_{'Interactive_Plots'}_{fc_period}.zip",
                    mime="application/zip"
                )

                download_tab.warning("🚀 Please help us save resources and costs: kindly shut down the application once you're done. 🙏✨")


if __name__ == "__main__":
    main()
