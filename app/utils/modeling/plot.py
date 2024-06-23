import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_forecasts(historical_data, forecast_data):
    
    # Determine the start date for the forecasts
    forecast_start_date = forecast_data['ds'].min()

    # Create the base line plot for historical data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_data['ds'], y=historical_data['y'], mode='lines', name='Historical Data',
                             line=dict(color='red')))

    # Add forecast data
    fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['y_pred'], mode='lines', name='Forecast',
                             line=dict(color='seagreen')))

    # Add min and max prediction intervals as dotted lines
    fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['min_pred'], mode='lines', name='Min Prediction',
                             line=dict(color='blue', dash='dot')))
    fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['max_pred'], mode='lines', name='Max Prediction',
                             line=dict(color='blue', dash='dot')))

    # Add a vertical dotted line to indicate the start of the forecasts
    fig.add_trace(go.Scatter(x=[forecast_start_date, forecast_start_date], y=[min(historical_data['y'].min(), forecast_data['min_pred'].min()), max(historical_data['y'].max(), forecast_data['max_pred'].max())],
                             mode='lines', name='Start of Forecast', line=dict(color='black', dash='dot')))

    fig.update_layout(autosize=False, width=1000, height=600)  # Set the size of the figure
    fig.update_xaxes(dtick="M1")  # Set the x-axis ticks to be month-based

    return fig


def create_pivot_table(data, index_unit='month', aggfunc='mean'):
    """
    Creates a pivot table with different time units as index (month, day of the week, or week number).
    Includes proper names for months and days, and adds percentage change from the previous year.
    Ensures correct ordering of indices for months and days of the week.
    """
    # Ensure 'ds' is in datetime format
    data['ds'] = pd.to_datetime(data['ds'])
    
    # Define the order for months and days of the week
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Extract time unit information based on the index_unit argument
    if index_unit == 'month':
        data['time_unit'] = data['ds'].dt.month_name()
        order = month_order
    elif index_unit == 'day_of_week':
        data['time_unit'] = data['ds'].dt.day_name()
        order = day_order
    elif index_unit == 'week_of_year':
        data['time_unit'] = data['ds'].dt.isocalendar().week
        order = None  # Week numbers are already in correct order
    else:
        raise ValueError("Invalid index_unit. Choose 'month', 'day_of_week', or 'week_of_year'")

    # Extract year for column headers
    data['year'] = data['ds'].dt.year
    
    # Pivot table to compare values across years
    result = data.pivot_table(index='time_unit', columns='year', values='y', aggfunc=aggfunc)

    # Reorder the index if a specific order is defined (for months or days)
    if order:
        result = result.reindex(order)

    # Calculate percentage change from the previous year
    result = result.join(result.pct_change(axis='columns') * 100, rsuffix='% Change')

    # Clean up infinite and NaN values resulted from zero division or empty previous year data
    result = result.replace([np.inf, -np.inf], np.nan)
    
    return result


def plot_time_series(data):
    """
    Generates a Plotly line plot from a pivot table with different years as lines.
    Assumes the data includes only columns for years, excluding percentage change columns.
    """
    fig = go.Figure()

    # Prepare a color palette
    colors = px.colors.qualitative.Plotly

    # Filter out any percentage change columns
    data = data[[col for col in data.columns if not '% Change' in col]]

    # Iterate through each column (year) in the DataFrame to create a line trace
    for idx, year in enumerate(data.columns):
        fig.add_trace(go.Scatter(x=data.index, y=data[year], mode='lines', name=str(year),
                                 line=dict(color=colors[idx % len(colors)])))

    # Update the layout to add titles and axis labels
    fig.update_layout(legend_title='Year',
                      xaxis=dict(type='category'))  # Ensure categorical handling of x-axis for clarity

    # Show the plot
    return fig