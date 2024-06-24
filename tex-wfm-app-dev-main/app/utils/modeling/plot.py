from typing import Optional, Tuple

import numpy as np
import pandas as pd

import plotly.graph_objects as go



def create_base_figure(width: int = 1500, height: int = 600) -> go.Figure:
    """
    Creates a base figure with specific width and height.
    """
    fig = go.Figure()
    fig.update_layout(autosize=False, width=width, height=height)  # Set the size of the figure
    fig.update_xaxes(dtick="M1")  # Set the x-axis ticks to be month-based
    return fig


def plot_forecast(df_predictions: pd.DataFrame, df_forecast: pd.DataFrame) -> go.Figure:
    """
    Plot the actual values for the training set and the forecast period
    along with the corresponding prediction intervals.
    """
    fig = create_base_figure()  # Create a base figure
    
    train_ds_index = pd.DatetimeIndex(df_predictions["ds"])
    first_prediction_date = df_predictions.loc[df_predictions["y_pred"].notnull(), "ds"].iloc[0]
    
    fig.add_trace(go.Scatter(
        x=df_predictions["ds"],
        y=df_predictions["y_pred"],
        mode='lines',
        name='Predicted',
        line=dict(color='firebrick', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_predictions["ds"],
        y=df_predictions["y_true"],
        mode='lines',
        name='Actual',
        line=dict(color='rgba(143,188,143,0.75)')
    ))
    
    first_prediction_date =  df_predictions.loc[df_predictions["y_pred"].notnull(), "ds"].iloc[0]
    warmup_count = len(df_predictions) - len(df_predictions.loc[df_predictions["y_pred"].notnull(), "ds"])
    
    fig.add_trace(go.Scatter(
        x=df_predictions["ds"][warmup_count:],
        y=df_predictions["min_pred"][warmup_count:],
        name='Lower PI - Predicted',
        mode='lines',
        line=dict(color='rgba(158,202,225,0.2)'),
    ))
    
    fig.add_trace(go.Scatter(
        x=df_predictions["ds"][warmup_count:],
        y=df_predictions["max_pred"][warmup_count:],
        name='Upper PI - Predicted',
        mode='lines',
        line=dict(color='rgba(158,202,225,0.2)'),
        fill='tonexty'
    ))

    forecast_ds_index = pd.DatetimeIndex(df_forecast["ds"]) 
    last_date = pd.DatetimeIndex([df_predictions['ds'].values[-1]])
    combined_forecast_x = last_date.append(forecast_ds_index)
    combined_forecast_y = np.append(df_predictions["y_pred"].values[-1], df_forecast["y_pred"].values)
    
    fig.add_trace(go.Scatter(
        x=combined_forecast_x,
        y=combined_forecast_y,
        mode='lines',
        name='Forecast',
        line=dict(color='orange')
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_ds_index,
        y=df_forecast["min_pred"],
        name='Lower PI - Forecast',
        mode='lines',
        line=dict(color='rgba(158,202,225,0.1)'),
        fill='tonexty'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_ds_index,
        y=df_forecast["max_pred"],
        name='Upper PI - Forecast',
        mode='lines',
        line=dict(color='rgba(158,202,225,0.1)'),
        fill='tonexty'
    ))
    
    # Add a dotted line to mark the beginning of the forecast
    
    middle_train_date = train_ds_index[warmup_count // 3]
    middle_prediction_date = train_ds_index[warmup_count + ((len(df_predictions) - warmup_count) // 3)]
    first_forecast_date = forecast_ds_index[0]
    middle_forecast_date = forecast_ds_index[len(forecast_ds_index) // 3]
    
    if warmup_count > 0:
        # Add a dotted line to mark the first non-null date
        fig.add_shape(
            type="line",
            x0=first_prediction_date,
            y0=0,
            x1=first_prediction_date,
            y1=max(df_predictions["y_true"].max(), df_predictions["y_pred"].max())*1.1,
            line=dict(
                color="black",
                width=3,
                dash="dot"
            )
        )

    
    fig.add_shape(
        type="line",
        x0=first_forecast_date,
        y0=0,
        x1=first_forecast_date,
        y1=max(df_predictions["y_true"].max(), df_predictions["y_pred"].max())*1.1,
        line=dict(
            color="purple",
            width=3,
            dash="dash"
        )
    )
    
    if warmup_count > 0:
        # Add an annotation to indicate the forecast period
        fig.add_annotation(
            x=middle_train_date,
            y=max(df_predictions["y_true"].max(), df_predictions["y_pred"].max()),
            xanchor="left",
            yanchor="bottom",
            text="Train Warmup Period",
            showarrow=False,
            font=dict(size=15)
        )

    # Add an annotation to indicate the forecast period
    fig.add_annotation(
        x=middle_prediction_date,
        y=max(df_predictions["y_true"].max(), df_predictions["y_pred"].max()),
        xanchor="left",
        yanchor="bottom",
        text="Train Evaluation Period",
        showarrow=False,
        font=dict(size=15)
    )
    
    fig.add_annotation(
        x=middle_forecast_date,
        y=max(df_predictions["y_true"].max(), df_predictions["y_pred"].max()),
        xanchor="left",
        yanchor="bottom",
        text="Forecast Period",
        showarrow=False,
        font=dict(size=15)
    )

    fig.update_layout(
        title=f"Actual vs. Predicted from {df_predictions['ds'].min().strftime('%d-%m-%Y')} to {df_predictions['ds'].max().strftime('%d-%m-%Y')} & Forecast from {df_forecast['ds'].min().strftime('%d-%m-%Y')} to {df_forecast['ds'].max().strftime('%d-%m-%Y')}"
    )
    
    fig.update(layout_yaxis_range = [0, max(df_predictions["y_true"].max(), df_predictions["y_pred"].max())*1.1])
    
    fig.update_layout(
        legend_title=None,
        legend=dict(x=1.05, y=0.5),
        autosize=True,
        height=300,
        margin=dict(t=1, b=1, r=1, l=1)
    )
    
    return fig


__all__ = ['plot_forecast']