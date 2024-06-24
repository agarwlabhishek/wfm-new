import streamlit as st
from datetime import datetime, timedelta


def html_download_button(tab, html_buffer, fname):
        # Prepare download button for exporting the plot as an HTML file
        tab.download_button(
            label='Export Plot (.html)',
            data=html_buffer.getvalue().encode(),
            # Format the file name with the current date-time
            file_name=f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{fname}_Plot.html",
            mime='text/html',
            key=f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{fname}_Plot",
            type="primary"
        )
        

def csv_download_button(tab, data, fname):
    # Prepare download button for exporting the plot as an CSV file
    tab.download_button(
        label="Export Data (.csv)",
        data=data.to_csv(),
        # Format the file name with the current date-time
        file_name=f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{fname}_Data.csv",
        mime='text/csv',
        key=f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{fname}_Data",
        type="primary"
    )