import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

def load_and_process_data(file_path):
    # Load data
    data = pd.read_csv(file_path)
    
    # Convert date column to datetime
    data['date'] = pd.to_datetime(data['date'])

    # Extract year, month from the date
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month

    # Group data by month and sum the receipts
    monthly_data = data.groupby(['year', 'month'])['receipts'].sum().reset_index()

    # Normalize the data for better performance in machine learning models
    scaler = MinMaxScaler()
    monthly_data['receipts'] = scaler.fit_transform(monthly_data[['receipts']])
    
    return monthly_data, scaler
