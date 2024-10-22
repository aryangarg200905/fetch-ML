# Fetch Rewards Take-home Exercise - Machine Learning Engineer

## Project Overview

This project involves building a machine learning model from scratch to predict the number of scanned receipts for each month of 2022, based on daily receipt data from 2021. The solution includes:

- **Data Processing**: Preprocessing daily data and aggregating it by month.
- **Modeling**: A Long Short-Term Memory (LSTM) neural network for time series forecasting.
- **Prediction**: Using the trained model to forecast the number of receipts for 2022.
- **Web App**: A Flask-based web app providing an API to retrieve the predictions.
- **Containerization**: The project is packaged in a Docker container for easy deployment.

## Project Structure

```bash
.
├── data_processing.py     # Handles data loading and preprocessing
├── model.py               # Defines and trains the ML model
├── predict.py             # Contains prediction logic
├── app.py                 # Flask app serving the prediction API
├── Dockerfile             # Docker configuration for containerizing the app
└── data_daily.csv         # CSV file containing daily receipt data for 2021
