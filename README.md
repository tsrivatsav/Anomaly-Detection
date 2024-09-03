# Stock Market Anomaly Detection

This project aims to apply various anomaly detection techniques to detect outliers in stock market data. We specifically focus on Intel's (NASDAQ: INTC) stock performance from 1985 to 2014. 

## Anomaly Detection Techniques:
* Statistical Outlier Detection (z-score and mahalanobis distance)
* Isolation Forest
* One-Class SVM
* K-Means
* Local Outlier Factor
* Angle-Based Outlier Detection

This analysis could be extended to autoencoders, gaussian mixture models, ARIMA, LSTM, and Prophet. 

## Usage:
* Python 3.9 -> see `requirements.txt` for dependencies
* See `data_preprocessing.ipynb` for EDA
* Run each Jupyter notebook individually

## Data:
* The project uses historical stock market data obtained from Yahoo Finance using the yfinance python module