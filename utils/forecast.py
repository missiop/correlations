from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def prophet_forecast(df, periods=30):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

def arima_forecast(data, periods=30, order=(5, 1, 0)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=periods)
    return predictions
