import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

def forecast_with_model(historical_data, model_choice):
    """
    Perform forecasting using the specified model (Prophet or ARIMA).
    """
    if not historical_data:
        raise ValueError("Historical data is empty or missing.")

    # Convert historical data to DataFrame
    try:
        df = pd.DataFrame.from_dict(historical_data, orient="index")
        df.reset_index(inplace=True)
        df.rename(columns={"index": "date", "5. adjusted close": "value"}, inplace=True)

        # Ensure the 'date' column is in datetime format
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        raise ValueError(f"Error processing historical data: {e}")

    # Prepare data for forecasting
    df = df[["date", "value"]]
    df.rename(columns={"date": "ds", "value": "y"}, inplace=True)

    # Apply the selected model
    if model_choice == "Prophet":
        from fbprophet import Prophet
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=30)  # 30 days forecast
        forecast = model.predict(future)
        return forecast[["ds", "yhat"]]
    elif model_choice == "ARIMA":
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(df["y"], order=(5, 1, 0))  # Example ARIMA(p,d,q) order
        model_fit = model.fit()
        forecast = model_fit.get_forecast(steps=30)
        forecast_df = pd.DataFrame({
            "ds": pd.date_range(start=df["ds"].iloc[-1], periods=30, freq="D"),
            "yhat": forecast.predicted_mean
        })
        return forecast_df
    else:
        raise ValueError("Unsupported model choice. Choose 'Prophet' or 'ARIMA'.")
