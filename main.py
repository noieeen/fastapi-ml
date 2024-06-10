from fastapi import FastAPI, Request
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from pydantic import BaseModel
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from typing import List

app = FastAPI()

class DataInput(BaseModel):
    XAxis: List[str]
    YAxis: List[float]

@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")

@app.get("/")
def root():
    message = { "message": "FastAPI Prediction!", "version":"0.0.1" }
    return message , 201
    
@app.post("/predict")
@cache(expire=60)  # Cache for 60 seconds
async def predict(data: DataInput):
    result = await time_series_prediction(data)
    result_dict = format_result(result)
    return result_dict

@app.post("/predict-prophet")
@cache(expire=60)  # Cache for 60 seconds
async def predict_prophet(data: DataInput):
    result = await time_series_prediction_prophet(data)
    print(result)
    # result_dict = format_result(result)
    return {"msg" : "true"}

async def time_series_prediction(data: DataInput):
    df = pd.DataFrame(data.dict())
    df['XAxis'] = pd.to_datetime(df['XAxis'])
    df.set_index('XAxis', inplace=True)

    model = SARIMAX(df['YAxis'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit(disp=False)

    forecast_steps = 12
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_df = forecast.conf_int()
    forecast_df['ForecastedCount'] = forecast.predicted_mean

    forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='MS')[1:]
    forecast_df.index = forecast_dates

    df_with_forecast = pd.concat([df, forecast_df[['ForecastedCount']]], axis=1)
    return df_with_forecast

async def time_series_prediction_prophet(data: DataInput):
    df = pd.DataFrame(data.dict())
    df['XAxis'] = pd.to_datetime(df['XAxis'])
    df.rename(columns={'XAxis': 'ds', 'YAxis': 'y'}, inplace=True)

    model = Prophet()
    model.fit(df)

    forecast_steps = 12
    future = model.make_future_dataframe(periods=forecast_steps, freq='MS')
    forecast = model.predict(future)

    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_df.rename(columns={'ds': 'Date', 'yhat': 'ForecastedCount'}, inplace=True)

    # Align forecast dates with the original dataframe
    forecast_dates = pd.date_range(start=df['ds'].iloc[-1], periods=forecast_steps + 1, freq='MS')[1:]
    forecast_df.set_index('Date', inplace=True)
    forecast_df = forecast_df.loc[forecast_dates]

    df.set_index('ds', inplace=True)
    df_with_forecast = pd.concat([df, forecast_df[['ForecastedCount']]], axis=1)

    return df_with_forecast

def format_result(df):
    return {
        "XAxis": df.index.strftime('%Y-%m-%d').tolist(),
        "YAxis": df['YAxis'].fillna(0).tolist(),
        "ForecastedCount": df['ForecastedCount'].fillna(0).tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
