from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import List

app = FastAPI()

class DataInput(BaseModel):
    XAxis: List[str]
    YAxis: List[float]

@app.post("/predict")
async def predict(data: DataInput):
    result = await time_series_prediction(data)
    result_dict = format_result(result)
    return result_dict

async def time_series_prediction(data: DataInput):
    df = pd.DataFrame(data.dict())
    df['XAxis'] = pd.to_datetime(df['XAxis'])
    df.set_index('XAxis', inplace=True)

    model = SARIMAX(df['YAxis'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit(disp=False)

    forecast_steps = 3
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_df = forecast.conf_int()
    forecast_df['ForecastedCount'] = forecast.predicted_mean

    forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='MS')[1:]
    forecast_df.index = forecast_dates

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
