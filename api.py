# importing utilities we are going to use
from fastapi import FastAPI, HTTPException
from prophet.serialize import model_from_json
import os
import uvicorn
import pandas as pd
import joblib
import json

# REFERENCES
CITIES = [
    "balikpapan",
    "bandung",
    "batam",
    "jakarta",
    "makassar",
    "medan",
    "palembang",
    "pekanbaru",
    "surabaya",
    "yogyakarta",
]
CUTOFF_DATE = "2022-11-18"

description = """
Cabe Merah Price Anomaly Detector for 10 City : 
1. balikpapan
2. bandung
3. batam 
4. jakarta
5. makassar
6. medan
7. palembang
8. pekanbaru
9. surabaya
10. yogyakarta

## Endpoint 
1. /nowcasting_price -> Get Anomaly Trigger only at exact date. 

"""
tags_metadata = [
    {
        "name": "nowcasting_price",
        "description": "Predicting whether the price change is outlier or not ",
    }
]

BASE_PROPHET_MODEL_PATH = "prophet_model"
BASE_ISOLATION_FOREST_MODEL_PATH = "isolation_forest_model"
PAST_DATA = "past_data/cities_dict.joblib"

app = FastAPI(
    title="Comodity Price Anomaly Detector",
    description=description,
    openapi_tags=tags_metadata,
)


def outlier_mapping(outlier_data, y_true):
    if outlier_data == -1 and y_true >= 0:
        return "outlier"
    else:
        return "inlier"


@app.get("/")
async def root():
    return {
        "message": "Head to endpoint /nowcasting_price to fetch forecast data or to /docs to see documentation"
    }


def compute_resid(y_true, y_hat, ds):
    # create empty dataframe to contain squared error of requested city
    error_container = pd.DataFrame()
    error_container["ds"] = ds
    error_container["ytrue"] = y_true
    error_container["squared_error"] = (y_true - y_hat) ** 2
    return error_container


@app.get("/past_outlier_data/{city}")
async def return_forecast(city: str):

    if city not in CITIES:
        raise HTTPException(
            status_code=404, detail=f"{city} not in choice. available choices {CITIES}"
        )
    # past_data
    # load dictionary contain data for each city
    data = joblib.load(PAST_DATA).get(city)["resid"]
    data = data[["ds", "ytrue", "outlier"]]

    data["outlier"] = data.apply(
        lambda x: outlier_mapping(x["outlier"], x["ytrue"]), axis=1
    )
    data["outlier"]
    pre_json_response = data.to_dict(orient="records")
    json_ = json.dumps(pre_json_response)
    return json_


@app.get("/nowcasting_price/{city}/{price_date}/{delta_price}")
async def return_forecast(city: str, price_date: str, delta_price: float):

    # validation
    if city not in CITIES:
        raise HTTPException(
            status_code=404, detail=f"{city} not in choice. available choices {CITIES}"
        )

    with open(
        os.path.join(BASE_PROPHET_MODEL_PATH, f"serialized_model_{city}.json"), "r"
    ) as model_final:
        model = model_from_json(model_final.read())  # Load Model
    # generate prediction

    # init empty df
    df = pd.DataFrame()
    df["cutoff_date"] = [CUTOFF_DATE]
    df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])
    REQ_DATE = price_date
    req_date = pd.DataFrame()
    req_date["request_date"] = [REQ_DATE]
    req_date["request_date"] = pd.to_datetime(req_date["request_date"])

    distance_from_cutoff = (
        req_date["request_date"] - df["cutoff_date"]
    ).dt.days.values.squeeze()
    # validation for date
    if distance_from_cutoff <= 0:
        raise HTTPException(
            status_code=404, detail="Minimal Start Date from 2022-11-19"
        )
    # generate prediction
    forecast_df = model.make_future_dataframe(periods=distance_from_cutoff)
    prediction = model.predict(forecast_df)
    latest_date = prediction.iloc[-1]
    # calculate squarred error
    residual = compute_resid(
        y_true=delta_price, y_hat=latest_date["yhat"], ds=req_date["request_date"]
    )

    # load isolation forest model
    ISOLATION_FOREST_MODEL = os.path.join(
        BASE_ISOLATION_FOREST_MODEL_PATH, f"{city}_isoforest.joblib"
    )
    isoforest_model = joblib.load(ISOLATION_FOREST_MODEL)
    # prediction
    outlier_prediction = isoforest_model.predict(
        residual["squared_error"].values.reshape(-1, 1)
    )

    text_outlier_result = (
        "outlier" if outlier_prediction == -1 and delta_price > 0 else "inlier"
    )
    # create dataframe contain prediction
    response_df = pd.DataFrame()
    response_df["date"] = [price_date]
    response_df["delta_price"] = [delta_price]
    response_df["city"] = [city]
    response_df["category"] = [text_outlier_result]
    # convert to json response
    pre_json_response = response_df.to_dict(orient="records")
    json_ = json.dumps(pre_json_response)
    return json_


# running the server
if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=5000, log_level="info")
