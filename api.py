#importing utilities we are going to use
from fastapi import FastAPI

from enum import Enum
#pydantic to create request body
from pydantic import BaseModel
# typing.Optional to create requets body that is not mandatory with default value
from typing import Optional 
# we are going to use datetime manipulation package to create timestamp
from datetime import date
from dateutil import relativedelta
#loading trained forecast model
from prophet.serialize import  model_from_json

import os
#import uvicorn for server 
import uvicorn
#for data manipulation
import pandas as pd
import math 
import json

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
1. /get_past_outlier_data -> Get All Historical Data Not Including NA Values 
2. /nowcasting_price -> Get Anomaly Trigger only at exact date. 


#
"""

BASE_PROPHET_MODEL_PATH = 'prophet_model'
BASE_ISOLATION_FOREST_MODEL_PATH = 'isolation_forest_model'

app = FastAPI(title='Comodity Price Anomaly Detector',description=description)


@app.get("/")
async def root():
    return {"message": "Head to endpoint /nowcasting_price to fetch forecast data or to /docs to see documentation"}


# class City(str,Enum) : 
#     balikpapan = 'balikpapan'
#     bandung = 'bandung'
#     batam = 'batam'
#     jakarta = 'jakarta'
#     makassar = 'makassar'
#     medan = 'medan'
#     palembang = 'palembang'
#     pekanbaru = 'pekanbaru'
#     surabaya = 'surabaya'
#     yogyakarta = 'yogyakarta'

#creating request body for endpoint /timeseris_forecasting
#we use pydantic BaseModel
class api_request(BaseModel) : 
    #month_limit has following format YYYY-MM-01 , the forecast is monthly basis  
    city : str 
    
    price_date : str
    # show_all_data is optional and default falue is True
    delta_price : float
    #window_size is related with model rolling average number, i picked 12
    # since forecast is monthly basis with 12 months in a year
class past_data(BaseModel) : 
    #month_limit has following format YYYY-MM-01 , the forecast is monthly basis  
    city : str 
    
    # show_all_data is optional and default falue is True
    date_range : str 
    
def compute_resid(y_true,y_hat,ds) : 
  #create empty dataframe to contain squared error of requested city 
  error_container = pd.DataFrame()
  error_container['ds'] = ds
  error_container['ytrue'] = y_true
  error_container['squared_error'] = (y_true-y_hat) ** 2 
  return error_container
    

@app.post("/get_past_outlier_data")
async def get_past_outlier(req : past_data) : 
    city = req.city 
    



@app.get("/nowcasting_price/{city}/{price_date}/{delta_price}") 
async def return_forecast(city : str,price_date:str,delta_price:float) : 
    #load city prophet model 

    with open(os.path.join(BASE_PROPHET_MODEL_PATH,f'serialized_model_{city}.json'), 'r') as model_final:
        model = model_from_json(model_final.read())  # Load Model 
    #generate prediction 
    CUTOFF_DATE = '2022-11-18'
    #init empty df 
    df = pd.DataFrame()
    df['cutoff_date'] = [CUTOFF_DATE]
    df['cutoff_date']  = pd.to_datetime(df['cutoff_date'] )
    REQ_DATE = price_date
    req_date = pd.DataFrame()
    req_date['request_date'] = [REQ_DATE]
    req_date['request_date']  = pd.to_datetime(req_date['request_date'] )
    
    distance_from_cutoff =  (req_date['request_date'] - df['cutoff_date']).dt.days.values.squeeze()
    forecast_df = model.make_future_dataframe(periods=distance_from_cutoff)
    prediction = model.predict(forecast_df)
    latest_date = prediction.iloc[-1]
    #calculate squarred error 
    residual = compute_resid(y_true=delta_price,y_hat=latest_date['yhat'],ds=req_date['request_date'])
    
    #load isolation forest model 
    ISOLATION_FOREST_MODEL = os.path.join(BASE_ISOLATION_FOREST_MODEL_PATH,f'{city}_isoforest.joblib')
    isoforest_model = joblib.load(ISOLATION_FOREST_MODEL)
    outlier_prediction = isoforest_model.predict(residual['squared_error'].values.reshape(-1, 1))
    
    text_outlier_result = 'outlier' if outlier_prediction==-1 else 'inlier'
    
    response_df = pd.DataFrame()
    response_df['date'] = [price_date] 
    response_df['delta_price']  = [delta_price]
    response_df['city'] = [city] 
    response_df['category'] = [text_outlier_result]
    #
    
    print(response_df)
    
    
   
    
    
    
    
    pre_json_response = response_df.to_dict(orient='records')
    json_ = json.dumps(pre_json_response)
    return json_
    # json_compatible_item_data = jsonable_encoder(output_before_json)
    # return JSONResponse(content=json_compatible_item_data)
    #start creating figure 
    
# running the server
if __name__ == '__main__' : 
    uvicorn.run(app=app,host="127.0.0.1", port=5000, log_level="info")