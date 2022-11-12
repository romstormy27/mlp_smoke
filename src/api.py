from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np

import src.utils as utils
import src.data_collection as data_collection
import src.praprocessing as praprocessing


# read config
config = utils.load_config()

# load model
std_scaler = utils.load_pickle(config["scaler_path"])
prod_model = utils.load_pickle(config["production_model_path"])

# define input data
class InputData(BaseModel):

    temperature_c_: float
    humidity_percent_: float
    tvoc_ppb_: float
    eco2_ppm_: float
    raw_h2: float
    raw_ethanol: float
    pressure_hpa: float
    pm10: float
    pm25: float
    nc05: float
    nc10: float
    nc25: float

app = FastAPI()

# landing page
@app.get("/")
def home():
    return "Hello, FastAPI up!"

# prediction page
@app.post("/predict/")
def predict(data: InputData):    
    # Convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop=True)

    # # convert to float
    data = data_collection.int_to_float(data)

    # standard scaling
    data = praprocessing.std_scaler_transform(data, std_scaler)

    # Predict data
    y_pred = str(prod_model.predict(data))[1]

    return {"prediction" : y_pred}
