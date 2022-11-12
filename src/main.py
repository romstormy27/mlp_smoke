import data_collection
import praprocessing
import modelling

import pandas as pd

from utils import load_config, load_pickle

from sklearn.metrics import roc_auc_score

def train():

    data_collection.main()

    praprocessing.main()

    modelling.main()


if __name__ == "__main__":

    config = load_config()

    # train()

    model = load_pickle(config["production_model_path"])

    scaler = load_pickle(config["scaler_path"])

    # x_valid = load_pickle(config["prep_valid_path"]["x_valid"])
    # y_valid = load_pickle(config["prep_valid_path"]["y_valid"])

    x_valid = {
    "temperature_c_": 0,
    "humidity_percent_": 0,
    "tvoc_ppb_": 0,
    "eco2_ppm_": 0,
    "raw_h2": 0,
    "raw_ethanol": 0,
    "pressure_hpa": 0,
    "pm10": 0,
    "pm25": 0,
    "nc05": 0,
    "nc10": 0,
    "nc25": 0
    }

    x_valid = pd.DataFrame(x_valid, index=[0])

    x_valid = data_collection.int_to_float(x_valid)

    x_valid = praprocessing.std_scaler_transform(x_valid, scaler)

    pred = model.predict(x_valid)

    print(pred)



