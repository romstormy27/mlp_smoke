from src.data_collection import int_to_float, split_data
from src.utils import load_config

import pandas as pd
import numpy as np

np.random.seed(90)

def test_int_to_float():

    config = load_config()

    # arrange
    ## grab float columns from config file
    float_columns = config["float_columns"]
    
    ## make mock data with columns == float columns and value are 1 which is int
    mock_data = {k:[1] for k in float_columns}

    ## convert into dataframe
    mock_df = pd.DataFrame(mock_data, columns=float_columns)

    # act
    mock_df = int_to_float(mock_df)

    # assert
    for col in mock_df.columns:
        assert mock_df[col].dtypes == "float64"

def test_split_data():

    config = load_config()
    X_columns = config["predictors"]
    y_columns = config["label"]

    # arrange
    ## make data with predictors and target column and 10 rows
    mock_X = {k:[i for i in range(10)] for k in X_columns}
    mock_y = [i for i in np.random.randint(0,2,10)]
    mock_X = pd.DataFrame(mock_X)
    mock_y = pd.DataFrame(mock_y, columns=y_columns)
    mock_df = pd.DataFrame(pd.concat([mock_X, mock_y], axis=1))

    # act
    x_train, _, _, _, _, _ = split_data(mock_df)

    # assert
    assert x_train.shape[0] == 6