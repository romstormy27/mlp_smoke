from src.data_collection import int_to_float
from src.utils import load_config
import pandas as pd

def test_int_to_float():

    config = load_config()

    # arrange
    float_columns = config["float_columns"]
    mock_data = {k:[1] for k in float_columns}
    mock_df = pd.DataFrame(mock_data, columns=float_columns)

    # act

    print(mock_df)

    # assert

if __name__ == "__main__":

    test_int_to_float()