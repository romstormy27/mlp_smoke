import pandas as pd
import copy

from sklearn.model_selection import train_test_split

from utils import load_config, load_pickle, dump_pickle

config = load_config()

def read_raw_df(raw_data_path: str)->pd.DataFrame:

    df = pd.read_csv(raw_data_path, index_col=0)

    # drop utc and cnt columns
    df.drop(config["unwanted_columns"], axis=1, inplace=True)

    # change columns name
    df.columns = config["new_columns_name"]

    return df

def int_to_float(dataframe: pd.DataFrame)->pd.DataFrame:
    
    df = copy.deepcopy(dataframe)

    df[config["float_columns"]] = df[config["float_columns"]].astype("float64")

    return df

def split_data(dataframe: pd.DataFrame) -> tuple:

    df = copy.deepcopy(dataframe)

    X = df[config["predictors"]]
    y = df[config["label"]]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)

    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test, random_state=42)

    return x_train, x_valid, x_test, y_train, y_valid, y_test

def main():
    df = read_raw_df(config["raw_dataset_path"])

    dump_pickle(df, config["raw_df_path"])

    df = int_to_float(df)

    x_train, x_valid, x_test, y_train, y_valid, y_test = split_data(df)

    dump_pickle(x_train, config["train_set_path"]["x_train"])
    dump_pickle(y_train, config["train_set_path"]["y_train"])

    dump_pickle(x_valid, config["valid_set_path"]["x_valid"])
    dump_pickle(y_valid, config["valid_set_path"]["y_valid"])

    dump_pickle(x_test, config["test_set_path"]["x_test"])
    dump_pickle(y_test, config["test_set_path"]["y_test"])


if __name__ == "__main__":

    main()

    
