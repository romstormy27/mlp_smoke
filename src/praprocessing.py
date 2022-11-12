from sklearn.preprocessing import StandardScaler
import pandas as pd
import copy
from utils import load_config, load_pickle, dump_pickle
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

def std_scaler_fit(x_train: pd.DataFrame):

    std_scaler = StandardScaler()

    std_scaler.fit(x_train)

    return std_scaler

def std_scaler_transform(features, scaler):
    
    col_names = scaler.feature_names_in_

    feat = copy.deepcopy(features)

    scaled = scaler.transform(feat)

    scaled_df = pd.DataFrame(scaled, columns=col_names)

    return scaled_df

def rus_balancer(x_train, y_train):

    rus = RandomUnderSampler(random_state=42)

    x_rus, y_rus = rus.fit_resample(x_train, y_train)

    return x_rus, y_rus

def sm_balancer(x_train, y_train):

    sm = SMOTE(random_state=42)

    x_sm, y_sm = sm.fit_resample(x_train, y_train)

    return x_sm, y_sm

def main():

    config = load_config()

    x_train = load_pickle(config["train_set_path"]["x_train"])
    x_valid = load_pickle(config["valid_set_path"]["x_valid"])
    x_test = load_pickle(config["test_set_path"]["x_test"])

    y_train = load_pickle(config["train_set_path"]["y_train"])
    y_valid = load_pickle(config["valid_set_path"]["y_valid"])
    y_test = load_pickle(config["test_set_path"]["y_test"])

    # reset index
    x_train = x_train.reset_index(drop=True)
    x_valid = x_valid.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)

    y_train = y_train.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # standardizing
    scaler = std_scaler_fit(x_train)

    x_train_scaled = std_scaler_transform(x_train, scaler)
    x_valid_scaled = std_scaler_transform(x_valid, scaler)
    x_test_scaled = std_scaler_transform(x_test, scaler)

    # class balancer - rus
    x_rus, y_rus = rus_balancer(x_train_scaled, y_train)

    # class balancer - smote
    x_sm, y_sm = sm_balancer(x_train_scaled, y_train)

    # dump everything
    dump_pickle(x_rus, config["prep_rus_path"]["x_rus"])
    dump_pickle(y_rus, config["prep_rus_path"]["y_rus"])

    dump_pickle(x_sm, config["prep_sm_path"]["x_sm"])
    dump_pickle(y_sm, config["prep_sm_path"]["y_sm"])

    dump_pickle(x_valid_scaled, config["prep_valid_path"]["x_valid"])
    dump_pickle(x_test_scaled, config["prep_test_path"]["x_test"])

if __name__ == "__main__":

    main()

