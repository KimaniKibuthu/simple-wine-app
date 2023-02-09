import os
import argparse
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn import metrics
from get_data import read_params

def train_and_evaluate(config_path):
    # Get data
    config = read_params(config_path)
    train_path = config["split_data"]["train_path"]
    test_path =config["split_data"]["test_path"]
    target = config["base"]["target_col"]
    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]


    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    x_train = train_data.drop(f"{target}", axis=1)
    y_train = train_data[f"{target}"]

    x_test = test_data.drop(f"{target}", axis=1)
    y_test = test_data[f"{target}"]

    # Train and evaluate model
    elastic = ElasticNet(l1_ratio=l1_ratio, alpha=alpha)
    elastic.fit(x_train, y_train)
    preds = elastic.predict(x_test)
    mae = metrics.mean_absolute_error(y_test, preds)
    print(mae)

    # Save model and accuracy


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    train_and_evaluate(config_path=parsed_args.config)

