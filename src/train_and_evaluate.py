import os
import argparse
import json
import joblib
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn import metrics
from get_data import read_params


def eval_metrics(actual, pred):
    rmse = metrics.mean_squared_error(actual, pred, squared=False)
    mae = metrics.mean_absolute_error(actual, pred)
    r2 = metrics.r2_score(actual, pred)
    return rmse, mae, r2

def train_and_evaluate(config_path):
    # Get data
    config = read_params(config_path)
    train_path = config["split_data"]["train_path"]
    test_path =config["split_data"]["test_path"]
    target = config["base"]["target_col"]
    model_dir = config["model_dir"]
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
    rmse, mae, r2 = eval_metrics(y_test, preds)

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "alpha": alpha,
            "l1_ratio": l1_ratio,
        }
        json.dump(params, f, indent=4)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(elastic, model_path)
    
    # Save model and accuracy


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    train_and_evaluate(config_path=parsed_args.config)

