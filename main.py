import time

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from configparser import ConfigParser

from utils import create_log_file, configure_logging
from preprocessing import (
    prepare_data,
    process_categorical,
    prepare_data_split,
    split_on_sensitive_attr,
)
from models import Classifier
from train import train_classifier
from nsga.main import run as run_nsga


def run(dataset, data_path, model_type):

    (df, features, label, categorical_features, sensitive_features) = prepare_data(
        dataset, data_path
    )
    X, y = prepare_data_with_categorical(df, features, label, categorical_features)
    (X_train, X_test, y_train, y_test) = prepare_data_split(X, y)

    # split_func = split_on_sensitive_attr(X_train)

    model = Classifier(
        dataset,
        model_type,
        X_train,
        y_train,
        X_test,
        y_test,
        features,
        sensitive_features,
    )
    model.fit()

    X_m = model.X_m
    X_f = model.X_f
    y_m = model.y_m
    y_f = model.y_f

    X_test_m = model.X_test_m
    X_test_f = model.X_test_f
    y_test_m = model.y_test_m
    y_test_f = model.y_test_f
    try:
        run_nsga(model_type, X_m, y_m, X_f, y_f, X_test_m, y_test_m, X_test_f, y_test_f)
    except Exception as e:
        pass


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Training script for MOEFC",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="compas",
        choices=["compas", "adult", "bank"],
        type=str,
        help="""Specify the data.
            Choices are compas, adult, and bank dataset.""",
    )
    parser.add_argument(
        "--model",
        default="lr",
        choices=["lr", "svm", "bayes"],
        type=str,
        help="""Specify the model.
            Choices are logisticRegression(lr), Support Vector Machines (svm), 
            and Bayes Classifier(bayes).""",
    )
    args = parser.parse_args()
    selected_data = args.dataset

    cfg_parser = ConfigParser()
    cfg_parser.read("config.ini")
    data_file_path = cfg_parser.get(selected_data, "data-path")
    log_dir = cfg_parser.get("logging", "log-directory")

    log_prefix = "{0}_{1}".format(selected_data, args.model)
    log_file = create_log_file(log_dir, log_prefix)
    logger = configure_logging("machine_bias", log_file)
    logger.info("---starting logging-----")

    start = time.time()
    run(dataset, data_file_path, args.model)
    elapsed = time.time() - start
    logger.info("Time taken: %.3f Hours" % (elapsed / 3600.0))
