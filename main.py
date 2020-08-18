import time
import json
import numpy as np

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
from nsga.main import run as run_nsga
from nsga.config import NSGAConfig


np.random.seed(42)


def run(dataset, data_path, model_type, generations, populations):

    (df, features, label, categorical_features, sensitive_features) = prepare_data(
        dataset, data_path
    )
    X, y = process_categorical(df, features, label, categorical_features)
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

    nsga_cfg = NSGAConfig(
        generations=generations,
        populations=populations,
        model_type=model_type,
        X_sensitive_a1=model.X_m,
    )

    X_m = model.X_m
    X_f = model.X_f
    y_m = model.y_m
    y_f = model.y_f

    X_test_m = model.X_test_m
    X_test_f = model.X_test_f
    y_test_m = model.y_test_m
    y_test_f = model.y_test_f
    try:
        run_nsga(nsga_cfg)
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
    selected_model = args.model
    # selected_nsga_params = args.nsga

    cfg_parser = ConfigParser()
    cfg_parser.read("config.ini")

    data_file_path = cfg_parser.get(selected_data, "data-path")
    nsga_params = dict(cfg_parser.items("nsga"))
    generations = json.loads(nsga_params.get("generations", None))
    populations = json.loads(nsga_params.get("populations", None))
    log_dir = cfg_parser.get("logging", "log-directory")

    log_prefix = "{0}_{1}".format(selected_data, selected_model)
    log_file = create_log_file(log_dir, log_prefix)
    logger = configure_logging("machine_bias", log_file)
    logger.info("---starting run-----")

    start = time.time()
    run(selected_data, data_file_path, selected_model, generations, populations)
    elapsed = time.time() - start
    logger.info("Time taken: %.3f Hours" % (elapsed / 3600.0))
