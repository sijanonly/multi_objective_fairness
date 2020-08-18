import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def prepare_data(dataset, data_path):

    if dataset.strip().lower() == "bank":
        df = pd.read_csv("bank/Dataset.csv")
        columns = list(df.columns)
        columns = [
            col for col in columns if col not in ["RowNumber", "CustomerId", "Surname"]
        ]
        # categorical fields
        categorical_features = ["Gender", "Geography"]
        continous_features = [
            "CreditScore",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
        ]

        sensitive_features = ["Gender"]

        label = "Exited"
        features = list(set(categorical_features + continous_features))

    if dataset.strip().lower() == "adult":

        def convert_workclass(value):
            for key, values in workclass_mapper.items():
                if value in values:
                    return key

        df = pd.read_csv("adult/adult.csv")

        # drop entries having '?' in any column values
        df = df[~(df == "?").any(axis=1)]

        df["income"] = df["income"].replace("<=50K", 0)
        df["income"] = df["income"].replace(">50K", 1)

        workclass_mapper = {
            "private": ["Private"],
            "gov": ["Local-gov", "Federal-gov", "State-gov"],
            "self": ["Self-emp-inc", "Self-emp-not-inc"],
            "umemploy": ["Never-worked"],
        }

        df["workclass"] = df["workclass"].apply(convert_workclass)

        df.replace(
            [
                "Divorced",
                "Married-AF-spouse",
                "Married-civ-spouse",
                "Married-spouse-absent",
                "Never-married",
                "Separated",
                "Widowed",
            ],
            [
                "unmarried",
                "married",
                "married",
                "married",
                "unmarried",
                "unmarried",
                "unmarried",
            ],
            inplace=True,
        )

        # categorical fields
        categorical_features = [
            "workclass",
            "race",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "gender",
            "native-country",
        ]
        continous_features = [
            "age",
            "fnlwgt",
            "educational-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]

        sensitive_features = ["gender", "race"]

        label = "income"
        features = list(set(categorical_features + continous_features))

    if dataset.strip().lower() == "compas":
        df3 = pd.read_csv(data_path)

        # convert object to datetime object
        df3[["dob", "c_jail_in", "c_jail_out"]] = df3[
            ["dob", "c_jail_in", "c_jail_out"]
        ].apply(pd.to_datetime)

        # age when the person was in jail : factor based on c_jail_in and dob
        df3["charge_age"] = df3["c_jail_in"].sub(df3["dob"], axis=0) / np.timedelta64(
            1, "Y"
        )

        # we saw that c_jail_in is null for rows, let's remove them
        df3 = df3[~df3.c_jail_in.isnull()]

        df3["charge_age"] = df3["charge_age"].apply(np.floor).astype(int)
        # total number of days the person spent on jail, which

        df3["jail_time"] = df3["c_jail_out"].sub(
            df3["c_jail_in"], axis=0
        ) / np.timedelta64(1, "h")

        # convert jail_time hours to total days : if hrs > 23, 1 day
        def convert_hrs_day(hours):
            if hours < 24:
                return 1
            else:
                return math.ceil(hours // 24)

        df3["jail_time"] = df3["jail_time"].apply(convert_hrs_day)

        # categorical fields
        categorical_features = ["c_charge_degree", "sex", "race"]
        # total attributes for analysis
        continous_features = [
            "age",
            "juv_fel_count",
            "priors_count",
            "charge_age",
            "jail_time",
            "juv_misd_count",
        ]
        sensitive_features = ["sex", "race"]
        label = "is_recid"
        features = list(set(categorical_features + continous_features))

        # filter Native American only
        df_race_white_1 = df3[df3["race"] == "Native American"]
        # filter Caucasian only
        df_race_white_2 = df3[df3["race"] == "Caucasian"]

        # filter only 'African-American'
        df_race_black = df3[df3["race"] == "African-American"]

        # combine two dfs
        df_combine = pd.concat(
            [df_race_white_1, df_race_white_2, df_race_black], ignore_index=True
        )

        # replace both races to name 'white'
        df_combine["race"] = df_combine["race"].replace(
            ["Native American", "Caucasian"], "White"
        )
        df_combine["race"] = df_combine["race"].replace(["African-American"], "Black")

        df = df_combine

    return df, features, label, categorical_features, sensitive_features


def process_categorical(df, features, label, categorical_features):
    X = df[features]
    y = df[label]

    # drop_first : if 4 categories, 3 encoding is enough, 100, 010, 001,
    # and one is covered by 000
    X = pd.get_dummies(X, columns=categorical_features, drop_first=False)

    return X, y


def prepare_data_split(X, y):
    scaler = MinMaxScaler()

    # X = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    # 70% for train and 30% for test also define stratified on target value
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42
    )
    normalized_train_X = pd.DataFrame(
        scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns
    )

    normalized_test_X = pd.DataFrame(
        scaler.transform(X_test), index=X_test.index, columns=X_test.columns
    )

    return normalized_train_X, normalized_test_X, y_train, y_test


def _get_split_index(dataset, X):
    if dataset.strip().lower() == "adult":
        attr1_indexes = X.index[X["gender_Female"] == 1].tolist()
        attr2_indexes = X.index[X["gender_Male"] == 1].tolist()
    elif dataset.strip().lower() == "compas":
        attr1_indexes = X.index[X["race_White"] == 0].tolist()
        attr2_indexes = X.index[X["race_White"] == 1].tolist()

    else:
        attr1_indexes = X.index[X["Gender_Female"] == 1].tolist()
        attr2_indexes = X.index[X["Gender_Male"] == 1].tolist()

    return attr1_indexes, attr2_indexes


def split_on_sensitive_attr(dataset, X, y):

    attr1_idx, attr2_idx = _get_split_index(dataset, X)

    X_1 = X[X.index.isin(attr1_idx)]
    y_1 = y[y.index.isin(attr1_idx)]

    X_2 = X[X.index.isin(attr2_idx)]
    y_2 = y[y.index.isin(attr2_idx)]

    return X_1, y_1, X_2, y_2
