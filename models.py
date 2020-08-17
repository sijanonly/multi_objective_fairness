import random
import numpy as np
import logging

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB

from preprocessing import split_on_sensitive_attr
from metrics import Fairness

logger = logging.getLogger(__name__)

CLASSIFIER_TYPE = {
    "svm": SVC(kernel="linear", class_weight="balanced", probability=True),
    "lr": LogisticRegression(C=1.0, class_weight="balanced", n_jobs=-1),
    "bayes": GaussianNB(),
}

N_MODELS = 12


def _remove_features(data, features):
    for feature in features:
        try:
            data = data.loc[:, ~data.columns.str.startswith(feature)]
        except Exception as e:
            pass
    return data


def _prepare_probabilities(prediction, actual):
    """
    prediction : given by model has format ([[prob(0), prob(1)],[prob(0), prob(1)]... ])
    """

    predictions = []
    for idx, item in enumerate(actual):
        predictions.append(prediction[idx][1])
    return predictions


def _prepare_feature_subsample(features, size):
    return random.sample(features, k=size)


class Classifier:
    def __init__(
        self,
        dataset,
        model_type,
        X_train,
        y_train,
        X_test,
        y_test,
        features,
        sensitive_features,
    ):
        self.dataset = dataset
        self.model_type = model_type
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.features = features
        self.sensitive_features = sensitive_features

        self._prepare_train_test()

        self.model = CLASSIFIER_TYPE.get(self.model_type)

    def _prepare_train_test(self):
        self.Xtrain_f, self.ytrain_f, self.Xtrain_m, self.ytrain_m = split_on_sensitive_attr(
            self.X_train, self.y_train, self.dataset
        )
        self.Xtest_f, self.ytest_f, self.Xtest_m, self.ytest_m = split_on_sensitive_attr(
            self.X_test, self.y_test, self.dataset
        )

    def prepare_test_probabilities(self, model, removal_features):
        Xtest_m = _remove_features(self.Xtest_m[:], removal_features)
        Xtest_f = _remove_features(self.Xtest_f[:], removal_features)

        clf_0_male_prob = model.predict_proba(Xtest_m)
        clf_0_male_prob = _prepare_probabilities(clf_0_male_prob, self.ytest_m)

        clf_0_female_prob = model.predict_proba(Xtest_f)
        clf_0_female_prob = _prepare_probabilities(clf_0_female_prob, self.ytest_f)

        return {
            "male_prediction": clf_0_male_prob,
            "male_labels": self.ytest_m.tolist(),
            "female_prediction": clf_0_female_prob,
            "female_labels": self.ytest_f.tolist(),
        }

    def _prepare_classifier(self, removal_features):

        clf_0 = self.model

        # remove sensitive attrributes from train data, NOTE : also need to remove from X_test
        X_train = _remove_features(self.X_train[:], removal_features)

        clf_0.fit(X_train, self.y_train)

        # remove sensitive features from Xtrain_m and Xtrain_f
        train_accuracy = clf_0.score(X_train, self.y_train)

        Xtrain_m = _remove_features(self.Xtrain_m[:], removal_features)
        Xtrain_f = _remove_features(self.Xtrain_f[:], removal_features)
        clf_0_male_prob = clf_0.predict_proba(Xtrain_m)
        clf_0_male_prob = _prepare_probabilities(clf_0_male_prob, self.ytrain_m)

        clf_0_female_prob = clf_0.predict_proba(Xtrain_f)
        clf_0_female_prob = _prepare_probabilities(clf_0_female_prob, self.ytrain_f)

        male_accuracy = clf_0.score(Xtrain_m, self.ytrain_m)
        female_accuracy = clf_0.score(Xtrain_f, self.ytrain_f)

        train_fairness = Fairness(
            clf_0, Xtrain_m, self.ytrain_m, Xtrain_f, self.ytrain_f
        ).fairness_measures()

        # test probabilities
        test_prob = self.prepare_test_probabilities(clf_0, removal_features)

        model = {
            "male_prediction": clf_0_male_prob,
            "male_labels": self.ytrain_m.tolist(),
            "female_prediction": clf_0_female_prob,
            "female_labels": self.ytrain_f.tolist(),
            "model": clf_0,
            "train_accuracy": train_accuracy,
            "train_fairness": train_fairness,
            "test_data": test_prob,
        }
        return model

    def _main_classifier(self):
        logger.info("......starting main classifer.....")
        # without sensitive features
        m0 = self._prepare_classifier(self.sensitive_features)

        # with all features
        m1 = self._prepare_classifier([])

        # model only female :
        m2 = self._prepare_classifier(["Gender_Male"])

        # model only male :
        m3 = self._prepare_classifier(["Gender_Female"])

        logger.info("......ending main classifer.....")

        return m0, m1, m2, m3

    def prepare_bagging_classifier(self, model):
        logger.info("......starting bagging classifer.....")
        clf_1_male_prob = model.predict_proba(self.Xtrain_m)
        clf_1_male_prob = _prepare_probabilities(clf_1_male_prob, self.ytrain_m)

        clf_1_female_prob = model.predict_proba(self.Xtrain_f)
        clf_1_female_prob = _prepare_probabilities(clf_1_female_prob, self.ytrain_f)

        male_accuracy = model.score(self.Xtrain_m, self.ytrain_m)
        female_accuracy = model.score(self.Xtrain_f, self.ytrain_f)

        train_fairness = Fairness(
            model, self.Xtrain_m, self.ytrain_m, self.Xtrain_f, self.ytrain_f
        ).fairness_measures()

        # test probabilities
        clf_0_male_prob = model.predict_proba(self.Xtest_m)
        clf_0_male_prob = _prepare_probabilities(clf_0_male_prob, self.ytest_m)

        clf_0_female_prob = model.predict_proba(self.Xtest_f)
        clf_0_female_prob = _prepare_probabilities(clf_0_female_prob, self.ytest_f)

        test_prob = {
            "male_prediction": clf_0_male_prob,
            "male_labels": self.ytest_m.tolist(),
            "female_prediction": clf_0_female_prob,
            "female_labels": self.ytest_f.tolist(),
        }

        model = {
            "male_prediction": clf_1_male_prob,
            "male_labels": self.ytrain_m.tolist(),
            "female_prediction": clf_1_female_prob,
            "female_labels": self.ytrain_f.tolist(),
            "model": model,
            "train_fairness": train_fairness,
            "test_data": test_prob,
        }
        logger.info("......ending bagging classifer.....")
        return model

    def _bagging_classifiers(self):
        """models with subsets of the training set"""

        # TODO : changed to self.model
        clf = BaggingClassifier(
            self.model,
            n_estimators=4,
            bootstrap=True,
            n_jobs=-1,
            random_state=123,
            max_samples=1 / 2.0,
            oob_score=True,
        )

        clf.fit(self.X_train, self.y_train)
        #         print('Accuracy of Bagging classifier on training set: {:.2f}'.format(clf.score(self.X_train, self.y_train)))

        model1 = clf.estimators_[0]
        model2 = clf.estimators_[1]
        model3 = clf.estimators_[2]
        model4 = clf.estimators_[3]

        m0 = self.prepare_bagging_classifier(model1)
        m0["train_accuracy"] = model1.score(self.X_train, self.y_train)

        m1 = self.prepare_bagging_classifier(model2)
        m1["train_accuracy"] = model2.score(self.X_train, self.y_train)

        m2 = self.prepare_bagging_classifier(model3)
        m2["train_accuracy"] = model3.score(self.X_train, self.y_train)

        m3 = self.prepare_bagging_classifier(model4)
        m3["train_accuracy"] = model4.score(self.X_train, self.y_train)

        return m0, m1, m2, m3

    def _prepare_featuresubsample_classifier(self, selected_features):

        neglected_features = [
            feature for feature in self.features if feature not in selected_features
        ]
        m0 = self._prepare_classifier(neglected_features)
        return m0

    def _featuresubsample_classifier(self):
        logger.info("......starting feature subsample classifer.....")
        feature1 = _prepare_feature_subsample(self.features, size=6)
        feature2 = _prepare_feature_subsample(self.features, size=7)
        feature3 = _prepare_feature_subsample(self.features, size=8)
        feature4 = _prepare_feature_subsample(self.features, size=9)

        m8 = self._prepare_featuresubsample_classifier(feature1)
        m9 = self._prepare_featuresubsample_classifier(feature2)
        m10 = self._prepare_featuresubsample_classifier(feature3)
        m11 = self._prepare_featuresubsample_classifier(feature4)
        logger.info("......ending feature subsample classifer.....")
        return m8, m9, m10, m11

    def fit(self):
        logger.info("inside prepare ouput.....")
        overall_accuracy = 0
        fairness_accuracy = 0
        fairness_recall = 0
        fairness_precision = 0
        fairness_independence = 0
        prediction_input_m = []
        prediction_input_f = []
        prediction_input_test_m = []
        prediction_input_test_f = []

        m0, m1, m2, m3 = self._main_classifier()
        m4, m5, m6, m7 = self._bagging_classifiers()
        m8, m9, m10, m11 = self._featuresubsample_classifier()

        for i in range(N_MODELS):
            column_str = "%s%s" % ("m", i)
            col_var = eval(column_str)

            prediction_input_m.append(np.array(col_var["male_prediction"]))
            prediction_input_f.append(np.array(col_var["female_prediction"]))
            train_accuracy = col_var["train_accuracy"]
            overall_accuracy += train_accuracy
            fairness = col_var["train_fairness"]
            fairness_accuracy += fairness["accuracy"]
            fairness_recall += fairness["recall"]
            fairness_precision += fairness["precision"]
            fairness_independence += fairness["independence"]

            test_data = col_var["test_data"]
            prediction_input_test_m.append(np.array(test_data["male_prediction"]))
            prediction_input_test_f.append(np.array(test_data["female_prediction"]))

        self.model_output = {}
        self.model_output["train_overall_acc"] = overall_accuracy / float(N_MODELS)
        self.model_output["train_fairness_acc"] = fairness_accuracy / float(N_MODELS)
        self.model_output["train_fairness_recall"] = fairness_recall / float(N_MODELS)
        self.model_output["train_fairness_precision"] = fairness_precision / float(
            N_MODELS
        )
        self.model_output[
            "train_fairness_independence"
        ] = fairness_independence / float(N_MODELS)

        print(
            "Train fairness : independence: {}".format(
                fairness_independence / float(N_MODELS)
            )
        )

        self.X_m = np.column_stack((prediction_input_m))
        self.X_f = np.column_stack((prediction_input_f))
        self.y_m = np.array(m1["male_labels"])
        self.y_f = np.array(m1["female_labels"])

        self.X_test_m = np.column_stack((prediction_input_test_m))
        self.X_test_f = np.column_stack((prediction_input_test_f))
        self.y_test_m = np.array(m1["test_data"]["male_labels"])
        self.y_test_f = np.array(m1["test_data"]["female_labels"])

        logger.info("prepare output completes....")

