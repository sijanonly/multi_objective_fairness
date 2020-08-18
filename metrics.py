from sklearn.metrics import confusion_matrix as cm


def _get_ratio(v1, v2):
    denom = v1
    numer = v2
    if v1 < v2:
        denom = v2
        numer = v1
    ratio = numer / denom
    return ratio


def _calculate_confusion_matrix(labels, predictions):
    conf_mat = cm(labels, predictions)
    TP, FN, FP, TN = conf_mat.ravel()
    return TP, FN, FP, TN


class Fairness:
    def __init__(self, model, Xtrain_m, ytrain_m, Xtrain_f, ytrain_f):
        self.model = model
        self.Xtrain_m = Xtrain_m
        self.ytrain_m = ytrain_m
        self.Xtrain_f = Xtrain_f
        self.ytrain_f = ytrain_f

    @property
    def predictions_male(self):
        return self.model.predict(self.Xtrain_m)

    @property
    def predictions_female(self):
        return self.model.predict(self.Xtrain_f)

    def fairness_precision(self):
        TP_m, FN_m, FP_m, TN_m = _calculate_confusion_matrix(
            self.ytrain_m, self.predictions_male
        )
        precision_m = TP_m / (TP_m + FP_m)
        TP_f, FN_f, FP_f, TN_f = _calculate_confusion_matrix(
            self.ytrain_f, self.predictions_female
        )
        precision_f = TP_f / (TP_f + FP_f)
        return _get_ratio(precision_m, precision_f)

    def fairness_recall(self):
        TP_m, FN_m, FP_m, TN_m = _calculate_confusion_matrix(
            self.ytrain_m, self.predictions_male
        )
        recall_m = TP_m / (TP_m + FN_m)
        TP_f, FN_f, FP_f, TN_f = _calculate_confusion_matrix(
            self.ytrain_f, self.predictions_female
        )
        recall_f = TP_f / (TP_f + FN_f)
        return _get_ratio(recall_m, recall_f)

    def fairness_accuracy(self):
        male_accuracy = self.model.score(self.Xtrain_m, self.ytrain_m)
        female_accuracy = self.model.score(self.Xtrain_f, self.ytrain_f)
        return _get_ratio(male_accuracy, female_accuracy)

    def fairness_independence(self):
        positive_count_1 = list(self.predictions_male).count(1)
        positive_count_1_norm = positive_count_1 / len(self.predictions_male)
        positive_count_2 = list(self.predictions_female).count(1)
        positive_count_2_norm = positive_count_2 / len(self.predictions_female)
        return _get_ratio(positive_count_1, positive_count_2)

    def fairness_measures(self):
        return {
            "accuracy": self.fairness_accuracy(),
            "recall": self.fairness_recall(),
            "precision": self.fairness_precision(),
            "independence": self.fairness_independence(),
        }
