from sklearn.metrics import confusion_matrix as cm
from nsga.utils import weighted_sum, prepare_normalized_weighted_sum, accuracy, error


def objective_accuracy(W, X1, X2, X1_labels, X2_labels):
    """
    
    W : weight matrix,
    X1 : prediction by m1
    X2 : prediction by m2
    X1_label : actual label for instance1
    X2_label : actual label for instance2
    
    abs(1-ratio) ?
    
    Objective : 1 - ratio
    
    normalized weights here as well.
    
    check which one is small and use that as numerator.
    """
    weighted_sum_m_norm, weighted_sum_f_norm = prepare_normalized_weighted_sum(
        W, X1, X2
    )

    acc1 = accuracy(weighted_sum_m_norm, X1_labels)
    acc2 = accuracy(weighted_sum_f_norm, X2_labels)
    denom = acc1
    numer = acc2
    if acc1 < acc2:
        denom = acc2
        numer = acc1

    ratio = numer / denom

    return -(1 - ratio)


def objective_error(W, X1, X2, X1_label, X2_label):
    """
    Calculates prediction error
    
    each instance will have different prediction and different associated label
    
    Can we combine the error ? (error1+error2)/2 ?
    
    Error : normalized by weights
    """

    weighted_sum_m_norm, weighted_sum_f_norm = prepare_normalized_weighted_sum(
        W, X1, X2
    )
    combined_weighted_sum = np.hstack((weighted_sum_m_norm, weighted_sum_f_norm))
    combined_labels = np.hstack((X1_label, X2_label))
    combined_error = error(combined_weighted_sum, combined_labels)

    return -combined_error


def objective_precision(W, X1, X2, X1_label, X2_label):
    """
    Fairness objective based on precision
    """
    weighted_sum_m_norm, weighted_sum_f_norm = prepare_normalized_weighted_sum(
        W, X1, X2
    )

    predictions1 = [1 if w >= 0.5 else 0 for w in weighted_sum_m_norm]
    conf_mat1 = cm(X1_label, predictions1)
    TP1, FN1, FP1, TN1 = conf_mat1.ravel()
    precision1 = TP1 / (TP1 + FP1)

    predictions2 = [1 if w >= 0.5 else 0 for w in weighted_sum_f_norm]

    conf_mat2 = cm(X2_label, predictions2)

    TP2, FN2, FP2, TN2 = conf_mat2.ravel()
    precision2 = TP2 / (TP2 + FP2)

    ratio = get_ratio(precision1, precision2)

    return -(1 - ratio)


def objective_recall(W, X1, X2, X1_label, X2_label):
    """
    Fairness objective based on recall
    """
    weighted_sum_m_norm, weighted_sum_f_norm = prepare_normalized_weighted_sum(
        W, X1, X2
    )

    predictions1 = [1 if w >= 0.5 else 0 for w in weighted_sum_m_norm]
    conf_mat1 = cm(X1_label, predictions1)
    TP1, FN1, FP1, TN1 = conf_mat1.ravel()
    recall1 = TP1 / (TP1 + FN1)

    predictions2 = [1 if w >= 0.5 else 0 for w in weighted_sum_f_norm]

    conf_mat2 = cm(X2_label, predictions2)

    TP2, FN2, FP2, TN2 = conf_mat2.ravel()
    recall2 = TP2 / (TP2 + FN2)

    ratio = get_ratio(recall1, recall2)

    return -(1 - ratio)


def objective_independence(W, X1, X2, X1_label, X2_label):
    """
    Fairness objective based on independence
    """
    # combined effect
    weighted_sum_m_norm, weighted_sum_f_norm = prepare_normalized_weighted_sum(
        W, X1, X2
    )
    predictions1 = [1 if w >= 0.5 else 0 for w in weighted_sum_m_norm]

    positive_count_1 = predictions1.count(1)

    predictions2 = [1 if w >= 0.5 else 0 for w in weighted_sum_f_norm]

    positive_count_2 = predictions2.count(1)
    positive_count_1_norm = positive_count_1 / len(predictions1)
    positive_count_2_norm = positive_count_2 / len(predictions2)
    ratio = get_ratio(positive_count_1_norm, positive_count_2_norm)

    return -(1 - ratio)
