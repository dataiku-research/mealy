# -*- coding: utf-8 -*-
from mealy.constants import ErrorAnalyzerConstants
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np


def compute_confidence_decision(primary_model_true_accuracy, primary_model_predicted_accuracy):
    difference_true_pred_accuracy = np.abs(primary_model_true_accuracy - primary_model_predicted_accuracy)
    decision = difference_true_pred_accuracy <= ErrorAnalyzerConstants.MPP_ACCURACY_TOLERANCE

    fidelity = 1. - difference_true_pred_accuracy

    # TODO Binomial test
    return fidelity, decision


def compute_mpp_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def compute_primary_model_accuracy(y):
    n_test_samples = y.shape[0]
    return float(np.count_nonzero(y == ErrorAnalyzerConstants.CORRECT_PREDICTION)) / n_test_samples


def fidelity_score(y_true, y_pred):
    difference_true_pred_accuracy = np.abs(compute_primary_model_accuracy(y_true) -
                                           compute_primary_model_accuracy(y_pred))
    fidelity = 1. - difference_true_pred_accuracy

    return fidelity


def fidelity_balanced_accuracy_score(y_true, y_pred):
    return fidelity_score(y_true, y_pred) + balanced_accuracy_score(y_true, y_pred)


def mpp_report(y_true, y_pred, output_dict=False):
    """Build a text report showing the main Model Performance Predictor (MPP) metrics.

    Args:
        y_true (numpy.ndarray): Ground truth values of wrong/correct predictions of the MPP primary model.
            Expected values in [ErrorAnalyzerConstants.WRONG_PREDICTION, ErrorAnalyzerConstants.CORRECT_PREDICTION].
        y_pred (numpy.ndarray): Estimated targets as returned by a Model Performance Predictor. Expected values in
            [ErrorAnalyzerConstants.WRONG_PREDICTION, ErrorAnalyzerConstants.CORRECT_PREDICTION].
        output_dict (bool): If True, return output as dict (default = False).

    Return:
        dict or str: metrics regarding the Model Performance Predictor.
    """

    mpp_accuracy_score = compute_mpp_accuracy(y_true, y_pred)
    mpp_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    primary_model_predicted_accuracy = compute_primary_model_accuracy(y_pred)
    primary_model_true_accuracy = compute_primary_model_accuracy(y_true)
    fidelity, confidence_decision = compute_confidence_decision(primary_model_true_accuracy,
                                                                primary_model_predicted_accuracy)
    if output_dict:
        report_dict = dict()
        report_dict[ErrorAnalyzerConstants.MPP_ACCURACY] = mpp_accuracy_score
        report_dict[ErrorAnalyzerConstants.MPP_BALANCED_ACCURACY] = mpp_balanced_accuracy
        report_dict[ErrorAnalyzerConstants.MPP_FIDELITY] = fidelity
        report_dict[ErrorAnalyzerConstants.PRIMARY_MODEL_TRUE_ACCURACY] = primary_model_true_accuracy
        report_dict[ErrorAnalyzerConstants.PRIMARY_MODEL_PREDICTED_ACCURACY] = primary_model_predicted_accuracy
        report_dict[ErrorAnalyzerConstants.CONFIDENCE_DECISION] = confidence_decision
        return report_dict

    report = 'The MPP was trained with accuracy %.2f%% and balanced accuracy %.2f%%.' % (mpp_accuracy_score * 100,
                                                                                         mpp_balanced_accuracy * 100)
    report += '\n'
    report += 'The Decision Tree estimated the primary model''s accuracy to %.2f%%.' % \
              (primary_model_predicted_accuracy * 100)
    report += '\n'
    report += 'The true accuracy of the primary model is %.2f.%%' % (primary_model_true_accuracy * 100)
    report += '\n'
    report += 'The Fidelity of the MPP is %.2f%%.' % \
              (fidelity * 100)
    report += '\n'
    if not confidence_decision:
        report += 'Warning: the built MPP might not be representative of the primary model performances.'
        report += '\n'
        report += 'The MPP predicted model accuracy is considered too different from the true model accuracy.'
        report += '\n'
    else:
        report += 'The MPP is considered representative of the primary model performances.'
        report += '\n'

    return report
