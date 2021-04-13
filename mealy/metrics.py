# -*- coding: utf-8 -*-
from mealy.constants import ErrorAnalyzerConstants
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np


def compute_confidence_decision(primary_model_true_accuracy, primary_model_predicted_accuracy):
    difference_true_pred_accuracy = np.abs(primary_model_true_accuracy - primary_model_predicted_accuracy)
    decision = difference_true_pred_accuracy <= ErrorAnalyzerConstants.TREE_ACCURACY_TOLERANCE

    fidelity = 1. - difference_true_pred_accuracy

    # TODO Binomial test
    return fidelity, decision


def compute_accuracy_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def compute_primary_model_accuracy(y):
    n_test_samples = y.shape[0]
    return float(np.count_nonzero(y == ErrorAnalyzerConstants.CORRECT_PREDICTION)) / n_test_samples


def compute_fidelity_score(y_true, y_pred):
    difference_true_pred_accuracy = np.abs(compute_primary_model_accuracy(y_true) -
                                           compute_primary_model_accuracy(y_pred))
    fidelity = 1. - difference_true_pred_accuracy

    return fidelity


def fidelity_balanced_accuracy_score(y_true, y_pred):
    return compute_fidelity_score(y_true, y_pred) + balanced_accuracy_score(y_true, y_pred)


def error_decision_tree_report(y_true, y_pred, output_format='str'):
    """Return a report showing the main Error Decision Tree metrics.

    Args:
        y_true (numpy.ndarray): Ground truth values of wrong/correct predictions of the error tree primary model.
            Expected values in [ErrorAnalyzerConstants.WRONG_PREDICTION, ErrorAnalyzerConstants.CORRECT_PREDICTION].
        y_pred (numpy.ndarray): Estimated targets as returned by the error tree. Expected values in
            [ErrorAnalyzerConstants.WRONG_PREDICTION, ErrorAnalyzerConstants.CORRECT_PREDICTION].
        output_format (string): Return format used for the report. Valid values are 'dict' or 'str'.

    Return:
        dict or str: dictionary or string report storing different metrics regarding the Error Decision Tree.
    """

    tree_accuracy_score = compute_accuracy_score(y_true, y_pred)
    tree_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    primary_model_predicted_accuracy = compute_primary_model_accuracy(y_pred)
    primary_model_true_accuracy = compute_primary_model_accuracy(y_true)
    fidelity, confidence_decision = compute_confidence_decision(primary_model_true_accuracy,
                                                                primary_model_predicted_accuracy)
    if output_format == 'dict':
        report_dict = dict()
        report_dict[ErrorAnalyzerConstants.TREE_ACCURACY] = tree_accuracy_score
        report_dict[ErrorAnalyzerConstants.TREE_BALANCED_ACCURACY] = tree_balanced_accuracy
        report_dict[ErrorAnalyzerConstants.TREE_FIDELITY] = fidelity
        report_dict[ErrorAnalyzerConstants.PRIMARY_MODEL_TRUE_ACCURACY] = primary_model_true_accuracy
        report_dict[ErrorAnalyzerConstants.PRIMARY_MODEL_PREDICTED_ACCURACY] = primary_model_predicted_accuracy
        report_dict[ErrorAnalyzerConstants.CONFIDENCE_DECISION] = confidence_decision
        return report_dict

    if output_format == 'str':

        report = 'The Error Decision Tree was trained with accuracy %.2f%% and balanced accuracy %.2f%%.' % (tree_accuracy_score * 100, tree_balanced_accuracy * 100)
        report += '\n'
        report += 'The Decision Tree estimated the primary model''s accuracy to %.2f%%.' % \
                  (primary_model_predicted_accuracy * 100)
        report += '\n'
        report += 'The true accuracy of the primary model is %.2f.%%' % (primary_model_true_accuracy * 100)
        report += '\n'
        report += 'The Fidelity of the error tree is %.2f%%.' % \
                  (fidelity * 100)
        report += '\n'
        if not confidence_decision:
            report += 'Warning: the built tree might not be representative of the primary model performances.'
            report += '\n'
            report += 'The error tree predicted model accuracy is considered too different from the true model accuracy.'
            report += '\n'
        else:
            report += 'The error tree is considered representative of the primary model performances.'
            report += '\n'

        return report

    else:
        raise ValueError("Output format should either be 'dict' or 'str'")
