# -*- coding: utf-8 -*-
import numpy as np

def check_enough_data(df, min_len):
    """
    Compare length of dataframe to minimum lenght of the test data.
    Used in the relevance of the measure.

    :param df: Input dataframe
    :param min_len:
    :return:
    """
    if df.shape[0] < min_len:
        raise ValueError('The original dataset is too small ({} rows) to have stable result, it needs to have at least {} rows'.format(df.shape[0], min_len))


def rank_features_by_error_correlation(feature_importances):
    sorted_feature_indices = np.argsort(- feature_importances)
    cut = len(np.where(feature_importances != 0)[0])
    sorted_feature_indices = sorted_feature_indices[:cut]
    return sorted_feature_indices


class ErrorAnalyzerConstants(object):

    WRONG_PREDICTION = "Wrong prediction"
    CORRECT_PREDICTION = "Correct prediction"
    MAX_DEPTH_GRID = [5, 10, 15, 20, 30, 50]
    TEST_SIZE = 0.2

    MIN_NUM_ROWS = 100 #500  # heuristic choice
    MAX_NUM_ROW = 100000  # heuristic choice

    MPP_ACCURACY_TOLERANCE = 0.1
    CRITERION = 'entropy'
    NUMBER_EPSILON_VALUES = 50

    ERROR_TREE_COLORS = {CORRECT_PREDICTION: '#538BC8', WRONG_PREDICTION: '#EC6547'}

    TOP_K_FEATURES = 3

    MPP_ACCURACY = 'mpp_accuracy_score'
    PRIMARY_MODEL_TRUE_ACCURACY = 'primary_model_true_accuracy'
    PRIMARY_MODEL_PREDICTED_ACCURACY = 'primary_model_predicted_accuracy'
    CONFIDENCE_DECISION = 'confidence_decision'

    NUMBER_PURITY_LEVELS = 10
