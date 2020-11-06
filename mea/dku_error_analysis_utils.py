import sys
import numpy as np


def safe_str(val):
    if sys.version_info > (3, 0):
        return str(val)
    if isinstance(val, unicode):
        return val.encode("utf-8")
    return str(val)


def check_enough_data(df, min_len):
    """Check the input dataset is not too small.
    Parameters
    ----------
    df : pandas.DataFrame, input dataset.
    min_len: int, minimum required length.
    """
    if df.shape[0] < min_len:
        raise ValueError(
                'The original dataset is too small ({} rows) to have stable result, '
                'it needs to have at least {} rows'.format(df.shape[0], min_len))


def rank_features_by_error_correlation(feature_importances):
    """Rank indices from the most to the least important according to the input weights.
    Parameters
    ----------
    feature_importances : numpy.array, feature importance weights.

    Returns
    -------
    sorted_feature_indices: list, list of feature indices sorted from the most to the least important.

    """
    sorted_feature_indices = np.argsort(- feature_importances)
    cut = len(np.where(feature_importances != 0)[0])
    sorted_feature_indices = sorted_feature_indices[:cut]
    return sorted_feature_indices


class ErrorAnalyzerConstants(object):
    """
    ErrorAnalyzerConstants stores the values for the configuration parameters of ErrorAnalyzer and ErrorVisualizer.
    """
    WRONG_PREDICTION = "Wrong prediction"
    CORRECT_PREDICTION = "Correct prediction"
    MAX_DEPTH_GRID = [5, 10, 15, 20, 30, 50]
    TEST_SIZE = 0.2

    MIN_NUM_ROWS = 100  # heuristic choice
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
