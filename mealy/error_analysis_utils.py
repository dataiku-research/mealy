# -*- coding: utf-8 -*-
from sklearn.pipeline import Pipeline
import numpy as np
from mealy.constants import ErrorAnalyzerConstants
from kneed import KneeLocator


def get_epsilon(difference):
    """
    Compute the threshold used to decide whether a prediction is wrong or correct (for regression tasks).

    Args:
           difference (1D-array): The absolute differences between the true target values and the predicted ones (by the primary model).

    Return:
           epsilon (float): The value of the threshold used to decide whether the prediction for a regression task is wrong or correct
    """
    epsilon_range = np.linspace(min(difference), max(difference), num=ErrorAnalyzerConstants.NUMBER_EPSILON_VALUES)
    cdf_error = []
    n_samples = difference.shape[0]
    for epsilon in epsilon_range:
        correct_predictions = difference <= epsilon
        cdf_error.append(np.count_nonzero(correct_predictions) / float(n_samples))
    return KneeLocator(epsilon_range, cdf_error).knee

def get_feature_list_from_column_transformer(ct_preprocessor):
    all_features, categorical_features = [], []
    for transformer_name, transformer, transformer_feature_names in ct_preprocessor.transformers_:
        if transformer_name == 'remainder' and transformer == 'drop':
            continue
        all_features.extend(transformer_feature_names)

        # check for categorical features
        if isinstance(transformer, Pipeline):
            for step in transformer.steps:
                if isinstance(step[1], ErrorAnalyzerConstants.VALID_CATEGORICAL_STEPS):
                    categorical_features.extend(transformer_feature_names)
                    break
        elif isinstance(transformer, ErrorAnalyzerConstants.VALID_CATEGORICAL_STEPS):
            categorical_features.extend(transformer_feature_names)
    return all_features, categorical_features


def check_lists_having_same_elements(list_A, list_B):
    return set(list_A) == set(list_B)

def check_enough_data(df, min_len):
    """
    Compare length of dataframe to minimum lenght of the test data.
    Used in the relevance of the measure.

    :param df: Input dataframe
    :param min_len:
    :return:
    """
    if df.shape[0] < min_len:
        raise ValueError(
            'The original dataset is too small ({} rows) to have stable result, it needs to have at least {} rows'.format(
                df.shape[0], min_len))
