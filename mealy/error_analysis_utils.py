# -*- coding: utf-8 -*-
from sklearn.pipeline import Pipeline
import numpy as np
import collections
from mealy.constants import ErrorAnalyzerConstants
from kneed import KneeLocator

def get_epsilon(difference, mode='rec'):
    """ Compute epsilon to define errors in regression task """
    assert (mode in ['std', 'rec'])
    epsilon = None
    if mode == 'std':
        std_diff = np.std(difference)
        mean_diff = np.mean(difference)
        epsilon = mean_diff + std_diff
    elif mode == 'rec':
        n_points = ErrorAnalyzerConstants.NUMBER_EPSILON_VALUES
        epsilon_range = np.linspace(min(difference), max(difference), num=n_points)
        cdf_error = np.zeros_like(epsilon_range)
        n_samples = difference.shape[0]
        for i, epsilon in enumerate(epsilon_range):
            correct = difference <= epsilon
            cdf_error[i] = float(np.count_nonzero(correct)) / n_samples
        kneedle = KneeLocator(epsilon_range, cdf_error)
        epsilon = kneedle.knee
    return epsilon

def get_feature_list_from_column_transformer(ct_preprocessor):
    all_feature = []
    categorical_features = []
    for i, (transformer_name, transformer, transformer_feature_names) in enumerate(ct_preprocessor.transformers_):
        if transformer_name == 'remainder' and transformer == 'drop':
            continue
        else:
            all_feature.extend(transformer_feature_names)

        # check for categorical features
        if isinstance(transformer, Pipeline):
            for step in transformer.steps:
                if isinstance(step[1], ErrorAnalyzerConstants.VALID_CATEGORICAL_STEPS):
                    categorical_features.extend(transformer_feature_names)
                    break
        elif isinstance(transformer, ErrorAnalyzerConstants.VALID_CATEGORICAL_STEPS):
            categorical_features.extend(transformer_feature_names)
        else:
            continue

    return all_feature, categorical_features


def check_lists_having_same_elements(list_A, list_B):
    return collections.Counter(list_A) == collections.Counter(list_B)


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
    return np.argsort(- feature_importances)
