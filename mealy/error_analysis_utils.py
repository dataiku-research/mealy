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

def generate_preprocessing_steps(transformer, invert_order=False):
    if isinstance(transformer, Pipeline):
        steps = [step for name, step in transformer.steps]
        if invert_order:
            steps = reversed(steps)
    else:
        steps = [transformer]
    for step in steps:
        if step == 'drop':
            # Skip the drop step of ColumnTransformer
            continue
        if step != 'passthrough' and not isinstance(step, ErrorAnalyzerConstants.SUPPORTED_STEPS):
            # Check all the preprocessing steps are supported by mealy
            unsupported_class = step.__class__
            raise TypeError('Mealy package does not support {}. '.format(unsupported_class) +
                        'It might be because it changes output dimension without ' +
                        'providing a get_feature_names function to keep track of the ' +
                        'generated features, or that it does not provide an ' +
                        'inverse_tranform method.')
        yield step

def invert_transform_via_identity(step):
    if isinstance(step, ErrorAnalyzerConstants.STEPS_THAT_CAN_BE_INVERSED_WITH_IDENTICAL_FUNCTION):
        return True
    if step == 'passthrough' or step is None:
        return True
    return False

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

def format_float(number, decimals):
    """
    Format a number to have the required number of decimals. Ensure no trailing zeros remain.

    Args:
        number (float or int): The number to format
        decimals (int): The number of decimals required

    Return:
        formatted (str): The number as a formatted string

    """
    formatted = ("{:." + str(decimals) + "f}").format(number).rstrip("0")
    if formatted.endswith("."):
        return formatted[:-1]
    return formatted
