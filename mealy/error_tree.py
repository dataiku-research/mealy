# -*- coding: utf-8 -*-
import numpy as np
from sklearn.exceptions import NotFittedError
from mealy.constants import ErrorAnalyzerConstants
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='mealy | %(levelname)s - %(message)s')


class ErrorTree(object):
    """ ErrorTree analyzes the errors of a prediction model on a test set.

    It uses model predictions and ground truth target to compute the model errors on the test set.
    It then trains a Decision Tree, called a Error Analyzer Tree, on the same test set by using the model error
    as target. The nodes of the decision tree are different segments of errors to be studied individually.

    Args:
        error_decision_tree (sklearn.tree.DecisionTreeClassifier): The estimator used to train the Error Tree.

    Attributes:
        estimator_ (sklearn.tree.DecisionTreeClassifier): The estimator used to train the Error Tree.
        impurity (numpy.ndarray): Impurity of leaves.
        quantized_impurity (numpy.ndarray): Impurity of leaves quantized into ErrorAnalyzerConstants.NUMBER_PURITY_LEVELS
            levels.
        difference (numpy.ndarray): Difference of number of wrongly and correctly predicted samples in leaves.
        total_error_fraction (numpy.ndarray): Percentage of incorrectly predicted samples in leaves over the total
            number of errors (used to rank the nodes).
        error_class_idx (int): Index of class of wrongly predicted samples in the Error Tree.
        n_total_errors (int): Number of total errors.
        wrongly_predicted_leaves (numpy.ndarray): Array of number of wrongly predicted samples in leaves.
        correctly_predicted_leaves (numpy.ndarray): Array of number of correctly predicted samples in leaves.
        leaf_ids (numpy.ndarray): List of all leaves indices.

    """
    def __init__(self, error_decision_tree):

        self._estimator = error_decision_tree
        self._leaf_ids = None
        self._impurity = None
        self._quantized_impurity = None
        self._difference = None
        self._total_error_fraction = None
        self._error_class_idx = None
        self._wrongly_predicted_leaves = None
        self._correctly_predicted_leaves = None

        self._check_error_tree()

    @property
    def estimator_(self):
        if self._estimator is None:
            raise NotFittedError("You should fit the ErrorAnalyzer first")
        return self._estimator

    @property
    def impurity(self):
        if self._impurity is None:
            self._impurity = self.correctly_predicted_leaves / (self.wrongly_predicted_leaves + self.correctly_predicted_leaves)
        return self._impurity

    @property
    def quantized_impurity(self):
        if self._quantized_impurity is None:
            purity_bins = np.linspace(0, 1., ErrorAnalyzerConstants.NUMBER_PURITY_LEVELS)
            self._quantized_impurity = np.digitize(self.impurity, purity_bins)
        return self._quantized_impurity

    @property
    def difference(self):
        if self._difference is None:
            self._difference = self.correctly_predicted_leaves - self.wrongly_predicted_leaves  # only negative numbers
        return self._difference

    @property
    def total_error_fraction(self):
        if self._total_error_fraction is None:
            self._total_error_fraction = self.wrongly_predicted_leaves / self.n_total_errors
        return self._total_error_fraction

    @property
    def error_class_idx(self):
        if self._error_class_idx is None:
            self._error_class_idx = np.where(self.estimator_.classes_ == ErrorAnalyzerConstants.WRONG_PREDICTION)[0][0]
        return self._error_class_idx

    @property
    def n_total_errors(self):
        return self.estimator_.tree_.value[0, 0, self.error_class_idx]

    @property
    def wrongly_predicted_leaves(self):
        if self._wrongly_predicted_leaves is None:
            self._wrongly_predicted_leaves = self.estimator_.tree_.value[self.leaf_ids, 0, self.error_class_idx]
        return self._wrongly_predicted_leaves

    @property
    def correctly_predicted_leaves(self):
        if self._correctly_predicted_leaves is None:
            self._correctly_predicted_leaves = self.estimator_.tree_.value[self.leaf_ids, 0, 1 - self.error_class_idx]
        return self._correctly_predicted_leaves

    @property
    def leaf_ids(self):
        if self._leaf_ids is None:
            self._leaf_ids = np.where(self.estimator_.tree_.feature < 0)[0]
        return self._leaf_ids

    def _check_error_tree(self):
        if self.estimator_.tree_.node_count == 1:
            logger.warning("The error tree has only 1 node, there will be problem when using it with ErrorVisualizer")