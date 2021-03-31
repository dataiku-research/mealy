# -*- coding: utf-8 -*-
import numpy as np
from sklearn.exceptions import NotFittedError
from mealy.constants import ErrorAnalyzerConstants
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='mealy | %(levelname)s - %(message)s')

class ErrorTree(object):

    def __init__(self, error_decision_tree):

        self._estimator = error_decision_tree
        self._leaf_ids = None
        self._impurity = None
        self._quantized_impurity = None
        self._difference = None
        self._total_error_fraction = None
        self.error_class_idx = np.where(self.estimator_.classes_ == ErrorAnalyzerConstants.WRONG_PREDICTION)[0][0]

        self._check_error_tree()

    @property
    def estimator_(self):
        if self._estimator is None:
            raise NotFittedError("You should fit the ErrorAnalyzer first")
        return self._estimator

    @property
    def impurity(self):
        if self._impurity is None:
            self._compute_ranking_arrays()
        return self._impurity

    @property
    def quantized_impurity(self):
        if self._quantized_impurity is None:
            self._compute_ranking_arrays()
        return self._quantized_impurity

    @property
    def difference(self):
        if self._difference is None:
            self._compute_ranking_arrays()
        return self._difference

    @property
    def total_error_fraction(self):
        if self._total_error_fraction is None:
            self._compute_ranking_arrays()
        return self._total_error_fraction

    @property
    def leaf_ids(self):
        if self._leaf_ids is None:
            self._compute_leaf_ids()
        return self._leaf_ids

    def get_error_leaves(self):
        error_node_ids = np.where(self.estimator_.tree_.value[:, 0, :].argmax(axis=1) == self.error_class_idx)[0]
        return np.in1d(self._leaf_ids, error_node_ids)

    def _check_error_tree(self):
        if self.estimator_.tree_.node_count == 1:
            logger.warning("The error tree has only 1 node, there will be problem when using it with ErrorVisualizer")

    def _compute_leaf_ids(self):
        """ Compute indices of leaf nodes """
        self._leaf_ids = np.where(self.estimator_.tree_.feature < 0)[0]

    def _compute_ranking_arrays(self, n_purity_levels=ErrorAnalyzerConstants.NUMBER_PURITY_LEVELS):
        """ Compute ranking array """
        correct_class_idx = 1 - self.error_class_idx

        wrongly_predicted_samples = self.estimator_.tree_.value[self.leaf_ids, 0, self.error_class_idx]
        correctly_predicted_samples = self.estimator_.tree_.value[self.leaf_ids, 0, correct_class_idx]

        self._impurity = correctly_predicted_samples / (wrongly_predicted_samples + correctly_predicted_samples)

        purity_bins = np.linspace(0, 1., n_purity_levels)
        self._quantized_impurity = np.digitize(self._impurity, purity_bins)
        self._difference = correctly_predicted_samples - wrongly_predicted_samples  # only negative numbers

        n_total_errors = np.sum(wrongly_predicted_samples)
        self._total_error_fraction = wrongly_predicted_samples / float(n_total_errors)