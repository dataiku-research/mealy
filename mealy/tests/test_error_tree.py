import numpy as np
import random
import unittest

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import NotFittedError
from mealy import ErrorTree, ErrorAnalyzerConstants

default_seed = 10
np.random.seed(default_seed)
random.seed(default_seed)


class TestErrorTree(unittest.TestCase):
    def setUp(self):
        target_mapping_dict = {0: ErrorAnalyzerConstants.WRONG_PREDICTION,
                               1: ErrorAnalyzerConstants.CORRECT_PREDICTION}

        X, target = load_breast_cancer(return_X_y=True)
        y = list(map(lambda x: target_mapping_dict[x], target))
        X_train, _, y_train, _ = train_test_split(X, y, random_state=0)

        clf = DecisionTreeClassifier(max_depth=2, random_state=0)
        clf.fit(X_train, y_train)
        self.tree = ErrorTree(clf)

    def test_empty_tree(self):
        with self.assertRaises(NotFittedError):
            ErrorTree(None)

    def test_tree(self):
        self.assertEqual(self.tree.error_class_idx, 1)
        self.assertEqual(self.tree.n_total_errors, 159)
        self.assertListEqual(self.tree.correctly_predicted_leaves.tolist(), [245., 2., 17., 3.])
        self.assertListEqual(self.tree.difference.tolist(), [238., -4., 4., -130.])
        self.assertListEqual(self.tree.impurity.tolist(), [0.9722222222222222, 0.25, 0.5666666666666667, 0.022058823529411766])
        self.assertListEqual(self.tree.leaf_ids.tolist(), [2, 3, 5, 6])
        self.assertListEqual(self.tree.quantized_impurity.tolist(), [9, 3, 6, 1])
        self.assertListEqual(self.tree.total_error_fraction.tolist(), [0.0440251572327044, 0.03773584905660377, 0.08176100628930817, 0.8364779874213837])
        self.assertListEqual(self.tree.wrongly_predicted_leaves.tolist(), [7.0, 6.0, 13.0, 133.0])
