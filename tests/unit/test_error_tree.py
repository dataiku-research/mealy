import numpy as np
from unittest import TestCase
from unittest.mock import Mock

from sklearn.exceptions import NotFittedError
from .. import ErrorTree, ErrorAnalyzerConstants


class TestErrorTree(TestCase):
    def test_empty_tree(self):
        with self.assertRaisesRegex(NotFittedError, "You should fit the ErrorAnalyzer first"):
            ErrorTree(None)

    def test_small_tree(self):
        clf = Mock()
        clf.tree_.node_count = 1
        with self.assertLogs("mealy.error_tree", level="WARNING") as caplog:
            error_tree = ErrorTree(clf)
            self.assertEqual(caplog.output, [
                "WARNING:mealy.error_tree:The error tree has only one node, " +
                "there will be problems when using it with ErrorVisualizer"
            ])

    def test_tree(self):
        tree_ = Mock(value=np.array([
            [[42, 69]],
            [[2, 9]],
            [[40, 58]],
            [[30, 0]],
            [[10, 10]],
            [[2, 9]],
            [[8, 1]],
            [[2, 30]],
            [[28, 28]],
            [[1, 10]],
            [[27, 18]]
            ]), feature=np.array([1, 2, -2, -2, 0, -2, -2, 0, 1]))

        clf = Mock(classes_=np.array([ErrorAnalyzerConstants.CORRECT_PREDICTION,
            ErrorAnalyzerConstants.WRONG_PREDICTION]), tree_=tree_)

        error_tree = ErrorTree(clf)

        self.assertEqual(error_tree.error_class_idx, 1)
        self.assertEqual(error_tree.n_total_errors, 69)
        np.testing.assert_array_equal(error_tree.leaf_ids, [2, 3, 5, 6])

        # Ranking arrays
        np.testing.assert_array_equal(error_tree.correctly_predicted_leaves, [40, 30, 2, 8])
        np.testing.assert_array_equal(error_tree.wrongly_predicted_leaves, [58, 0, 9, 1])
        np.testing.assert_array_equal(error_tree.difference, [-18, 30, -7, 7])
        np.testing.assert_array_equal(error_tree.impurity, [20/49, 1, 2/11, 8/9])
        np.testing.assert_array_equal(error_tree.quantized_impurity, [4, 10, 2, 9])
        np.testing.assert_array_equal(error_tree.total_error_fraction, [58/69, 0, 9/69, 1/69])
