import numpy as np
import pandas as pd
import logging
from unittest import TestCase, skip
from unittest.mock import patch, Mock
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier

from .. import ErrorTree, ErrorAnalyzer, ErrorAnalyzerConstants, DummyPipelinePreprocessor, PipelinePreprocessor

logger = logging.getLogger(__name__)

class TestErrorAnalyzer(TestCase):
    @patch("numpy.count_nonzero", wraps=np.count_nonzero)
    def test_get_leaf_selector(self, cnz):
        leaves = [4, 80, 13, 3, 1, 0, 12]
        selector = ErrorAnalyzer._get_leaf_selector(list(leaves))
        self.assertFalse(cnz.called)
        self.assertTrue(selector(True))
        np.testing.assert_array_equal(selector(leaves), leaves)

        cnz.reset_mock()
        selector = ErrorAnalyzer._get_leaf_selector(list(leaves), [3, 4])
        cnz.assert_called_once()
        np.testing.assert_array_equal(cnz.call_args[0][0],
            [True, False, False, True, False, False, False])
        np.testing.assert_array_equal(selector(np.array(leaves)), [4, 3])

        cnz.reset_mock()
        selector = ErrorAnalyzer._get_leaf_selector(list(leaves), 1)
        cnz.assert_called_once()
        np.testing.assert_array_equal(cnz.call_args[0][0],
            [False, False, False, False, True, False, False])
        np.testing.assert_array_equal(selector(np.array(leaves)), [1])

        cnz.reset_mock()
        with self.assertLogs("mealy.error_analyzer") as caplog:
            selector = ErrorAnalyzer._get_leaf_selector(list(leaves), -1)
        self.assertEqual(len(caplog.records), 1)
        self.assertEqual(caplog.records[-1].msg,
            "None of the ids provided correspond to a leaf id.")
        cnz.assert_called_once()
        np.testing.assert_array_equal(cnz.call_args[0][0],
            [False, False, False, False, False, False, False])
        self.assertEqual(selector(np.array(leaves)).size, 0)

        cnz.reset_mock()
        with self.assertLogs("mealy.error_analyzer") as caplog:
            selector = ErrorAnalyzer._get_leaf_selector(list(leaves), [100, 0, -1])
        self.assertEqual(len(caplog.records), 1)
        self.assertEqual(caplog.records[-1].msg,
            "Some of the ids provided do not belong to leaves. Only leaf ids are kept.")
        cnz.assert_called_once()
        np.testing.assert_array_equal(cnz.call_args[0][0],
            [False, False, False, False, False, True, False])
        np.testing.assert_array_equal(selector(np.array(leaves)), [0])

    def test_init(self):
        # Analyzer with single estimator
        single_estimator = Mock(spec=BaseEstimator)
        with patch("mealy.error_analyzer.DummyPipelinePreprocessor.__init__", return_value=None) as dummy:
            analyzer = ErrorAnalyzer(single_estimator)
            dummy.assert_called_once_with(None)
        self.assertEqual(single_estimator, analyzer._primary_model)
        self.assertTrue(isinstance(analyzer.pipeline_preprocessor, DummyPipelinePreprocessor))

        # Analyzer with pipeline
        ct_transformer = Mock(spec=ColumnTransformer)
        ok_pipe = Mock(spec=Pipeline, steps=[
            ("transformer", ct_transformer),
            ("single_estimator", single_estimator)
        ])

        with patch("mealy.error_analyzer.PipelinePreprocessor.__init__", return_value=None) as pipe,\
            self.assertLogs() as caplog:
            analyzer = ErrorAnalyzer(ok_pipe)
            logger.info("Dummy log")
            pipe.assert_called_once_with(ct_transformer, None)
            self.assertEqual(len(caplog.records), 1)
            self.assertEqual(caplog.records[-1].msg, "Dummy log")
        self.assertEqual(single_estimator, analyzer._primary_model)
        self.assertTrue(isinstance(analyzer.pipeline_preprocessor, PipelinePreprocessor))

        # Analyzer with pipeline (wrong attribute type)
        wrong_primary_model = Mock()
        with self.assertRaisesRegex(TypeError, "ErrorAnalyzer needs as input either a scikit BaseEstimator or a scikit Pipeline."):
            analyzer = ErrorAnalyzer(wrong_primary_model)

        # Analyzer with pipeline (missing col transformer)
        incomplete_pipe = Mock(spec=Pipeline, steps=[
            ("single_estimator", single_estimator)
        ])

        with patch("mealy.error_analyzer.PipelinePreprocessor.__init__", return_value=None) as pipe,\
            self.assertLogs("mealy.error_analyzer", level="WARNING") as caplog,\
            self.assertRaisesRegex(TypeError, "The input preprocessor has to be a ColumnTransformer."):
            analyzer = ErrorAnalyzer(incomplete_pipe)
            self.assertEqual(len(caplog.records), 1)
            self.assertEqual(caplog.records[-1].msg,
                "Pipeline should have two steps: the preprocessing of the features, and the primary model to analyze.")

        # Analyzer with pipeline (missing estimator)
        incomplete_pipe = Mock(spec=Pipeline, steps=[
            ("wrong_type", Mock())
        ])

        with patch("mealy.error_analyzer.PipelinePreprocessor.__init__", return_value=None) as pipe,\
            self.assertLogs("mealy.error_analyzer", level="WARNING") as caplog,\
            self.assertRaisesRegex(TypeError, "The last step of the pipeline has to be a BaseEstimator"):
            analyzer = ErrorAnalyzer(incomplete_pipe)
            self.assertEqual(len(caplog.records), 1)
            self.assertEqual(caplog.records[-1].msg,
            "Pipeline should have two steps: the preprocessing of the features, and the primary model to analyze.")

    @staticmethod
    def get_analyzer(**kwargs):
        with patch("mealy.error_analyzer.ErrorAnalyzer.__init__", return_value=None):
            analyzer = ErrorAnalyzer(Mock(spec=BaseEstimator))
        error_tree = Mock(leaf_ids=np.array([4, 80, 13, 3, 1, 0, 12]), **kwargs)
        analyzer._error_tree = error_tree
        analyzer._random_state = 42
        analyzer._param_grid = None
        return analyzer

    def test_get_ranked_leaf_ids__tot_error_fraction(self):
        analyzer = TestErrorAnalyzer.get_analyzer(
            total_error_fraction=np.array([.4, .1, .2, .5, .6, .7, .3])
        )
        with patch.object(analyzer, "_get_leaf_selector", return_value=lambda array: array):
            sorted_leaves = analyzer._get_ranked_leaf_ids()
        np.testing.assert_array_equal(sorted_leaves, [0, 1, 3, 4, 12, 13, 80])

    def test_get_ranked_leaf_ids__purity(self):
        analyzer = TestErrorAnalyzer.get_analyzer(
            quantized_impurity=np.array([.5, .5, .5, .2, .1, .1, .5]),
            difference=np.array([0, 3, 2, 100, 11, 10, 1])
        )
        with patch.object(analyzer, "_get_leaf_selector", return_value=lambda array: array):
            sorted_leaves = analyzer._get_ranked_leaf_ids(rank_by='purity')
        np.testing.assert_array_equal(sorted_leaves, [0, 1, 3, 4, 12, 13, 80])

    def test_get_ranked_leaf_ids__class_difference(self):
        analyzer = TestErrorAnalyzer.get_analyzer(
            impurity=np.array([.5, -1, 1, .2, 1000, 1000, .8]),
            difference=np.array([0, 10, 0, 0, -5, -10, 0])
        )
        with patch.object(analyzer, "_get_leaf_selector", return_value=lambda array: array):
            sorted_leaves = analyzer._get_ranked_leaf_ids(rank_by='class_difference')
        np.testing.assert_array_equal(sorted_leaves, [0, 1, 3, 4, 12, 13, 80])

    def test_get_ranked_leaf_ids__invalid(self):
        analyzer = TestErrorAnalyzer.get_analyzer()
        with patch.object(analyzer, "_get_leaf_selector", return_value=lambda array: array),\
            self.assertRaisesRegex(ValueError,
            "Input argument for rank_by is invalid. Should be 'total_error_fraction', 'purity' or 'class_difference'"):
            sorted_leaves = analyzer._get_ranked_leaf_ids(rank_by='toto')

    def test_get_ranked_leaf_ids__empty(self):
        analyzer = TestErrorAnalyzer.get_analyzer()
        with patch.object(analyzer, "_get_leaf_selector", return_value=lambda array: np.array([])):
            sorted_leaves = analyzer._get_ranked_leaf_ids()
        self.assertEqual(sorted_leaves.size, 0)

    def test_fit__not_enough_data(self):
        analyzer = TestErrorAnalyzer.get_analyzer()
        analyzer.pipeline_preprocessor = Mock()
        analyzer.pipeline_preprocessor.transform.side_effect = lambda x: x
        X = np.array([[1,2,3,4,5]]*5)
        y = None
        with patch("mealy.error_analyzer.ErrorAnalyzer._compute_primary_model_error", side_effect=lambda a, b: (a, .3)),\
            self.assertRaises(ValueError):
            analyzer.fit(X, y)

    def test_fit__default_grid(self):
        analyzer = TestErrorAnalyzer.get_analyzer()
        analyzer.pipeline_preprocessor = Mock()
        analyzer.pipeline_preprocessor.transform.side_effect = lambda x: x
        self.get_analyzer.param_grid = None
        X = np.array([[1,2,3,4,5]]*100)
        y = np.array([1]*100)
        with patch("mealy.error_analyzer.ErrorAnalyzer._compute_primary_model_error", side_effect=lambda a, b: (b, .3)), \
            patch("mealy.error_analyzer.GridSearchCV") as patched:
            analyzer.fit(X, y)
            clf = patched.call_args[0][0]
            param_grid, cv = patched.call_args[1]["param_grid"], patched.call_args[1]["cv"]
        self.assertTrue(isinstance(clf, DecisionTreeClassifier))
        self.assertListEqual(param_grid["max_depth"], [5, 10])
        np.testing.assert_array_equal(param_grid["min_samples_leaf"],  np.linspace(.01/5, .01, 5))
        self.assertEqual(cv, 5)
        np.testing.assert_array_equal(analyzer._error_train_y, y)
        np.testing.assert_array_equal(analyzer._error_train_x, X)
        self.assertTrue(isinstance(analyzer.error_tree, ErrorTree))

    @skip(reason="Not done yet")
    def test_get_path_to_node(self):
        pass

    @skip(reason="Not done yet")
    def test_get_error_leaf_summary(self):
        pass


class TestErrorAnalyzerPrimaryModelMethods(TestCase):
    def setUp(self):
        with patch("mealy.error_analyzer.ErrorAnalyzer.__init__", return_value=None):
            self.analyzer = ErrorAnalyzer(Mock(spec=BaseEstimator))
        self.analyzer.epsilon = None
        self.analyzer._primary_model = Mock()

    def test_compute_primary_model_error__regressor(self):
        self.analyzer._primary_model._estimator_type = "regressor"
        self.analyzer._primary_model.predict.side_effect = lambda array: array+1
        X = np.array([1,1,1,1,1])
        y = np.array([0,0,0,0,1])
        with patch.object(self.analyzer, "_evaluate_primary_model_predictions",
            side_effect= lambda y_true, y_pred: (y_true, y_pred)):
            y_true, y_pred = self.analyzer._compute_primary_model_error(X, y)
        np.testing.assert_array_equal(y_true, y)
        np.testing.assert_array_equal(y_pred, [2,2,2,2,2])

    def test_compute_primary_model_error__multi(self):
        self.analyzer._primary_model.predict.side_effect = lambda array: array+1
        X = np.array([1,1,1,1,1])
        y = np.array([0,1,2,3,4])
        with patch.object(self.analyzer, "_evaluate_primary_model_predictions",
            side_effect= lambda y_true, y_pred: (y_true, y_pred)):
            y_true, y_pred = self.analyzer._compute_primary_model_error(X, y)
        np.testing.assert_array_equal(y_true, y)
        np.testing.assert_array_equal(y_pred, [2,2,2,2,2])

    def test_compute_primary_model_error__binary(self):
        self.analyzer._primary_model.classes_ = ["W", "C"]
        self.analyzer._primary_model.predict_proba.side_effect = lambda array: array
        self.analyzer.probability_threshold = .2
        X = np.array([
            [0, .1],
            [0, .8],
            [0, .1],
            [0, .8],
            [0, .8],
        ])
        y = np.array([0,1,1,0,0])
        with patch.object(self.analyzer, "_evaluate_primary_model_predictions",
            side_effect= lambda y_true, y_pred: (y_true, y_pred)):
            y_true, y_pred = self.analyzer._compute_primary_model_error(X, y)
        np.testing.assert_array_equal(y_true, y)
        np.testing.assert_array_equal(y_pred, ["W","C","W","C","C"])

    def test_evaluate_primary_model_predictions__regression(self):
        self.analyzer._primary_model._estimator_type = "regressor"
        self.analyzer.epsilon = .4
        y_true = np.array([1,.2,-.3,-.7,0])
        y_pred = np.array([.1,0,0,0.2,1])
        with patch("mealy.error_analyzer.get_epsilon", return_value="") as patched,\
            self.assertLogs("mealy.error_analyzer") as caplog:
            error_y, error_rate = self.analyzer._evaluate_primary_model_predictions(y_true, y_pred)
            self.assertFalse(patched.called)
            self.assertEqual(len(caplog.records), 1)
            self.assertEqual((caplog.records[-1].levelname, caplog.records[-1].msg),
                ("INFO", 'The primary model has an error rate of 0.6'))
        np.testing.assert_array_equal(error_y, [
            ErrorAnalyzerConstants.WRONG_PREDICTION,
            ErrorAnalyzerConstants.CORRECT_PREDICTION,
            ErrorAnalyzerConstants.CORRECT_PREDICTION,
            ErrorAnalyzerConstants.WRONG_PREDICTION,
            ErrorAnalyzerConstants.WRONG_PREDICTION
        ])
        np.testing.assert_array_equal(error_rate, .6)

    def test_evaluate_primary_model_predictions__classif(self):
        y_true = np.array(["A","B","A","C","A"])
        y_pred = np.array(["B","B","A","A","B"])
        with patch("mealy.error_analyzer.get_epsilon", return_value="") as patched,\
            self.assertLogs("mealy.error_analyzer") as caplog:
            error_y, error_rate = self.analyzer._evaluate_primary_model_predictions(y_true, y_pred)
            self.assertFalse(patched.called)
            self.assertEqual(len(caplog.records), 1)
            self.assertEqual((caplog.records[-1].levelname, caplog.records[-1].msg),
                ("INFO", 'The primary model has an error rate of 0.6'))
        np.testing.assert_array_equal(error_y, [
            ErrorAnalyzerConstants.WRONG_PREDICTION,
            ErrorAnalyzerConstants.CORRECT_PREDICTION,
            ErrorAnalyzerConstants.CORRECT_PREDICTION,
            ErrorAnalyzerConstants.WRONG_PREDICTION,
            ErrorAnalyzerConstants.WRONG_PREDICTION
        ])
        np.testing.assert_array_equal(error_rate, .6)

    def test_evaluate_primary_model_predictions__all_wrong(self):
        self.analyzer._primary_model._estimator_type = "regressor"
        y_true = np.array([1,.2,-.3,-.7,0])
        y_pred = np.array([.1,0,0,0.2,1])
        with patch("mealy.error_analyzer.get_epsilon", return_value=.1) as patched,\
            self.assertLogs("mealy.error_analyzer") as caplog:
            error_y, error_rate = self.analyzer._evaluate_primary_model_predictions(y_true, y_pred)
            patched.assert_called_once()
            self.assertEqual(len(caplog.records), 2)
            self.assertEqual((caplog.records[-1].levelname, caplog.records[-1].msg),
                ("INFO", 'The primary model has an error rate of 1'))
            self.assertEqual((caplog.records[-2].levelname, caplog.records[-2].msg),
                ("WARNING", 'All predictions are Wrong prediction. To build a proper ErrorAnalyzer decision tree we need both correct and incorrect predictions'))
        np.testing.assert_array_equal(error_y, [
            ErrorAnalyzerConstants.WRONG_PREDICTION,
            ErrorAnalyzerConstants.WRONG_PREDICTION,
            ErrorAnalyzerConstants.WRONG_PREDICTION,
            ErrorAnalyzerConstants.WRONG_PREDICTION,
            ErrorAnalyzerConstants.WRONG_PREDICTION
        ])
        np.testing.assert_array_equal(error_rate, 1)

    def test_evaluate_primary_model_predictions__all_correct(self):
        self.analyzer._primary_model._estimator_type = "regressor"
        y_true = np.array([1,.2,-.3,-.7,0])
        y_pred = np.array([.1,0,0,0.2,1])
        with patch("mealy.error_analyzer.get_epsilon", return_value=1.1) as patched,\
            self.assertLogs("mealy.error_analyzer") as caplog:
            error_y, error_rate = self.analyzer._evaluate_primary_model_predictions(y_true, y_pred)
            patched.assert_called_once()
            self.assertEqual(len(caplog.records), 2)
            self.assertEqual((caplog.records[-1].levelname, caplog.records[-1].msg),
                ("INFO", 'The primary model has an error rate of 0'))
            self.assertEqual((caplog.records[-2].levelname, caplog.records[-2].msg),
                ("WARNING", 'All predictions are Correct prediction. To build a proper ErrorAnalyzer decision tree we need both correct and incorrect predictions'))
        np.testing.assert_array_equal(error_y, [
            ErrorAnalyzerConstants.CORRECT_PREDICTION,
            ErrorAnalyzerConstants.CORRECT_PREDICTION,
            ErrorAnalyzerConstants.CORRECT_PREDICTION,
            ErrorAnalyzerConstants.CORRECT_PREDICTION,
            ErrorAnalyzerConstants.CORRECT_PREDICTION
        ])
        np.testing.assert_array_equal(error_rate, 0)
