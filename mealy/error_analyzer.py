# -*- coding: utf-8 -*-
import numpy as np
import collections
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.base import is_regressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from mealy.error_analysis_utils import check_enough_data, get_epsilon
from mealy.constants import ErrorAnalyzerConstants
from mealy.metrics import error_decision_tree_report, fidelity_balanced_accuracy_score
from mealy.preprocessing import PipelinePreprocessor, DummyPipelinePreprocessor
from mealy.error_tree import ErrorTree
from sklearn.exceptions import NotFittedError

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='mealy | %(levelname)s - %(message)s')


class ErrorAnalyzer(BaseEstimator):
    """ ErrorAnalyzer analyzes the errors of a prediction model on a test set.

    It uses model predictions and ground truth target to compute the model errors on the test set.
    It then trains a Decision Tree, called a Error Analyzer Tree, on the same test set by using the model error
    as target. The nodes of the decision tree are different segments of errors to be studied individually.

    Args:
        primary_model (sklearn.base.BaseEstimator or sklearn.pipeline.Pipeline): a sklearn model to analyze. Either an estimator
            or a Pipeline containing a ColumnTransformer with the preprocessing steps and an estimator as last step.
        feature_names (list of str): list of feature names. Defaults to None.
        param_grid (dict): sklearn.tree.DecisionTree hyper-parameters values for grid search.
        random_state (int): random seed.

    Attributes:
        _error_tree (DecisionTreeClassifier): the estimator used to train the Error Analyzer Tree
    """

    def __init__(self, primary_model,
                feature_names=None,
                param_grid=None,
                probability_threshold=0.5,
                random_state=65537):

        self.param_grid = param_grid
        self.probability_threshold = probability_threshold
        self.random_state = random_state

        if isinstance(primary_model, Pipeline):
            if len(primary_model.steps) != 2:
                logger.warning("Pipeline should have two steps: the preprocessing of the features, and the primary model to analyze.")
            estimator = primary_model.steps[-1][1]
            if not isinstance(estimator, BaseEstimator):
                raise TypeError("The last step of the pipeline has to be a BaseEstimator.")
            self._primary_model = estimator

            ct_preprocessor = primary_model.steps[0][1]
            if not isinstance(ct_preprocessor, ColumnTransformer):
                raise TypeError("The input preprocessor has to be a ColumnTransformer.")
            self.pipeline_preprocessor = PipelinePreprocessor(ct_preprocessor, feature_names)
        elif isinstance(primary_model, BaseEstimator):
            self._primary_model = primary_model
            self.pipeline_preprocessor = DummyPipelinePreprocessor(feature_names)
        else:
            raise TypeError('ErrorAnalyzer needs as input either a scikit BaseEstimator or a scikit Pipeline.')

        self._error_tree = None
        self._error_train_x = None
        self._error_train_y = None
        self.epsilon = None

    @property
    def param_grid(self):
        return self._param_grid

    @param_grid.setter
    def param_grid(self, value):
        self._param_grid = value

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, value):
        self._random_state = value

    @property
    def error_tree(self):
        if self._error_tree is None:
            raise NotFittedError("The error tree is not fitted yet. Call 'fit' method with appropriate arguments before using this estimator.")
        return self._error_tree

    @error_tree.setter
    def error_tree(self, tree):
        if self.pipeline_preprocessor.get_preprocessed_feature_names() is None:
            self.pipeline_preprocessor.preprocessed_feature_names = ["feature#%s" % feature_index
                for feature_index in range(tree.estimator_.n_features_)]
        self._error_tree = tree

    @property
    def preprocessed_feature_names(self):
        return self.pipeline_preprocessor.get_preprocessed_feature_names()

    def fit(self, X, y):
        """
        Fit the Error Analyzer Tree.

        Trains the Error Analyzer Tree, a Decision Tree to discriminate between samples that are correctly
        predicted or wrongly predicted (errors) by a primary model.

        Args:
            X (numpy.ndarray or pandas.DataFrame): feature data from a test set to evaluate the primary predictor and
                train a Error Analyzer Tree.
            y (numpy.ndarray or pandas.DataFrame): target data from a test set to evaluate the primary predictor and
                train a Error Analyzer Tree.
        """
        logger.info("Preparing the Error Analyzer Tree...")

        np.random.seed(self._random_state)
        preprocessed_X = self.pipeline_preprocessor.transform(X)

        check_enough_data(preprocessed_X, min_len=ErrorAnalyzerConstants.MIN_NUM_ROWS)
        self._error_train_y, error_rate = self._compute_primary_model_error(preprocessed_X, y)
        self._error_train_x = preprocessed_X

        logger.info("Fitting the Error Analyzer Tree...")
        # entropy/mutual information is used to split nodes in Microsoft Pandora system
        dt_clf = tree.DecisionTreeClassifier(criterion=ErrorAnalyzerConstants.CRITERION,
                                             random_state=self._random_state)
        param_grid = self.param_grid
        if param_grid is None:
            min_samples_leaf_max = min(error_rate, ErrorAnalyzerConstants.MIN_SAMPLES_LEAF_LOWEST_UPPER_BOUND)
            param_grid = {
                'max_depth': ErrorAnalyzerConstants.MAX_DEPTH,
                'min_samples_leaf': np.linspace(min_samples_leaf_max/5, min_samples_leaf_max, 5)
            }

        logger.info('Grid search the Error Tree with the following grid: {}'.format(param_grid))
        gs_clf = GridSearchCV(dt_clf,
                              param_grid=param_grid,
                              cv=5,
                              scoring=make_scorer(fidelity_balanced_accuracy_score))

        gs_clf.fit(self._error_train_x, self._error_train_y)
        self.error_tree = ErrorTree(error_decision_tree=gs_clf.best_estimator_)
        logger.info('Chosen parameters: {}'.format(gs_clf.best_params_))

    #TODO rewrite this method using the ranking arrays
    def get_error_leaf_summary(self, leaf_selector=None, add_path_to_leaves=False, output_format='dict', rank_by='total_error_fraction'):
        """ Return summary information regarding leaves.

        Args:
            leaf_selector (None, int or array-like): The leaves whose information will be returned
                * int: Only return information of the leaf with the corresponding id
                * array-like: Only return information of the leaves corresponding to these ids
                * None (default): Return information of all the leaves
            add_path_to_leaves (bool): Whether to add information of the path across the tree till the selected node. Defaults to False.
            output_format (string): Return format used for the report. Valid values are 'dict' or 'str'. Defaults to 'dict'.
            rank_by (str): Ranking criterion for the leaves. Valid values are:
                * 'total_error_fraction' (default): rank by the fraction of total error in the node
                * 'purity': rank by the purity (ratio of wrongly predicted samples over the total number of node samples)
                * 'class_difference': rank by the difference of number of wrongly and correctly predicted samples
                in a node.

        Return:
            dict or str: list of reports (as dictionary or string) with different information on each selected leaf.
        """

        leaf_nodes = self._get_ranked_leaf_ids(leaf_selector=leaf_selector, rank_by=rank_by)

        leaves_summary = []
        for leaf_id in leaf_nodes:
            n_errors = int(self.error_tree.estimator_.tree_.value[leaf_id, 0, self.error_tree.error_class_idx])
            n_samples = self.error_tree.estimator_.tree_.n_node_samples[leaf_id]
            local_error = n_errors / n_samples
            total_error_fraction = n_errors / self.error_tree.n_total_errors
            n_corrects = n_samples - n_errors

            if output_format == 'dict':
                leaf_dict = {
                    "id": leaf_id,
                    "n_corrects": n_corrects,
                    "n_errors": n_errors,
                    "local_error": local_error,
                    "total_error_fraction": total_error_fraction
                }
                if add_path_to_leaves:
                    leaf_dict["path_to_leaf"] = self._get_path_to_node(leaf_id)
                leaves_summary.append(leaf_dict)

            elif output_format == 'str':
                leaf_summary = 'LEAF %d:\n' % leaf_id
                leaf_summary += '     Correct predictions: %d | Wrong predictions: %d | Local error (purity): %.2f | Fraction of total error: %.2f\n' % (n_corrects, n_errors, local_error, total_error_fraction)

                if add_path_to_leaves:
                    leaf_summary += '     Path to leaf:\n'
                    for (step_idx, step) in enumerate(self._get_path_to_node(leaf_id)):
                        leaf_summary += '     ' + '   ' * step_idx + step  + '\n'

                leaves_summary.append(leaf_summary)

            else:
                raise ValueError("Output format should either be 'dict' or 'str'")

        return leaves_summary

    def evaluate(self, X, y, output_format='str'):
        """
        Evaluate performance of ErrorAnalyzer on the given test data and labels.
        Return ErrorAnalyzer summary metrics regarding the Error Tree.

        Args:
            X (numpy.ndarray or pandas.DataFrame): feature data from a test set to evaluate the primary predictor
                and train a Error Analyzer Tree.
            y (numpy.ndarray or pandas.DataFrame): target data from a test set to evaluate the primary predictor and
                train a Error Analyzer Tree.
            output_format (string): Return format used for the report. Valid values are 'dict' or 'str'. Defaults to 'str'.

        Return:
            dict or str: dictionary or string report storing different metrics regarding the Error Decision Tree.
        """
        prep_x, prep_y = self.pipeline_preprocessor.transform(X), np.array(y)
        y_pred = self.error_tree.estimator_.predict(prep_x)
        y_true, _ = self._compute_primary_model_error(prep_x, prep_y)
        return error_decision_tree_report(y_true, y_pred, output_format)

    def _compute_primary_model_error(self, X, y):
        """
        Computes the errors of the primary model predictions and samples

        Args:
            X: array-like of shape (n_samples, n_features)
            Input samples.

            y: array-like of shape (n_samples,)
            True target values for `X`.

        Returns:
             sampled_X: ndarray
             A sample of `X`.

             error_y: array of string of shape (n_sampled_X, )
             Boolean value of whether or not the primary model predicted correctly or incorrectly the samples in sampled_X.
        """
        if is_regressor(self._primary_model) or len(np.unique(y)) > 2:
            # regression or multiclass classification models: no proba threshold
            y_pred = self._primary_model.predict(X)
        else: # binary -> need to check the proba threshold
            prediction_index = (self._primary_model.predict_proba(X)[:, 1] > self.probability_threshold).astype(int)
            # map the prediction indexes to the original target values
            y_pred = np.array([self._primary_model.classes_[i] for i in prediction_index])

        error_y, error_rate = self._evaluate_primary_model_predictions(y_true=y, y_pred=y_pred)
        return error_y, error_rate

    def _evaluate_primary_model_predictions(self, y_true, y_pred):
        """
        Compute errors of the primary model on the test set

        Args:
            y_true: 1D array
            True target values.

            y_pred: 1D array
            Predictions of the primary model.

        Return:
            error_y: array of string of len(y_true)
            Boolean value of whether or not the primary model got the prediction right.

            error_rate: float
            Accuracy of the primary model
        """

        error_y = np.full_like(y_true, ErrorAnalyzerConstants.CORRECT_PREDICTION, dtype="O")
        if is_regressor(self._primary_model):
            difference = np.abs(y_true - y_pred)
            if self.epsilon is None:
                # only compute epsilon when fitting the model (not while evaluating)
                self.epsilon = get_epsilon(difference)
            error_mask = difference > self.epsilon
        else:
            error_mask = y_true != y_pred

        n_wrong_preds = np.count_nonzero(error_mask)
        error_y[error_mask] = ErrorAnalyzerConstants.WRONG_PREDICTION

        if n_wrong_preds == 0 or n_wrong_preds == len(error_y):
            logger.warning('All predictions are {}. To build a proper ErrorAnalyzer decision tree we need both correct and incorrect predictions'.format(error_y[0]))

        error_rate = n_wrong_preds / len(error_y)
        logger.info('The primary model has an error rate of {}'.format(error_rate))
        return error_y, error_rate

    def _get_ranked_leaf_ids(self, leaf_selector=None, rank_by='total_error_fraction'):
        """ Select error nodes and rank them by importance.

        Args:
            leaf_selector (None, int or array-like): the leaves whose information will be returned
                * int: Only return information of the leaf with the corresponding id
                * array-like: Only return information of the leaves corresponding to these ids
                * None (default): Return information of all the leaves
            rank_by (str): ranking criterion for the leaves. Valid values are:
                * 'total_error_fraction': rank by the fraction of total error in the node
                * 'purity': rank by the purity (ratio of wrongly predicted samples over the total number of node samples)
                * 'class_difference': rank by the difference of number of wrongly and correctly predicted samples
                in a node.

        Return:
            list or numpy.ndarray: list of selected leaves indices.

        """
        apply_leaf_selector = self._get_leaf_selector(leaf_selector)
        selected_leaves = apply_leaf_selector(self.error_tree.leaf_ids)
        if selected_leaves.size == 0:
            return selected_leaves
        if rank_by == 'total_error_fraction':
            sorted_ids = np.argsort(-apply_leaf_selector(self.error_tree.total_error_fraction))
        elif rank_by == 'purity':
            sorted_ids = np.lexsort((apply_leaf_selector(self.error_tree.difference), apply_leaf_selector(self.error_tree.quantized_impurity)))
        elif rank_by == 'class_difference':
            sorted_ids = np.lexsort((apply_leaf_selector(self.error_tree.impurity), apply_leaf_selector(self.error_tree.difference)))
        else:
            raise ValueError(
                "Input argument for rank_by is invalid. Should be 'total_error_fraction', 'purity' or 'class_difference'")
        return selected_leaves.take(sorted_ids)

    #TODO leaf_selector is taking too many different types of data ?
    def _get_leaf_selector(self, leaf_selector):
        """
        Return a function that select rows of provided arrays. Arrays must be of shape (1, number of leaves)
            Args:
                leaf_selector: None, int or array-like
                    How to select the rows of the array
                      * int: Only keep the row corresponding to this leaf id
                      * array-like: Only keep the rows corresponding to these leaf ids
                      * None (default): Keep the whole array of leaf ids

            Return:
                A function with one argument array as a selector of leaf ids
                Args:
                    array: numpy array of shape (1, number of leaves)
                    An array of which we only want to keep some rows
        """
        if leaf_selector is None:
            return lambda array: array

        leaf_selector_as_array = np.array(leaf_selector)
        leaf_selector = np.in1d(self.error_tree.leaf_ids, leaf_selector_as_array)
        nr_kept_leaves = np.count_nonzero(leaf_selector)
        if nr_kept_leaves == 0:
            print("None of the ids provided correspond to a leaf id.")
        elif nr_kept_leaves < leaf_selector_as_array.size:
            print("Some of the ids provided do not belong to leaves. Only leaf ids are kept.")
        return lambda array: array[leaf_selector]

    def _get_path_to_node(self, node_id):
        """ Return path to node as a list of split steps from the nodes of the sklearn Tree object """
        feature_names = self.pipeline_preprocessor.get_original_feature_names()
        children_left = list(self.error_tree.estimator_.tree_.children_left)
        children_right = list(self.error_tree.estimator_.tree_.children_right)
        threshold = self._inverse_transform_thresholds()
        feature = self._inverse_transform_features()

        cur_node_id = node_id
        path_to_node = collections.deque()
        while cur_node_id > 0:

            node_is_left_child = cur_node_id in children_left
            if node_is_left_child:
                parent_id = children_left.index(cur_node_id)
            else:
                parent_id = children_right.index(cur_node_id)

            feat = feature[parent_id]
            thresh = threshold[parent_id]

            is_categorical = self.pipeline_preprocessor.is_categorical(feat)
            thresh = str(thresh) if is_categorical else "{:.2f}".format(thresh)

            decision_rule = ''
            if node_is_left_child:
                decision_rule += ' <= ' if not is_categorical else ' != '
            else:
                decision_rule += " > " if not is_categorical else ' == '

            decision_rule = str(feature_names[feat]) + decision_rule + thresh
            path_to_node.appendleft(decision_rule)
            cur_node_id = parent_id

        return path_to_node

    #TODO naming is not very clear ?
    def _inverse_transform_features(self):
        """ Undo preprocessing of feature values.

        If the predictor comes with a Pipeline preprocessor, map the features indices of the Error Analysis
        Tree back to their indices in the original unpreprocessed space of features.
        Otherwise simply return the feature indices of the decision tree. The feature indices of a decision tree
        indicate what features are used to split the training set at each node.
        See https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html.

        Return:
            list or numpy.ndarray:
                indices of features of the Error Analyzer Tree, possibly mapped back to the
                original unprocessed feature space.
        """
        return [self.pipeline_preprocessor.inverse_transform_feature_id(feat_idx) if feat_idx > 0 else feat_idx
            for feat_idx in self.error_tree.estimator_.tree_.feature]

    #TODO naming is not very clear ?
    def _inverse_transform_thresholds(self):
        """  Undo preprocessing of feature threshold values.

        If the predictor comes with a Pipeline preprocessor, undo the preprocessing on the thresholds of the Error Analyzer
        Tree for an easier plot interpretation. Otherwise simply return the thresholds of
        the decision tree. The thresholds of a decision tree are the feature values used to split the training set at
        each node. See https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html.

        Return:
            numpy.ndarray:
                thresholds of the Error Tree, possibly with preprocessing undone.
        """
        return self.pipeline_preprocessor.inverse_thresholds(self.error_tree.estimator_.tree_, self._error_train_x.shape[1])
