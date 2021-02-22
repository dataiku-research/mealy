# -*- coding: utf-8 -*-
import numpy as np
import collections
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.base import is_regressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, accuracy_score
from mealy.error_analysis_utils import check_enough_data, get_epsilon
from mealy.constants import ErrorAnalyzerConstants
from mealy.metrics import mpp_report, fidelity_balanced_accuracy_score
from mealy.preprocessing import PipelinePreprocessor, DummyPipelinePreprocessor
from mealy.error_tree import ErrorTree
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='mealy | %(levelname)s - %(message)s')


class ErrorAnalyzer(BaseEstimator):
    """ ErrorAnalyzer analyzes the errors of a prediction model on a test set.

    It uses model predictions and ground truth target to compute the model errors on the test set.
    It then trains a Decision Tree, called a Error Analyzer Tree, on the same test set by using the model error
    as target. The nodes of the decision tree are different segments of errors to be studied individually.

    Args:
        original_model (sklearn.base.BaseEstimator or sklearn.pipeline.Pipeline): a sklearn model to analyze. Either an estimator
            or a Pipeline containing a ColumnTransformer with the preprocessing steps and an estimator as last step.
        feature_names (list): list of feature names, default=None.
        max_num_row (int): maximum number of rows to process.
        param_grid (dict): sklearn.tree.DecisionTree hyper-parameters values for grid search.
        random_state (int): random seed.

    Attributes:
        _error_tree (DecisionTreeClassifier): the estimator used to train the Error Analyzer Tree
    """

    def __init__(self, original_model,
                 feature_names=None,
                 max_num_row=ErrorAnalyzerConstants.MAX_NUM_ROW,
                 param_grid=ErrorAnalyzerConstants.PARAMETERS_GRID,
                 random_state=65537):

        self.feature_names = feature_names
        self.max_num_row = max_num_row
        self.param_grid = param_grid
        self.random_state = random_state

        if isinstance(original_model, Pipeline):
            estimator = original_model.steps[-1][1]
            if not isinstance(estimator, BaseEstimator):
                raise NotImplementedError("The last step of the pipeline has to be a BaseEstimator.")
            self._original_model = estimator
            ct_preprocessor = Pipeline(original_model.steps[:-1]).steps[0][1]
            if not isinstance(ct_preprocessor, ColumnTransformer):
                raise NotImplementedError("The input preprocessor has to be a ColumnTransformer.")
            self.pipeline_preprocessor = PipelinePreprocessor(ct_preprocessor, feature_names)
            self._preprocessed_feature_names = self.pipeline_preprocessor.get_preprocessed_feature_names()
        elif isinstance(original_model, BaseEstimator):
            self._original_model = original_model
            self._preprocessed_feature_names = feature_names
            self.pipeline_preprocessor = DummyPipelinePreprocessor(feature_names)
        else:
            raise ValueError('ErrorAnalyzer needs as input either a scikit Estimator or a scikit Pipeline.')

        self._error_tree = None
        self._is_regression = is_regressor(self._original_model)
        self._error_train_x = None
        self._error_train_y = None

    @property
    def feature_names(self):
        return self._feature_names

    @feature_names.setter
    def feature_names(self, value):
        self._feature_names = value

    @property
    def original_model(self):
        return self._original_model

    @original_model.setter
    def original_model(self, value):
        self._original_model = value

    @property
    def max_num_row(self):
        return self._max_num_row

    @max_num_row.setter
    def max_num_row(self, value):
        self._max_num_row = value

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
        return self._error_tree

    @property
    def preprocessed_feature_names(self):
        if self._preprocessed_feature_names is None:
            self._preprocessed_feature_names = ["feature#%s" % feature_index
                                                for feature_index in
                                                range(self._error_tree.estimator_.n_features_)]

        return self._preprocessed_feature_names

    """ 
    def get_preprocessed_feature_names(self):
        if self._preprocessed_feature_names is None:
            self._preprocessed_feature_names = ["feature#%s" % feature_index
                                                for feature_index in
                                                range(self._error_tree.estimator_.n_features_)]

        return self._preprocessed_feature_names
    """

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
        preprocessed_x = self.pipeline_preprocessor.transform(X)
        self._error_train_x, self._error_train_y, error_rate = self._compute_primary_model_error(preprocessed_x, y)

        logger.info("Fitting the Error Analyzer Tree...")
        # entropy/mutual information is used to split nodes in Microsoft Pandora system
        dt_clf = tree.DecisionTreeClassifier(criterion=ErrorAnalyzerConstants.CRITERION,
                                             random_state=self._random_state)
        param_grid = {
            'max_depth': [3, 5, 10],
            'min_samples_leaf': np.linspace(error_rate/5, error_rate, 5)
        }

        logger.info('Grid search the Error Tree with the following grid: {}'.format(param_grid))
        gs_clf = GridSearchCV(dt_clf,
                              param_grid=param_grid, #self._param_grid,
                              cv=5,
                              scoring=make_scorer(fidelity_balanced_accuracy_score))

        gs_clf.fit(self._error_train_x, self._error_train_y)
        self._error_tree = ErrorTree(error_decision_tree=gs_clf.best_estimator_)
        logger.info('Chosen parameters: {}'.format(gs_clf.best_params_))

    #TODO: rewrite this method using the ranking arrays
    def get_error_node_summary(self, leaf_selector='all_errors', add_path_to_leaves=False, print_summary=False):
        """ Return summary information regarding input nodes.

        Args:
            leaf_selector (int or list or str): the desired leaf nodes to visualize. When int it represents the
                number of the leaf node, when a list it represents a list of leaf nodes. When a string, the valid values
                are either 'all_error' to plot all leaves of class 'Wrong prediction' or 'all' to plot all leaf nodes.
            add_path_to_leaves (bool): add information of the feature path across the tree till the selected node.
            print_summary (bool): print summary for the selected nodes.

        Return:
            dict: dictionary of metrics for each selected node of the Error Analyzer Tree.
        """

        leaf_nodes = self._get_ranked_leaf_ids(leaf_selector=leaf_selector)

        y = self._error_train_y
        n_total_errors = y[y == ErrorAnalyzerConstants.WRONG_PREDICTION].shape[0]
        error_class_idx = np.where(self._error_tree.estimator_.classes_ == ErrorAnalyzerConstants.WRONG_PREDICTION)[0][0]
        correct_class_idx = 1 - error_class_idx

        leaves_summary = []
        path_to_node = None
        for leaf_id in leaf_nodes:
            values = self._error_tree.estimator_.tree_.value[leaf_id, :]
            n_errors = int(np.ceil(values[0, error_class_idx]))
            n_corrects = int(np.ceil(values[0, correct_class_idx]))
            local_error = float(n_errors) / (n_corrects + n_errors)
            global_error = float(n_errors) / n_total_errors

            leaf_dict = {
                "id": leaf_id,
                "n_corrects": n_corrects,
                "n_errors": n_errors,
                "local_error": local_error,
                "global_error": global_error
            }

            leaves_summary.append(leaf_dict)

            if add_path_to_leaves:
                path_to_node = self._get_path_to_node(leaf_id)
                leaf_dict["path_to_leaf"] = path_to_node

            if print_summary:
                print("LEAF %d:" % leaf_id)
                print("     Correct predictions: %d | Wrong predictions: %d | "
                      "Local error (purity): %.2f | Global error: %.2f" %
                      (n_corrects, n_errors, local_error, global_error))

                if add_path_to_leaves:
                    print('     Path to leaf:')
                    for (step_idx, step) in enumerate(path_to_node):
                        print('     ' + '   ' * step_idx + step)

        return leaves_summary

    def evaluate(self, X, y, output_format='text'):
        """
        Evaluate performance of ErrorAnalyzer on new the given test data and labels.
        Print ErrorAnalyzer summary metrics regarding the Error Analyzer Tree.

        Args:
            X (numpy.ndarray or pandas.DataFrame): feature data from a test set to evaluate the primary predictor
                and train a Error Analyzer Tree.
            y (numpy.ndarray or pandas.DataFrame): target data from a test set to evaluate the primary predictor and
                train a Error Analyzer Tree.
            output_format (string): whether to return a "dict" or a "text"

        Return:
            dict or str: metrics regarding the Error Analyzer Tree.
        """
        prep_x, prep_y = self.pipeline_preprocessor.transform(X), np.array(y)
        prep_x, y_true, _ = self._compute_primary_model_error(prep_x, prep_y)
        y_pred = self._error_tree.estimator_.predict(prep_x)
        return mpp_report(y_true, y_pred, output_format)

    def _prepare_data(self, X, y):
        """Check and sample data

        Args:
            X: array-like of shape (n_samples, n_features)
            Input samples.

            y: array-like of shape (n_samples,)
            The target values

        Returns:
            sampled_X: ndarray of shape (new_n_samples, n_features)
            sampled_y: array of shape (new_n_samples,)
        """

        check_enough_data(X, min_len=ErrorAnalyzerConstants.MIN_NUM_ROWS)
        logger.info("Sampling data: original dataset had {} rows, selecting the first {}.".format(X.shape[0], self._max_num_row))
        sampled_X = X[:self._max_num_row, :]
        sampled_y = y[:self._max_num_row]

        return sampled_X, sampled_y

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
             Boolean value of whether or not the original model predicted correctly or incorrectly the samples in sampled_X.
        """
        logger.info('Prepare data with model for Error Analyzer Tree')

        sampled_X, sampled_y = self._prepare_data(X, y)
        y_pred = self._original_model.predict(sampled_X)
        error_y, error_rate = self._evaluate_primary_model_predictions(y_true=sampled_y, y_pred=y_pred)
        return sampled_X, error_y, error_rate

    def _evaluate_primary_model_predictions(self, y_true, y_pred):
        """
        Compute errors of the primary model on the test set

        Args:
            y_true: 1D array
            True target values.

            y_pred: 1D array
            Predictions of the primary model.

        Return:
            error_y: array of string of len(y_trye)
            Boolean value of whether or not the primary model got the prediction right.

            error_rate: float
            Accuracy of the original model
        """

        if self._is_regression:
            difference = np.abs(y_true - y_pred)
            epsilon = get_epsilon(difference, mode='rec')
            error_array = np.array(difference > epsilon)
        else:
            error_array = np.array(y_true != y_pred)

        target_mapping_dict = {True: ErrorAnalyzerConstants.WRONG_PREDICTION, False: ErrorAnalyzerConstants.CORRECT_PREDICTION}
        error_y = np.array([target_mapping_dict[elem] for elem in error_array], dtype=object)

        possible_outcomes = list(set(error_y.tolist()))
        if len(possible_outcomes) == 1:
            logger.warning(
                'All predictions are {}. To build a proper ErrorAnalyzer decision tree we need both correct and incorrect predictions'.format(
                    possible_outcomes[0]))

        error_rate = np.sum(error_array, dtype=float)/len(error_array)
        logger.info('The original model has an error rate of {}'.format(round(error_rate, 3)))
        return error_y, error_rate

    def _get_ranked_leaf_ids(self, leaf_selector, rank_leaves_by='purity'):
        """ Select error nodes and rank them by importance.

        Args:
            leaf_selector (int or list or str): the desired leaf nodes to visualize. When int it represents the
                number of the leaf node, when a list it represents a list of leaf nodes. When a string, the valid values are
                either 'all_error' to plot all leaves of class 'Wrong prediction' or 'all' to plot all leaf nodes.
            rank_leaves_by (str): ranking criterium for the leaf nodes. It can be either 'purity' to rank by the leaf
                node purity (ratio of wrongly predicted samples over the total for an error node) or 'class_difference'
                (difference of number of wrongly and correctly predicted samples in a node).

        Return:
            list or numpy.ndarray: list of selected leaf nodes indices.

        """
        apply_leaf_selector = self._get_leaf_selector(leaf_selector)
        selected_leaves = apply_leaf_selector(self._error_tree.leaf_ids)
        if selected_leaves.size == 0:
            return selected_leaves
        if rank_leaves_by == 'purity':
            sorted_ids = np.lexsort(
                (apply_leaf_selector(self._error_tree.difference), apply_leaf_selector(self._error_tree.quantized_impurity)))
        elif rank_leaves_by == 'class_difference':
            sorted_ids = np.lexsort((apply_leaf_selector(self._error_tree.impurity), apply_leaf_selector(self._error_tree.difference)))
        else:
            raise NotImplementedError("Input value for 'rank_leaves_by' is invalid. It must be 'purity' or 'class_difference'.")
        return selected_leaves.take(sorted_ids)

    #TODO leaf_selector is taking too many different types of data ?
    def _get_leaf_selector(self, leaf_selector):
        """
        Return a function that select rows of provided arrays. Arrays must be of shape (1, number of leaves)
            Args:
                leaf_selector: int, str, or array-like
                How to select the rows of the array
                  * int: Only keep the row corresponding to this leaf id
                  * array-like: Only keep the rows corresponding to these leaf ids
                  * str:
                    - "all": Keep the whole array
                    - "all_errors": Keep the rows with indices corresponding to the leaf ids classifying the primary model prediction as wrong

            Return:
                A function with one argument array as a selector of leaf ids
                Args:
                    array: numpy array of shape (1, number of leaves)
                    An array of which we only want to keep some rows
        """
        if isinstance(leaf_selector, str):
            if leaf_selector == "all":
                return lambda array: array
            elif leaf_selector == "all_errors":
                return lambda array: array[self._error_tree.get_error_leaves()]
            else:
                raise ValueError('Unknown string value "{}" for leaf_selector, please choose either "all" or "all_errors".'.format(leaf_selector))

        leaf_selector_as_array = np.array(leaf_selector)
        leaf_selector = np.in1d(self._error_tree.leaf_ids, leaf_selector_as_array)
        nr_kept_leaves = np.count_nonzero(leaf_selector)
        if nr_kept_leaves == 0:
            print("None of the ids provided correspond to a leaf id.")
        elif nr_kept_leaves < leaf_selector_as_array.size:
            print("Some of the ids provided do not belong to leaves. Only leaf ids are kept.")
        return lambda array: array[leaf_selector]

    def _get_path_to_node(self, node_id):
        """ Return path to node as a list of split steps from the nodes of the sklearn Tree object """
        feature_names = self.pipeline_preprocessor.get_original_feature_names()
        children_left = self._error_tree.estimator_.tree_.children_left
        children_right = self._error_tree.estimator_.tree_.children_right
        threshold = self._inverse_transform_thresholds()
        feature = self._inverse_transform_features()

        cur_node_id = node_id
        path_to_node = collections.deque()
        while cur_node_id > 0:

            if cur_node_id in children_left:
                parent_id = list(children_left).index(cur_node_id)
            else:
                parent_id = list(children_right).index(cur_node_id)

            feat = feature[parent_id]
            thresh = threshold[parent_id]

            is_categorical = self.pipeline_preprocessor.is_categorical(feat)
            thresh = str(thresh if is_categorical else ("%.2f" % thresh))

            decision_rule = ''
            if cur_node_id in children_left:
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
        Decision Tree back to their indices in the original unpreprocessed space of features.
        Otherwise simply return the feature indices of the decision tree. The feature indices of a decision tree
        indicate what features are used to split the training set at each node.
        See https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html.

        Return:
            list or numpy.ndarray:
                indices of features of the Error Analyzer Tree, possibly mapped back to the
                original unprocessed feature space.
        """
        feats_idx = self._error_tree.estimator_.tree_.feature.copy()

        for i, f in enumerate(feats_idx):
            if f > 0:
                feats_idx[i] = self.pipeline_preprocessor.inverse_transform_feature_id(f)

        return feats_idx

    #TODO naming is not very clear ?
    def _inverse_transform_thresholds(self):
        """  Undo preprocessing of feature threshold values.

        If the predictor comes with a Pipeline preprocessor, undo the preprocessing on the thresholds of the Error Analyzer
        Tree for an easier plot interpretation. Otherwise simply return the thresholds of
        the decision tree. The thresholds of a decision tree are the feature values used to split the training set at
        each node. See https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html.

        Return:
            numpy.ndarray:
                thresholds of the Error Analyzer Tree, possibly with preprocessing undone.
        """

        feats_idx = self._error_tree.estimator_.tree_.feature[self._error_tree.estimator_.tree_.feature > 0]
        thresholds = self._error_tree.estimator_.tree_.threshold.copy().astype('O')
        thresh = thresholds[self._error_tree.estimator_.tree_.feature > 0]
        n_rows = np.count_nonzero(self._error_tree.estimator_.tree_.feature[self._error_tree.estimator_.tree_.feature > 0])
        n_cols = self._error_train_x.shape[1]
        dummy_x = np.zeros((n_rows, n_cols))

        indices = []
        i = 0

        for f, t in zip(feats_idx, thresh):
            dummy_x[i, f] = t
            indices.append((i, self.pipeline_preprocessor.inverse_transform_feature_id(f)))
            i += 1

        undo_dummy_x = self.pipeline_preprocessor.inverse_transform(dummy_x)
        descaled_thresh = [undo_dummy_x[i, j] for i, j in indices]
        thresholds[self._error_tree.estimator_.tree_.feature > 0] = descaled_thresh

        return thresholds

