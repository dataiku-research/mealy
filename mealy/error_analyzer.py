# -*- coding: utf-8 -*-
import numpy as np
import collections
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.base import is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from kneed import KneeLocator
import logging

from mealy.error_analysis_utils import check_enough_data
from mealy.constants import ErrorAnalyzerConstants
from mealy.metrics import mpp_report, fidelity_balanced_accuracy_score
from mealy.preprocessing import PipelinePreprocessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='mealy | %(levelname)s - %(message)s')


class ErrorAnalyzer(object):
    """ ErrorAnalyzer analyzes the errors of a prediction model on a test set.

    It uses model predictions and ground truth target to compute the model errors on the test set.
    It then trains a Decision Tree, called a Model Performance Predictor, on the same test set by using the model error
    as target. The nodes of the decision tree are different segments of errors to be studied individually.

    Args:
        predictor (sklearn.base.BaseEstimator or sklearn.pipeline.Pipeline): a sklearn model to analyze. Either an estimator
            or a Pipeline containing a ColumnTransformer with the preprocessing steps and an estimator as last step.
        feature_names (list): list of feature names, default=None.
        seed (int): random seed.

    Attributes:
        error_train_x (numpy.ndarray): features used to train the Model performance Predictor.
        error_train_y (numpy.ndarray): target used to train the Model performance Predictor, it is abinary variable
            representing whether the input predictor predicted correctly or incorrectly the samples in error_train_x.
        model_performance_predictor_features (list): feature names used in the Model Performance Predictor.
        model_performance_predictor (sklearn.tree.DecisionTreeClassifier): performance predictor decision tree.
        train_leaf_ids (numpy.ndarray): indices of leaf in the Model Performance Predictor, where each of the training
            sample falls.
        impurity (numpy.ndarray): impurity of leaf nodes (used for ranking the nodes).
        quantized_impurity (numpy.ndarray): quantized impurity of leaf nodes (used for ranking the nodes).
        difference (numpy.ndarray): difference of number of correctly and incorrectly predicted samples in leaf nodes
            (used for ranking the nodes).
        leaf_ids (numpy.ndarray): list of all leaf nodes indices.
    """

    def __init__(self, predictor, feature_names=None, seed=65537):
        if isinstance(predictor, Pipeline):
            estimator = predictor.steps[-1][1]
            if not isinstance(estimator, BaseEstimator):
                raise NotImplementedError("The last step of the pipeline has to be a BaseEstimator.")
            self._predictor = estimator

            ct_preprocessor = Pipeline(predictor.steps[:-1]).steps[0][1]

            if not isinstance(ct_preprocessor, ColumnTransformer):
                raise NotImplementedError("The input preprocessor has to be a ColumnTransformer.")
            self.pipeline_preprocessor = PipelinePreprocessor(ct_preprocessor, feature_names)

            self._features_in_model_performance_predictor = self.pipeline_preprocessor.get_preprocessed_feature_names()

        else:
            self._predictor = predictor
            self._features_in_model_performance_predictor = feature_names
            self.pipeline_preprocessor = None

        self._is_regression = is_regressor(self._predictor)

        self._error_clf = None
        self._error_train_x = None
        self._error_train_y = None

        self._error_train_leaf_id = None
        self._leaf_ids = None

        self._impurity = None
        self._quantized_impurity = None
        self._difference = None

        self._error_clf_thresholds = None
        self._error_clf_features = None

        self._seed = seed

    @property
    def error_train_x(self):
        return self._error_train_x

    @property
    def error_train_y(self):
        return self._error_train_y

    @property
    def model_performance_predictor_features(self):
        if self._features_in_model_performance_predictor is None:
            self._features_in_model_performance_predictor = ["feature#%s" % feature_index
                                                             for feature_index in range(self._error_clf.n_features_)]

        return self._features_in_model_performance_predictor

    @property
    def model_performance_predictor(self):
        if self._error_clf is None:
            raise NotFittedError("You should fit a model performance predictor first")
        return self._error_clf

    @property
    def train_leaf_ids(self):
        if self._error_train_leaf_id is None:
            self._compute_train_leaf_ids()
        return self._error_train_leaf_id

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
    def leaf_ids(self):
        if self._leaf_ids is None:
            self._compute_leaf_ids()
        return self._leaf_ids

    def fit(self, x, y, max_nr_rows=ErrorAnalyzerConstants.MAX_NUM_ROW):
        """ Fit the Model Performance Predictor.

        Trains the Model Performance Predictor, a Decision Tree to discriminate between samples that are correctly
        predicted or wrongly predicted (errors) by a primary model.

        Args:
            x (numpy.ndarray or pandas.DataFrame): feature data from a test set to evaluate the primary predictor and
                train a Model Performance Predictor.
            y (numpy.ndarray or pandas.DataFrame): target data from a test set to evaluate the primary predictor and
                train a Model Performance Predictor.
            max_nr_rows (int): maximum number of rows to process.
        """
        logger.info("Preparing the model performance predictor...")

        np.random.seed(self._seed)

        if self.pipeline_preprocessor is None:
            prep_x, prep_y = x, y
        else:
            prep_x, prep_y = self.pipeline_preprocessor.transform(x), np.array(y)

        self._error_train_x, self._error_train_y = self._compute_primary_model_error(prep_x, prep_y, max_nr_rows)

        possible_outcomes = list(set(self._error_train_y.tolist()))
        if len(possible_outcomes) == 1:
            logger.warning('All predictions are {}. To build a proper MPP decision tree we need both correct and incorrect predictions'.format(possible_outcomes[0]))

        logger.info("Fitting the model performance predictor...")

        # entropy/mutual information is used to split nodes in Microsoft Pandora system
        criterion = ErrorAnalyzerConstants.CRITERION

        dt_clf = tree.DecisionTreeClassifier(criterion=criterion, random_state=self._seed)
        gs_clf = GridSearchCV(dt_clf, param_grid=ErrorAnalyzerConstants.PARAMETERS_GRID,
                              cv=5, scoring=make_scorer(fidelity_balanced_accuracy_score))

        gs_clf.fit(self._error_train_x, self._error_train_y)
        self._error_clf = gs_clf.best_estimator_

        logger.info('Grid search selected parameters:')
        logger.info(gs_clf.best_params_)

        if sum(self._error_clf.tree_.feature > 0) == 0:
            logger.warning("The MPP tree has only 1 node, there will be problem when using this with ErrorVisualizer")

    def _compute_primary_model_error(self, x, y, max_nr_rows):
        """
        Computes the errors of the primary model predictions and samples
        :return: an array with error target (correctly predicted vs wrongly predicted)
        """

        logger.info('Prepare data with model for model performance predictor')

        check_enough_data(x, min_len=ErrorAnalyzerConstants.MIN_NUM_ROWS)

        if x.shape[0] > max_nr_rows:
            logger.info("Rebalancing data: original dataset had {} rows, selecting the first {}.".format(x.shape[0],
                                                                                                         max_nr_rows))

            x = x[:max_nr_rows, :]
            y = y[:max_nr_rows]

        y_pred = self._predictor.predict(x)

        error_y = self._get_errors(y, y_pred)

        return x, error_y

    def predict(self, x):
        """ Predict model performance on samples

        Args:
            x (numpy.ndarray or pandas.DataFrame): dataset where to apply the Model Performance Predictor.

        Return:
            numpy.ndarray: predictions from the Model Performance Predictor (Wrong/Correct primary predictions).
        """
        if self.pipeline_preprocessor is None:
            prep_x = x
        else:
            prep_x = self.pipeline_preprocessor.transform(x)
        return self.model_performance_predictor.predict(prep_x)

    @staticmethod
    def _get_epsilon(difference, mode='rec'):
        """ Compute epsilon to define errors in regression task """
        assert (mode in ['std', 'rec'])
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

    def _get_errors(self, y, y_pred):
        """ Compute errors of the primary model on the test set """
        if self._is_regression:

            difference = np.abs(y - y_pred)

            epsilon = ErrorAnalyzer._get_epsilon(difference, mode='rec')

            error = difference > epsilon
        else:

            error = (y != y_pred)

        error = list(error)
        transdict = {True: ErrorAnalyzerConstants.WRONG_PREDICTION, False: ErrorAnalyzerConstants.CORRECT_PREDICTION}
        error = np.array([transdict[elem] for elem in error], dtype=object)

        return error

    def _compute_train_leaf_ids(self):
        """ Compute indices of leaf nodes for the train set """
        self._error_train_leaf_id = self.model_performance_predictor.apply(self._error_train_x)

    def _compute_leaf_ids(self):
        """ Compute indices of leaf nodes """
        self._leaf_ids = np.where(self.model_performance_predictor.tree_.feature < 0)[0]

    def _get_error_leaves(self):
        error_class_idx = \
        np.where(self.model_performance_predictor.classes_ == ErrorAnalyzerConstants.WRONG_PREDICTION)[0][0]
        error_node_ids = \
        np.where(self.model_performance_predictor.tree_.value[:, 0, :].argmax(axis=1) == error_class_idx)[0]
        return np.in1d(self.leaf_ids, error_node_ids)

    def _compute_ranking_arrays(self, n_purity_levels=ErrorAnalyzerConstants.NUMBER_PURITY_LEVELS):
        """ Compute ranking array """
        error_class_idx = \
        np.where(self.model_performance_predictor.classes_ == ErrorAnalyzerConstants.WRONG_PREDICTION)[0][0]
        correct_class_idx = 1 - error_class_idx

        wrongly_predicted_samples = self.model_performance_predictor.tree_.value[self.leaf_ids, 0, error_class_idx]
        correctly_predicted_samples = self.model_performance_predictor.tree_.value[self.leaf_ids, 0, correct_class_idx]

        self._impurity = correctly_predicted_samples / (wrongly_predicted_samples + correctly_predicted_samples)

        purity_bins = np.linspace(0, 1., n_purity_levels)
        self._quantized_impurity = np.digitize(self._impurity, purity_bins)
        self._difference = correctly_predicted_samples - wrongly_predicted_samples  # only negative numbers

    def get_ranked_leaf_ids(self, leaf_selector, rank_by='purity'):
        """ Select error nodes and rank them by importance.

        Args:
            leaf_selector (int or list or str): the desired leaf nodes to visualize. When int it represents the
                number of the leaf node, when a list it represents a list of leaf nodes. When a string, the valid values are
                either 'all_error' to plot all leaves of class 'Wrong prediction' or 'all' to plot all leaf nodes.
            rank_by (str): ranking criterium for the leaf nodes. It can be either 'purity' to rank by the leaf
                node purity (ratio of wrongly predicted samples over the total for an error node) or 'class_difference'
                (difference of number of wrongly and correctly predicted samples in a node).

        Return:
            list or numpy.ndarray: list of selected leaf nodes indices.

        """
        apply_leaf_selector = self._get_leaf_selector(leaf_selector)
        selected_leaves = apply_leaf_selector(self.leaf_ids)
        if selected_leaves.size == 0:
            return selected_leaves
        if rank_by == 'purity':
            sorted_ids = np.lexsort(
                (apply_leaf_selector(self.difference), apply_leaf_selector(self.quantized_impurity)))
        elif rank_by == 'class_difference':
            sorted_ids = np.lexsort((apply_leaf_selector(self.impurity), apply_leaf_selector(self.difference)))
        else:
            raise NotImplementedError("Input argument 'rank_by' is invalid. Should be 'purity' or 'class_difference'")
        return selected_leaves.take(sorted_ids)

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
            if leaf_selector == "all_errors":
                return lambda array: array[self._get_error_leaves()]

        leaf_selector_as_array = np.array(leaf_selector)
        leaf_selector = np.in1d(self.leaf_ids, leaf_selector_as_array)
        nr_kept_leaves = np.count_nonzero(leaf_selector)
        if nr_kept_leaves == 0:
            print("None of the ids provided correspond to a leaf id.")
        elif nr_kept_leaves < leaf_selector_as_array.size:
            print("Some of the ids provided do not belong to leaves. Only leaf ids are kept.")
        return lambda array: array[leaf_selector]

    def _get_path_to_node(self, node_id):
        """ Return path to node as a list of split steps from the nodes of the sklearn Tree object """
        if self.pipeline_preprocessor is None:
            feature_names = self.model_performance_predictor_features
        else:
            feature_names = self.pipeline_preprocessor.get_original_feature_names()

        children_left = self.model_performance_predictor.tree_.children_left
        children_right = self.model_performance_predictor.tree_.children_right
        # feature = self.model_performance_predictor.tree_.feature
        # threshold = self.model_performance_predictor.tree_.threshold

        threshold = self.inverse_transform_thresholds()
        feature = self.inverse_transform_features()

        cur_node_id = node_id
        path_to_node = collections.deque()
        while cur_node_id > 0:

            if cur_node_id in children_left:
                parent_id = list(children_left).index(cur_node_id)
            else:
                parent_id = list(children_right).index(cur_node_id)

            feat = feature[parent_id]
            thresh = threshold[parent_id]

            is_categorical = False
            if self.pipeline_preprocessor is not None:
                is_categorical = self.pipeline_preprocessor.is_categorical(feat)

            thresh = thresh if is_categorical else ("%.2f" % thresh)

            decision_rule = ''
            if cur_node_id in children_left:
                decision_rule += ' <= ' if not is_categorical else ' != '
            else:
                decision_rule += " > " if not is_categorical else ' == '

            decision_rule = str(feature_names[feat]) + decision_rule + thresh
            path_to_node.appendleft(decision_rule)
            cur_node_id = parent_id

        return path_to_node

    def inverse_transform_features(self):
        """ Undo preprocessing of feature values.

        If the predictor comes with a Pipeline preprocessor, map the features indices of the Model
        Performance Predictor Decision Tree back to their indices in the original unpreprocessed space of features.
        Otherwise simply return the feature indices of the decision tree. The feature indices of a decision tree
        indicate what features are used to split the training set at each node.
        See https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html.

        Return:
            list or numpy.ndarray:
                indices of features of the Model Performance Predictor Decision Tree, possibly mapped back to the
                original unprocessed feature space.
        """

        if self.pipeline_preprocessor is None:
            return self._error_clf.tree_.feature

        if self._error_clf_features is not None:
            return self._error_clf_features

        feats_idx = self._error_clf.tree_.feature.copy()

        for i, f in enumerate(feats_idx):
            if f > 0:
                feats_idx[i] = self.pipeline_preprocessor.inverse_transform_feature_id(f)

        self._error_clf_features = feats_idx

        return self._error_clf_features

    def inverse_transform_thresholds(self):
        """  Undo preprocessing of feature threshold values.

        If the predictor comes with a Pipeline preprocessor, undo the preprocessing on the thresholds of the Model
        Performance Predictor Decision Tree for an easier plot interpretation. Otherwise simply return the thresholds of
        the decision tree. The thresholds of a decision tree are the feature values used to split the training set at
        each node. See https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html.

        Return:
            numpy.ndarray:
                thresholds of the Model Performance Predictor Decision Tree, possibly with preprocessing undone.
        """

        if self.pipeline_preprocessor is None:
            return self._error_clf.tree_.threshold

        if self._error_clf_thresholds is not None:
            return self._error_clf_thresholds

        feats_idx = self._error_clf.tree_.feature[self._error_clf.tree_.feature > 0]
        thresholds = self._error_clf.tree_.threshold.copy().astype('O')
        thresh = thresholds[self._error_clf.tree_.feature > 0]
        n_rows = np.count_nonzero(self._error_clf.tree_.feature[self._error_clf.tree_.feature > 0])
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

        thresholds[self._error_clf.tree_.feature > 0] = descaled_thresh

        self._error_clf_thresholds = thresholds

        return self._error_clf_thresholds

    # TODO: rewrite this method using the ranking arrays
    def error_node_summary(self, leaf_selector='all_errors', add_path_to_leaves=False, print_summary=False):
        """ Return summary information regarding input nodes.

        Args:
            leaf_selector (int or list or str): the desired leaf nodes to visualize. When int it represents the
                number of the leaf node, when a list it represents a list of leaf nodes. When a string, the valid values
                are either 'all_error' to plot all leaves of class 'Wrong prediction' or 'all' to plot all leaf nodes.
            add_path_to_leaves (bool): add information of the feature path across the tree till the selected node.
            print_summary (bool): print summary for the selected nodes.

        Return:
            dict: dictionary of metrics for each selected node of the Model Performance Predictor.
        """

        leaf_nodes = self.get_ranked_leaf_ids(leaf_selector=leaf_selector)

        y = self._error_train_y
        n_total_errors = y[y == ErrorAnalyzerConstants.WRONG_PREDICTION].shape[0]
        error_class_idx = \
        np.where(self.model_performance_predictor.classes_ == ErrorAnalyzerConstants.WRONG_PREDICTION)[0][0]
        correct_class_idx = 1 - error_class_idx

        leaves_summary = []
        for leaf_id in leaf_nodes:
            values = self.model_performance_predictor.tree_.value[leaf_id, :]
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

    def mpp_summary(self, x_test, y_test, nr_max_rows=ErrorAnalyzerConstants.MAX_NUM_ROW, output_dict=False):
        """ Print ErrorAnalyzer summary metrics regarding the Model Performance Predictor.

        Args:
            x_test (numpy.ndarray or pandas.DataFrame): feature data from a test set to evaluate the primary predictor
                and train a Model Performance Predictor.
            y_test (numpy.ndarray or pandas.DataFrame): target data from a test set to evaluate the primary predictor and
                train a Model Performance Predictor.
            nr_max_rows (int): maximum number of rows to process.
            output_dict (bool): whether to return a dict or a string with metrics.

        Return:
            dict or str: metrics regarding the Model Performance Predictor.
        """
        
        if self.pipeline_preprocessor is None:
            prep_x, prep_y = x_test, y_test
        else:
            prep_x, prep_y = self.pipeline_preprocessor.transform(x_test), np.array(y_test)

        prep_x, y_true = self._compute_primary_model_error(prep_x, prep_y, nr_max_rows)
        y_pred = self.model_performance_predictor.predict(prep_x)
        return mpp_report(y_true, y_pred, output_dict)
