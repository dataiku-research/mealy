# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import collections
from sklearn.model_selection import train_test_split
from dku_error_analysis_utils import ErrorAnalyzerConstants
from dku_error_analysis_tree_parsing.tree_parser import TreeParser
from dku_error_analysis_decision_tree.node import Node
from dku_error_analysis_mpp.error_analyzer import ErrorAnalyzer
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Error Analysis Plugin | %(levelname)s - %(message)s')


class DkuErrorAnalyzer(ErrorAnalyzer):
    """
    DkuErrorAnalyzer analyzes the errors of a DSS prediction model on its test set.
    It uses model predictions and ground truth target to compute the model errors on the test set.
    It then trains a Decision Tree on the same test set by using the model error as target.
    The nodes of the decision tree are different segments of errors to be studied individually.
    """

    def __init__(self, model_accessor, seed=65537):

        if model_accessor is None:
            raise NotImplementedError('you need to define a model accessor.')

        self._model_accessor = model_accessor
        self._target = self._model_accessor.get_target_variable()
        self._model_predictor = self._model_accessor.get_predictor()
        feature_names = self._model_predictor.get_features()

        super(DkuErrorAnalyzer, self).__init__(self._model_accessor.get_clf(), feature_names, seed)

        self._train_x = None
        self._test_x = None
        self._train_y = None
        self._test_y = None

        self._error_df = None

        self._tree = None
        self._tree_parser = None

    @property
    def tree(self):
        if self._tree is None:
            self.parse_tree()
        return self._tree

    @property
    def tree_parser(self):
        return self._tree_parser

    @property
    def features_dict(self):
        return self._model_accessor.get_per_feature()

    @property
    def error_df(self):
        return self._error_df

    def fit(self):
        """
        Trains a Decision Tree to discriminate between samples that are correctly predicted or wrongly predicted
        (errors) by a primary model.
        """

        self._prepare_data_from_dku_saved_model()

        super(DkuErrorAnalyzer, self).fit(self._train_x, self._train_y)

    def _preprocess_dataframe(self, df, with_target=True):
        """ Preprocess input DataFrame with primary model preprocessor """
        if with_target and self._target not in df:
            raise ValueError('The dataset does not contain target "{}".'.format(self._target))
        if with_target:
            x, input_mf_index, _, y = self._model_predictor.preprocessing.preprocess(
                df,
                with_target=True)
            return x, y, input_mf_index
        return self._model_predictor.preprocessing.preprocess(df)[0]

    def _prepare_data_from_dku_saved_model(self):
        """ Preprocess and split original test set from Dku saved model
        into train and test set for the error analyzer """
        np.random.seed(self._seed)

        original_df = self._model_accessor.get_original_test_df(ErrorAnalyzerConstants.MAX_NUM_ROW)

        preprocessed_x, y, input_mf_index = self._preprocess_dataframe(original_df)

        x_df = pd.DataFrame(preprocessed_x, index=input_mf_index)
        y_df = pd.Series(y, index=input_mf_index)

        train_x_df, test_x_df, train_y_df, test_y_df = train_test_split(
            x_df, y_df, test_size=ErrorAnalyzerConstants.TEST_SIZE
        )

        self._train_x = train_x_df.values
        self._train_y = train_y_df.values

        self._test_x = test_x_df.values
        self._test_y = test_y_df.values

        original_train_df = original_df.loc[train_x_df.index]

        self._error_df = original_train_df.drop(self._target, axis=1)

    def parse_tree(self):
        """ Parse Decision Tree and get features information used to display distributions """
        self._error_df.loc[:, ErrorAnalyzerConstants.ERROR_COLUMN] = self.error_train_y

        self._tree_parser = TreeParser(self._model_accessor.model_handler, self._error_clf)
        self._tree = self._tree_parser.build_tree(self._error_df, self._features_in_model_performance_predictor)
        self._tree.parse_nodes(self._tree_parser,
                               self._features_in_model_performance_predictor,
                               self.error_train_x)

    def _get_path_to_node(self, node_id):
        """ return path to node as a list of split steps from the nodes of the de-processed
        dku_error_analysis_decision_tree.tree.InteractiveTree object """
        cur_node = self.tree.get_node(node_id)
        path_to_node = collections.deque()
        while cur_node is not None and cur_node.id != 0:
            path_to_node.appendleft(cur_node.print_decision_rule())
            cur_node = self.tree.get_node(cur_node.parent_id)

        return path_to_node

    def mpp_summary(self, dku_test_dataset=None, output_dict=False):
        """ print ErrorAnalyzer summary metrics """
        if dku_test_dataset is None:
            return super(DkuErrorAnalyzer, self).mpp_summary(self._test_x, self._test_y, output_dict=output_dict)
        else:
            test_df = dku_test_dataset.get_dataframe()
            test_x, test_y, _ = self._preprocess_dataframe(test_df)
            return super(DkuErrorAnalyzer, self).mpp_summary(test_x, test_y.values, output_dict=output_dict)

    def predict(self, dku_test_dataset):
        """ Predict model performance on Dku dataset """
        test_df = dku_test_dataset.get_dataframe()

        test_x = self._preprocess_dataframe(test_df, with_target=False)

        return super(DkuErrorAnalyzer, self).predict(test_x)
