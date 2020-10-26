# -*- coding: utf-8 -*-
import numpy as np
from graphviz import Source
from sklearn.tree import export_graphviz
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from dku_error_analysis_utils import ErrorAnalyzerConstants, rank_features_by_error_correlation
from dku_error_analysis_mpp.error_analyzer import ErrorAnalyzer
from dku_error_analysis_mpp.dku_error_analyzer import DkuErrorAnalyzer

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Error Analysis Plugin | %(levelname)s - %(message)s')

plt.rc('font', family="sans-serif")
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 8, 10, 12
plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE) 
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc("hatch", color="white", linewidth=4)

class _BaseErrorVisualizer(object):
    def __init__(self, error_analyzer):
        if not isinstance(error_analyzer, ErrorAnalyzer):
            raise NotImplementedError('You need to input an ErrorAnalyzer object.')

        self._error_analyzer = error_analyzer
        self.get_ranked_leaf_ids = lambda leaf_selector, rank_by: error_analyzer.get_ranked_leaf_ids(leaf_selector, rank_by)

    @staticmethod
    def _plot_histograms(hist_data, label, **params):
        bottom = None
        for class_value, bar_heights in hist_data.items():
            plt.bar(height=bar_heights,
                    label="{} ({})".format(class_value, label),
                    color=ErrorAnalyzerConstants.ERROR_TREE_COLORS[class_value],
                    bottom=bottom,
                    align="edge",
                    alpha=.8,
                    **params)
            bottom = bar_heights

    @staticmethod
    def _add_new_plot(figsize, bins, feature_name, leaf_id):
        x_ticks = range(len(bins))
        plt.figure(figsize=figsize)
        plt.xticks(x_ticks)
        plt.gca().set_xticklabels(labels=bins)
        plt.xlabel('{}'.format(feature_name))
        plt.ylabel('Proportion of samples')
        plt.title('Distribution of {} in leaf {}'.format(feature_name, leaf_id))

        return x_ticks

    @staticmethod
    def _plot_feature_distribution(x_ticks, feature_is_numerical, leaf_data, root_data=None):
        width, x = 1.0, x_ticks
        if root_data is not None:
            width /= 2
            if feature_is_numerical:
                x = x_ticks[1:]
            _BaseErrorVisualizer._plot_histograms(root_data, label="global data", x=x, hatch="///", width=-width)
        if feature_is_numerical:
            x = x_ticks[:-1]
        _BaseErrorVisualizer._plot_histograms(leaf_data, label="leaf data", x=x, width=width)

        plt.legend()
        plt.pause(0.05)

class ErrorVisualizer(_BaseErrorVisualizer):
    """
    ErrorVisualizer provides visual utilities to analyze the error classifier in ErrorAnalyzer
    """

    def __init__(self, error_analyzer):
        super(ErrorVisualizer, self).__init__(error_analyzer)

        self._error_clf = error_analyzer.model_performance_predictor
        self._error_train_x = error_analyzer.error_train_x
        self._error_train_y = error_analyzer.error_train_y

        self._train_leaf_ids = error_analyzer.train_leaf_ids

        self._features_in_model_performance_predictor = error_analyzer.model_performance_predictor_features
        if self._features_in_model_performance_predictor is None:
            self._features_in_model_performance_predictor = list(range(self._error_clf.max_features_))

    def plot_error_tree(self, size=None):
        """ Plot the graph of the decision tree """
        return Source(export_graphviz(self._error_clf, feature_names=self._features_in_model_performance_predictor,
                                    class_names=self._error_clf.classes_, node_ids=True, proportion=True, out_file=None,
                                    filled=True, rounded=True))

    def plot_feature_distributions_on_leaves(self, leaf_selector='all_errors', top_k_features=ErrorAnalyzerConstants.TOP_K_FEATURES,
                                            show_global=True, show_class=False, rank_leaves_by="purity", nr_bins=10, figsize=(15, 10)):
        """ Return plot of error node feature distribution and compare to global baseline """

        error_class_idx = np.where(self._error_clf.classes_ == ErrorAnalyzerConstants.WRONG_PREDICTION)[0][0]
        correct_class_idx = 1 - error_class_idx

        ranked_feature_ids = rank_features_by_error_correlation(self._error_clf.feature_importances_)
        if top_k_features > 0:
            ranked_feature_ids = ranked_feature_ids[:top_k_features]
        x, y = self._error_train_x[:,ranked_feature_ids], self._error_train_y
        min_values, max_values = x.min(axis=0), x.max(axis=0)

        global_error_sample_ids = y == ErrorAnalyzerConstants.WRONG_PREDICTION
        nr_wrong, nr_correct = self._error_clf.tree_.value[:, 0, error_class_idx], self._error_clf.tree_.value[:, 0, correct_class_idx]            

        leaf_nodes = self.get_ranked_leaf_ids(leaf_selector, rank_leaves_by)
        for leaf in leaf_nodes:
            leaf_sample_ids = self._train_leaf_ids == leaf
            nr_leaf_samples = np.count_nonzero(leaf_sample_ids)
            proba_wrong_leaf, proba_correct_leaf = nr_wrong[leaf]/nr_leaf_samples, nr_correct[leaf]/nr_leaf_samples
            print('Leaf {} (Wrong prediction: {:.3f}, Correct prediction: {:.3f})'.format(leaf, proba_wrong_leaf, proba_correct_leaf))

            for i, feature_idx in enumerate(ranked_feature_ids):
                feature_name = self._features_in_model_performance_predictor[feature_idx]
                bins = np.round(np.linspace(min_values[i], max_values[i], nr_bins + 1), 2)
                feature_column = x[:,i]
                if show_global:
                    if show_class:
                        root_hist_data = {
                            ErrorAnalyzerConstants.WRONG_PREDICTION: np.histogram(feature_column[global_error_sample_ids], bins=bins, density=True)[0],
                            ErrorAnalyzerConstants.CORRECT_PREDICTION: np.histogram(feature_column[~global_error_sample_ids], bins=bins, density=True)[0]
                        }
                    else:
                        root_prediction = ErrorAnalyzerConstants.CORRECT_PREDICTION if nr_correct[0] > nr_wrong[0] else ErrorAnalyzerConstants.WRONG_PREDICTION
                        root_hist_data = {root_prediction: np.histogram(feature_column, bins=bins, density=True)[0]}

                leaf_hist_data = {}
                if show_class:
                    leaf_hist_data = {
                        ErrorAnalyzerConstants.WRONG_PREDICTION: np.histogram(feature_column[leaf_sample_ids & global_error_sample_ids], bins=bins, density=True)[0],
                        ErrorAnalyzerConstants.CORRECT_PREDICTION: np.histogram(feature_column[leaf_sample_ids & ~global_error_sample_ids], bins=bins, density=True)[0]
                    }
                else:
                    leaf_prediction = ErrorAnalyzerConstants.CORRECT_PREDICTION if proba_correct_leaf > proba_wrong_leaf else ErrorAnalyzerConstants.WRONG_PREDICTION
                    leaf_hist_data = {leaf_prediction: np.histogram(feature_column[leaf_sample_ids], bins=bins, density=True)[0]}

                feature_is_numerical = True # TODO: change this once we have unprocessing done for sklearn models
                x_ticks = _BaseErrorVisualizer._add_new_plot(figsize, bins, feature_name, leaf)
                _BaseErrorVisualizer._plot_feature_distribution(x_ticks, feature_is_numerical, leaf_hist_data, root_hist_data if show_global else None)

        plt.show()

class DkuErrorVisualizer(_BaseErrorVisualizer):
    """
    ErrorVisualizer provides visual utilities to analyze the error classifier in ErrorAnalyzer and DkuErrorAnalyzer.
    """

    def __init__(self, error_analyzer):

        if not isinstance(error_analyzer, DkuErrorAnalyzer):
            raise NotImplementedError('You need to input a DkuErrorAnalyzer object.')

        super(DkuErrorVisualizer, self).__init__(error_analyzer)

        self._tree = error_analyzer.tree
        self._tree_parser = error_analyzer.tree_parser

    def plot_error_tree(self, size=None):
        """ Plot the graph of the decision tree """

        return Source(self._tree.to_dot_string())

    def plot_feature_distributions_on_leaves(self, leaf_selector='all_errors', top_k_features=ErrorAnalyzerConstants.TOP_K_FEATURES,
                                            show_global=True, show_class=False, rank_leaves_by="purity", nr_bins=10, figsize=(15, 10)):
        """ Return plot of error node feature distribution and compare to global baseline """

        leaf_nodes = self.get_ranked_leaf_ids(leaf_selector, rank_leaves_by)
        ranked_features = self._tree.ranked_features[:top_k_features]
        if show_global:
            if not show_class:
                root_prediction = self._tree.get_node(0).prediction
            root_hist_data_all_features = {}

        for leaf_id in leaf_nodes:
            for feature_name in ranked_features:
                leaf = self._tree.get_node(leaf_id)
                node_summary = 'Leaf {} ({}: {:.3f}'.format(leaf.id, *leaf.probabilities[0])
                if len(leaf.probabilities) > 1:
                    node_summary += ', {}: {:.3f})'.format(*leaf.probabilities[1])
                else:
                    node_summary += ')'
                print(node_summary)

                leaf_stats = self._tree.get_stats(leaf.id, feature_name, nr_bins)
                feature_is_numerical = feature_name in self._tree.features
                bins = leaf_stats["bin_edge"] if feature_is_numerical else leaf_stats["bin_value"]

                if show_global:
                    if feature_name not in root_hist_data_all_features:
                        root_hist_data_all_features[feature_name] = self._tree.get_stats(0, feature_name, min(len(bins), nr_bins))
                    if show_class:
                        root_hist_data = root_hist_data_all_features[feature_name]["target_distrib"]
                    else:
                        root_hist_data = {root_prediction: root_hist_data_all_features[feature_name]["count"]}

                leaf_hist_data = {}
                if show_class:
                    leaf_hist_data = leaf_stats["target_distrib"]
                else:
                    leaf_hist_data = {leaf.prediction: leaf_stats["count"]}

                x_ticks = _BaseErrorVisualizer._add_new_plot(figsize, bins, feature_name, leaf.id)
                _BaseErrorVisualizer._plot_feature_distribution(x_ticks, feature_is_numerical, leaf_hist_data, root_hist_data if show_global else None)

        plt.show()
