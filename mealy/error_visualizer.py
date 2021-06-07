# -*- coding: utf-8 -*-
import numpy as np
from graphviz import Source
from collections import deque
import matplotlib.pyplot as plt
from .constants import ErrorAnalyzerConstants
from .error_analyzer import ErrorAnalyzer
from .error_analysis_utils import format_float

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
            raise TypeError('You need to input an ErrorAnalyzer object.')

        self._error_analyzer = error_analyzer

        self._get_ranked_leaf_ids = lambda leaf_selector, rank_by: \
            error_analyzer._get_ranked_leaf_ids(leaf_selector, rank_by)

    @staticmethod
    def _plot_histograms(hist_data, label, **params):
        bottom = None
        for class_value in [ErrorAnalyzerConstants.CORRECT_PREDICTION, ErrorAnalyzerConstants.WRONG_PREDICTION]:
            bar_heights = hist_data.get(class_value)
            if bar_heights is not None:
                plt.bar(height=bar_heights,
                        label="{} ({})".format(class_value, label),
                        edgecolor="white",
                        linewidth=1,
                        color=ErrorAnalyzerConstants.ERROR_TREE_COLORS[class_value],
                        bottom=bottom,
                        **params)
                bottom = bar_heights

    @staticmethod
    def _add_new_plot(figsize, bins, x_ticks, feature_name, suptitle):
        plt.figure(figsize=figsize)
        plt.xticks(x_ticks, rotation="90")
        plt.gca().set_xticklabels(labels=bins)
        plt.ylabel('Proportion of samples')
        plt.title('Distribution of {}'.format(feature_name))
        plt.suptitle(suptitle)

    @staticmethod
    def _plot_feature_distribution(x_ticks, feature_is_numerical, leaf_data, root_data=None):
        width, x = 1.0, x_ticks
        align = "edge"
        if root_data is not None:
            width /= 2
            if feature_is_numerical:
                x = x_ticks[1:]
            _BaseErrorVisualizer._plot_histograms(root_data, label="global data", x=x, hatch="///",
                                                  width=-width, align=align)
        if leaf_data is not None:
            if feature_is_numerical:
                x = x_ticks[:-1]
            elif root_data is None:
                align = "center"
            _BaseErrorVisualizer._plot_histograms(leaf_data, label="leaf data", x=x,
                                                  align=align, width=width)
        plt.legend()
        plt.pause(0.05)


class ErrorVisualizer(_BaseErrorVisualizer):
    """
    ErrorVisualizer provides visual utilities to analyze the Error Tree in ErrorAnalyzer

    Args:
        error_analyzer (ErrorAnalyzer): fitted ErrorAnalyzer representing the performance of a primary model.
    """

    def __init__(self, error_analyzer):
        super(ErrorVisualizer, self).__init__(error_analyzer)

        self._error_tree = self._error_analyzer.error_tree
        self._error_clf = self._error_tree.estimator_
        self._train_leaf_ids = self._error_clf.apply(self._error_analyzer._error_train_x)
        self._thresholds = None
        self._features = None

        self._original_feature_names = self._error_analyzer.pipeline_preprocessor.get_original_feature_names()
        self._numerical_feature_names = [f for f in self._original_feature_names if not self._error_analyzer.pipeline_preprocessor.is_categorical(name=f)]

    @property
    def thresholds_(self):
        if self._thresholds is None:
            self._thresholds = self._error_analyzer._inverse_transform_thresholds()
        return self._thresholds

    @property
    def features_(self):
        if self._features is None:
            self._features = self._error_analyzer._inverse_transform_features()
        return self._features

    def plot_error_tree(self, size=(50, 50)):
        """
        Plot the graph of the decision tree.

        Args:
            size (tuple): size of the output plot.

        Return:
            graphviz.Source: graph of the Error Analyzer Tree.

        """
        dot_str = 'digraph Tree {{\n size="{0},{1}!";\n'.format(size[0], size[1])
        dot_str += 'node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;\n'
        dot_str += 'edge [fontname=helvetica] ;\ngraph [ranksep=equally, splines=polyline] ;\n'
        color = ErrorAnalyzerConstants.ERROR_TREE_COLORS[ErrorAnalyzerConstants.WRONG_PREDICTION]
        leaves, left_child_to_parent, right_child_to_parent = set(), {}, {}
        ids = deque()
        ids.append(0)

        while ids:
            node_id = ids.popleft()
            dot_str += '{0} [label="node #{0}\n'.format(node_id)

            parent_id = left_child_to_parent.get(node_id, right_child_to_parent.get(node_id))
            if parent_id is not None:
                rule = self.node_decision_rule(parent_id, node_id in left_child_to_parent)
                dot_str += "{}\n".format((rule[:32] + "...") if len(rule) > 35 else rule)

            n_wrong_preds = self._error_clf.tree_.value[node_id, 0, self._error_tree.error_class_idx]
            total_error_fraction = n_wrong_preds / self._error_tree.n_total_errors
            samples = self._error_clf.tree_.n_node_samples[node_id]
            local_error = n_wrong_preds / samples
            dot_str += 'samples = {}%\n'.format(format_float(100 * samples / self._error_clf.tree_.n_node_samples[0], 3))
            dot_str += 'local error = {}%\n'.format(format_float(100 * local_error, 3))
            dot_str += 'fraction of total error = {}%\n'.format(format_float(100 * total_error_fraction, 3))

            alpha = "{:02x}".format(int(local_error*255))
            dot_str += '", fillcolor="{}", tooltip="{}"] ;\n'.format(color+alpha, "root" if parent_id is None else rule)

            if parent_id is not None:
                edge_width = max(1, ErrorAnalyzerConstants.GRAPH_MAX_EDGE_WIDTH * total_error_fraction)
                dot_str += '{} -> {} [penwidth={}];\n'.format(parent_id, node_id, edge_width)
            left_child_id, right_child_id = self._error_clf.tree_.children_left[node_id], self._error_clf.tree_.children_right[node_id]
            if left_child_id > 0:
                ids += [left_child_id, right_child_id]
                left_child_to_parent[left_child_id] = node_id
                right_child_to_parent[right_child_id] = node_id
            else:
                leaves.add(node_id)

        dot_str += '{rank=same ; '+ '; '.join(map(str, leaves)) + '} ;\n'
        dot_str += "}"
        return Source(dot_str)

    def node_decision_rule(self, parent_id, left_child):
        feature = self._original_feature_names[self.features_[parent_id]]
        value = self.thresholds_[parent_id]
        numerical_split = feature in self._numerical_feature_names
        if numerical_split:
            if left_child:
                return '{} <= {}'.format(feature, format_float(value, 2))
            return '{} < {}'.format(format_float(value, 2), feature)
        return feature + ' is ' + ( '' if left_child else 'not ') + str(value)

    def plot_feature_distributions_on_leaves(self, leaf_selector=None,
                                             top_k_features=ErrorAnalyzerConstants.TOP_K_FEATURES,
                                             show_global=True, show_class=True, rank_leaves_by="total_error_fraction",
                                             nr_bins=10, figsize=(15, 10)):

        """
        Return feature distribution plots at the selected leaves.

        The leaves for which the distributions are plotted are determined by the leaf_selector argument.
        By default, no specific leaves are selected, and so the distributions are plotted for all the leaves.
        The leaves are ranked following a criterion set via the argument rank_leaves_by.

        The features are sorted by feature importance in the Error Tree. The more important a feature is, the more correlated with the errors it is.
        The number of feature distributions to plot is set via top_k_features.

        Args:
            leaf_selector (None, int or array-like): the leaves whose information will be returned
                * int: Only plot the feature distributions for the leaf matching the id
                * array-like of int: Only plot the feature distributions for the leaves matching the ids
                * None (default): Plot the feature distributions for all the leaves

            top_k_features (int): Number of features to plot per node.
                * If a positive integer k is given, the distributions of the first k features (first in the sense of their importance) are plotted
                * If a negative integer k is given, the distributions of all but the k last features (last in the sense of their importance) are plotted
                * If k is 0, all the feature distributions are plotted

            show_global (bool): Whether to plot the feature distributions for the whole data (global baseline) along with the ones for the leaf samples.

            show_class (bool): Whether to show the proportion of Wrongly and Correctly predicted samples for each bin.

            rank_leaves_by (str): Ranking criterion for the leaves. Valid values are:
                * 'total_error_fraction': rank by the fraction of total error in the node
                * 'purity': rank by the purity (ratio of wrongly predicted samples over the total number of node samples)
                * 'class_difference': rank by the difference of number of wrongly and correctly predicted samples
                in a node.

            nr_bins (int): Number of bins in the feature distribution plots. Defaults to 10.

            figsize (tuple of float): Tuple of size 2 for the size of the plots as (width, height) in inches. Defaults to (15, 10).

        """
        ranked_feature_ids = self._error_analyzer.pipeline_preprocessor.get_top_ranked_feature_ids(self._error_clf.feature_importances_, top_k_features)

        x = self._error_analyzer.pipeline_preprocessor.inverse_transform(self._error_analyzer._error_train_x)[:, ranked_feature_ids]
        y = self._error_analyzer._error_train_y
        feature_names = self._original_feature_names

        min_values, max_values = x.min(axis=0), x.max(axis=0)
        total_error_fraction_sample_ids = y == ErrorAnalyzerConstants.WRONG_PREDICTION
        nr_wrong = self._error_clf.tree_.value[:, 0, self._error_tree.error_class_idx]

        leaf_nodes = self._get_ranked_leaf_ids(leaf_selector, rank_leaves_by)
        for leaf in leaf_nodes:
            leaf_sample_ids = self._train_leaf_ids == leaf
            nr_leaf_samples = self._error_clf.tree_.n_node_samples[leaf]
            proba_wrong_leaf = nr_wrong[leaf] / nr_leaf_samples
            suptitle = 'Leaf {} (Wrong prediction: {},'.format(leaf, format_float(proba_wrong_leaf, 3))
            suptitle += ' Correct prediction: {})'.format(format_float(1 - proba_wrong_leaf, 3))

            for i, feature_idx in enumerate(ranked_feature_ids):
                feature_name = feature_names[feature_idx]
                # TODO: use self._numerical_feature_names instead
                feature_is_numerical = not self._error_analyzer.pipeline_preprocessor.is_categorical(feature_idx)

                feature_column = x[:, i]

                if feature_is_numerical:
                    bins = np.round(np.linspace(min_values[i], max_values[i], nr_bins + 1), 2)
                    if show_class:
                        histogram_func = lambda f_samples: np.histogram(f_samples, bins=bins, density=False)[0]
                    else:
                        histogram_func = lambda f_samples: np.histogram(f_samples, bins=bins, density=True)[0]

                else:
                    bins = np.unique(feature_column)[:nr_bins]
                    if show_class:
                        histogram_func = lambda f_samples: np.bincount(np.searchsorted(bins, f_samples), minlength=len(bins))[:nr_bins]
                    else:
                        histogram_func = lambda f_samples: np.bincount(np.searchsorted(bins, f_samples), minlength=len(bins))[:nr_bins] / len(f_samples)

                if show_global:
                    if show_class:
                        hist_wrong = histogram_func(feature_column[total_error_fraction_sample_ids])
                        hist_correct = histogram_func(feature_column[~total_error_fraction_sample_ids])
                        n_samples = np.sum(hist_wrong + hist_correct)
                        normalized_hist_wrong = hist_wrong / n_samples
                        normalized_hist_correct = hist_correct / n_samples
                        root_hist_data = {
                            ErrorAnalyzerConstants.WRONG_PREDICTION: normalized_hist_wrong,
                            ErrorAnalyzerConstants.CORRECT_PREDICTION: normalized_hist_correct
                        }
                    else:
                        root_prediction = ErrorAnalyzerConstants.CORRECT_PREDICTION if nr_wrong[0] < self._error_tree.n_total_errors / 2 else ErrorAnalyzerConstants.WRONG_PREDICTION
                        root_hist_data = {root_prediction: histogram_func(feature_column)}
                else:
                    root_hist_data = None

                if show_class:
                    hist_wrong = histogram_func(feature_column[leaf_sample_ids & total_error_fraction_sample_ids])
                    hist_correct = histogram_func(feature_column[leaf_sample_ids & ~total_error_fraction_sample_ids])
                    n_samples = np.sum(hist_wrong + hist_correct)
                    normalized_hist_wrong = hist_wrong / n_samples
                    normalized_hist_correct = hist_correct / n_samples
                    leaf_hist_data = {
                        ErrorAnalyzerConstants.WRONG_PREDICTION: normalized_hist_wrong,
                        ErrorAnalyzerConstants.CORRECT_PREDICTION: normalized_hist_correct
                    }
                else:
                    leaf_prediction = ErrorAnalyzerConstants.CORRECT_PREDICTION if proba_wrong_leaf < .5 else ErrorAnalyzerConstants.WRONG_PREDICTION
                    leaf_hist_data = {leaf_prediction: histogram_func(feature_column[leaf_sample_ids])}

                x_ticks = range(len(bins))
                _BaseErrorVisualizer._add_new_plot(figsize, bins, x_ticks, feature_name, suptitle)
                _BaseErrorVisualizer._plot_feature_distribution(x_ticks, feature_is_numerical, leaf_hist_data, root_hist_data)

        plt.show()
