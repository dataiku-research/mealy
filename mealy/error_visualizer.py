# -*- coding: utf-8 -*-
import numpy as np
import graphviz as gv
import pydotplus
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from mealy.constants import ErrorAnalyzerConstants
from mealy.error_analyzer import ErrorAnalyzer

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

        self._get_ranked_leaf_ids = lambda leaf_selector, rank_by: error_analyzer._get_ranked_leaf_ids(leaf_selector, rank_by)

    @staticmethod
    def _plot_histograms(hist_data, label, **params):
        bottom = None
        for class_value in [ErrorAnalyzerConstants.CORRECT_PREDICTION, ErrorAnalyzerConstants.WRONG_PREDICTION]:
            bar_heights = hist_data.get(class_value)
            if bar_heights is not None:
                plt.bar(height=bar_heights,
                        label="{} ({})".format(class_value, label),
                        color=ErrorAnalyzerConstants.ERROR_TREE_COLORS[class_value],
                        bottom=bottom,
                        align="edge",
                        **params)
                bottom = bar_heights

    @staticmethod
    def _add_new_plot(figsize, bins, x_ticks, feature_name, leaf_id):
        plt.figure(figsize=figsize)
        plt.xticks(x_ticks)
        plt.gca().set_xticklabels(labels=bins)
        plt.xlabel('{}'.format(feature_name))
        plt.ylabel('Proportion of samples')
        plt.title('Distribution of {} in leaf {}'.format(feature_name, leaf_id))

    @staticmethod
    def _plot_feature_distribution(x_ticks, feature_is_numerical, leaf_data, root_data=None):
        width, x = 1.0, x_ticks
        if root_data is not None:
            width /= 2
            if feature_is_numerical:
                x = x_ticks[1:]
            _BaseErrorVisualizer._plot_histograms(root_data, label="global data", x=x, hatch="///", width=-width)
        if leaf_data is not None:
            if feature_is_numerical:
                x = x_ticks[:-1]
            _BaseErrorVisualizer._plot_histograms(leaf_data, label="leaf data", x=x, width=width)

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
        self._thresholds = self._error_analyzer._inverse_transform_thresholds()
        self._features = self._error_analyzer._inverse_transform_features()

        self._original_feature_names = self._error_analyzer.pipeline_preprocessor.get_original_feature_names()
        self._numerical_feature_names = [f for f in self._original_feature_names if not self._error_analyzer.pipeline_preprocessor.is_categorical(name=f)]

    def plot_error_tree(self, size=None):
        """
        Plot the graph of the decision tree.

        Args:
            size (tuple): size of the output plot.

        Return:
            graphviz.Source: graph of the Error Analyzer Tree.

        """
        digraph_tree = export_graphviz(self._error_clf,
                                       feature_names=self._error_analyzer.preprocessed_feature_names,
                                       class_names=self._error_clf.classes_,
                                       node_ids=True,
                                       proportion=True,
                                       rotate=False,
                                       out_file=None,
                                       filled=True,
                                       rounded=True,
                                       impurity=False)

        pydot_graph = pydotplus.graph_from_dot_data(str(digraph_tree))

        # descale threshold value
        thresholds = self._thresholds
        features = self._features

        nodes = pydot_graph.get_node_list()

        for node in nodes:
            if node.get_label():
                node_label = node.get_label().strip('"')
                idx = int(node_label.split('node #')[1].split('\\n')[0])
                n_wrong_preds = self._error_clf.tree_.value[idx, 0, self._error_tree.error_class_idx]
                total_error_fraction = n_wrong_preds / self._error_tree.n_total_errors
                local_error = n_wrong_preds / self._error_clf.tree_.n_node_samples[idx]
                if ' <= ' in node_label:
                    lte_split = node_label.split(' <= ')
                    samples_split = lte_split[1].split('\\nsamples')

                    split_feature = self._original_feature_names[features[idx]]
                    descaled_value = thresholds[idx]

                    if split_feature in self._numerical_feature_names:
                        descaled_value = '%.2f' % descaled_value
                        lte_modified = ' <= '.join([lte_split[0], descaled_value])
                    else:
                        lte_split_without_feature = lte_split[0].split('\\n')[0]
                        lte_split_with_new_feature = lte_split_without_feature + '\\n' + split_feature
                        lte_modified = ' != '.join([lte_split_with_new_feature, str(descaled_value)])
                    new_label = lte_modified
                else:
                    samples_split = node_label.split('\\nsamples')
                    new_label = samples_split[0]

                value_split = samples_split[1].split('\\nvalue')

                new_label += '\\nsamples' + value_split[0] + \
                             '\\nlocal error = %.3f %%' % (local_error * 100) + \
                             '\\nfraction of total error = %.3f %%\\n' % (total_error_fraction * 100)

                node.set_label(new_label)

                alpha = 0.0
                if 'value = [' in node_label:
                    # transparency as the local error
                    if local_error >= ErrorAnalyzerConstants.GRAPH_MIN_LOCAL_ERROR_OPAQUE:
                        alpha = 1.0
                    else:
                        alpha = local_error

                node_class = ErrorAnalyzerConstants.CORRECT_PREDICTION if total_error_fraction == 0 else ErrorAnalyzerConstants.WRONG_PREDICTION
                class_color = ErrorAnalyzerConstants.ERROR_TREE_COLORS[node_class].strip('#')
                class_color_rgb = tuple(int(class_color[i:i + 2], 16) for i in (0, 2, 4))
                # compute the color as alpha against white
                color_rgb = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in class_color_rgb]
                color = '#{:02x}{:02x}{:02x}'.format(color_rgb[0], color_rgb[1], color_rgb[2])
                node.set_fillcolor(color)

                if idx in self._error_clf.tree_.children_left:
                    parent_id = np.where(self._error_clf.tree_.children_left == idx)[0][0]
                elif idx in self._error_clf.tree_.children_right:
                    parent_id = np.where(self._error_clf.tree_.children_right == idx)[0][0]
                else:
                    parent_id = None

                if parent_id is not None:
                    parent_edge = pydot_graph.get_edge(str(parent_id), node.get_name())[0]
                    parent_edge.set_penwidth(max(1, ErrorAnalyzerConstants.GRAPH_MAX_EDGE_WIDTH * total_error_fraction))

        if size is not None:
            pydot_graph.set_size('"%d,%d!"' % (size[0], size[1]))
        gvz_graph = gv.Source(pydot_graph.to_string())

        return gvz_graph

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
        # TODO to do what ?
        feature_names = self._original_feature_names

        min_values, max_values = x.min(axis=0), x.max(axis=0)
        total_error_fraction_sample_ids = y == ErrorAnalyzerConstants.WRONG_PREDICTION
        nr_wrong = self._error_clf.tree_.value[:, 0, self._error_tree.error_class_idx]

        leaf_nodes = self._get_ranked_leaf_ids(leaf_selector, rank_leaves_by)
        for leaf in leaf_nodes:
            leaf_sample_ids = self._train_leaf_ids == leaf
            nr_leaf_samples = self._error_clf.tree_.n_node_samples[leaf]
            proba_wrong_leaf = nr_wrong[leaf] / nr_leaf_samples
            proba_correct_leaf = 1 - proba_wrong_leaf
            print('Leaf {} (Wrong prediction: {:.3f}, Correct prediction: {:.3f})'.format(leaf, proba_wrong_leaf, proba_correct_leaf))

            for i, feature_idx in enumerate(ranked_feature_ids):

                feature_name = feature_names[feature_idx]
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

                root_hist_data = {}
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
                    leaf_prediction = ErrorAnalyzerConstants.CORRECT_PREDICTION if proba_correct_leaf > proba_wrong_leaf else ErrorAnalyzerConstants.WRONG_PREDICTION
                    leaf_hist_data = {leaf_prediction: histogram_func(feature_column[leaf_sample_ids])}

                x_ticks = range(len(bins))
                _BaseErrorVisualizer._add_new_plot(figsize, bins, x_ticks, feature_name, leaf)
                _BaseErrorVisualizer._plot_feature_distribution(x_ticks, feature_is_numerical, leaf_hist_data, root_hist_data)

        plt.show()
