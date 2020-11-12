# -*- coding: utf-8 -*-
import numpy as np
import graphviz as gv
import pydotplus
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from mealy.error_analysis_utils import rank_features_by_error_correlation
from mealy.constants import ErrorAnalyzerConstants
from mealy.error_analyzer import ErrorAnalyzer

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='mealy | %(levelname)s - %(message)s')

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
    ErrorVisualizer provides visual utilities to analyze the Model Performance Predictor in ErrorAnalyzer

    Args:
        error_analyzer (ErrorAnalyzer): fitted ErrorAnalyzer representing the performance of a primary model.
    """

    def __init__(self, error_analyzer):
        super(ErrorVisualizer, self).__init__(error_analyzer)

        self._error_clf = error_analyzer.model_performance_predictor
        self._error_train_x = error_analyzer.error_train_x
        self._error_train_y = error_analyzer.error_train_y

        self.pipeline_preprocessor = error_analyzer.pipeline_preprocessor
        self.thresholds = error_analyzer.inverse_transform_thresholds()
        self.features = error_analyzer.inverse_transform_features()

        self.mpp_feature_names = error_analyzer.model_performance_predictor_features

        if self.pipeline_preprocessor is None:
            self.original_feature_names = self.mpp_feature_names

            self.numerical_feature_names = self.mpp_feature_names
        else:
            self.original_feature_names = self.pipeline_preprocessor.get_original_feature_names()

            self.numerical_feature_names = [f for f in self.original_feature_names if
                                            not self.pipeline_preprocessor.is_categorical(name=f)]

        self._train_leaf_ids = error_analyzer.train_leaf_ids

    def plot_error_tree(self, size=None):
        """Plot the graph of the decision tree.

        Args:
            size (tuple): size of the output plot.

        Return:
            graphviz.Source: graph of the Model Performance Predictor decision tree.

        """
        digraph_tree = export_graphviz(self._error_clf,
                                       feature_names=self.mpp_feature_names,
                                       class_names=self._error_clf.classes_,
                                       node_ids=True,
                                       proportion=True,
                                       rotate=False,
                                       out_file=None,
                                       filled=True,
                                       rounded=True)

        pydot_graph = pydotplus.graph_from_dot_data(str(digraph_tree))

        # descale threshold value
        thresholds = self.thresholds
        features = self.features

        nodes = pydot_graph.get_node_list()
        for node in nodes:
            if node.get_label():
                node_label = node.get_label()

                if ' <= ' in node_label:
                    idx = int(node_label.split('node #')[1].split('\\n')[0])
                    lte_split = node_label.split(' <= ')
                    entropy_split = lte_split[1].split('\\nentropy')

                    split_feature = self.original_feature_names[features[idx]]

                    descaled_value = thresholds[idx]

                    if split_feature in self.numerical_feature_names:
                        descaled_value = '%.2f' % descaled_value
                        lte_modified = ' <= '.join([lte_split[0], descaled_value])
                    else:
                        lte_split_without_feature = lte_split[0].split('\\n')[0]
                        lte_split_with_new_feature = lte_split_without_feature + '\\n' + split_feature
                        lte_modified = ' != '.join([lte_split_with_new_feature, str(descaled_value)])
                    new_label = '\\nentropy'.join([lte_modified, entropy_split[1]])

                    node.set_label(new_label)

                alpha = 0.0
                node_class = ErrorAnalyzerConstants.CORRECT_PREDICTION
                if 'value = [' in node_label:
                    values = [float(ii) for ii in node_label.split('value = [')[1].split(']')[0].split(',')]
                    node_arg_class = np.argmax(values)
                    node_class = self._error_clf.classes_[node_arg_class]
                    # transparency as the entropy value
                    alpha = values[node_arg_class]
                class_color = ErrorAnalyzerConstants.ERROR_TREE_COLORS[node_class].strip('#')
                class_color_rgb = tuple(int(class_color[i:i + 2], 16) for i in (0, 2, 4))
                # compute the color as alpha against white
                color_rgb = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in class_color_rgb]
                color = '#{:02x}{:02x}{:02x}'.format(color_rgb[0], color_rgb[1], color_rgb[2])
                node.set_fillcolor(color)

        if size is not None:
            pydot_graph.set_size('"%d,%d!"' % (size[0], size[1]))
        gvz_graph = gv.Source(pydot_graph.to_string())

        return gvz_graph

    def plot_feature_distributions_on_leaves(self, leaf_selector='all_errors',
                                             top_k_features=ErrorAnalyzerConstants.TOP_K_FEATURES,
                                             show_global=True, show_class=False, rank_leaves_by="purity", nr_bins=10,
                                             figsize=(15, 10)):
        """Return plot of error node feature distribution and compare to global baseline.

        The top-k features are sorted by importance in the Model Performance Predictor.
        The most important are more correlated with the errors. When no specific node is selected,
        the leaf nodes are ranked by an importance criterium and presented in relevance order.

        Args:

            leaf_selector (int or list or str): the desired leaf nodes to visualize. When int it represents the
                number of the leaf node, when a list it represents a list of leaf nodes. When a string, the valid values
                are either 'all_error' to plot all leaves of class 'Wrong prediction' or 'all' to plot all leaf nodes.

            top_k_features (int): number of features to plot per node.

            show_global (bool): plot the feature distribution of samples in a node vs. samples in the whole data
                (global baseline).

            show_class (bool): show the proportion of Wrongly and Correctly predicted samples in the feature
                distributions.

            rank_leaves_by (str): ranking criterium for the leaf nodes. It can be either 'purity' to rank by the leaf
                node purity (ratio of wrongly predicted samples over the total for an error node) or 'class_difference'
                (difference of number of wrongly and correctly predicted samples in a node).

            nr_bins (int): number of bins in the feature distribution plots.

            figsize (tuple): size of the plots.

        """

        error_class_idx = np.where(self._error_clf.classes_ == ErrorAnalyzerConstants.WRONG_PREDICTION)[0][0]
        correct_class_idx = 1 - error_class_idx

        ranked_feature_ids = rank_features_by_error_correlation(self._error_clf.feature_importances_)
        if self.pipeline_preprocessor is None:
            if top_k_features > 0:
                ranked_feature_ids = ranked_feature_ids[:top_k_features]

            x, y = self._error_train_x[:, ranked_feature_ids], self._error_train_y
            min_values, max_values = x.min(axis=0), x.max(axis=0)
            feature_names = self.mpp_feature_names
        else:
            ranked_feature_ids = [self.pipeline_preprocessor.inverse_transform_feature_id(idx) for idx in
                                  ranked_feature_ids]
            if top_k_features > 0:
                ranked_feature_ids = ranked_feature_ids[:top_k_features]
            x, y = self.pipeline_preprocessor.inverse_transform(self._error_train_x)[:,
                   ranked_feature_ids], self._error_train_y
            # TODO
            min_values, max_values = x.min(axis=0), x.max(axis=0)
            feature_names = self.original_feature_names

        global_error_sample_ids = y == ErrorAnalyzerConstants.WRONG_PREDICTION
        nr_wrong, nr_correct = self._error_clf.tree_.value[:, 0, error_class_idx], self._error_clf.tree_.value[:, 0,
                                                                                   correct_class_idx]

        leaf_nodes = self.get_ranked_leaf_ids(leaf_selector, rank_leaves_by)
        for leaf in leaf_nodes:
            leaf_sample_ids = self._train_leaf_ids == leaf
            nr_leaf_samples = nr_wrong[leaf] + nr_correct[leaf]
            proba_wrong_leaf, proba_correct_leaf = nr_wrong[leaf] / nr_leaf_samples, nr_correct[leaf] / nr_leaf_samples
            print('Leaf {} (Wrong prediction: {:.3f}, Correct prediction: {:.3f})'.format(leaf, proba_wrong_leaf,
                                                                                          proba_correct_leaf))

            for i, feature_idx in enumerate(ranked_feature_ids):

                feature_name = feature_names[feature_idx]
                feature_is_numerical = True if self.pipeline_preprocessor is None else (
                    not self.pipeline_preprocessor.is_categorical(feature_idx))

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
                        histogram_func = lambda f_samples: \
                            np.bincount(np.searchsorted(bins, f_samples),
                                        minlength=len(bins))[:nr_bins].astype(float)
                    else:
                        histogram_func = lambda f_samples: \
                            np.bincount(np.searchsorted(bins, f_samples),
                                        minlength=len(bins))[:nr_bins].astype(float) / len(f_samples)

                if show_global:
                    if show_class:
                        hist_wrong = histogram_func(feature_column[global_error_sample_ids])
                        hist_correct = histogram_func(feature_column[~global_error_sample_ids])
                        n_samples = np.sum(hist_wrong + hist_correct)
                        normalized_hist_wrong = hist_wrong / n_samples
                        normalized_hist_correct = hist_correct / n_samples
                        root_hist_data = {
                            ErrorAnalyzerConstants.WRONG_PREDICTION:
                                normalized_hist_wrong,
                            ErrorAnalyzerConstants.CORRECT_PREDICTION:
                                normalized_hist_correct
                        }
                    else:
                        root_prediction = ErrorAnalyzerConstants.CORRECT_PREDICTION if int(nr_correct[0]) >= int(
                            nr_wrong[0]) else ErrorAnalyzerConstants.WRONG_PREDICTION
                        root_hist_data = {root_prediction: histogram_func(feature_column)}

                if show_class:
                    hist_wrong = histogram_func(feature_column[leaf_sample_ids & global_error_sample_ids])
                    hist_correct = histogram_func(feature_column[leaf_sample_ids & ~global_error_sample_ids])
                    n_samples = np.sum(hist_wrong + hist_correct)
                    normalized_hist_wrong = hist_wrong / n_samples
                    normalized_hist_correct = hist_correct / n_samples
                    leaf_hist_data = {
                        ErrorAnalyzerConstants.WRONG_PREDICTION:
                            normalized_hist_wrong,
                        ErrorAnalyzerConstants.CORRECT_PREDICTION:
                            normalized_hist_correct
                    }
                else:
                    leaf_prediction = ErrorAnalyzerConstants.CORRECT_PREDICTION if proba_correct_leaf > proba_wrong_leaf else ErrorAnalyzerConstants.WRONG_PREDICTION
                    leaf_hist_data = {
                        leaf_prediction: histogram_func(feature_column[leaf_sample_ids])}

                x_ticks = _BaseErrorVisualizer._add_new_plot(figsize, bins, feature_name, leaf)
                _BaseErrorVisualizer._plot_feature_distribution(x_ticks, feature_is_numerical, leaf_hist_data,
                                                                root_hist_data if show_global else None)

        plt.show()
