
import numpy as np
from unittest import TestCase, skip
from unittest.mock import Mock, patch

from .. import ErrorVisualizer, ErrorAnalyzer


class TestVisualizer(TestCase):
    def setUp(self):
        self.error_analyzer = Mock(spec=ErrorAnalyzer, _error_train_x=None, pipeline_preprocessor=Mock())


class TestPlotTree(TestVisualizer):
    def setUp(self):
        super(TestPlotTree, self).setUp()
        self.error_analyzer.error_tree.n_total_errors = 80
        self.error_analyzer.error_tree.error_class_idx = 1
        self.error_analyzer.error_tree.estimator_.tree_.value = np.array([
            [[40, 60]],
            [[5, 5]],
            [[0, 0]],
            [[0, 0]],
            [[30, 42]]
        ])
        self.error_analyzer.error_tree.estimator_.tree_.n_node_samples = np.array([100, 25, 40, 35, 72])
        self.error_analyzer.error_tree.estimator_.tree_.children_left = np.array([1, -2, -2, -2, -2])
        self.error_analyzer.error_tree.estimator_.tree_.children_right = np.array([4, -2, -2, -2, -2])
        self.error_analyzer.pipeline_preprocessor.get_original_feature_names.return_value = []

    def test_plot_error_tree(self):
        visualizer = ErrorVisualizer(self.error_analyzer)

        with patch("mealy.ErrorVisualizer.node_decision_rule",
            side_effect=lambda x, y: "decision rule" if y\
            else "pretty really very extremely fairly long decision rule"):
            result = visualizer.plot_error_tree((20, 30)).source.split("\n")

        digraph_desc, size, node_desc, edge_desc, graph_desc, leave_desc, final_line = result[:5] + result[-2:]
        root_label, root_samples, root_local_error, root_global_error, root_alpha = result[5:10]
        left_label, left_rule, left_samples, left_local_error, left_global_error, left_alpha, left_edge = result[10:17]
        right_label, right_rule, right_samples, right_local_error, right_global_error, right_alpha, right_edge = result[17:24]

        assert digraph_desc == 'digraph Tree {'
        assert size == ' size="20,30!";'
        assert node_desc == 'node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;'
        assert edge_desc == 'edge [fontname=helvetica] ;'
        assert graph_desc == 'graph [ranksep=equally, splines=polyline] ;'
        assert leave_desc == '{rank=same ; 1; 4} ;'
        assert final_line == '}'

        assert root_label == '0 [label="node #0'
        assert root_samples == 'samples = 100%'
        assert root_local_error == 'local error = 60%'
        assert root_global_error == 'fraction of total error = 75%'
        assert root_alpha == '", fillcolor="#CE122899", tooltip="root"] ;'

        assert left_label == '1 [label="node #1'
        assert left_rule == 'decision rule'
        assert left_samples == 'samples = 25%'
        assert left_local_error == 'local error = 20%'
        assert left_global_error == 'fraction of total error = 6.25%'
        assert left_alpha == '", fillcolor="#CE122833", tooltip="decision rule"] ;'
        assert left_edge == '0 -> 1 [penwidth=1];'

        assert right_label == '4 [label="node #4'
        assert right_rule == 'pretty really very extremely fai...'
        assert right_samples == 'samples = 72%'
        assert right_local_error == 'local error = 58.333%'
        assert right_global_error == 'fraction of total error = 52.5%'
        assert right_alpha == '", fillcolor="#CE122894", tooltip="pretty really very extremely fairly long decision rule"] ;'
        assert right_edge == '0 -> 4 [penwidth=5.25];'


class TestNodeDecisionRule(TestVisualizer):
    def setUp(self):
        super(TestNodeDecisionRule, self).setUp()
        self.error_analyzer.pipeline_preprocessor.get_original_feature_names.return_value = ["num", "cat"]
        self.error_analyzer.pipeline_preprocessor.is_categorical.side_effect = lambda name: name=="cat"
        self.error_analyzer._inverse_transform_thresholds.return_value = ["A", 6.45678]
        self.error_analyzer._inverse_transform_features.return_value = [1, 0]
        self.visualizer = ErrorVisualizer(self.error_analyzer)

    def test_categorical_left_node(self):
        cat_left = self.visualizer.node_decision_rule(0, True)
        self.assertEqual(cat_left, "cat is A")

    def test_categorical_right_node(self):
        cat_right = self.visualizer.node_decision_rule(0, False)
        self.assertEqual(cat_right, "cat is not A")

    def test_numerical_left_node(self):
        num_left = self.visualizer.node_decision_rule(1, True)
        self.assertEqual(num_left, "num <= 6.46")

    def test_numerical_right_node(self):
        num_right = self.visualizer.node_decision_rule(1, False)
        self.assertEqual(num_right, "6.46 < num")


# TODO: finish the tests
@skip(reason="Not done yet")
class TestPlotDistribution(TestVisualizer):
    def setUp(self):
        super(TestPlotDistribution, self).setUp()

    def test_show_global_show_class(self):
        pass

    def test_show_global_no_show_class(self):
        pass

    def test_no_show_global_show_class(self):
        pass

    def test_no_show_global_no_show_class(self):
        pass
