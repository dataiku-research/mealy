import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from unittest import TestCase
from unittest.mock import patch, Mock
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from .. import PipelinePreprocessor, DummyPipelinePreprocessor


class TestFeatureTransformer(TestCase):
    def setUp(self):
        self.feature_list = ["num_1", "num_2", "cat_1", "cat_2"]


class TestDummyPipeline(TestFeatureTransformer):
    def setUp(self):
        super(TestDummyPipeline, self).setUp()
        self.x = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        self.df = pd.DataFrame(self.x, columns=self.feature_list[:-1])
        self.sparse_x = csr_matrix(self.x)
        self.pipe = DummyPipelinePreprocessor(self.feature_list)

    def test_transform_array(self):
        preprocessed_x = self.pipe.transform(self.x)
        np.testing.assert_array_equal(preprocessed_x, self.x)

    def test_transform_df(self):
        preprocessed_x = self.pipe.transform(self.df)
        np.testing.assert_array_equal(preprocessed_x, self.x)

    def test_transform_sparse(self):
        preprocessed_x = self.pipe.transform(self.sparse_x)
        self.assertEqual((preprocessed_x != self.sparse_x).nnz, 0)

    def test_get_top_ranked_feature_ids(self):
        input_to_results = {
            0: np.array([0, 2, 1]),
            2: np.array([0, 2]),
            -2: np.array([0])
        }
        for max_nr_features, result in input_to_results.items():
            with self.subTest(max_nr_features=max_nr_features):
                ranked = self.pipe.get_top_ranked_feature_ids(np.array([0.5, 0.2, 0.3]),
                                                              max_nr_features=max_nr_features)
                np.testing.assert_array_equal(ranked, result)

    def test_original_features(self):
        self.assertListEqual(self.pipe.get_original_feature_names(), self.feature_list)

    def test_preprocessed_features(self):
        self.assertListEqual(self.pipe.get_preprocessed_feature_names(), self.feature_list)

    def test_is_cat(self):
        self.assertFalse(self.pipe.is_categorical(0))

    def test_inverse_transform(self):
        array = np.random.rand(3,4)
        np.testing.assert_array_equal(self.pipe.inverse_transform(array), array)

    def test_inverse_transform_feature_id(self):
        self.assertEqual(self.pipe.inverse_transform_feature_id(1), 1)


class TestPreprocessingPipeline(TestFeatureTransformer):
    def setUp(self):
        super(TestPreprocessingPipeline, self).setUp()
        col_transformer = Mock(spec=ColumnTransformer)
        with patch("mealy.preprocessing.PipelinePreprocessor._get_feature_list_from_column_transformer", return_value=self.feature_list),\
            patch("mealy.preprocessing.PipelinePreprocessor._create_feature_mapping", return_value=None):
            self.pipe = PipelinePreprocessor(col_transformer)

    def test_init(self):
        col_transformer = Mock(spec=ColumnTransformer)
        with patch("mealy.preprocessing.PipelinePreprocessor._get_feature_list_from_column_transformer", return_value=self.feature_list),\
            patch("mealy.preprocessing.PipelinePreprocessor._create_feature_mapping", return_value=None):
            pipe_with_feature_arg = PipelinePreprocessor(col_transformer, self.feature_list)
            pipe_without_feature_arg = PipelinePreprocessor(col_transformer)
            with self.assertRaisesRegex(ValueError,
                "The list of features given by user does not correspond to the list of features handled by the Pipeline."):
                error = PipelinePreprocessor(col_transformer, ["cat_1", "num_2"])
        self.assertListEqual(pipe_with_feature_arg.get_original_feature_names(), self.feature_list)
        self.assertListEqual(pipe_without_feature_arg.get_original_feature_names(), self.feature_list)

    @patch("mealy.preprocessing.PipelinePreprocessor._update_feature_mapping_dict_using_input_names", return_value=None)
    @patch("mealy.preprocessing.PipelinePreprocessor._update_feature_mapping_dict_using_output_names", return_value=None)
    @patch("mealy.preprocessing.generate_preprocessing_steps", side_effect=lambda transformer: transformer)
    def test_create_feature_mapping_output_dim_change(self, _, mocked_o, mocked_i):
        ohe = Mock(spec=OneHotEncoder)
        steps = [
            Mock(spec=StandardScaler),
            "drop",
            "passthrough",
            ohe
        ]
        self.pipe.ct_preprocessor.transformers_ = [("does_not_matter", steps, ["num_1", "cat_2"])]
        self.pipe._create_feature_mapping()
        step, features, feat_ids = mocked_o.call_args[0]
        self.assertEqual(step, ohe)
        self.assertListEqual(features, ["num_1", "cat_2"])
        np.testing.assert_array_equal(feat_ids, np.array([0,3]))
        self.assertEqual(mocked_o.call_count, 1)
        self.assertEqual(mocked_i.call_count, 0)

    @patch("mealy.preprocessing.PipelinePreprocessor._update_feature_mapping_dict_using_input_names", return_value=None)
    @patch("mealy.preprocessing.PipelinePreprocessor._update_feature_mapping_dict_using_output_names", return_value=None)
    @patch("mealy.preprocessing.generate_preprocessing_steps", side_effect=lambda transformer: transformer)
    def test_create_feature_mapping_no_change_in_output_dim(self, _, mocked_o, mocked_i):
        steps = [
            Mock(spec=StandardScaler),
            "drop",
            "passthrough",
            Mock(spec=SimpleImputer)
        ]
        self.pipe.ct_preprocessor.transformers_ = [("does_not_matter", steps, self.feature_list)]
        self.pipe._create_feature_mapping()
        self.assertEqual(mocked_o.call_count, 0)
        self.assertEqual(mocked_i.call_count, 1)
        features, feat_ids = mocked_i.call_args[0]
        self.assertListEqual(features, self.feature_list)
        np.testing.assert_array_equal(feat_ids, np.array([0,1,2,3]))

    def test_update_feature_mapping_dict_using_input_names(self):
        self.pipe._update_feature_mapping_dict_using_input_names(["transformed_name_1", "transformed_name_2"], [0, 4])
        self.assertListEqual(self.pipe.preprocessed_feature_names, ["transformed_name_1", "transformed_name_2"])
        self.assertDictEqual(self.pipe.original2preprocessed, {0: [0], 4: [1]})
        self.assertDictEqual(self.pipe.preprocessed2original, {0: 0, 1: 4})

    def test_update_feature_mapping_dict_using_output_names(self):
        single_tr = Mock()
        single_tr.get_feature_names.return_value = ["very_transformed_name_1_0", "even_more_transformed_name"]
        self.pipe._update_feature_mapping_dict_using_output_names(single_tr, ["transformed_name_1", "transformed_name_2"], [0, 4])
        self.assertListEqual(self.pipe.preprocessed_feature_names, ["very_transformed_name_1_0", "even_more_transformed_name"])
        self.assertDictEqual(self.pipe.original2preprocessed, {0: [0]})
        self.assertDictEqual(self.pipe.preprocessed2original, {0: 0})

    def test_original_features(self):
        self.assertListEqual(self.pipe.get_original_feature_names(), self.feature_list)

    def test_is_cat(self):
        self.pipe.categorical_features = ["cat_1", "cat_2"]
        self.assertFalse(self.pipe.is_categorical(1))
        self.assertFalse(self.pipe.is_categorical(name="num_1"))
        self.assertTrue(self.pipe.is_categorical(3))
        self.assertTrue(self.pipe.is_categorical(name="cat_1"))
        with self.assertRaisesRegex(ValueError, "Either the input index or its name should be specified."):
            self.pipe.is_categorical()

    @patch("mealy.preprocessing.PipelinePreprocessor.inverse_transform_feature_id", side_effect=lambda idx: idx)
    @patch("mealy.preprocessing.PipelinePreprocessor.inverse_transform", side_effect=lambda a: a+1)
    def test_inverse_thresholds(self, mocked_inverse_transform, mocked_inverse_transform_feature_id):
        tree = Mock(feature=np.array([0,-2,1,3,-2,-2,0]), threshold=np.array([1, -2, 42,6,-2,-2,12]))
        self.pipe.preprocessed_feature_names = ["a", "b", "c", "d", "e", "f"]
        thresholds = self.pipe.inverse_thresholds(tree)
        a = mocked_inverse_transform.call_args[0][0]
        self.assertTrue(mocked_inverse_transform.call_count == 1)
        np.testing.assert_array_equal(a, np.array([
            [1,0,0,0,0,0],
            [0,42,0,0,0,0],
            [0,0,0,6,0,0],
            [12,0,0,0,0,0],
        ]))
        self.assertTrue(mocked_inverse_transform_feature_id.call_count == 4)
        first, second, third, fourth = mocked_inverse_transform_feature_id.call_args_list
        self.assertEqual(first[0][0], 0)
        self.assertEqual(second[0][0], 1)
        self.assertEqual(third[0][0], 3)
        self.assertEqual(fourth[0][0], 0)
        np.testing.assert_array_equal(thresholds, [2, -2, 43, 7, -2, -2, 13])

    def test_get_feature_ids_related_to_transformer(self):
        self.pipe.original2preprocessed = {0: [42], 1: [6], 2: [7], 3: [6, 7]}
        orig_feature_ids, preprocessed_feature_ids = self.pipe._get_feature_ids_related_to_transformer(["num_1", "cat_2"])
        self.assertListEqual(preprocessed_feature_ids, [42, 6, 7])
        np.testing.assert_array_equal(orig_feature_ids, [0, 3])

    def test_get_top_ranked_feature_ids(self):
        importance = np.array([20, 1, -2, 43])
        self.pipe.preprocessed2original = {0:1, 1:2, 2:3, 3:0}
        self.assertListEqual(self.pipe.get_top_ranked_feature_ids(importance, 0), [0, 1, 2, 3])
        self.assertListEqual(self.pipe.get_top_ranked_feature_ids(importance, -2), [0, 1])
        self.assertListEqual(self.pipe.get_top_ranked_feature_ids(importance, 1), [0])

    def test_get_top_ranked_feature_ids_with_duplicates(self):
        importance = np.array([20, 1, -2, 43])
        self.pipe.preprocessed2original = {0:1, 1:1, 2:3, 3:3}
        self.assertListEqual(self.pipe.get_top_ranked_feature_ids(importance, 0), [3, 1])
        self.assertListEqual(self.pipe.get_top_ranked_feature_ids(importance, -1), [3, 1])
        self.assertListEqual(self.pipe.get_top_ranked_feature_ids(importance, 1), [3])

    @patch("mealy.preprocessing.generate_preprocessing_steps", side_effect=lambda transformer: transformer)
    def test_get_feature_list_from_column_transformer(self, _):
        steps = [
            "drop",
            "passthrough",
            Mock(spec=StandardScaler),
            Mock(spec=OneHotEncoder)
        ]

        other_steps = [
            "drop",
            "passthrough",
            Mock(spec=StandardScaler),
            Mock(spec=SimpleImputer)
        ]

        self.pipe.ct_preprocessor.transformers_ = [
            ("does_not_matter", steps, ["cat_1", "cat_2"]),
            ("does_not_matter", other_steps, ["num_1"])
        ]

        features = self.pipe._get_feature_list_from_column_transformer()
        self.assertListEqual(self.pipe.categorical_features, ["cat_1", "cat_2"])
        self.assertListEqual(features, ["cat_1", "cat_2", "num_1"])

    def test_inverse_transform(self):
        stsd = Mock(spec=StandardScaler)
        ohe = Mock(spec=OneHotEncoder)
        si = Mock(spec=SimpleImputer)

        self.pipe.ct_preprocessor.transformers_ = [
            ("does_not_matter", [stsd, ohe], ["num_1", "num_2"]),
            ("does_not_matter", [si], ["cat_1"])
        ]

        preprocessed_x = np.array([
            [0,1,2,3,4,5],
            [0,1,2,3,4,5],
            [0,1,2,3,4,5],
            [0,1,2,3,4,5],
            [0,1,2,3,4,5]
        ])

        expected = np.array([
            [5,6,2,0],
            [5,6,2,0],
            [5,6,2,0],
            [5,6,2,0],
            [5,6,2,0],
        ])

        def f(feature_names):
            if feature_names == ["cat_1"]:
                return 2, 1
            if feature_names == ["num_1", "num_2"]:
                return [0,1], [4,5]

        def g(step, transformer_output, feature_names):
            if step == si:
                return transformer_output * 2
            if step == ohe:
                return transformer_output + 4
            if step == stsd:
                return transformer_output - 3

        with patch("mealy.preprocessing.PipelinePreprocessor._get_feature_ids_related_to_transformer", side_effect=f),\
            patch("mealy.preprocessing.generate_preprocessing_steps", side_effect=lambda transformer, invert_order: reversed(transformer) if invert_order else transformer),\
            patch("mealy.preprocessing.PipelinePreprocessor._inverse_single_step", side_effect=g):
            result = self.pipe.inverse_transform(preprocessed_x)
        np.testing.assert_array_equal(result, expected)
