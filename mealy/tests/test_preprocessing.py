import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
import random
import unittest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from mealy.preprocessing import PipelinePreprocessor, DummyPipelinePreprocessor

default_seed = 10
np.random.seed(default_seed)
random.seed(default_seed)


class TestFeatureTransformer(unittest.TestCase):
    def setUp(self):
        self.feature_list = ["num_1", "num_2", "cat_1", "cat_2"]
        self.x = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        self.sparse_x = csr_matrix(self.x)
        self.feature_importances = np.array([0.5, 0.2, 0.3])


class TestDummyPipeline(TestFeatureTransformer):
    def setUp(self):
        super(TestDummyPipeline, self).setUp()
        self.df = pd.DataFrame(self.x, columns=self.feature_list[:-1])
        self.pipe = DummyPipelinePreprocessor(self.feature_list)

    def test_transform_array(self):
        preprocessed_x = self.pipe.transform(self.x)
        self.assertTrue((preprocessed_x == self.x).all())

    def test_transform_df(self):
        preprocessed_x = self.pipe.transform(self.df)
        self.assertTrue((preprocessed_x == self.x).all())

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
                ranked = self.pipe.get_top_ranked_feature_ids(self.feature_importances,
                                                              max_nr_features=max_nr_features)
                self.assertTrue((ranked == result).all())

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


class TestPreprocessingPipelineWithPipeline(TestFeatureTransformer):
    @unittest.mock.patch("mealy.preprocessing.PipelinePreprocessor._get_feature_list_from_column_transformer", return_value=["num_1", "num_2", "cat_1", "cat_2"])
    @unittest.mock.patch("mealy.preprocessing.PipelinePreprocessor._create_feature_mapping", return_value=None)
    def setUp(self, patched_create_mapping, patched_feature_list):
        super(TestPreprocessingPipelineWithPipeline, self).setUp()
        col_transformer = unittest.mock.Mock(spec=ColumnTransformer)
        self.pipe = PipelinePreprocessor(col_transformer)

    @unittest.mock.patch("mealy.preprocessing.PipelinePreprocessor._update_feature_mapping_dict_using_input_names", return_value=None)
    @unittest.mock.patch("mealy.preprocessing.PipelinePreprocessor._update_feature_mapping_dict_using_output_names", return_value=None)
    @unittest.mock.patch("mealy.preprocessing.generate_preprocessing_steps", side_effect=lambda transformer: transformer)
    def test_create_feature_mapping_output_dim_change(self, _, mocked_o, mocked_i):
        ohe = unittest.mock.Mock(spec=OneHotEncoder)
        steps = [
            unittest.mock.Mock(spec=StandardScaler),
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

    @unittest.mock.patch("mealy.preprocessing.PipelinePreprocessor._update_feature_mapping_dict_using_input_names", return_value=None)
    @unittest.mock.patch("mealy.preprocessing.PipelinePreprocessor._update_feature_mapping_dict_using_output_names", return_value=None)
    @unittest.mock.patch("mealy.preprocessing.generate_preprocessing_steps", side_effect=lambda transformer: transformer)
    def test_create_feature_mapping_no_change_in_output_dim(self, _, mocked_o, mocked_i):
        steps = [
            unittest.mock.Mock(spec=StandardScaler),
            "drop",
            "passthrough",
            unittest.mock.Mock(spec=SimpleImputer)
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
        single_tr = unittest.mock.Mock()
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
        with self.assertRaises(ValueError, msg="Either the input index or its name should be specified."):
            self.pipe.is_categorical()

    @unittest.mock.patch("mealy.preprocessing.PipelinePreprocessor.inverse_transform_feature_id", side_effect=lambda idx: idx)
    @unittest.mock.patch("mealy.preprocessing.PipelinePreprocessor.inverse_transform", side_effect=lambda a: a+1)
    def test_inverse_thresholds(self, mocked_inverse_transform, mocked_inverse_transform_feature_id):
        nr_cols = 6
        tree = unittest.mock.Mock(feature=np.array([0,-2,1,3,-2,-2,0]), threshold=np.array([1, -2, 42,6,-2,-2,12]))
        thresholds = self.pipe.inverse_thresholds(tree, nr_cols)
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
        self.assertTrue((thresholds == [2, -2, 43, 7, -2, -2, 13]).all())

    def test_get_feature_ids_related_to_transformer(self):
        self.pipe.original2preprocessed = {0: [42], 1: [6], 2: [7], 3: [6, 7]}
        orig_feature_ids, preprocessed_feature_ids = self.pipe._get_feature_ids_related_to_transformer(["num_1", "cat_2"])
        self.assertListEqual(preprocessed_feature_ids, [42, 6, 7])
        self.assertTrue((orig_feature_ids == [0, 3]).all())

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

    @unittest.mock.patch("mealy.preprocessing.generate_preprocessing_steps", side_effect=lambda transformer: transformer)
    def test_get_feature_list_from_column_transformer(self, _):
        steps = [
            "drop",
            "passthrough",
            unittest.mock.Mock(spec=StandardScaler),
            unittest.mock.Mock(spec=OneHotEncoder)
        ]

        other_steps = [
            "drop",
            "passthrough",
            unittest.mock.Mock(spec=StandardScaler),
            unittest.mock.Mock(spec=SimpleImputer)
        ]

        self.pipe.ct_preprocessor.transformers_ = [
            ("does_not_matter", steps, ["cat_1", "cat_2"]),
            ("does_not_matter", other_steps, ["num_1"])
        ]

        features = self.pipe._get_feature_list_from_column_transformer()
        self.assertListEqual(self.pipe.categorical_features, ["cat_1", "cat_2"])
        self.assertListEqual(features, ["cat_1", "cat_2", "num_1"])

    def test_inverse_transform(self):
        pass #TODO
