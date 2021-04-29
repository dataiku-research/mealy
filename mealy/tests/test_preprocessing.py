import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import random
import unittest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from mealy import ErrorAnalyzer
from mealy.metrics import compute_accuracy_score, balanced_accuracy_score, compute_primary_model_accuracy, compute_confidence_decision
from mealy.preprocessing import PipelinePreprocessor, DummyPipelinePreprocessor

default_seed = 10
np.random.seed(default_seed)
random.seed(default_seed)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.feature_list = ['a', 'b', 'c']
        self.x = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        self.y = [1, 1, 0]
        self.df = pd.DataFrame(self.x, columns=self.feature_list)
        self.sparse_x = csr_matrix(self.x)
        self.feature_importances = np.array([0.5, 0.2, 0.3])

    def test_with_dummy_pipeline(self):
        pipe = DummyPipelinePreprocessor(self.feature_list)
        preprocessed_x_from_array = pipe.transform(self.x)
        preprocessed_x_from_df = pipe.transform(self.df)
        preprocessed_x_from_sparse_matrix = pipe.transform(self.sparse_x)
        all_feature_importances = pipe.get_top_ranked_feature_ids(self.feature_importances, max_nr_features=0)
        top_2_feature_importances = pipe.get_top_ranked_feature_ids(self.feature_importances, max_nr_features=2)

        self.assertIsNone(np.testing.assert_array_equal(preprocessed_x_from_array, self.x))
        self.assertIsNone(np.testing.assert_array_equal(preprocessed_x_from_df, self.x))
        self.assertIsNone(np.testing.assert_array_equal(preprocessed_x_from_sparse_matrix.toarray(), self.x))
        self.assertListEqual(pipe.get_original_feature_names(), self.feature_list)
        self.assertListEqual(pipe.get_preprocessed_feature_names(), self.feature_list)
        self.assertFalse(pipe.is_categorical(index=0), False)
        self.assertEqual(pipe.inverse_transform_feature_id(index=0), 0)
        self.assertIsNone(np.testing.assert_array_equal(pipe.inverse_transform(preprocessed_x_from_array), self.x))
        self.assertListEqual(all_feature_importances.tolist(), [0, 2, 1])
        self.assertListEqual(top_2_feature_importances.tolist(), [0, 2])

    def test_with_scikit_pipeline(self):
        numeric_transformer = Pipeline(steps=[
            ('SimpleImputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('StandardScaler', StandardScaler()),
        ])

        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, self.feature_list)])
        scikit_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', DecisionTreeClassifier(max_depth=2, random_state=0))])

        scikit_pipeline.fit(self.df, self.y)

        pipe = PipelinePreprocessor(scikit_pipeline.steps[0][1], self.feature_list)

        ref_transformed_array = np.array([[-1.22474487, -1.22474487, -1.22474487],
                                       [ 0., 0., 0.],
                                       [ 1.22474487, 1.22474487, 1.22474487]])
        self.assertIsNone(np.testing.assert_allclose(pipe.transform(self.df), ref_transformed_array))
        # got a weird error with assert_allclose if we don't manually recast the result to np array
        self.assertIsNone(np.testing.assert_allclose(np.array(pipe.inverse_transform(ref_transformed_array).tolist()), self.x))

        self.assertEqual(pipe.inverse_transform_feature_id(0), 0)
        self.assertEqual(pipe.inverse_transform_feature_id(1), 1)
        self.assertEqual(pipe.inverse_transform_feature_id(2), 2)

        self.assertListEqual(pipe.get_top_ranked_feature_ids(self.feature_importances, 0), [0, 2, 1])
        self.assertListEqual(pipe.get_top_ranked_feature_ids(self.feature_importances, 2), [0, 2])