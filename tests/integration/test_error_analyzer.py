import os
import json
import numpy as np
import pandas as pd
import random
import unittest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .. import ErrorAnalyzer
from .. import compute_accuracy_score, balanced_accuracy_score, compute_primary_model_accuracy, compute_confidence_decision

default_seed = 10
np.random.seed(default_seed)
random.seed(default_seed)

DATASET_URL = 'https://www.openml.org/data/get_csv/54002/adult-census.arff'
df = pd.read_csv(DATASET_URL)
target = 'class'

script_dir = os.path.dirname(__file__)
reference_result_file_path = os.path.join(script_dir, 'reference_data.json')
with open(reference_result_file_path) as f:
    reference_data = json.load(f)

numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).drop([target], axis=1).columns.tolist()
X_numerical = df.dropna().drop(target, axis=1)[numeric_features]
X_all = df.dropna().drop(target, axis=1)
y = df.dropna()[target]


class TestErrorAnalyzer(unittest.TestCase):

    def test_with_only_scikit_model(self):
        X_train, X_test, y_train, y_test = train_test_split(X_numerical, y, test_size=0.2)
        rf = RandomForestClassifier(n_estimators=10)
        rf.fit(X_train, y_train)
        error_tree = ErrorAnalyzer(rf, feature_names=numeric_features)
        error_tree.fit(X_test, y_test)

        y_true, _ = error_tree._compute_primary_model_error(X_test.values, y_test)
        y_pred = error_tree._error_tree.estimator_.predict(X_test.values)

        error_tree_accuracy_score = compute_accuracy_score(y_true, y_pred)
        error_tree_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        primary_model_predicted_accuracy = compute_primary_model_accuracy(y_pred)
        primary_model_true_accuracy = compute_primary_model_accuracy(y_true)
        fidelity, confidence_decision = compute_confidence_decision(primary_model_true_accuracy,
                                                                    primary_model_predicted_accuracy)

        metric_to_check = {
            'error_tree_accuracy_score': error_tree_accuracy_score,
            'error_tree_balanced_accuracy': error_tree_balanced_accuracy,
            'primary_model_predicted_accuracy': primary_model_predicted_accuracy,
            'primary_model_true_accuracy': primary_model_true_accuracy,
            'fidelity': fidelity
        }

        reference_data_for_single_estimator = reference_data.get('single_estimator')
        metric_reference = reference_data_for_single_estimator.get('metric_reference')

        for metric_name, metric_value in metric_to_check.items():
            self.assertAlmostEqual(metric_value, metric_reference[metric_name], 5)

        leaf_summary_str = error_tree.get_error_leaf_summary(leaf_selector=98, output_format='str')
        leaf_summary_dct = error_tree.get_error_leaf_summary(leaf_selector=98, output_format='dict')
        single_leaf_summary_reference_dict = reference_data_for_single_estimator.get('single_leaf_summary_reference')
        self.assertListEqual(leaf_summary_str, single_leaf_summary_reference_dict['str'])
        self.assertListEqual(leaf_summary_dct, single_leaf_summary_reference_dict['dct'])

        all_leaves_summary_str = error_tree.get_error_leaf_summary(output_format='str')
        all_leaves_summary_dct = error_tree.get_error_leaf_summary(output_format='dict')
        all_leaves_summary_reference_dict = reference_data_for_single_estimator.get('all_leaves_summary_reference')
        self.assertListEqual(all_leaves_summary_str, all_leaves_summary_reference_dict['str'])
        self.assertListEqual(all_leaves_summary_dct, all_leaves_summary_reference_dict['dct'])

        evaluate_str = error_tree.evaluate(X_test, y_test, output_format='str')
        evaluate_dct = error_tree.evaluate(X_test, y_test, output_format='dict')
        evaluate_reference_dict = reference_data_for_single_estimator.get('evaluate_reference')
        self.assertEqual(evaluate_str, evaluate_reference_dict['str'])
        self.assertDictEqual(evaluate_dct, evaluate_reference_dict['dct'])


    def test_with_scikit_pipeline(self):
        X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2)
        numeric_transformer = Pipeline(steps=[
            ('SimpleImputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('StandardScaler', StandardScaler()),
        ])
        categorical_transformer = Pipeline(steps=[
            ('OneHotEncoder', OneHotEncoder(handle_unknown='ignore')),
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        rf = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', RandomForestClassifier(n_estimators=10))])

        rf.fit(X_train, y_train)
        error_tree = ErrorAnalyzer(rf)
        error_tree.fit(X_test, y_test)

        X_test_prep, y_test_prep = error_tree.pipeline_preprocessor.transform(X_test), np.array(y_test)

        y_true, _ = error_tree._compute_primary_model_error(X_test_prep, y_test_prep)
        y_pred = error_tree._error_tree.estimator_.predict(X_test_prep)

        error_tree_accuracy_score = compute_accuracy_score(y_true, y_pred)
        error_tree_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        primary_model_predicted_accuracy = compute_primary_model_accuracy(y_pred)
        primary_model_true_accuracy = compute_primary_model_accuracy(y_true)
        fidelity, confidence_decision = compute_confidence_decision(primary_model_true_accuracy,
                                                                    primary_model_predicted_accuracy)

        metric_to_check = {
            'error_tree_accuracy_score': error_tree_accuracy_score,
            'error_tree_balanced_accuracy': error_tree_balanced_accuracy,
            'primary_model_predicted_accuracy': primary_model_predicted_accuracy,
            'primary_model_true_accuracy': primary_model_true_accuracy,
            'fidelity': fidelity
        }

        reference_data_for_pipeline = reference_data.get('pipeline')
        metric_reference = reference_data_for_pipeline.get('metric_reference')

        for metric_name, metric_value in metric_to_check.items():
            self.assertAlmostEqual(metric_value, metric_reference[metric_name], 5)

        _ = error_tree.get_error_leaf_summary(leaf_selector=98, add_path_to_leaves=True, output_format='str')

        leaf_summary_str = error_tree.get_error_leaf_summary(leaf_selector=98, output_format='str')
        leaf_summary_dct = error_tree.get_error_leaf_summary(leaf_selector=98, output_format='dict')
        single_leaf_summary_reference_dict = reference_data_for_pipeline.get('single_leaf_summary_reference')
        self.assertListEqual(leaf_summary_str, single_leaf_summary_reference_dict['str'])
        self.assertListEqual(leaf_summary_dct, single_leaf_summary_reference_dict['dct'])

        all_leaves_summary_str = error_tree.get_error_leaf_summary(output_format='str')
        all_leaves_summary_dct = error_tree.get_error_leaf_summary(output_format='dict')
        all_leaves_summary_reference_dict = reference_data_for_pipeline.get('all_leaves_summary_reference')
        self.assertListEqual(all_leaves_summary_str, all_leaves_summary_reference_dict['str'])
        self.assertListEqual(all_leaves_summary_dct, all_leaves_summary_reference_dict['dct'])

        evaluate_str = error_tree.evaluate(X_test, y_test, output_format='str')
        evaluate_dct = error_tree.evaluate(X_test, y_test, output_format='dict')
        evaluate_reference_dict = reference_data_for_pipeline.get('evaluate_reference')
        self.assertEqual(evaluate_str, evaluate_reference_dict['str'])
        self.assertDictEqual(evaluate_dct, evaluate_reference_dict['dct'])
