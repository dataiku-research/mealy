import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from mealy import ErrorAnalyzer
from mealy.metrics import compute_mpp_accuracy, balanced_accuracy_score, compute_primary_model_accuracy, compute_confidence_decision

default_seed = 10
np.random.seed(default_seed)
random.seed(default_seed)

DATASET_URL = 'https://www.openml.org/data/get_csv/54002/adult-census.arff'
df = pd.read_csv(DATASET_URL)
target = 'class'

def test_with_only_scikit_model():
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    X = df.dropna().drop(target, axis=1)[numeric_features]
    y = df.dropna()[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    mpp = ErrorAnalyzer(rf, feature_names=numeric_features)
    mpp.fit(X_test, y_test)

    prep_x, y_true = mpp._compute_primary_model_error(X_test, y_test, max_nr_rows=100000)
    y_pred = mpp.model_performance_predictor.predict(prep_x)


    mpp_accuracy_score = compute_mpp_accuracy(y_true, y_pred)
    mpp_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    primary_model_predicted_accuracy = compute_primary_model_accuracy(y_pred)
    primary_model_true_accuracy = compute_primary_model_accuracy(y_true)
    fidelity, confidence_decision = compute_confidence_decision(primary_model_true_accuracy,
                                                            primary_model_predicted_accuracy)

    metric_to_check = {
        'mpp_accuracy_score': mpp_accuracy_score,
        'mpp_balanced_accuracy': mpp_balanced_accuracy,
        'primary_model_predicted_accuracy': primary_model_predicted_accuracy,
        'primary_model_true_accuracy': primary_model_true_accuracy,
        'fidelity': fidelity
    }

    metric_reference = {
        'mpp_accuracy_score': 0.8651926915399969,
        'mpp_balanced_accuracy': 0.7003443854497639,
        'primary_model_predicted_accuracy': 0.8836173806233687,
        'primary_model_true_accuracy': 0.8255796100107478,
        'fidelity': 0.9419622293873791,
    }

    for metric_name, metric_value in metric_to_check.items():
        print('Checking', metric_name)
        assert (metric_value - metric_reference[metric_name]) < 0.0001


def test_with_scikit_pipeline():
    X = df.dropna().drop(target, axis=1)
    y = df.dropna()[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns  # .tolist()
    categorical_features = df.select_dtypes(include=['object']).drop([target], axis=1).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('SimpleImputer', SimpleImputer(strategy='median')),
        ('StandardScaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('OneHotEncoder', OneHotEncoder(handle_unknown='ignore')),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    rf = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', RandomForestClassifier())])

    rf.fit(X_train, y_train)
    mpp = ErrorAnalyzer(rf)
    mpp.fit(X_test, y_test)

    X_test_prep, y_test_prep = mpp.pipeline_preprocessor.transform(X_test), np.array(y_test)

    prep_x, y_true = mpp._compute_primary_model_error(X_test_prep, y_test_prep, max_nr_rows=100000)
    y_pred = mpp.model_performance_predictor.predict(prep_x)

    mpp_accuracy_score = compute_mpp_accuracy(y_true, y_pred)
    mpp_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    primary_model_predicted_accuracy = compute_primary_model_accuracy(y_pred)
    primary_model_true_accuracy = compute_primary_model_accuracy(y_true)
    fidelity, confidence_decision = compute_confidence_decision(primary_model_true_accuracy,
                                                                primary_model_predicted_accuracy)

    metric_to_check = {
        'mpp_accuracy_score': mpp_accuracy_score,
        'mpp_balanced_accuracy': mpp_balanced_accuracy,
        'primary_model_predicted_accuracy': primary_model_predicted_accuracy,
        'primary_model_true_accuracy': primary_model_true_accuracy,
        'fidelity': fidelity
    }

    metric_reference = {
        'mpp_accuracy_score': 0.8952863503761708,
        'mpp_balanced_accuracy': 0.7209989546416461,
        'primary_model_predicted_accuracy': 0.9011208352525718,
        'primary_model_true_accuracy': 0.8602794411177644,
        'fidelity': 0.9591586058651926
    }

    for metric_name, metric_value in metric_to_check.items():
        print('Checking', metric_name)
        assert (metric_value - metric_reference[metric_name]) < 0.0001