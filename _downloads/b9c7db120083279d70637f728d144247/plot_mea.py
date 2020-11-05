"""
Model Error Analysis on Boston house
===================================================================

Here we train a primary model to predict the price of houses in Boston.
Then we build a Model Performance Predictor, a Decision Tree trained to
predict on what samples the primary model will yield Wrong or Correct
predictions. We then use the Model Performance Predictor to understand
what are the problematic samples and features where the majority of
model failures occurs.
"""


##############################################################################
# Those are the necessary imports and initializations

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import numpy as np

from mea.error_analyzer import ErrorAnalyzer
from mea.error_visualizer import ErrorVisualizer

np.random.seed(7)

##############################################################################
# Load the Boston houses dataset

dataset = load_boston()
X = dataset.data
y = dataset.target
feature_names = dataset.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

##############################################################################
# Train a RandomForestRegressor to predict the price.
# This is the primary model.

model = RandomForestRegressor()
model.fit(X_train, y_train)

r2_score = model.score(X_test, y_test)
print("R^2: {:.2f}".format(r2_score))

##############################################################################
# Fit a Model Performance Predictor on the primary model performances

error_analyzer = ErrorAnalyzer(model, feature_names=feature_names)
error_analyzer.fit(X_test, y_test)

##############################################################################
# Print metrics regarding the Model Performance Predictor

print(error_analyzer.mpp_summary(X_test, y_test, output_dict=False))

##############################################################################
# Plot the Model Performance Predictor Decision Tree

error_visualizer = ErrorVisualizer(error_analyzer)
error_visualizer.plot_error_tree()

##############################################################################
# Print the details regarding the decision tree nodes containing the majority of errors

error_analyzer.error_node_summary(leaf_selector="all_errors", add_path_to_leaves=True, print_summary=True)

##############################################################################
# Plot the feature distributions of samples in the nodes containing the majority of errors
# Rank features by correlation to error

error_visualizer.plot_feature_distributions_on_leaves(leaf_selector="all_errors", top_k_features=3)


##############################################################################
# Discussion
# ----------
#
# Model Performance Predictor Metrics
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We are facing a regression problem, but the primary predictions are thresholded
# and categorized into Wrong/Correct predictions. In this context, the primary task is
# translated into a binary classification and the primary model can be scored using an
# accuracy as the the average number of samples predicted as close enough to the true price.
# This accuracy in this example is of 92.9%, that is correctly learnt by the MPP
# estimating the very same value. The analysis will focus than on those 7.1% of test samples
# where the primary predictions failed, i.e. are not close enough to the true value.
#
#
# Model Failures
# ^^^^^^^^^^^^^^
#
# The majority of failures are highlighted first in the most relevant failure node, the LEAF 16.
# From the feature distribution, we see that most failures occur for high values of feature RM and AGE.
# In the next iteration of model design, we need a strategy to improve the primary model
# on those sub-populations.
#

