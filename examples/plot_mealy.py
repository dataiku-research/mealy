"""
Model Error Analysis for the Boston houses dataset
===================================================================

Here we train a RandomForestRegressor to predict the price of the houses
in Boston. This is our primary model. Then we build a secondary model,
called Model Performance Predictor (MPP), to predict on what samples
the primary model returns wrong or correct predictions. The MPP is a
DecisionTree returning a binary outcome success/failure. The leaf nodes
yielding failure outcome gather the samples mis-predicted by the primary
model. Plotting the feature distributions of these samples and comparing
to the whole data highlights the subpopulations where the model works poorly.
"""


##############################################################################
# When using a python notebook, set ``%matplotlib inline`` to enable display.


##############################################################################
# Those are the necessary imports and initializations.

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import numpy as np
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from mealy.error_analyzer import ErrorAnalyzer
from mealy.error_visualizer import ErrorVisualizer


default_seed = 10
np.random.seed(default_seed)
random.seed(default_seed)

##############################################################################
# Load Boston houses dataset.

dataset = load_boston()
X = dataset.data
y = dataset.target
feature_names = dataset.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y)

##############################################################################
# Train a RandomForestRegressor.

model = RandomForestRegressor()
model.fit(X_train, y_train)

r2_score = model.score(X_test, y_test)
print("R2 = %.2f" % r2_score)

##############################################################################
# Fit a Model Performance Predictor on the model performances.

error_analyzer = ErrorAnalyzer(model, feature_names=feature_names)
error_analyzer.fit(X_test, y_test)

##############################################################################
# Print metrics regarding the Model Performance Predictor.

print(error_analyzer.mpp_summary(X_test, y_test, output_dict=False))

##############################################################################
# Plot the Model Performance Predictor Decision Tree.

error_visualizer = ErrorVisualizer(error_analyzer)
tree_src = error_visualizer.plot_error_tree()

# the output of ``plot_error_tree`` is rendered automatically in a python notebook
# the following is for rendering in this sphinx gallery
tree_src.format = 'png'
tree_src.render('tree')
tree_img = mpimg.imread('tree.png')

plt.figure(figsize=(20, 20))
plt.imshow(tree_img)
plt.axis('off')

##############################################################################
# Print the details regarding the decision tree nodes containing the majority of errors.

error_analyzer.get_error_node_summary(leaf_selector="all_errors", add_path_to_leaves=True, print_summary=True);

##############################################################################
# Plot the feature distributions of samples in the leaf containing the majority of errors.
# Rank features by correlation to error.
leaf_id = error_analyzer._get_ranked_leaf_ids('all_errors')[0]
error_visualizer.plot_feature_distributions_on_leaves(leaf_selector=leaf_id, top_k_features=3)

##############################################################################
# Discussion
# ----------
#
# Model Performance Predictor Metrics
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We are dealing with a regression task, but the metrics highlight the accuracy
# of the primary model and its estimate given by the Model Performance Predictor.
# Here the primary predictions of price have been categorized in two classes:
# 'Correct prediction' and 'Wrong prediction' by thresholding the deviation of
# the prediction from the true value. Close enough predictions are Correct prediction,
# the others are Wrong prediction. For more details, have a look at the documentation.
# The accuracy is then the number of Correct predictions over the total.
# The MPP is representative of the behavior of the primary model as the true primary
# accuracy and the one estimated by the MPP are close.
#
# Model Failures
# ^^^^^^^^^^^^^^
#
# Let's focus on the nodes of the MPP DecisionTree, in particular the leaf nodes
# of class 'Wrong prediction'. These leaves contain the majority of errors, each
# leaf clustering a subpopulation of errors with different feature values. The largest
# and purest failure nodes are highlighted when printing the error node summary, and
# also when plotting the feature distributions in the node (``leaf_selector="all_errors"``).
# From the feature distributions, sorted by correlation with the error, we can see that
# the majority of problems occur for extreme values of features ``LSTAT`` and ``AGE``.
# In the next iteration of model design, the primary model needs to be improved for these
# subpopulations.
#
