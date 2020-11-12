Introduction to Model Error Analysis
====================================

Principle
----------

After training a ML model, data scientists need to investigate the model failures to build intuition on the critical sub-populations
on which the model is performing poorly. This analysis is essential in the iterative process of model design and feature engineering
and is usually performed manually.

The mealy package streamlines the analysis of the samples mostly contributing to model errors and provides the user with
automatic tools to break down the model errors into meaningful groups, easier to analyze, and to highlight the most frequent
type of errors, as well as the problematic features correlated with the failures.

We call the model under investigation the _primary_ model.

This approach relies on a Model Performance Predictor (MPP), a secondary model trained to predict whether the primary
model prediction is correct or wrong, i.e. a success or a failure. More precisely, the MPP is a binary DecisionTree classifier
predicting whether the primary model will yield a Correct Prediction or a Wrong Prediction.

The MPP can be trained on any dataset meant to evaluate the primary model performances, thus containing ground truth labels.
In particular the provided primary test set is split into a secondary training set to train the MPP and a secondary test set
to compute the MPP metrics.

In classification tasks the model failure is a wrong predicted class, whereas in case of regression tasks the failure is
defined as a large deviation of the predicted value from the true one. In the latter case, when the absolute difference
between the predicted and the true value is higher than a threshold ε, the model outcome is considered as a Wrong Prediction.
The threshold ε is computed as the knee point of the Regression Error Characteristic (REC) curve, ensuring the absolute error
of primary predictions to be within tolerable limits.

The leaves of the MPP decision tree break down the test dataset into smaller segments with similar features and similar
model performances. Analyzing the sub-population in the error leaves, and comparing with the global population, provides
insights about critical features correlated with the model failures.

The mealy package leads the user to focus on what are the problematic features and what are the typical values of these features
for the mis-predicted samples. This information can later be exploited to support the strategy selected by the user :
* improve model design: removing a problematic feature, removing samples likely to be mislabeled, ensemble with a model trained
on a problematic subpopulation, ...
* enhance data collection: gather more data regarding the most erroneous under-represented populations,
* select critical samples for manual inspection thanks to the MPP and avoid primary predictions on them, generating model assertions.

The typical workflow in the iterative model design supported by error analysis is illustrated in the figure below.

.. image:: _static/mea_flow.png
  :alt: Model Error Analysis workflow

Metrics
----------
