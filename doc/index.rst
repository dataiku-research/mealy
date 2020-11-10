.. Model Error Analysis documentation master file, created by
   sphinx-quickstart on Mon Oct 26 21:06:13 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mealy's documentation!
================================================

mealy is a Python package to perform Model Error Analysis of scikit-learn models,
leveraging a Model Performance Predictor, a Decision Tree predicting the failures
and successes of a ML model.

The code of the project is on Github: `mealy <https://github.com/dataiku/mealy>`_


.. _install:

Installation
============

Using Pypi
----------

mealy can be installed through Pypi using:

.. code-block:: bash

    $ pip install "mealy"

Installing locally
------------------

You can also fetch the code and install the package from your local repository.
Again, the preferred way is to use pip.

.. code-block:: bash

    $ git clone https://github.com/dataiku/mealy
    $ cd mealy
    $ pip install -e


.. toctree::
   :maxdepth: 2
   :caption: Model Error Analysis

   introduction

.. toctree::
   :maxdepth: 2
   :caption: Example galleries

   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: API and developer reference

   reference
   Fork mealy on Github <https://github.com/dataiku/mealy>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
