# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from collections import defaultdict
import logging
from mealy.error_analysis_utils import check_lists_having_same_elements, generate_preprocessing_steps, invert_transform_via_identity
from mealy.constants import ErrorAnalyzerConstants

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='mealy | %(levelname)s - %(message)s')


class FeatureNameTransformer(object):
    """ Transformer of feature names and indices.

        A FeatureNameTransformer parses an input Pipeline preprocessor and generate
        a mapping between the input unprocessed feature names/indices and the output
        preprocessed feature names/indices.

        Args:
            ct_preprocessor (sklearn.compose.ColumnTransformer): preprocessor.
            orig_feats (list): list of original unpreprocessed feature names, default=None.

        Attributes:
            original_feature_names (list): list of original unpreprocessed feature names.
            preprocessed_feature_names (list): list of preprocessed feature names.

    """
    def __init__(self, original_features, preprocessed_features):
        self.original_feature_names = original_features
        self.preprocessed_feature_names = preprocessed_features

    def get_original_feature_names(self):
        return self.original_feature_names

    def get_preprocessed_feature_names(self):
        return self.preprocessed_feature_names

    def is_categorical(self, index=None, name=None):
        raise NotImplementedError

    def inverse_transform_feature_id(self, index):
        raise NotImplementedError

    def inverse_transform(self, x):
        raise NotImplementedError

    def transform(self, x):
        raise NotImplementedError

    def get_top_ranked_feature_ids(self, feature_importances, max_nr_features):
        raise NotImplementedError

    def inverse_thresholds(self, tree):
        raise NotImplementedError


class PipelinePreprocessor(FeatureNameTransformer):
    """Transformer of feature values from the original values to preprocessed ones.

        A PipelinePreprocessor parses an input Pipeline preprocessor and generate
        a mapping between the input unprocessed feature values and the output
        preprocessed feature values.

        Args:
            ct_preprocessor (sklearn.compose.ColumnTransformer): preprocessing steps.
            original_features (list): list of original unpreprocessed feature names, default=None.

    """

    def __init__(self, ct_preprocessor, original_features=None):
        self.ct_preprocessor = ct_preprocessor
        self.original2preprocessed = defaultdict(list)
        self.preprocessed2original = {}
        self.categorical_features = []

        logger.info('Retrieving the list of features used in the pipeline')
        original_features_from_ct = self._get_feature_list_from_column_transformer()
        if original_features is None:
            original_features = original_features_from_ct
        elif not check_lists_having_same_elements(original_features, original_features_from_ct):
            # If user explicitly gives a list of input features, we compare it with the list derived from the ColumnTransformer
            raise ValueError('The list of features given by user does not correspond to the list of features handled by the Pipeline.')

        super(PipelinePreprocessor, self).__init__(original_features=original_features, preprocessed_features=[])

        logger.info('Generating the feature id mapping dict')
        self._create_feature_mapping()

    def _get_feature_list_from_column_transformer(self):
        all_features = []
        for _, transformer, feature_names in self.ct_preprocessor.transformers_:
            for step in generate_preprocessing_steps(transformer):
                if isinstance(step, ErrorAnalyzerConstants.VALID_CATEGORICAL_STEPS):
                    # Check for categorical features
                    self.categorical_features += feature_names
                    break

            all_features += feature_names
        return all_features

    def _create_feature_mapping(self):
        """
        Update the dicts of input <-> output feature id mapping: self.original2preprocessed and self.preprocessed2original
        """
        for _, transformer, feature_names in self.ct_preprocessor.transformers_:
            orig_feat_ids = np.where(np.in1d(self.original_feature_names, feature_names))[0]
            for step in generate_preprocessing_steps(transformer):
                output_dim_changed = False
                if isinstance(step, ErrorAnalyzerConstants.STEPS_THAT_CHANGE_OUTPUT_DIMENSION_WITH_OUTPUT_FEATURE_NAMES):
                    # It is assumed that for each pipeline, at most one step changes the feature's dimension
                    # For now, it can only be a OneHotEncoder step
                    self._update_feature_mapping_dict_using_output_names(step,
                                                                        feature_names,
                                                                        orig_feat_ids)
                    output_dim_changed = True
                    break
            if not output_dim_changed:
                self._update_feature_mapping_dict_using_input_names(feature_names, orig_feat_ids)

    def _update_feature_mapping_dict_using_input_names(self, transformer_feature_names, original_feature_ids):
        self.preprocessed_feature_names.extend(transformer_feature_names)
        for original_feat_id in original_feature_ids:
            idx = len(self.preprocessed2original)
            self.original2preprocessed[original_feat_id] = [idx]
            self.preprocessed2original[idx] = original_feat_id

    def _update_feature_mapping_dict_using_output_names(self, single_transformer, transformer_feature_names, original_feature_ids):
        out_feature_names = list(single_transformer.get_feature_names(input_features=transformer_feature_names))
        self.preprocessed_feature_names.extend(out_feature_names)
        for orig_id, orig_name in zip(original_feature_ids, transformer_feature_names):
            part_out_feature_names = [name for name in out_feature_names if orig_name + '_' in name]
            offset = len(self.preprocessed2original)
            for i in range(len(part_out_feature_names)):
                self.original2preprocessed[orig_id].append(offset + i)
                self.preprocessed2original[offset + i] = orig_id

    def _transform_feature_id(self, index):
        """
        Args:
            index: int

        Returns: index of output feature(s) generated by the requested feature.
        """
        return self.original2preprocessed[index]

    def transform(self, x):
        """Transform the input feature values according to the preprocessing pipeline.

        Args:
            x (array-like or dataframe of shape (number of samples, number of features)): input feature values.

        Return:
            numpy.ndarray: transformed feature values.
        """
        return self.ct_preprocessor.transform(x)

    def _get_feature_ids_related_to_transformer(self, transformer_feature_names):
        original_features = self.get_original_feature_names()
        original_feature_ids = np.where(np.in1d(original_features, transformer_feature_names))[0]
        preprocessed_feature_ids = []
        for i in original_feature_ids:
            preprocessed_feature_ids += self._transform_feature_id(i)
        return original_feature_ids, preprocessed_feature_ids

    @staticmethod
    def _inverse_single_step(single_step, step_output, transformer_feature_names):
        inverse_transform_function_available = getattr(single_step, "inverse_transform", None)
        if invert_transform_via_identity(single_step):
            logger.info("Reversing step using identity transformation on column(s): {}".format(single_step, ', '.join(transformer_feature_names)))
            return step_output
        if inverse_transform_function_available:
            logger.info("Reversing step using inverse_transform() method on column(s): {}".format(single_step, ', '.join(transformer_feature_names)))
            return single_step.inverse_transform(step_output)
        raise TypeError('The package does not support {} because it does not provide inverse_transform function.'.format(single_step))

    def inverse_transform(self, preprocessed_x):
        """Invert the preprocessing pipeline and inverse transform feature values.

        Args:
            preprocessed_x (numpy.ndarray or scipy sparse matrix): preprocessed feature values.

        Return:
            numpy.ndarray: feature values without preprocessing.

        """
        nr_original_features = len(self.get_original_feature_names())
        undo_prep_test_x = np.zeros((preprocessed_x.shape[0], nr_original_features), dtype='O')
        any_cat = np.vectorize(lambda x: self.is_categorical(x))

        for _, transformer, feature_names in reversed(self.ct_preprocessor.transformers_):
            original_feature_ids, preprocessed_feature_ids = self._get_feature_ids_related_to_transformer(feature_names)
            transformer_output = preprocessed_x[:, preprocessed_feature_ids]
            if issparse(transformer_output) and not np.any(any_cat(original_feature_ids)):
                transformer_output = transformer_output.todense()

            # TODO: could be simplified as sklearn.Pipeline implements inverse_transform
            for step in generate_preprocessing_steps(transformer, invert_order=True):
                transformer_input = PipelinePreprocessor._inverse_single_step(step, transformer_output, feature_names)
                transformer_output = transformer_input
            undo_prep_test_x[:, original_feature_ids] = transformer_input

        return undo_prep_test_x

    def is_categorical(self, index=None, name=None):
        """Check whether an unprocessed feature at a given index or with a given name is categorical.

        Args:
            index (int): feature index.
            name (str): feature name.

        Return:
            bool: True if the input feature is categorical, else False. If both index and name are provided, the index
                is retained.
        """
        if index is not None:
            name = self.original_feature_names[index]
        if name is not None:
            return name in self.categorical_features
        else:
            raise ValueError("Either the input index or its name should be specified.")

    def inverse_transform_feature_id(self, index):
        """Undo preprocessing of feature name.

        Transform the preprocessed feature name at given index back into the original unprocessed feature index.

        Args:
            index (int): feature index.

        Return:
            int : index of the unprocessed feature corresponding to the input preprocessed feature index.
        """
        return self.preprocessed2original[index]

    def get_top_ranked_feature_ids(self, feature_importances, max_nr_features):
        ranked_transformed_feature_ids = np.argsort(- feature_importances)
        if max_nr_features <= 0:
            max_nr_features += len(self.get_original_feature_names())

        ranked_feature_ids, seen = [], set()
        for idx in ranked_transformed_feature_ids:
            inverse_transformed_feature_id = self.inverse_transform_feature_id(idx)
            if inverse_transformed_feature_id not in seen:
                seen.add(inverse_transformed_feature_id)
                ranked_feature_ids.append(inverse_transformed_feature_id)
                if max_nr_features == len(ranked_feature_ids):
                    return ranked_feature_ids
        return ranked_feature_ids # should never be reached, but just in case

    def inverse_thresholds(self, tree):
        used_feature_mask = tree.feature >= 0
        feats_idx = tree.feature[used_feature_mask]
        thresholds = tree.threshold.astype('O')
        thresh = thresholds[used_feature_mask]
        n_cols = len(self.get_preprocessed_feature_names())

        dummy_x, indices= [], []
        for f, t in zip(feats_idx, thresh):
            row = [0]*n_cols
            row[f] = t
            dummy_x.append(row)
            indices.append(self.inverse_transform_feature_id(f))

        undo_dummy_x = self.inverse_transform(np.array(dummy_x))
        descaled_thresh = [undo_dummy_x[i, j] for i, j in enumerate(indices)]
        thresholds[used_feature_mask] = descaled_thresh
        return thresholds


class DummyPipelinePreprocessor(FeatureNameTransformer):

    def __init__(self, model_performance_predictor_features):
        super(DummyPipelinePreprocessor, self).__init__(
            original_features=model_performance_predictor_features,
            preprocessed_features=model_performance_predictor_features)

    def transform(self, x):
        """
        Args:
            x (array-like or dataframe of shape (number of samples, number of features)): input feature values.
        Returns:
            ndarray
        """
        if isinstance(x, pd.DataFrame):
            return x.values
        if isinstance(x, np.ndarray) or issparse(x):
            return x
        raise TypeError('x should be either a pandas dataframe, a numpy ndarray or a scipy sparse matrix')

    def is_categorical(self, index=None, name=None):
        return False

    def inverse_transform_feature_id(self, index):
        return index

    def inverse_transform(self, x):
        return x

    def get_top_ranked_feature_ids(self, feature_importances, max_nr_features):
        if max_nr_features == 0:
            return np.argsort(- feature_importances)
        return np.argsort(- feature_importances)[:max_nr_features]

    def inverse_thresholds(self, tree):
        return tree.threshold.astype('O')
