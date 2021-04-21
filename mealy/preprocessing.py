# -*- coding: utf-8 -*-
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from scipy.sparse import issparse
import logging
from mealy.error_analysis_utils import check_lists_having_same_elements, get_feature_list_from_column_transformer
from mealy.constants import ErrorAnalyzerConstants

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='mealy | %(levelname)s - %(message)s')


class FeatureNameTransformer(object):
    """ Transformer of feature names and indices.

        A FeatureNameTransformer parses an input Pipeline preprocessor and generate
        a mapping between the input unprocessed feature names/indices and the output
        preprocessed feature names/indices.

        Args:
            ct_preprocessor (sklearn.compose.ColumnTransformer): preprocessor
            orig_feats (list): list of original unpreprocessed feature names, default=None.

        Attributes:
            original_feature_names (list): list of original unpreprocessed feature names.
            preprocessed_feature_names (list): list of preprocessed feature names

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

    def inverse_thresholds(self, tree, n_cols):
        raise NotImplementedError


class PipelinePreprocessor(FeatureNameTransformer):
    """Transformer of feature values from the original values to preprocessed ones.

        A PipelinePreprocessor parses an input Pipeline preprocessor and generate
        a mapping between the input unprocessed feature values and the output
        preprocessed feature values.

        Args:
            ct_preprocessor (sklearn.compose.ColumnTransformer): preprocessing steps
            original_features (list): list of original unpreprocessed feature names, default=None.

    """

    def __init__(self, ct_preprocessor, original_features=None):
        self.ct_preprocessor = ct_preprocessor
        self.original2preprocessed = {}
        self.preprocessed2original = {}
        self.len_preproc = 0

        logger.info('Retrieving the list of features used in the pipeline')
        original_features_from_ct, self.categorical_features = get_feature_list_from_column_transformer(self.ct_preprocessor)
        if original_features is None:
            original_features = original_features_from_ct
        elif not check_lists_having_same_elements(original_features, original_features_from_ct):
            # If user explicitly gives a list of input features, we compare it with the list derived from the ColumnTransformer
            raise ValueError('The list of features given by user does not correspond to the list of features handled by the Pipeline.')

        super(PipelinePreprocessor, self).__init__(original_features=original_features, preprocessed_features=[])

        logger.info('Generating the feature id mapping dict')
        self._create_feature_mapping(ct_preprocessor)

    def _create_feature_mapping(self, ct_preprocessor):
        """
        Update the dicts of input <-> output feature id mapping: self.original2preprocessed and self.preprocessed2original

        Args:
            ct_preprocessor: a ColumnTransformer object
        """
        for i, (transformer_name, transformer, transformer_feature_names) in enumerate(ct_preprocessor.transformers_):
            orig_feats_ids = np.where(np.in1d(self.original_feature_names, transformer_feature_names))[0]
            if isinstance(transformer, Pipeline):
                # The assumption here is that for each pipeline there is at most one step that change feature dimension
                # For now, the only possible function is OneHotEncoder

                # We take by default the first step in the pipeline
                single_tr = transformer.steps[0][1]
                # Check if there is a step than changes the output dimension, if that's the case single_tr will be it
                for (step_name, step) in transformer.steps:
                    if isinstance(step, ErrorAnalyzerConstants.STEPS_THAT_CHANGE_OUTPUT_DIMENSION_WITH_OUTPUT_FEATURE_NAMES):
                        single_tr = step
                        break
                if isinstance(single_tr, ErrorAnalyzerConstants.STEPS_THAT_DOES_NOT_CHANGE_OUTPUT_DIMENSION):
                    self._update_feature_mapping_dict_using_input_names(transformer_feature_names, orig_feats_ids)
                elif isinstance(single_tr, ErrorAnalyzerConstants.STEPS_THAT_CHANGE_OUTPUT_DIMENSION_WITH_OUTPUT_FEATURE_NAMES):
                    self._update_feature_mapping_dict_using_output_names(single_tr, transformer_feature_names, orig_feats_ids)
                else:
                    raise TypeError('The package does not support {}, probably because it changes output dimension '
                                     'but does not provide get_feature_names function to keep track of new features '
                                     'generated.'.format(single_tr))

            elif isinstance(transformer, ErrorAnalyzerConstants.STEPS_THAT_DOES_NOT_CHANGE_OUTPUT_DIMENSION):
                self._update_feature_mapping_dict_using_input_names(transformer_feature_names, orig_feats_ids)
            elif isinstance(transformer, ErrorAnalyzerConstants.STEPS_THAT_CHANGE_OUTPUT_DIMENSION_WITH_OUTPUT_FEATURE_NAMES):
                self._update_feature_mapping_dict_using_output_names(transformer, transformer_feature_names, orig_feats_ids)
            elif transformer_name == 'remainder' and transformer == 'drop':
                # skip the default drop step of ColumnTransformer
                continue
            else:
                raise TypeError('The package does not support {}, probably because it changes output dimension but '
                                 'does not provide get_feature_names function to keep track of new '
                                 'features generated.'.format(transformer))

    def _update_feature_mapping_dict_using_input_names(self, transformer_feature_names, orig_feats_ids):
        self.preprocessed_feature_names.extend(transformer_feature_names)
        self.original2preprocessed.update({in_i: self.len_preproc + i for i, in_i in enumerate(orig_feats_ids)})
        self.preprocessed2original.update({self.len_preproc + i: in_i for i, in_i in enumerate(orig_feats_ids)})
        self.len_preproc += len(transformer_feature_names)

    def _update_feature_mapping_dict_using_output_names(self, single_transformer, transformer_feature_names, original_feature_ids):
        """
        For now, this functions only applies for OnehotEncoder
        """
        out_feature_names = list(single_transformer.get_feature_names(input_features=transformer_feature_names))
        self.preprocessed_feature_names.extend(out_feature_names)
        for orig_id, orig_name in zip(original_feature_ids, transformer_feature_names):
            part_out_feature_names = [i for i, name in enumerate(out_feature_names) if orig_name + '_' in name]
            self.original2preprocessed.update({orig_id: [self.len_preproc + i for i in range(len(part_out_feature_names))]})
            self.preprocessed2original.update({self.len_preproc + i: orig_id for i in range(len(part_out_feature_names))})
            self.len_preproc += len(part_out_feature_names)

    def _transform_feature_id(self, index):
        """
        Args:
            index: int

        Returns: index of output feature(s) generated by the requested feature
        """
        return self.original2preprocessed[index]

    def transform(self, x):
        """Transform the input feature values according to the preprocessing pipeline.

        Args:
            x (numpy.ndarray or pandas.DataFrame): input feature values.

        Return:
            numpy.ndarray: transformed feature values
        """
        return self.ct_preprocessor.transform(x)

    def _get_feature_ids_related_to_transformer(self, transformer_feature_names):
        original_features = self.get_original_feature_names()
        original_feature_ids = np.where(np.in1d(original_features, transformer_feature_names))[0]
        preprocessed_feature_ids = []
        for i in original_feature_ids:
            out_ids = self._transform_feature_id(i)
            if isinstance(out_ids, int):
                preprocessed_feature_ids.append(out_ids)
            else:  # list of ids
                preprocessed_feature_ids.extend(out_ids)
        return original_feature_ids, preprocessed_feature_ids

    @staticmethod
    def _inverse_single_step(single_step, step_output, transformer_feature_names):
        inverse_transform_function_available = getattr(single_step, "inverse_transform", None)
        if inverse_transform_function_available:
            logger.info("Reversing step {} using inverse_transform() function on column(s): {}".format(single_step, ', '.join([f for f in transformer_feature_names])))
            return single_step.inverse_transform(step_output)
        if isinstance(single_step, ErrorAnalyzerConstants.STEPS_THAT_CAN_BE_INVERSED_WITH_IDENTICAL_FUNCTION):
            logger.info("Reversing step {} using identity transformation on column(s): {}".format(single_step, ', '.join([f for f in transformer_feature_names])))
            return step_output
        raise TypeError('The package does not support {} because it does not provide inverse_transform function.'.format(single_step))

    def inverse_transform(self, preprocessed_x):
        """Invert the preprocessing pipeline and inverse transform feature values.

        Args:
            preprocessed_x (numpy.ndarray): preprocessed feature values.

        Return:
            numpy.ndarray: feature values without preprocessing

        """
        original_features = self.get_original_feature_names()
        undo_prep_test_x = np.zeros((preprocessed_x.shape[0], len(original_features)), dtype='O')

        for (transformer_name, transformer, transformer_feature_names) in self.ct_preprocessor.transformers_:
            if transformer_name == 'remainder' and transformer == 'drop':
                continue
            original_feature_ids, preprocessed_feature_ids = self._get_feature_ids_related_to_transformer(transformer_feature_names)
            output_of_transformer = preprocessed_x[:, preprocessed_feature_ids]

            any_numeric = np.any(np.vectorize(lambda x: not self.is_categorical(x)))
            if issparse(output_of_transformer) and any_numeric:
                output_of_transformer = output_of_transformer.todense()

            input_of_transformer = None
            if isinstance(transformer, Pipeline):
                for _, step in reversed(transformer.steps):
                    input_of_transformer = PipelinePreprocessor._inverse_single_step(step, output_of_transformer, transformer_feature_names)
                    output_of_transformer = input_of_transformer
                undo_prep_test_x[:, original_feature_ids] = input_of_transformer
            else:
                input_of_transformer = PipelinePreprocessor._inverse_single_step(transformer, output_of_transformer, transformer_feature_names)
                undo_prep_test_x[:, original_feature_ids] = input_of_transformer

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

    def inverse_thresholds(self, tree, n_cols):
        used_feature_mask = tree.feature > 0
        feats_idx = tree.feature[used_feature_mask]
        thresholds = tree.threshold.astype('O')
        thresh = thresholds[used_feature_mask]

        dummy_x = np.zeros((len(feats_idx), n_cols))
        indices, i = [], 0
        for f, t in zip(feats_idx, thresh):
            dummy_x[i, f] = t
            indices.append((i, self.inverse_transform_feature_id(f)))
            i += 1

        undo_dummy_x = self.inverse_transform(dummy_x)
        descaled_thresh = [undo_dummy_x[i, j] for i, j in indices]
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
            x: dataframe or ndarray
        Returns:
            ndarray
        """
        if isinstance(x, pd.DataFrame):
            return x.values
        if isinstance(x, np.ndarray):
            return x
        raise TypeError('x should be either a pandas dataframe or a numpy ndarray')

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

    def inverse_thresholds(self, tree, n_cols):
        return tree.threshold.astype('O')
