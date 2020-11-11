# -*- coding: utf-8 -*-
from sklearn.pipeline import Pipeline
import numpy as np
from mea.error_analysis_utils import  ErrorAnalyzerConstants
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Error Analysis | %(levelname)s - %(message)s')


class FeatureNameTransformer(object):

    def __init__(self, ct_preprocessor, original_features=None):

        self.original_feature_names = None
        self.preprocessed_feature_names = None
        self.categorical_features = []
        self.original2preprocessed = dict()
        self.preprocessed2original = dict()
        self.preprocessed_feature_names = list()
        self.len_preproc = 0

        original_features_from_ct, self.categorical_features = self._get_feature_list_from_column_transformer(
            ct_preprocessor)
        if original_features is None:
            self.original_feature_names = original_features_from_ct
        else:
            self._compare_original_feature_list(original_features, original_features_from_ct)
            self.original_feature_names = original_features

        self.create_feature_mapping(ct_preprocessor)

    def create_feature_mapping(self, ct_preprocessor):
        """
        Update the dicts of input <-> output feature mapping: self.original2preprocessed and self.preprocessed2original

        Args:
            ct_preprocessor: a ColumnTransformer object

        """
        for i, (tr_name, tr, tr_feature_names) in enumerate(ct_preprocessor.transformers_):
            orig_feats_ids = np.where(np.in1d(self.original_feature_names, tr_feature_names))[0]
            if isinstance(tr, Pipeline):
                number_of_steps_that_change_dimension = 0
                # we take by default the first step in the pipeline
                single_tr = tr.steps[0][1]
                # Now we check if there is a step than changes the output dimension
                for step in tr.steps:
                    # step is a tuple (step_name, step_function)
                    if isinstance(step[1], ErrorAnalyzerConstants.STEPS_THAT_CHANGE_OUTPUT_DIMENSION_WITH_OUTPUT_FEATURE_NAMES):
                        single_tr = step[1]
                        number_of_steps_that_change_dimension += 1
                if number_of_steps_that_change_dimension > 1:
                    raise ValueError('Each pipeline can only have one step that changes feature dimension, '
                                     'here we got {}'.format(number_of_steps_that_change_dimension))
                if isinstance(single_tr, ErrorAnalyzerConstants.STEPS_THAT_DOES_NOT_CHANGE_OUTPUT_DIMENSION):
                    self.update_feature_mapping_dict_using_input_names(tr_feature_names, orig_feats_ids)
                elif isinstance(single_tr, ErrorAnalyzerConstants.STEPS_THAT_CHANGE_OUTPUT_DIMENSION_WITH_OUTPUT_FEATURE_NAMES):
                    self.update_feature_mapping_dict_using_output_names(single_tr, tr_feature_names, orig_feats_ids)
                else:
                    raise ValueError('The package does not support {}, probably because it changes output dimension '
                                     'but does not provide get_feature_names function to keep track of new features '
                                     'generated.'.format(single_tr))

            elif isinstance(tr, ErrorAnalyzerConstants.STEPS_THAT_DOES_NOT_CHANGE_OUTPUT_DIMENSION):
                self.update_feature_mapping_dict_using_input_names(tr_feature_names, orig_feats_ids)
            elif isinstance(tr, ErrorAnalyzerConstants.STEPS_THAT_CHANGE_OUTPUT_DIMENSION_WITH_OUTPUT_FEATURE_NAMES):
                self.update_feature_mapping_dict_using_output_names(tr, tr_feature_names, orig_feats_ids)
            elif tr_name == 'remainder' and tr == 'drop':
                # skip the default drop step of ColumnTransformer
                continue
            else:
                raise ValueError('The package does not support {}, probably because it changes output dimension but '
                                 'does not provide get_feature_names function to keep track of new '
                                 'features generated.'.format(tr))

    def update_feature_mapping_dict_using_input_names(self, tr_feature_names, orig_feats_ids):
        self.preprocessed_feature_names.extend(tr_feature_names)
        self.original2preprocessed.update({in_i: self.len_preproc + i for i, in_i in enumerate(orig_feats_ids)})
        self.preprocessed2original.update({self.len_preproc + i: in_i for i, in_i in enumerate(orig_feats_ids)})
        self.len_preproc += len(tr_feature_names)

    def update_feature_mapping_dict_using_output_names(self, single_tr, tr_feature_names, orig_feats_ids):
        """
        For now, this functions only applies for OnehotEncoder

        Args:
            single_tr:
            tr_feature_names:
            orig_feats_ids:
        """
        out_feature_names = list(single_tr.get_feature_names(input_features=tr_feature_names))
        self.preprocessed_feature_names.extend(out_feature_names)
        for orig_id, orig_name in zip(orig_feats_ids, tr_feature_names):
            #TODO This approach only works for OnehotEncoder
            part_out_feature_names = [i for i, name in enumerate(out_feature_names) if orig_name + '_' in name]

            self.original2preprocessed.update({orig_id: [self.len_preproc + i for i in range(len(part_out_feature_names))]})
            self.preprocessed2original.update({self.len_preproc + i: orig_id for i in range(len(part_out_feature_names))})
            self.len_preproc += len(part_out_feature_names)

    def _compare_original_feature_list(self, feature_list_from_user, feature_list_from_ct):
        if len(feature_list_from_user) != len(feature_list_from_ct):
            raise ValueError('The list of input original feature does not correspond to the list of features handled '
                             'by the ct_preprocessor.')

    def _get_feature_list_from_column_transformer(self, ct_preprocessor):
        all_feature = []
        categorical_features = []
        for i, (tr_name, tr, tr_feature_names) in enumerate(ct_preprocessor.transformers_):
            if tr_name == 'remainder' and tr == 'drop':
                continue
            else:
                all_feature.extend(tr_feature_names)

            # check for categorical features
            if isinstance(tr, Pipeline):
                for step in tr.steps:
                    if isinstance(step[1], ErrorAnalyzerConstants.VALID_CATEGORICAL_STEPS):
                        categorical_features.extend(tr_feature_names)
                        break
            elif isinstance(tr, ErrorAnalyzerConstants.VALID_CATEGORICAL_STEPS):
                categorical_features.extend(tr_feature_names)
            else:
                continue

        return all_feature, categorical_features

    def transform(self, index=None, name=None):
        if index is not None:
            return self.original2preprocessed[index]
        elif name is not None:
            index = self.original_feature_names.index(name)
            new_index = self.original2preprocessed[index]
            return [self.preprocessed_feature_names[idx] for idx in new_index]
        else:
            raise ValueError("One of the input index or name should be specified.")

    def inverse_transform(self, index=None, name=None):
        if index is not None:
            return self.preprocessed2original[index]
        elif name is not None:
            index = self.preprocessed_feature_names.index(name)
            new_index = self.preprocessed2original[index]
            return self.original_feature_names[new_index]
        else:
            raise ValueError("One of the input index or name should be specified.")

    def is_categorical(self, index=None, name=None):
        if index is not None:
            name = self.original_feature_names[index]
            return name in self.categorical_features
        elif name is not None:
            return name in self.categorical_features
        else:
            raise ValueError("One of the input index or name should be specified.")


class PipelinePreprocessor(object):

    def __init__(self, ct_preprocessor, original_features=None):

        self.fn_transformer = FeatureNameTransformer(ct_preprocessor, original_features)
        self.ct_preprocessor = ct_preprocessor

    def transform(self, x):
        return self.ct_preprocessor.transform(x)

    def inverse_transform(self, preprocessed_x):

        def _inverse_single_step(single_step, step_output):
            inverse_transform_function_available = getattr(single_step, "inverse_transform", None)
            if inverse_transform_function_available:
                step_input = single_step.inverse_transform(step_output)
                logger.info("Reversing 'step' {} on {}".format(tr_name, ' '.join([f for f in tr_feature_names])))
            elif isinstance(single_step, ErrorAnalyzerConstants.STEPS_THAT_CAN_BE_INVERSED_WITH_IDENTICAL_FUNCTION):
                logger.info("Apply identity transformation.")
                step_input = step_output
            else:
                raise ValueError('The package does not support {} because it does not provide inverse_transform function.'.format(single_step))
            return step_input

        original_features = self.fn_transformer.original_feature_names
        undo_prep_test_x = np.zeros((preprocessed_x.shape[0], len(original_features)), dtype='O')

        for (tr_name, tr, tr_feature_names) in self.ct_preprocessor.transformers_:
            orig_feats_ids = np.where(np.in1d(original_features, tr_feature_names))[0]
            prep_feats_ids = []
            for i in orig_feats_ids:
                out_ids = self.fn_transformer.transform(i)
                if isinstance(out_ids, int):
                    prep_feats_ids.append(out_ids)
                else:  # list of ids
                    prep_feats_ids.extend(out_ids)

            step_out_x = preprocessed_x[:, prep_feats_ids]
            step_in_x = None
            if isinstance(tr, Pipeline):
                for step_name, step in reversed(tr.steps):
                    step_in_x = _inverse_single_step(step, step_out_x)
                    step_out_x = step_in_x
                undo_prep_test_x[:, orig_feats_ids] = step_in_x
            elif tr_name == 'remainder' and tr == 'drop':
                continue
            else:
                step_in_x = _inverse_single_step(tr, step_out_x)
                undo_prep_test_x[:, orig_feats_ids] = step_in_x

        return undo_prep_test_x
