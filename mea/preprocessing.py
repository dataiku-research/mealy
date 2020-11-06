from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
import numpy as np

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Error Analysis Plugin | %(levelname)s - %(message)s')


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
    def __init__(self, ct_preprocessor, orig_feats=None):

        self.original_feature_names = None
        self.preprocessed_feature_names = None
        self.original2preprocessed = None
        self.preprocessed2original = None

        self.categorical_features = []

        feature_names = []
        for i, (tr_type, tr, tr_feature_names) in enumerate(ct_preprocessor.transformers_):
            feature_names.extend(tr_feature_names)
            if isinstance(tr, Pipeline):
                single_tr = tr.steps[-1][1]
            else:
                single_tr = tr

            if isinstance(single_tr, OneHotEncoder) or isinstance(tr, OrdinalEncoder):
                self.categorical_features.extend(tr_feature_names)

        if orig_feats is None:
            self.original_feature_names = feature_names
        else:
            self.original_feature_names = orig_feats

        assert len(feature_names) == len(self.original_feature_names)

        self.original2preprocessed = dict()
        self.preprocessed2original = dict()
        self.preprocessed_feature_names = list()

        len_preproc = 0
        for i, (tr_type, tr, tr_feature_names) in enumerate(ct_preprocessor.transformers_):

            orig_feats_ids = np.where(np.in1d(orig_feats, tr_feature_names))[0]
            if isinstance(tr, Pipeline):
                single_tr = tr.steps[-1][1]
            else:
                single_tr = tr

            try:
                out_feature_names = list(single_tr.get_feature_names(input_features=tr_feature_names))
                self.preprocessed_feature_names.extend(out_feature_names)

                for orig_id, orig_name in zip(orig_feats_ids, tr_feature_names):
                    part_out_feature_names = [i for i, name in enumerate(out_feature_names) if orig_name + '_' in name]

                    self.original2preprocessed.update(
                        {orig_id: [len_preproc + i for i in range(len(part_out_feature_names))]})
                    self.preprocessed2original.update(
                        {len_preproc + i: orig_id for i in range(len(part_out_feature_names))})
                    len_preproc += len(part_out_feature_names)
            except Exception as e:
                logger.info(e)
                logger.info('Get feature names not supported. Using input feature names.')
                self.preprocessed_feature_names.extend(tr_feature_names)

                self.original2preprocessed.update({in_i: len_preproc + i for i, in_i in enumerate(orig_feats_ids)})
                self.preprocessed2original.update({len_preproc + i: in_i for i, in_i in enumerate(orig_feats_ids)})
                len_preproc += len(tr_feature_names)

    def transform(self, index=None, name=None):
        """ Apply preprocessing to feature name.

        Transform the unprocessed feature name at given index or with given name
        into the output preprocessed feature names or indices.

        Args:
            index (int): feature index.
            name (str): feature name.

        Return:
            list: list of indices (resp. names) of the preprocessed features corresponding to the input feature index
                (resp.name). If both index and name are provided, the index is retained and output indices are returned.
        """
        if index is not None:
            return self.original2preprocessed[index]
        elif name is not None:
            index = self.original_feature_names.index(name)
            new_index = self.original2preprocessed[index]
            return [self.preprocessed_feature_names[idx] for idx in new_index]
        else:
            raise ValueError("One of the input index or name should be specified.")

    def inverse_transform(self, index=None, name=None):
        """Undo preprocessing of feature name.

        Transform the preprocessed feature name at given index or with given name
        back into the original unprocessed feature name or index.

        Args:
            index (int): feature index.
            name (str): feature name.

        Return:
            int or str: index (resp. name) of the unprocessed feature corresponding to the input preprocessed feature
                index (resp.name). If both index and name are provided, the index is retained and an output index is
                returned.
        """
        if index is not None:
            return self.preprocessed2original[index]
        elif name is not None:
            index = self.preprocessed_feature_names.index(name)
            new_index = self.preprocessed2original[index]
            return self.original_feature_names[new_index]
        else:
            raise ValueError("One of the input index or name should be specified.")

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
            return name in self.categorical_features
        elif name is not None:
            return name in self.categorical_features
        else:
            raise ValueError("One of the input index or name should be specified.")


class PipelinePreprocessor(object):
    """ Transformer of feature values from the original values to preprocessed ones.

        A PipelinePreprocessor parses an input Pipeline preprocessor and generate
        a mapping between the input unprocessed feature values and the output
        preprocessed feature values.

        Args:
            ct_preprocessor (sklearn.compose.ColumnTransformer): preprocessing steps
            orig_feats (list): list of original unpreprocessed feature names, default=None.

        Attributes:
            fn_transformer (FeatureNameTransformer): transformer managing the mapping between original and
                preprocessed feature names.

    """
    def __init__(self, ct_preprocessor, orig_feats=None):

        self.fn_transformer = FeatureNameTransformer(ct_preprocessor, orig_feats)
        self.ct_preprocessor = ct_preprocessor

    def transform(self, x):
        """Transform the input feature values according to the preprocessing pipeline.

        Args:
            x (numpy.ndarray or pandas.DataFrame): input feature values.

        Return:
            numpy.ndarray: transformed feature values
        """
        return self.ct_preprocessor.transform(x).toarray()

    def inverse_transform(self, preprocessed_x):
        """Invert the preprocessing pipeline and inverse transform feature values.

        Args:
            preprocessed_x (numpy.ndarray): preprocessed feature values.

        Return:
            numpy.ndarray: feature values without preprocessing

        """

        orig_feats = self.fn_transformer.original_feature_names

        undo_prep_test_x = np.zeros((preprocessed_x.shape[0], len(orig_feats)), dtype='O')

        for (tr_name, tr, tr_feats) in self.ct_preprocessor.transformers_:

            try:
                orig_feats_ids = np.where(np.in1d(orig_feats, tr_feats))[0]
                prep_feats_ids = []
                for i in orig_feats_ids:
                    out_ids = self.fn_transformer.transform(i)
                    if isinstance(out_ids, int):
                        prep_feats_ids.append(out_ids)
                    else:
                        prep_feats_ids.extend(out_ids)

                if isinstance(tr, Pipeline):
                    for step_name, step in tr.steps:
                        try:
                            undo_prep_test_x[:, orig_feats_ids] = step.inverse_transform(
                                preprocessed_x[:, prep_feats_ids])
                            logger.info("Reversing %s on %s" % (step_name, ' '.join([f for f in tr_feats])))
                        except Exception as e:
                            logger.info(e)
                            logger.info("Step does not support inverse_transform. Skipping.")
                            assert (len(prep_feats_ids) == len(orig_feats_ids))
                            undo_prep_test_x[:, orig_feats_ids] = preprocessed_x[:, prep_feats_ids]
                            continue
                else:
                    undo_prep_test_x[:, orig_feats_ids] = tr.inverse_transform(preprocessed_x[:, prep_feats_ids])
                    logger.info("Reversing %s on %s" % (tr_name, ' '.join([f for f in tr_feats])))
            except Exception as e:
                logger.info(e)
                logger.info("Step does not support inverse_transform. Skipping.")
                continue
        return undo_prep_test_x
