from unittest import TestCase
from unittest.mock import patch, Mock
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from .. import generate_preprocessing_steps


class TestFeatureTransformer(TestCase):
    def test_generate_preprocessing_steps_pipeline(self):
        ohe = Mock(spec=OneHotEncoder)
        pipe = Mock(spec=Pipeline, steps=[
            ("d", "drop"),
            ("p", "passthrough"),
            ("o", ohe)
        ])
        generator = generate_preprocessing_steps(pipe)
        self.assertEqual(next(generator), "passthrough")
        self.assertEqual(next(generator), ohe)
        with self.assertRaises(StopIteration):
            next(generator)

    def test_generate_preprocessing_steps_pipeline_inverted(self):
        ohe = Mock(spec=OneHotEncoder)
        pipe = Mock(spec=Pipeline, steps=[
            ("d", "drop"),
            ("p", "passthrough"),
            ("o", ohe)
        ])
        generator = generate_preprocessing_steps(pipe, True)
        self.assertEqual(next(generator), ohe)
        self.assertEqual(next(generator), "passthrough")
        with self.assertRaises(StopIteration):
            next(generator)

    def test_generate_preprocessing_steps_only_drop(self):
        generator = generate_preprocessing_steps("drop")
        with self.assertRaises(StopIteration):
            next(generator)

    def test_generate_preprocessing_steps_only_supported(self):
        ohe = Mock(spec=OneHotEncoder)
        generator = generate_preprocessing_steps(ohe)
        self.assertEqual(next(generator), ohe)
        with self.assertRaises(StopIteration):
            next(generator)

    def test_generate_preprocessing_steps_only_passthrough(self):
        generator = generate_preprocessing_steps("passthrough")
        self.assertEqual(next(generator), "passthrough")
        with self.assertRaises(StopIteration):
            next(generator)

    def test_generate_preprocessing_steps_only_unsupported(self):
        msg = 'Mealy package does not support {}. '.format(Mock) + \
            'It might be because it changes output dimension without ' +\
            'providing a get_feature_names function to keep track of the ' + \
            'generated features, or that it does not provide an inverse_tranform method.'
        g = generate_preprocessing_steps(Mock())
        with self.assertRaisesRegex(TypeError, msg):
            next(g)
