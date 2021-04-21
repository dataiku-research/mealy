from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, PowerTransformer, QuantileTransformer, \
    MaxAbsScaler, Binarizer, Normalizer, MinMaxScaler, RobustScaler


class ErrorAnalyzerConstants(object):

    WRONG_PREDICTION = "Wrong prediction"
    CORRECT_PREDICTION = "Correct prediction"
    MAX_DEPTH = [5, 10]
    MIN_SAMPLES_LEAF_LOWEST_UPPER_BOUND = .01 # for min_samples_leaf, the min upper bound value should be 0.01
    TEST_SIZE = 0.2

    MIN_NUM_ROWS = 100  # heuristic choice

    TREE_ACCURACY_TOLERANCE = 0.1
    CRITERION = 'entropy'
    NUMBER_EPSILON_VALUES = 50

    ERROR_TREE_COLORS = {CORRECT_PREDICTION: '#CCCCCC', WRONG_PREDICTION: '#CE1228'}

    TOP_K_FEATURES = 3

    TREE_ACCURACY = 'error_tree_accuracy_score'
    TREE_FIDELITY = 'error_tree_fidelity_score'
    TREE_BALANCED_ACCURACY = 'error_tree_balanced_accuracy_score'
    PRIMARY_MODEL_TRUE_ACCURACY = 'primary_model_true_accuracy'
    PRIMARY_MODEL_PREDICTED_ACCURACY = 'primary_model_predicted_accuracy'
    CONFIDENCE_DECISION = 'confidence_decision'

    NUMBER_PURITY_LEVELS = 10

    # use tuple because isinstance() takes only tuple as input type
    VALID_CATEGORICAL_STEPS = (OneHotEncoder, OrdinalEncoder)
    STEPS_THAT_DOES_NOT_CHANGE_OUTPUT_DIMENSION = (StandardScaler, PowerTransformer, QuantileTransformer, MaxAbsScaler,
                                                   Binarizer, Normalizer, MinMaxScaler, RobustScaler, SimpleImputer,
                                                   OrdinalEncoder)
    STEPS_THAT_CHANGE_OUTPUT_DIMENSION_WITH_OUTPUT_FEATURE_NAMES = (OneHotEncoder,)
    GRAPH_MAX_EDGE_WIDTH = 10
    GRAPH_MIN_LOCAL_ERROR_OPAQUE = 0.5
    
    # for imputers we don't need inverse function
    STEPS_THAT_CAN_BE_INVERSED_WITH_IDENTICAL_FUNCTION = (SimpleImputer,)
