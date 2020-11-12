from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, PowerTransformer, QuantileTransformer, \
    MaxAbsScaler, Binarizer, Normalizer, MinMaxScaler, RobustScaler


class ErrorAnalyzerConstants(object):

    WRONG_PREDICTION = "Wrong prediction"
    CORRECT_PREDICTION = "Correct prediction"
    PARAMETERS_GRID = {'max_depth': [5, 10, 20, None], 'min_samples_leaf': [10, 20]}
    TEST_SIZE = 0.2

    MIN_NUM_ROWS = 100 #500  # heuristic choice
    MAX_NUM_ROW = 100000  # heuristic choice

    MPP_ACCURACY_TOLERANCE = 0.1
    CRITERION = 'entropy'
    NUMBER_EPSILON_VALUES = 50

    ERROR_TREE_COLORS = {CORRECT_PREDICTION: '#538BC8', WRONG_PREDICTION: '#EC6547'}

    TOP_K_FEATURES = 3

    MPP_ACCURACY = 'mpp_accuracy_score'
    MPP_FIDELITY = 'mpp_fidelity_score'
    MPP_BALANCED_ACCURACY = 'mpp_balanced_accuracy_score'
    PRIMARY_MODEL_TRUE_ACCURACY = 'primary_model_true_accuracy'
    PRIMARY_MODEL_PREDICTED_ACCURACY = 'primary_model_predicted_accuracy'
    CONFIDENCE_DECISION = 'confidence_decision'

    NUMBER_PURITY_LEVELS = 10

    # use tuple because isinstance() takes only tuple as input type
    VALID_CATEGORICAL_STEPS = (OneHotEncoder, OrdinalEncoder)
    STEPS_THAT_DOES_NOT_CHANGE_OUTPUT_DIMENSION = (StandardScaler, PowerTransformer, QuantileTransformer, MaxAbsScaler,
                                                   Binarizer, Normalizer, MinMaxScaler, RobustScaler, SimpleImputer,
                                                   KNNImputer, OrdinalEncoder)
    STEPS_THAT_CHANGE_OUTPUT_DIMENSION_WITH_OUTPUT_FEATURE_NAMES = (OneHotEncoder,)
    # for imputers we don't need inverse function
    STEPS_THAT_CAN_BE_INVERSED_WITH_IDENTICAL_FUNCTION = (SimpleImputer, KNNImputer)