BIOSENT_FILE_NAME = 'bioSent2Vec.bin'

UMLS_MODEL_NAME = 'GanjinZero/UMLSBert_ENG'


class TestTrainSetConfig:
    TEST_SIZE = 0.2


class XGBTrainConfig:
    subsample = 0.8
    n_estimators = 400
    min_child_weight = 5
    max_depth = 6
    learning_rate = 0.1
    gamma = 2
    colsample_bytree = 0.8
    alpha = 5
    eval_metric = 'auc'
    objective = 'binary:logistic'
    use_label_encoder = False
