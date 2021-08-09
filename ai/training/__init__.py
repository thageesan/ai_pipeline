from shared.tools.os import getenv
from shared.tools.utils import pd

from numpy import array
from params import XGBTrainConfig
from xgboost import XGBClassifier
from joblib import dump


def app():
    """
    Fits an xgboost model using training data.
    Returns:

    """
    xgb_params = {
        "subsample": XGBTrainConfig.subsample,
        "n_estimators": XGBTrainConfig.n_estimators,
        "min_child_weight": XGBTrainConfig.min_child_weight,
        "max_depth": XGBTrainConfig.max_depth,
        "learning_rate": XGBTrainConfig.learning_rate,
        "gamma": XGBTrainConfig.gamma,
        "colsample_bytree": XGBTrainConfig.colsample_bytree,
        "alpha": XGBTrainConfig.alpha,
        "eval_metric": XGBTrainConfig.eval_metric,
        "objective": XGBTrainConfig.objective,
        "use_label_encoder": XGBTrainConfig.use_label_encoder,
    }

    # load training set
    model_path = getenv('DATA_FOLDER')
    model_file_name = getenv('MODEL_FILE_NAME')
    training_set_file_name = getenv('TRAINING_SET')
    data_frame = pd.read_parquet(f'{model_path}/{training_set_file_name}')

    x = array(data_frame['x'].tolist())
    y = data_frame['y'].to_numpy()

    xgb = XGBClassifier(**xgb_params)
    model = xgb.fit(x, y)

    dump(model, f'{model_path}/{model_file_name}')
