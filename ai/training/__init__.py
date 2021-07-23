from ai.data import app as data_wrangling_app


def train_xgb(X, y, xgb_params=None):
    """
    Fits an xgboost model on the data

    Args:
        X(array): training numpy array
        y(array): labels
        xgb_params(dict): params for the xgb model which mostly are from the grid search findings

    Returns:
        xgb: trained model
    """
    if not xgb_params:
        xgb_params = {
            "subsample": 0.8,
            "n_estimators": 500,
            "min_child_weight": 5,
            "max_depth": 6,
            "learning_rate": 0.1,
            "gamma": 2,
            "colsample_bytree": 0.8,
            "alpha": 5,
            "eval_metric": "auc",
            "objective": "binary:logistic",
            "use_label_encoder": False
        }

    # xgb = xgboost.XGBClassifier(**xgb_params)
    # xgb.fit(X, y)
    # return xgb


def app():
    corpus, data_frame = data_wrangling_app()


if __name__ == '__main__':
    app()
