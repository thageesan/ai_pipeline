from shared.tools.os import getenv
from shared.tools.utils import pd

from joblib import load
from json import dump
from numpy import array
from sklearn.metrics import accuracy_score


def app():
    data_path = getenv('DATA_FOLDER')
    model_file_name = getenv('MODEL_FILE_NAME')
    test_data_file_name = getenv('TESTING_SET')

    metric_folder_path = getenv('METRICS_FOLDER')
    metric_file_name = getenv('TEST_METRIC_FILE_NAME')

    # load model
    model = load(f'{data_path}/{model_file_name}')

    # load test data
    data_frame = pd.read_parquet(f'{data_path}/{test_data_file_name}')

    x_test = array(data_frame['x'].tolist())
    y_test = array(data_frame['y'].tolist())

    y_predict = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_predict)

    with open(f'{metric_folder_path}/{metric_file_name}', 'w') as outfile:
        dump({
            'test_model': {
                'accuracy': accuracy,
                'loss': 1 - accuracy
            }
        }, outfile)
