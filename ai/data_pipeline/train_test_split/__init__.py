from shared.tools.os import getenv
from shared.tools.utils import pd
from params import TestTrainSetConfig


from sklearn.model_selection import train_test_split


def app():
    data_path = getenv('DATA_FOLDER')
    features_file_name = getenv('FEATURE_EXTRACTION_FILE')

    training_set_file_name = getenv('TRAINING_SET')
    testing_set_file_name = getenv('TESTING_SET')

    data_frame = pd.read_parquet(f'{data_path}/{features_file_name}')
    data_frame_train, data_frame_test = train_test_split(data_frame, test_size=TestTrainSetConfig.TEST_SIZE)

    data_frame_train.to_parquet(f'{data_path}/{training_set_file_name}')

    data_frame_test.to_parquet(f'{data_path}/{testing_set_file_name}')
