from shared.tools.utils import pd, check_cudf, pandas_series_to_cudf_series
from shared.tools.utils.text import clean_sentence
from shared.tools.os import getenv


def add_negative_samples(positive_training_samples, rows_needed, columns):
    """
    Takes positive training samples and adds negative training samples to it.
    Args:
        positive_training_samples:
        rows_needed:
        columns:

    Returns:
        positive training sample with additional negative training samples.

    """

    df_sample = positive_training_samples.sample(n=rows_needed).reset_index(drop=True)
    df_mismatch = df_sample.sample(frac=1)

    if check_cudf(df_mismatch):
        # cudf cannot do column string manipulation
        df_mismatch['snippet'] = df_mismatch['snippet'].to_pandas().sample(frac=1).values
    else:
        df_mismatch['snippet'] = df_mismatch['snippet'].sample(frac=1).values

    df_mismatch['label'] = [0] * len(df_mismatch)

    df_temp = pd.concat([df_mismatch, df_sample]).reset_index(drop=True)

    idx = find_anomaly_ids(df_temp, columns)

    df_mismatch = df_mismatch.drop(idx)
    return pd.concat([positive_training_samples, df_mismatch])


def find_anomaly_ids(data_frames, columns):
    """
    This will examine the data frames (based on the columns passed) to see if the shuffling created an anomaly.  If
    detected it will return an array of ids
    :param data_frames:
    :param columns:
    :return:
    """
    data_frame_group_by = data_frames.groupby(list(data_frames[columns]))

    return [x[0] for x in data_frame_group_by.groups.values() if len(x) != 1]


def prepare_data_frame(csv_file_path, columns):
    data_frame = pd.read_csv(csv_file_path)

    for column in columns:
        if hasattr(data_frame[column], 'to_pandas'):
            panda_series = data_frame[column].to_pandas()
            panda_series = panda_series.map(lambda s: clean_sentence(s))
            data_frame[column] = pandas_series_to_cudf_series(pandas_series=panda_series)
        else:
            data_frame[column] = data_frame[column].map(lambda s: clean_sentence(s))

    return data_frame.drop_duplicates(columns).reset_index()


def app():
    # TODO: columns as an env variable?
    columns = ['finding', 'snippet']
    data_path = getenv('DATA_FOLDER')
    positive_training_samples_file = getenv('POSITIVE_CSV_FILE')
    negative_training_samples_file = getenv('NEGATIVE_CSV_FILE')

    positive_training_samples = prepare_data_frame(csv_file_path=f'{data_path}/{positive_training_samples_file}', columns=columns)
    negative_training_samples = prepare_data_frame(csv_file_path=f'{data_path}/{negative_training_samples_file}', columns=columns)

    training_samples_file = getenv('TRAINING_SAMPLES_FILE')
    training_samples_file_path = f'{data_path}/{training_samples_file}'

    rows_needed = len(positive_training_samples) - len(negative_training_samples)

    if rows_needed != 0:
        training_samples = add_negative_samples(positive_training_samples=positive_training_samples, rows_needed=rows_needed, columns=columns)

    training_samples = pd.concat(
        [
            training_samples[['finding', 'snippet', 'label']],
            negative_training_samples[['finding', 'snippet', 'label']]
        ]
    ).reset_index()

    training_samples.to_parquet(training_samples_file_path)
