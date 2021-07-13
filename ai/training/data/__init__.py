from shared.tools.data import read_csv
from shared.tools.os import getenv
from shared.tools.execution_time import calculate_execution_time


def app():
    data_folder = '/app/data/'
    csv_file = getenv('CSV_FILE', None)
    calculate_execution_time(method=read_csv, label='GPU execution time', file_path=f'{data_folder}{csv_file}')
    calculate_execution_time(method=read_csv, label='CPU execution time', file_path=f'{data_folder}{csv_file}', use_cpu=True)


if __name__ == '__main__':
    app()
