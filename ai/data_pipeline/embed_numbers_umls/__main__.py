from ai.data_pipeline.embed_numbers_umls import app

from shared.tools.execution_time import calculate_execution_time

if __name__ == '__main__':
    calculate_execution_time(
        method=app,
        label='data wrangling execution time',
    )
