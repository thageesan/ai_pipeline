from ai.data_pipeline.generate_training_samples import app

from shared.tools.execution_time import calculate_execution_time

if __name__ == '__main__':
    calculate_execution_time(
        method=app,
        label='data wrangling execution time',
    )
