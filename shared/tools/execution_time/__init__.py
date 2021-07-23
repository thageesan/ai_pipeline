import time

from shared.logger import logger, EInfoCode


def __get_time():
    return time.time()


def calculate_execution_time(method, label: str, **kwargs):
    start_time = __get_time()
    results = method(**kwargs)
    logger.info(EInfoCode.I00002.value, label, (__get_time() - start_time), extra={'code': EInfoCode.I00002.name})
    return results
