from torch import cuda
import pandas as pd
from shared.logger import logger, EInfoCode

use_cuda = cuda.is_available()
logger.info(EInfoCode.I00001.value, use_cuda, extra={'code': EInfoCode.I00001.name})

if use_cuda:
    import cudf


def read_csv(file_path, use_cpu=False):
    if use_cuda and not use_cpu:
        return cudf.read_csv(file_path)
    else:
        logger.info(EInfoCode.I00003.value, extra={'code': EInfoCode.I00003.name})
        return pd.read_csv(file_path)
