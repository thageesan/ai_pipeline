from torch import cuda
from shared.logger import logger, EInfoCode
from shared.tools.os import getenv

use_cuda = cuda.is_available()
use_cpu = getenv("USE_CPU", False).capitalize() == 'True'
logger.info(EInfoCode.I00001.value, use_cuda, extra={'code': EInfoCode.I00001.name})

if use_cuda and not use_cpu:
    import cudf as pd
    logger.info(EInfoCode.I00004.value, extra={'code': EInfoCode.I00004.name})
else:
    import pandas as pd
    logger.info(EInfoCode.I00003.value, extra={'code': EInfoCode.I00003.name})


def pandas_series_to_cudf_series(pandas_series):
    return pd.Series.from_pandas(pandas_series)
