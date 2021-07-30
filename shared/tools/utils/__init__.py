import GPUtil
from shared.logger import logger, EInfoCode
from shared.tools.os import getenv

deviceIDs = GPUtil.getAvailable()
gpu_available = False

if len(deviceIDs) > 0:
    gpu_available = True

use_cpu = getenv("USE_CPU", False).capitalize() == 'True'
logger.info(EInfoCode.I00001.value, gpu_available, extra={'code': EInfoCode.I00001.name})

if gpu_available and not use_cpu:
    import cudf as pd
    logger.info(EInfoCode.I00004.value, extra={'code': EInfoCode.I00004.name})
else:
    import pandas as pd
    logger.info(EInfoCode.I00003.value, extra={'code': EInfoCode.I00003.name})


def pandas_series_to_cudf_series(pandas_series):
    return pd.Series.from_pandas(pandas_series)


def check_cudf(data_frame):
    return hasattr(data_frame, 'to_pandas')
