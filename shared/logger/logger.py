import logging

from shared.tools.os import getenv

from .info_codes import EInfoCode


class Logger:

    def __init__(self):
        self.logger = logging.getLogger(getenv("APP_NAME", None))
        self.logger.setLevel(logging.DEBUG)
        # create a new instance of the stream handler
        console_handler = logging.StreamHandler()
        # create a new instance of the formatter
        formatter = logging.Formatter(
            fmt=f'[{getenv("APP_ENVIRONMENT")} AI_PROJECT %(name)s] [%(code)s] [%(asctime)s] [%(pathname)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S'
        )
        # set the formatter to the handler
        console_handler.setFormatter(formatter)
        # add the handler to the logger
        self.logger.addHandler(console_handler)
        self.logger.info(EInfoCode.I00000.value, extra={'code': EInfoCode.I00000.name})


logger = Logger().logger
