import json
from json.decoder import JSONDecodeError

from oauth2client.service_account import ServiceAccountCredentials

from shared.logger import EErrorCode, logger


def get_google_credentials(google_service_account_dict):
    """
    Obtains google credentials by parsing the google service account dict.
    :param google_service_account_dict:
    :return:
    """
    try:
        google_service_account_key_file_dict = json.loads(json.loads(google_service_account_dict))
    except JSONDecodeError as e:
        logger.error(EErrorCode.E00023.value, e, extra={'code': EErrorCode.E00023.name})
        raise
    except TypeError as e:
        logger.error(EErrorCode.E00023.value, e, extra={'code': EErrorCode.E00023.name})
        raise

    return ServiceAccountCredentials.from_json_keyfile_dict(keyfile_dict=google_service_account_key_file_dict)
