from enum import Enum


class EErrorCode(Enum):
    E00000 = 'Failed to load biosent2vec_embed pickle file. %s'
    E00001 = 'Failed to load umlsbert_embed pickle file. %s'
    E00023 = 'Did not specify GOOGLE_SERVICE_ACCOUNT_KF_DICT environment variable. %s'
