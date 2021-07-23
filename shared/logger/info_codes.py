from enum import Enum


class EInfoCode(Enum):
    I00000 = 'Logger Initialized'
    I00001 = 'Using cuda? %s'
    I00002 = 'method:  %s ---- %s seconds ----'
    I00003 = 'Using CPU'
    I00004 = 'Using GPU'
    I00005 = 'Could not find biosent2vec_embed_file_path %s'
    I00006 = 'Could not find umlsbert_embed_file_path %s'
    I00007 = 'Google Drive File Download Progress %s'
    I00008 = 'test %s'
