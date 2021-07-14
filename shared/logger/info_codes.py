from enum import Enum


class EInfoCode(Enum):
    I00000 = 'Logger Initialized'
    I00001 = 'Using cuda? %s'
    I00002 = 'method:  %s ---- %s seconds ----'
    I00003 = 'Using CPU'
    I00004 = 'Using GPU'
