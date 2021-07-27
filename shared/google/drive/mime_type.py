from enum import Enum


class MimeType(Enum):
    MAC_BINARY = 'application/macbinary'
    CSV = 'text/csv'
    FOLDER = 'application/vnd.google-apps.folder'
    JSON = 'application/json'
    TEXT = 'text/plain'
