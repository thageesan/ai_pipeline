from params import BIOSENT_FILE_NAME

from shared.google.credentials import get_google_credentials
from shared.google.drive import Drive
from shared.google.drive.mime_type import MimeType
from shared.tools.os import getenv


def app():
    drive_id = getenv('GDRIVE_ID')
    key = getenv('GOOGLE_SERVICE_ACCOUNT_KF_DICT')
    google_credentials = get_google_credentials(key)
    file_name = BIOSENT_FILE_NAME
    data_path = getenv('DATA_FOLDER')
    save_location = f'{data_path}'

    drive = Drive(drive_id=drive_id, credentials=google_credentials)
    drive.download(file_name=file_name, mime_type=MimeType.MAC_BINARY.value,
                   save_location=save_location)


if __name__ == '__main__':
    app()
