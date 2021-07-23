from shared.google.credentials import get_google_credentials
from shared.google.drive import Drive
from shared.google.drive.mime_type import MimeType
from shared.tools.os import getenv, path, getcwd

import os


def app():
    drive_id = getenv('GDRIVE_ID')
    key = os.environ.get('GOOGLE_SERVICE_ACCOUNT_KF_DICT')
    google_credentials = get_google_credentials(key)
    file_name = 'bioSent2Vec.bin'
    save_location = path.join(getcwd(), 'data')
    print(f'save location: {save_location}')

    drive = Drive(drive_id=drive_id, credentials=google_credentials)
    drive.download(file_name=file_name, mime_type=MimeType.MAC_BINARY.value, save_location=save_location)


if __name__ == '__main__':
    app()