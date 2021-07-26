from shared.google.credentials import get_google_credentials
from shared.google.drive import Drive
from shared.google.drive.mime_type import MimeType
from shared.tools.os import getenv, path, getcwd

import os


def app():
    drive_id = getenv('GDRIVE_ID')
    key = os.environ.get('GOOGLE_SERVICE_ACCOUNT_KF_DICT')
    google_credentials = get_google_credentials(key)
    file_name = 'df_positive_w_id_for_train_and_stats.csv'
    save_location = path.join(getcwd(), 'data')

    drive = Drive(drive_id=drive_id, credentials=google_credentials)
    drive.download(file_name=file_name, mime_type=MimeType.CSV.value, save_location=save_location)


if __name__ == '__main__':
    app()
