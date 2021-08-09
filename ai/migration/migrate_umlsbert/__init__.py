from shared.aws.session import Session
from shared.aws.s3 import S3
from shared.google.credentials import get_google_credentials
from shared.google.drive import Drive
from shared.google.drive.mime_type import MimeType
from shared.tools.os import getenv


def app():
    drive_id = getenv('GDRIVE_ID')
    key = getenv('GOOGLE_SERVICE_ACCOUNT_KF_DICT')
    google_credentials = get_google_credentials(key)
    folder_name = getenv('UMLS_FOLDER_PATH')
    save_location = '/tmp'
    bucket_name = getenv('S3_BUCKET')
    bucket_path = getenv('S3_BUCKET_PATH')

    drive = Drive(drive_id=drive_id, credentials=google_credentials)
    folder_id = drive.check_file_exists(file_name=folder_name, mime_type=MimeType.FOLDER.value)
    files = drive.get_files(folder_id=folder_id)
    for file in files:
        if file['mimeType'] is not MimeType.FOLDER.value:
            saved_file_path = drive.download(file_name=file['name'], mime_type=file['mimeType'], save_location=save_location)
            downloaded_file_name = file['name']
            session = Session(aws_secret_access_key=getenv('AWS_SECRET_ACCESS_KEY'), aws_access_key_id=getenv('AWS_ACCESS_KEY_ID'), region_name=getenv('AWS_REGION'))
            s3 = S3(session=session)
            s3.upload(file_path=saved_file_path, bucket_name=bucket_name, storage_path=f'{bucket_path}/{folder_name}/{downloaded_file_name}')

