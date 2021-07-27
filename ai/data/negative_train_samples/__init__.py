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
    file_name = getenv('NEGATIVE_CSV_FILE')
    save_location = '/tmp'
    bucket_name = getenv('S3_BUCKET')
    bucket_path = getenv('S3_BUCKET_PATH')

    drive = Drive(drive_id=drive_id, credentials=google_credentials)
    saved_file_path = drive.download(file_name=file_name, mime_type=MimeType.CSV.value, save_location=save_location)
    session = Session(aws_secret_access_key=getenv('AWS_SECRET_ACCESS_KEY'), aws_access_key_id=getenv('AWS_ACCESS_KEY_ID'), region_name=getenv('AWS_REGION'))
    s3 = S3(session=session)
    s3.upload(file_path=saved_file_path, bucket_name=bucket_name, storage_path=f'{bucket_path}/{file_name}')


if __name__ == '__main__':
    app()
