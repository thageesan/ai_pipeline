from boto3 import Session


class S3:
    __service = None

    def __init__(self, session: Session):
        self.__service = session.resource('s3')

    def upload(self, file_path, bucket_name, storage_path):
        self.__service.meta.client.upload_file(Filename=file_path, Bucket=bucket_name, Key=storage_path)
