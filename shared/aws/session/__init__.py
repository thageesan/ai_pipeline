from boto3 import Session


def get_session(aws_access_key_id, aws_secret_access_key):
    return Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
