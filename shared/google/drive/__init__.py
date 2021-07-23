from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from shared.logger import logger, EInfoCode, EDebugCode
from .mime_type import MimeType

import os
from io import FileIO
from sys import exit


class Drive:
    __service = None
    __drive_id = None

    def __init__(self, credentials, drive_id=None):
        self.__service = build('drive', 'v3', credentials=credentials)
        self.__drive_id = drive_id

    def download(self, file_name, mime_type: MimeType, save_location):
        save_file_path = os.path.join(save_location, file_name)
        fh = FileIO(save_file_path, 'wb')

        file_id = self.check_file_exists(file_name=file_name, mime_type=mime_type)

        request = self.__service.files().get_media(fileId=file_id)

        downloader = MediaIoBaseDownload(fd=fh, request=request, chunksize=20480 * 1024 * 50)

        done = False

        while not done:
            try:
                status, done = downloader.next_chunk()
                logger.info(EInfoCode.I00007.value, str(status.progress() * 100), extra={'code': EInfoCode.I00007.name})
            except Exception:
                fh.close()
                os.remove(save_file_path)
                exit(1)

    def check_file_exists(self, file_name, mime_type):
        """
        Checks to see if file exists.
        :param file_name:
        :param mime_type:
        :return:
        """
        response = self.__service.files().list(
            q=f"name='{file_name}' and mimeType='{mime_type}' and trashed=false",
            driveId=self.__drive_id,
            corpora='drive',
            includeItemsFromAllDrives=True,
            supportsAllDrives=True
        ).execute()

        files = response.get('files', [])

        if len(files) == 0:
            logger.debug(EDebugCode.D00013.value, extra={'code': EDebugCode.D00013.name})
            return False
        else:
            logger.debug(EDebugCode.D00012.value, files[0], extra={'code': EDebugCode.D00012.name})
            # NOTE: this does not consider multiple folders with the same name it grabs the first one
            return files[0]['id']
