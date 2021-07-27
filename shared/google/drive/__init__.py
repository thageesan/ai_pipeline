from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

from shared.logger import logger, EInfoCode, EDebugCode
from .mime_type import MimeType

import os
from io import FileIO
from sys import exit

# google has not fixed the socket timeout issue
import socket
socket.setdefaulttimeout(15 * 60)


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

        return save_file_path

    def upload(self, file_name, file_path, mime_type, parent_folder=None):
        """
        Uploads a file to google drive.
        :param file_name:
        :param file_path:
        :param mime_type:
        :param parent_folder:
        :return:
        """
        folder_id = self.check_file_exists(file_name=parent_folder, mime_type=MimeType.FOLDER.value)
        logger.info(EInfoCode.I00029.value, folder_id, extra={'code': EInfoCode.I00029.name})
        if not folder_id:
            folder_id = self.create_folder(folder_name=parent_folder)

        file_metadata = {
            'name': file_name,
            'driveId': self.__drive_id,
            'parents': [self.__drive_id] if parent_folder is None else [folder_id]
        }

        file_id = self.check_file_exists(file_name=file_name, mime_type=mime_type)

        media = MediaFileUpload(file_path, mimetype=mime_type, chunksize=1024*1024, resumable=True)
        if not file_id:
            file = self.__service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id',
                supportsAllDrives=True
            ).execute()
        else:
            file = self.__service.files().update(
                media_body=media,
                fileId=file_id,
                keepRevisionForever=True,
                supportsAllDrives=True
            ).execute()
        logger.info(EInfoCode.I00028.value, file.get('id'), extra={'code': EInfoCode.I00028.name})

        return file.get('id')

    def create_folder(self, folder_name, parent_folder=None):
        """
        Creates a folder on google drive
        :param folder_name:
        :param parent_folder:
        :return:
        """
        if parent_folder is not None:
            parent_folder_id = self.check_file_exists(file_name=parent_folder,
                                                      mime_type=MimeType.FOLDER.value)

        file_metadata = {
            'name': folder_name,
            'driveId': self.__drive_id,
            'mimeType': MimeType.FOLDER.value,
            'parents': [self.__drive_id] if parent_folder is None else [parent_folder_id]
        }

        file = self.__service.files().create(
            body=file_metadata,
            fields='id',
            supportsAllDrives=True
        ).execute()

        folder_id = file.get('id')
        logger.info(EInfoCode.I00027.value, folder_id, extra={'code': EInfoCode.I00027.name})
        return folder_id

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

    def get_file(self, file_id):
        return self.__service.files().get(fileId=file_id, supportsAllDrives=True).execute()
