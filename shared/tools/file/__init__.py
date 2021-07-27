from shared.logger import logger, EDebugCode
from shared.tools.os import scandir, path, makedirs

from ntpath import basename


def get_file_paths(folder_path):
    """
    Retrieves all the series paths within the specified folder
    :param folder_path:
    :return:
    """
    files_object = {}
    for entry in scandir(folder_path):
        if entry.is_file():
            logger.debug(EDebugCode.D00018.value, folder_path, extra={'code': EDebugCode.D00018.name})
            files_object[str(entry.path)] = True
        elif entry.is_dir():
            # merge dictionary
            files_object.update(get_file_paths(entry.path))
    print(list(files_object.keys()))
    return list(files_object.keys())


def get_file_name(file_path):
    return basename(file_path)


def get_file_extension(file_name):
    _, file_extension = path.splitext(file_name)
    return file_extension


def create_directory(directory_path):
    """
    Creates directory if it does not exist
    :param directory_path:
    :return:
    """
    if not path.isdir(directory_path):
        makedirs(directory_path)
