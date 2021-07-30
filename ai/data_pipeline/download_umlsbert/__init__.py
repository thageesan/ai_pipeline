from shared.tools.os import path, getenv
from shared.tools.file import get_file_paths, get_file_name, get_file_extension, create_directory
from shared.google.credentials import get_google_credentials
from shared.google.drive import Drive, MimeType

from transformers import AutoTokenizer, AutoModel, WEIGHTS_NAME, CONFIG_NAME

import torch


def app():
    model_name = getenv('UML_SBERT_MODEL_NAME')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    output_dir = "/tmp/data/umlsbert"

    create_directory(output_dir)

    # If we have a distributed model, save only the encapsulated model
    # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
    model_to_save = model.module if hasattr(model, 'module') else model

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = path.join(output_dir, WEIGHTS_NAME)
    output_config_file = path.join(output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

    file_paths = get_file_paths(output_dir)

    drive_id = getenv('GDRIVE_ID')
    key = getenv('GOOGLE_SERVICE_ACCOUNT_KF_DICT')
    google_credentials = get_google_credentials(key)

    drive = Drive(drive_id=drive_id, credentials=google_credentials)

    for file_path in file_paths:
        file_name = get_file_name(file_path=file_path)
        extension = get_file_extension(file_name=file_name)
        mime_type = None

        if extension == '.json':
            mime_type = MimeType.JSON.value
        elif extension == '.csv':
            mime_type = MimeType.CSV.value
        elif extension == '.bin':
            mime_type = MimeType.MAC_BINARY.value
        elif extension == '.txt':
            mime_type = MimeType.TEXT.value

        if mime_type is not None:
            drive.upload(file_name=file_name, file_path=file_path, mime_type=mime_type, parent_folder='UMLSBert')

