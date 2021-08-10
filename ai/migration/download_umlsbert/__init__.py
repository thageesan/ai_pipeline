from params import UMLS_MODEL_NAME

from shared.tools.os import path, getenv
from shared.tools.file import get_file_paths, get_file_name, get_file_extension, create_directory
from shared.google.credentials import get_google_credentials
from shared.google.drive import Drive, MimeType

from transformers import AutoTokenizer, AutoModel, WEIGHTS_NAME, CONFIG_NAME

import torch


def app():
    model_name = UMLS_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    data_path = getenv('DATA_FOLDER')

    model_output_folder = getenv('UMLS_FOLDER_PATH')

    output_dir = f'{data_path}/{model_output_folder}'

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


