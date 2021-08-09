from shared.embed_sentences.umls_bert_embeddor import UMLSEmbedder, get_model, get_tokenizer

from shared.tools.os import getenv
from shared.tools.utils.text import convert_num_to_words
from shared.tools.utils import pd

from numpy import vectorize
import re


def app():
    data_folder = getenv('DATA_FOLDER')
    training_samples_file_name = getenv('TRAINING_SAMPLES_FILE')
    training_samples_file_path = f'{data_folder}/{training_samples_file_name}'
    modle_path = getenv('UMLS_FOLDER_PATH')
    embedded_numbers_file_name = getenv('EMBED_NUMBERS_FILE')
    embedded_numbers_file_path = f'{data_folder}/{embedded_numbers_file_name}'
    model_file_path = f'{data_folder}/{modle_path}'

    findings_data_frame = pd.read_parquet(training_samples_file_path)

    model = get_model(model_file_path=model_file_path)
    tokenizer = get_tokenizer(model_file_path=model_file_path)

    embedder = UMLSEmbedder()

    list_of_numbers_in_words = []

    def has_numbers(regular_expression, string):
        numbers = re.findall(regular_expression, string)
        if len(numbers) > 0:
            list_of_numbers_in_words.extend([convert_num_to_words(d) for d in numbers])
            return True
        else:
            return False

    # list_of_numbers_in_words is updated by the vectorization below
    findings_data_frame['finding_has_numbers'] = vectorize(has_numbers)(r'\d+\.\d+|\d+', findings_data_frame['finding'])
    findings_data_frame['snippets_has_numbers'] = vectorize(has_numbers)(r'\d+\.\d+|\d+',
                                                                         findings_data_frame['snippet'])

    embedder.embed_sentences(model=model, tokenizer=tokenizer, sentences=list_of_numbers_in_words,
                             embedded_sentence_file_path=embedded_numbers_file_path)
