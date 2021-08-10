from params import BIOSENT_FILE_NAME
from shared.embed_sentences import SentenceEmbedder
from shared.tools.os import getenv
from shared.tools.utils import pd
from shared.tools.utils.text import clean_sentence, convert_num_to_words

from functools import lru_cache
from json import loads
from numpy import vectorize, array, float32
from sent2vec import Sent2vecModel


def app():
    data_folder = getenv('DATA_FOLDER')
    training_samples_file_name = getenv('TRAINING_SAMPLES_FILE')
    embedded_snippets_file_path = getenv('EMBED_SNIPPETS_BIOSENT_FILE')
    snippet_file_path = f'{data_folder}/{training_samples_file_name}'

    embedded_snippets = EmbeddedSnippetsBioSent()

    model_file_name = BIOSENT_FILE_NAME

    model_file_path = f'{data_folder}/{model_file_name}'
    embedded_snippets.embed_sentences(model_file_path=model_file_path, sentence_file_path=snippet_file_path,
                                      embedded_sentence_file_path=f'{data_folder}/{embedded_snippets_file_path}')


class EmbeddedSnippetsBioSent(SentenceEmbedder):

    def embed_sentences(self, model_file_path, sentence_file_path, embedded_sentence_file_path):
        model = Sent2vecModel()
        model.load_model(model_file_path)

        snippets = pd.read_parquet(sentence_file_path)
        snippets['title'] = vectorize(clean_sentence)(snippets['snippet'])
        snippets['embedded_snippets'] = vectorize(self.compute_embed)(snippets['title'], model)
        snippets = snippets.drop(
            columns=['finding', 'label']
        )
        snippets = snippets.drop_duplicates()
        snippets.to_parquet(embedded_sentence_file_path)

    @lru_cache(maxsize=100)
    def compute_embed(self, string, model):
        string = convert_num_to_words(string)
        return str(model.embed_sentence(string).tolist())

    def load_embedded_sentences(self, embedded_snippets_file_path):
        snippets = pd.read_parquet(embedded_snippets_file_path)
        snippets = snippets.set_index('title').to_dict()['embedded_snippets']

        for key, value in snippets.items():
            string_array = array(loads(value))
            snippets[key] = string_array.astype(float32)
        return snippets
