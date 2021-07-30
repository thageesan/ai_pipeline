from shared.embed_snippets import EmbeddedSnippets
from shared.tools.os import getenv
from shared.tools.utils.text import clean_sentence, convert_num_to_words

from functools import lru_cache
from numpy import vectorize
from sent2vec import Sent2vecModel


def app():
    data_folder = getenv('DATA_FOLDER')
    embedded_snippets_file_path = getenv('EMBED_SNIPPETS_BIOSENT_FILE')
    snippet_file_path = f'{data_folder}/cleaned_snippets_with_org_name_new_rows.csv'

    embedded_snippets = EmbeddedSnippetsBioSent(
        snippet_file_path=snippet_file_path,
        embedded_snippets_file_path=f'{data_folder}/{embedded_snippets_file_path}'
    )

    model_file_name = getenv('BIO_SENT_FILE')

    model_file_path = f'{data_folder}/{model_file_name}'
    embedded_snippets.embed_snippets(model_file_path=model_file_path)


class EmbeddedSnippetsBioSent(EmbeddedSnippets):

    def __init__(self, snippet_file_path, embedded_snippets_file_path):
        super().__init__(snippet_file_path, embedded_snippets_file_path)

    def embed_snippets(self, model_file_path):
        model = Sent2vecModel()
        model.load_model(model_file_path)

        self.snippets['title'] = vectorize(clean_sentence)(self.snippets['Name'])
        self.snippets['embedded_snippets'] = vectorize(self.compute_embed)(self.snippets['title'], model)
        self.snippets = self.snippets.drop(
            columns=['Section', 'Organ', 'Name', 'Content', 'Original Name', 'Suggested E-Score']
        )
        self.snippets = self.snippets.drop_duplicates()
        self.snippets.to_parquet(self.embedded_snippets_file_path)

    @lru_cache(maxsize=100)
    def compute_embed(self, string, model):
        string = convert_num_to_words(string)
        return model.embed_sentence(string).tostring()
