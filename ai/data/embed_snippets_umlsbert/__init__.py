from shared.tools.os import getenv
from shared.tools.data.text import clean_sentence, convert_num_to_words
from shared.tools.data import pd

from functools import lru_cache
from transformers import AutoModel, AutoTokenizer

from numpy import vectorize


def app():
    data_folder = getenv('DATA_FOLDER')
    embedded_snippets_file_path = getenv('EMBED_SNIPPETS_UMLSBERT')
    snippet_file_path = f'{data_folder}/cleaned_snippets_with_org_name_new_rows.csv'
    model_file_path = f'{data_folder}/UMLSBert'
    embedded_snippets = EmbeddedSnippets(
        snippet_file_path=snippet_file_path,
        model_file_path=model_file_path,
        embedded_snippets_file_path=f'{data_folder}/{embedded_snippets_file_path}'
    )

    embedded_snippets.embed_snippets()


class EmbeddedSnippets:
    special_tokens_dict = {'additional_special_tokens': ['xxx']}

    def __init__(
            self,
            snippet_file_path,
            model_file_path,
            embedded_snippets_file_path
    ):
        self.snippet_file_path = snippet_file_path
        self.snippets = pd.read_csv(snippet_file_path)

        self.model_file_path = model_file_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_file_path)
        self.tokenizer.add_special_tokens(self.special_tokens_dict)

        self.model = AutoModel.from_pretrained(model_file_path)

        self.embedded_snippets_file_path = embedded_snippets_file_path

    def embed_snippets(self):
        self.snippets['title'] = vectorize(clean_sentence)(self.snippets['Name'])
        self.snippets['embedded_snippets'] = vectorize(self.compute_embed)(self.snippets['title'])
        self.snippets = self.snippets.drop(columns=['Section', 'Organ', 'Name', 'Content', 'Original Name', 'Suggested E-Score'])
        self.snippets = self.snippets.drop_duplicates()
        self.snippets.to_parquet(self.embedded_snippets_file_path)

    @lru_cache(maxsize=100)
    def compute_embed(self, string):
        string = convert_num_to_words(string)
        inputs = self.tokenizer(string, return_tensors='pt')
        return self.model(**inputs).last_hidden_state[0][0].detach().numpy().tostring()
