from shared.embed_snippets import EmbeddedSnippets
from shared.tools.os import getenv
from shared.tools.utils.text import clean_sentence, convert_num_to_words

from functools import lru_cache
from numpy import vectorize
from transformers import AutoModel, AutoTokenizer


def app():
    data_folder = getenv('DATA_FOLDER')
    embedded_snippets_file_path = getenv('EMBED_SNIPPETS_UMLSBERT_FILE')
    snippet_file_path = f'{data_folder}/cleaned_snippets_with_org_name_new_rows.csv'

    embedded_snippets = EmbeddedSnippetsUMLSBert(
        snippet_file_path=snippet_file_path,
        embedded_snippets_file_path=f'{data_folder}/{embedded_snippets_file_path}'
    )

    model_file_path = f'{data_folder}/UMLSBert'

    embedded_snippets.embed_snippets(model_file_path=model_file_path)


class EmbeddedSnippetsUMLSBert(EmbeddedSnippets):

    def __init__(self, snippet_file_path, embedded_snippets_file_path):
        super().__init__(snippet_file_path, embedded_snippets_file_path)

    def embed_snippets(self, model_file_path):
        tokenizer = AutoTokenizer.from_pretrained(model_file_path)
        tokenizer.add_special_tokens(self.special_tokens_dict)

        model = AutoModel.from_pretrained(model_file_path)

        self.snippets['title'] = vectorize(clean_sentence)(self.snippets['Name'])
        self.snippets['embedded_snippets'] = vectorize(self.compute_embed)(self.snippets['title'], tokenizer, model)
        self.snippets = self.snippets.drop(
            columns=['Section', 'Organ', 'Name', 'Content', 'Original Name', 'Suggested E-Score']
        )
        self.snippets = self.snippets.drop_duplicates()
        self.snippets.to_parquet(self.embedded_snippets_file_path)

    @lru_cache(maxsize=100)
    def compute_embed(self, string, tokenizer, model):
        string = convert_num_to_words(string)
        inputs = tokenizer(string, return_tensors='pt')
        return model(**inputs).last_hidden_state[0][0].detach().numpy().tostring()
