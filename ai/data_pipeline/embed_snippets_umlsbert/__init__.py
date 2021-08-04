from shared.embed_snippets import EmbeddedSnippets
from shared.tools.os import getenv
from shared.tools.utils import pd
from shared.tools.utils.text import clean_sentence, convert_num_to_words

from functools import lru_cache
from numpy import vectorize, float32, array
from transformers import AutoModel, AutoTokenizer
from json import loads


def app():
    data_folder = getenv('DATA_FOLDER')
    embedded_snippets_file_path = getenv('EMBED_SNIPPETS_UMLSBERT_FILE')
    snippet_file_path = f'{data_folder}/cleaned_snippets_with_org_name_new_rows.csv'

    embedded_snippets = EmbeddedSnippetsUMLSBert()

    model_file_path = f'{data_folder}/UMLSBert'

    embedded_snippets.embed_snippets(model_file_path=model_file_path, snippet_file_path=snippet_file_path,
                                     embedded_snippets_file_path=f'{data_folder}/{embedded_snippets_file_path}')


class EmbeddedSnippetsUMLSBert(EmbeddedSnippets):

    def embed_snippets(self, model_file_path, snippet_file_path, embedded_snippets_file_path):
        tokenizer = AutoTokenizer.from_pretrained(model_file_path)
        tokenizer.add_special_tokens(self.special_tokens_dict)

        model = AutoModel.from_pretrained(model_file_path)

        snippets = pd.read_csv(snippet_file_path)

        snippets['title'] = vectorize(clean_sentence)(snippets['Name'])
        snippets['embedded_snippets'] = vectorize(self.compute_embed)(snippets['title'], tokenizer, model)
        snippets = snippets.drop(
            columns=['Section', 'Organ', 'Name', 'Content', 'Original Name', 'Suggested E-Score']
        )
        snippets = snippets.drop_duplicates()
        snippets.to_parquet(embedded_snippets_file_path)

    @lru_cache(maxsize=100)
    def compute_embed(self, string, tokenizer, model):
        string = convert_num_to_words(string)
        inputs = tokenizer(string, return_tensors='pt')
        return str(model(**inputs).last_hidden_state[0][0].detach().numpy().tolist())

    def load_embedded_snippets(self, embedded_snippets_file_path):
        snippets = pd.read_parquet(embedded_snippets_file_path)
        snippets = snippets.set_index('title').to_dict()['embedded_snippets']

        for key, value in snippets.items():
            string_array = array(loads(value))
            snippets[key] = string_array.astype(float32)
        return snippets
