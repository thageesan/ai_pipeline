from shared.embed_sentences import SentenceEmbedder
from shared.tools.os import getenv
from shared.tools.utils import pd
from shared.tools.utils.text import clean_sentence, convert_num_to_words

from functools import lru_cache
from numpy import vectorize, float32, array
from transformers import AutoModel, AutoTokenizer
from json import loads


def app():
    data_folder = getenv('DATA_FOLDER')
    training_samples_file_name = getenv('TRAINING_SAMPLES_FILE')
    model_folder_name = getenv('UMLS_FOLDER_PATH')
    embedded_finding_file_path = getenv('EMBED_FINDING_UMLS_FILE')
    training_samples_file_path = f'{data_folder}/{training_samples_file_name}'

    embedded_snippets = EmbeddedFindingUMLSBert()

    model_file_path = f'{data_folder}/{model_folder_name}'

    embedded_snippets.embed_sentences(model_file_path=model_file_path, sentence_file_path=training_samples_file_path,
                                      embedded_sentence_file_path=f'{data_folder}/{embedded_finding_file_path}')


def get_model(model_file_path):
    return AutoModel.from_pretrained(model_file_path)


def get_tokenizer(model_file_path):
    tokenizer = AutoTokenizer.from_pretrained(model_file_path)
    return tokenizer


class EmbeddedFindingUMLSBert(SentenceEmbedder):

    def embed_sentences(self, model_file_path, sentence_file_path, embedded_sentence_file_path):
        tokenizer = get_tokenizer(model_file_path=model_file_path)
        tokenizer.add_special_tokens(self.special_tokens_dict)

        model = get_model(model_file_path=model_file_path)

        snippets = pd.read_parquet(sentence_file_path)

        snippets['title'] = vectorize(clean_sentence)(snippets['finding'])
        snippets['embedded_finding'] = vectorize(self.compute_embed)(snippets['title'], tokenizer, model)
        snippets = snippets.drop(
            columns=['snippet', 'label']
        )
        snippets = snippets.drop_duplicates()
        snippets.to_parquet(embedded_sentence_file_path)

    @lru_cache(maxsize=100)
    def compute_embed(self, string, tokenizer, model):
        string = convert_num_to_words(string)
        inputs = tokenizer(string, return_tensors='pt')
        return str(model(**inputs).last_hidden_state[0][0].detach().numpy().tolist())

    def load_embedded_sentences(self, embedded_finding_file_path):
        snippets = pd.read_parquet(embedded_finding_file_path)
        snippets = snippets.set_index('title').to_dict()['embedded_finding']

        for key, value in snippets.items():
            string_array = array(loads(value))
            snippets[key] = string_array.astype(float32)
        return snippets
