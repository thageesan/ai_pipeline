from shared.tools.utils.text import clean_sentence, convert_num_to_words
from shared.tools.utils import pd

from . import SentenceEmbedder

from functools import lru_cache
from json import loads
from numpy import vectorize, float32, array
from transformers import AutoModel, AutoTokenizer


def get_model(model_file_path):
    return AutoModel.from_pretrained(model_file_path)


def get_tokenizer(model_file_path):
    tokenizer = AutoTokenizer.from_pretrained(model_file_path)
    return tokenizer


class UMLSEmbedder(SentenceEmbedder):
    special_tokens_dict = {'additional_special_tokens': ['xxx']}

    def embed_sentences(self, model, tokenizer, sentences, embedded_sentence_file_path):
        tokenizer.add_special_tokens(self.special_tokens_dict)

        data_frame = pd.DataFrame(sentences, columns=['title'])
        data_frame['title'] = vectorize(clean_sentence)(data_frame['title'])
        data_frame['embedded'] = vectorize(self.compute_embed)(data_frame['title'], tokenizer, model)
        data_frame = data_frame.drop_duplicates()
        data_frame.to_parquet(embedded_sentence_file_path)

    @lru_cache(maxsize=100)
    def compute_embed(self, string, tokenizer, model):
        string = convert_num_to_words(string)
        inputs = tokenizer(string, return_tensors='pt')
        return str(model(**inputs).last_hidden_state[0][0].detach().numpy().tolist())

    def load_embedded_sentences(self, embedded_finding_file_path):
        data_frame = pd.read_parquet(embedded_finding_file_path)
        data_frame = data_frame.set_index('title').to_dict()['embedded']

        for key, value in data_frame.items():
            string_array = array(loads(value))
            data_frame[key] = string_array.astype(float32)
        return data_frame
