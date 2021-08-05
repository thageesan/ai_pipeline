from abc import ABC, abstractmethod


class EmbeddedSentences(ABC):
    special_tokens_dict = {'additional_special_tokens': ['xxx']}

    def __init__(self):
        pass

    @abstractmethod
    def embed_sentences(self, model_file_path, sentence_file_path, embedded_sentence_file_path):
        pass

    @abstractmethod
    def compute_embed(self, **kwargs):
        pass

    @abstractmethod
    def load_embedded_sentences(self, embedded_snippets_file_path):
        pass
