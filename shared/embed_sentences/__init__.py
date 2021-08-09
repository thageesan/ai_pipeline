from abc import ABC, abstractmethod


class SentenceEmbedder(ABC):
    special_tokens_dict = {'additional_special_tokens': ['xxx']}

    def __init__(self):
        pass

    @abstractmethod
    def embed_sentences(self, **kwargs):
        pass

    @abstractmethod
    def compute_embed(self, **kwargs):
        pass

    @abstractmethod
    def load_embedded_sentences(self, embedded_snippets_file_path):
        pass
