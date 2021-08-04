from abc import ABC, abstractmethod


class EmbeddedSnippets(ABC):
    special_tokens_dict = {'additional_special_tokens': ['xxx']}

    def __init__(self):
        pass

    @abstractmethod
    def embed_snippets(self, model_file_path, snippet_file_path, embedded_snippets_file_path):
        pass

    @abstractmethod
    def compute_embed(self, **kwargs):
        pass

    @abstractmethod
    def load_embedded_snippets(self, embedded_snippets_file_path):
        pass
