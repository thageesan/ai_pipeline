from abc import ABC, abstractmethod


from shared.tools.utils import pd


class EmbeddedSnippets(ABC):
    special_tokens_dict = {'additional_special_tokens': ['xxx']}

    def __init__(
            self,
            snippet_file_path,
            embedded_snippets_file_path
    ):
        self.snippet_file_path = snippet_file_path
        self.snippets = pd.read_csv(snippet_file_path)

        self.embedded_snippets_file_path = embedded_snippets_file_path

    @abstractmethod
    def embed_snippets(self, model_file_path):
        pass

    @abstractmethod
    def compute_embed(self, **kwargs):
        pass
