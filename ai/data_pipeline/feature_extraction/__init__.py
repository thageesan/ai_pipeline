from ai.data_pipeline.embed_snippets_biosent import EmbeddedSnippetsBioSent
from ai.data_pipeline.embed_snippets_umlsbert import EmbeddedSnippetsUMLSBert
from shared.tools.os import getenv


def app():
    data_path = getenv('DATA_FOLDER')
    biosent_snippet_file = getenv('EMBED_SNIPPETS_BIOSENT_FILE')
    embedded_biosent_snippets_file_path = f'{data_path}/{biosent_snippet_file}'
    uml_snippet_file = getenv('EMBED_SNIPPETS_UMLSBERT_FILE')
    embedded_uml_snippets_file_path = f'{data_path}/{uml_snippet_file}'
    # columns = ['finding', 'snippet']

    embedded_biosent_snippets = EmbeddedSnippetsBioSent()
    biosent_snippets = embedded_biosent_snippets.load_embedded_snippets(
        embedded_snippets_file_path=embedded_biosent_snippets_file_path)

    embedded_umlsbert_snippets = EmbeddedSnippetsUMLSBert()
    umlsbert_snippets = embedded_umlsbert_snippets.load_embedded_snippets(
        embedded_snippets_file_path=embedded_uml_snippets_file_path)

    # corpus = None
    # samples = None
