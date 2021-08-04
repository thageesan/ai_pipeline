from shared.tools.os import getenv
from shared.tools.utils import pd
from shared.tools.utils.text import clean_sentence, tokenize_sentence

from numpy import array, save


def extract_corpus(snippets, columns):
    """
    Creates a list of tokenized sentences.
    Args:
        snippets:
        columns:

    Returns:
        List of tokenized sentences.
    """
    corpus_list = []
    for _, row in snippets.iterrows():
        docs = [[row[c]] for c in columns]
        for doc in docs:
            cleaned_sentence = clean_sentence(doc[0])
            corpus_list.append(tokenize_sentence(cleaned_sentence))
    return array(corpus_list)


def serialize(corpus, file_path):
    save(file_path, corpus)


def app():
    data_path = getenv('DATA_FOLDER')
    snippet_file = getenv('SNIPPET_CSV_FILE')
    snippet_file_path = f'{data_path}/{snippet_file}'
    snippets = pd.read_csv(snippet_file_path)

    corpus_file = getenv('CORPUS_FILE')
    corpus_file_path = f'{data_path}/{corpus_file}'
    corpus_list = extract_corpus(snippets, ['Name'])

    serialize(corpus=corpus_list, file_path=corpus_file_path)
